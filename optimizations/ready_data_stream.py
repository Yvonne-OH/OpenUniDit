"""Opt-in ready-first DataLoader with exact, per-rank request cursors.

Only batches returned by next_batch are consumed. Worker-prefetched requests
remain replayable. Unknown timeout requests are never discarded. Checkpoints
must call gather_data_state on every rank, and restore via select_data_state.
Ready-first changes short-term sample order; it cannot fix insufficient average
I/O supply. A worker error is a tagged, consumed request, not an unknown timeout.
"""
from __future__ import annotations

from collections import deque
import hashlib
import inspect
import json
import time

import torch
import torch.distributed as dist

REQUEST_KEY = "__ready_request__"


def sampler_signature(sampler):
    # len() may initialize data-proportional weights; match training's ordering.
    steps = len(sampler)
    spec = {key: getattr(sampler, key, None) for key in
        ("seed", "world_size", "batch_size", "bucket_batch_sizes", "weights",
         "randomize_order_on_start", "max_repeat_per_epoch")}
    spec["bucket_counts"] = {key: len(value) for key, value in getattr(sampler, "bucket_indices", {}).items()}
    spec["steps_per_epoch"] = steps
    if getattr(sampler, "tar_locality_order", None) is not None:
        spec["tar_locality_order"] = sampler.tar_locality_order
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


class ConsumptionCursor:
    """Consumed prefix plus a compact interval set of out-of-order requests."""

    def __init__(self, epoch, length, frontier=0, ranges=()):
        self.epoch, self.length, self.frontier = int(epoch), int(length), int(frontier)
        self.ranges = [list(map(int, pair)) for pair in ranges]
        if self.length <= 0 or self.epoch < 0 or not 0 <= self.frontier <= self.length:
            raise ValueError("Invalid ready-data cursor frontier")
        previous = self.frontier
        for start, end in self.ranges:
            if not previous < start < end <= self.length:
                raise ValueError("Ready-data consumed ranges must be sorted, separated and beyond frontier")
            previous = end

    def contains(self, offset):
        return offset < self.frontier or any(start <= offset < end for start, end in self.ranges)

    def consume(self, offset):
        offset = int(offset)
        if not 0 <= offset < self.length or self.contains(offset):
            raise ValueError(f"Duplicate or invalid ready request {(self.epoch, offset)}")
        intervals = sorted([*self.ranges, [offset, offset+1]])
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        if merged and merged[0][0] == self.frontier:
            self.frontier = merged.pop(0)[1]
        self.ranges = merged

    def unconsume(self, offset):
        offset = int(offset)
        if not 0 <= offset < self.length or not self.contains(offset):
            raise ValueError(f"Cannot undo an unconsumed ready request {(self.epoch, offset)}")
        if offset < self.frontier:
            old_frontier, self.frontier = self.frontier, offset
            if offset+1 < old_frontier:
                self.ranges.insert(0, [offset+1, old_frontier])
            return
        ranges = []
        for start, end in self.ranges:
            if start <= offset < end:
                if start < offset:
                    ranges.append([start, offset])
                if offset+1 < end:
                    ranges.append([offset+1, end])
            else:
                ranges.append([start, end])
        self.ranges = ranges

    def state_dict(self):
        return {"epoch": self.epoch, "batch_offset": self.frontier,
                "consumed_ranges": [pair[:] for pair in self.ranges]}


def _error(exc):
    category = getattr(exc, "category", None)
    if not category:
        try:
            from data.datasets.video_decode import media_error_category
            category = media_error_category(exc)
        except Exception:
            category = "executor_bug"
    return {"category": str(category), "message": f"{type(exc).__name__}: {exc}"}


class TaggedDataError(RuntimeError):
    def __init__(self, tag, error):
        self.request_tag = tuple(tag)
        self.category = error["category"]
        super().__init__(f"Ready request {self.request_tag}: {error['message']}")


class _TaggedDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, request):
        epoch, offset, index = request
        try:
            return (epoch, offset), self.data[index], None
        except Exception as exc:
            return (epoch, offset), None, _error(exc)


class _TaggedCollate:
    def __init__(self, collate_fn):
        self.collate_fn = collate_fn

    def __call__(self, items):
        tag = items[0][0]
        if any(item[0] != tag for item in items):
            return tag, None, {"category": "executor_bug", "message": "Mixed request tags in one batch"}
        errors = [error for _, _, error in items if error is not None]
        if errors:
            error = next((value for value in errors if value["category"] == "executor_bug"), errors[0])
            return tag, None, error
        try:
            return tag, self.collate_fn([item for _, item, _ in items]), None
        except Exception as exc:
            return tag, None, _error(exc)


class _TaggedSampler:
    def __init__(self, sampler, cursor):
        self.sampler, self.cursor = sampler, cursor

    def __iter__(self):
        # Snapshot the cursor at iterator creation; parent consumption must not
        # mutate the identities of already-dispatched worker requests.
        epoch, frontier = self.cursor.epoch, self.cursor.frontier
        excluded = [pair[:] for pair in self.cursor.ranges]
        self.sampler.set_start_batch(frontier, epoch=epoch)
        for offset, indices in enumerate(self.sampler, start=frontier):
            if any(start <= offset < end for start, end in excluded):
                continue
            yield [(epoch, offset, index) for index in indices]

    def __len__(self):
        return self.cursor.length-self.cursor.frontier-sum(end-start for start, end in self.cursor.ranges)


def _request_tag(value):
    if isinstance(value, dict):
        value = value.get(REQUEST_KEY)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("A pending ready batch must carry its exact request tag")
    return tuple(map(int, value))


class ReadyDataStream:
    ready_first = True

    def __init__(self, data, sampler, training_cfg, logger, epoch=0, batch_offset=0,
                 resume_state=None, signature=None, collate_fn=None):
        self.data, self.sampler, self.training_cfg, self.logger = data, sampler, training_cfg, logger
        self.signature = signature or sampler_signature(sampler)
        self._future = {}
        state = resume_state
        if state is not None:
            if state.get("sampler_signature") != self.signature:
                raise ValueError("Resolve incompatible data state with select_data_state before constructing stream")
            if state.get("epoch_length", len(sampler)) != len(sampler):
                raise ValueError("Saved ready-data epoch length differs from current sampler")
            epoch, batch_offset = int(state["epoch"]), int(state["batch_offset"])
            for future in state.get("future_epochs", ()):
                cursor = ConsumptionCursor(future["epoch"], len(sampler), future["batch_offset"], future.get("consumed_ranges", ()))
                if cursor.epoch <= epoch or cursor.epoch in self._future:
                    raise ValueError("Invalid saved future ready-data epoch")
                self._future[cursor.epoch] = cursor
        self.cursor = ConsumptionCursor(epoch, len(sampler), batch_offset,
            state.get("consumed_ranges", ()) if state else ())
        if collate_fn is None:
            from data.lmdb_dataset import collate_image_batch
            collate_fn = collate_image_batch
        self._collate = _TaggedCollate(collate_fn)
        self._tagged_sampler = _TaggedSampler(sampler, self.cursor)
        self.loader = self.iterator = None
        self._warm_buffer = deque()
        self.last_fetch_seconds = 0.
        self.last_request = None
        self._start()

    @property
    def epoch(self):
        return self.last_request[0] if self.last_request is not None else self.cursor.epoch

    @property
    def batch_offset(self):
        # Training uses this property for per-request CFG dropout RNG, so it
        # follows the delivered tag. Serialized batch_offset is the frontier.
        return self.last_request[1]+1 if self.last_request is not None else self.cursor.frontier

    def _advance_epoch(self):
        while self.cursor.frontier == self.cursor.length:
            epoch = self.cursor.epoch+1
            self.cursor = self._future.pop(epoch, ConsumptionCursor(epoch, len(self.sampler)))
        self._tagged_sampler.cursor = self.cursor

    def _start(self):
        if "in_order" not in inspect.signature(torch.utils.data.DataLoader).parameters:
            raise RuntimeError("ready_data_pipeline requires DataLoader(in_order=False) support")
        self._advance_epoch()
        workers = int(self.training_cfg.get("num_workers", 4))
        kwargs = dict(batch_sampler=self._tagged_sampler, collate_fn=self._collate,
                      num_workers=workers, in_order=False)
        if workers:
            kwargs.update(prefetch_factor=int(self.training_cfg.get("dataloader_prefetch_factor", self.training_cfg.get("prefetch_factor", 4))),
                persistent_workers=True, timeout=float(self.training_cfg.get("dataloader_timeout_seconds", 180)))
        self.loader = torch.utils.data.DataLoader(_TaggedDataset(self.data), **kwargs)
        self.iterator = iter(self.loader)

    def next_batch(self):
        if self.cursor.frontier == self.cursor.length:
            self._advance_epoch()
            self.iterator = iter(self.loader)
        # A timeout has no known tag: leave the cursor intact, so restart replays
        # all outstanding requests while excluding every consumed request.
        tag, batch, error = self._warm_buffer.popleft() if self._warm_buffer else next(self.iterator)
        tag = _request_tag(tag)
        if tag[0] != self.cursor.epoch:
            raise RuntimeError(f"Stale ready-data epoch: got {tag}, expected {self.cursor.epoch}")
        self.cursor.consume(tag[1])
        self.last_request = tag
        if error is not None:
            raise TaggedDataError(tag, error)
        if not isinstance(batch, dict):
            raise TypeError("ReadyDataStream collate_fn must return a mapping")
        batch[REQUEST_KEY] = tag
        return batch

    def defer_batch(self, batch):
        """Retry a healthy rank's batch when another rank could not join."""
        tag = _request_tag(batch)
        if tag != self.last_request or tag[0] != self.cursor.epoch:
            raise ValueError(f"Cannot defer a different ready request: {tag}")
        self.cursor.unconsume(tag[1])
        self._warm_buffer.appendleft((tag, batch, None))

    def state_dict(self, pending_batch=False, unconsumed_batches=()):
        tags = [_request_tag(batch) for batch in unconsumed_batches]
        if pending_batch is True:
            if self.last_request is None:
                raise ValueError("Pending batch has no delivered ready request")
            tags.append(self.last_request)
        elif pending_batch is not False and pending_batch is not None:
            tags.append(_request_tag(pending_batch))
        states = {self.cursor.epoch: self.cursor.state_dict(),
                  **{epoch: cursor.state_dict() for epoch, cursor in self._future.items()}}
        if tags:
            first = min(epoch for epoch, _ in tags)
            for epoch in range(first, self.cursor.epoch):
                states.setdefault(epoch, {"epoch": epoch, "batch_offset": len(self.sampler), "consumed_ranges": []})
            for epoch, offset in set(tags):
                if epoch not in states:
                    raise ValueError(f"Pending ready tag has not been delivered: {(epoch, offset)}")
                state = states[epoch]
                cursor = ConsumptionCursor(epoch, len(self.sampler), state["batch_offset"], state["consumed_ranges"])
                cursor.unconsume(offset)
                states[epoch] = cursor.state_dict()
        first = min(states)
        result = {**states.pop(first), "version": 2, "stream_kind": "ready_first",
                "sampler_signature": self.signature, "epoch_length": len(self.sampler),
                "future_epochs": [states[epoch] for epoch in sorted(states)]}
        if getattr(self.sampler, "tar_locality_order", None) is not None:
            result["tar_locality_order"] = dict(self.sampler.tar_locality_order)
        if getattr(self, "tar_usage_reference", None) is not None:
            result["tar_usage_reference"] = self.tar_usage_reference
        return result

    def wait_ready(self, min_batches, timeout_seconds):
        """Verify decoded payloads without marking any request consumed.

        Raw DataLoader completions are moved into a bounded warm buffer; queue
        size alone cannot distinguish successful payloads from error sentinels.
        Buffered requests are absent from consumed state and replay on restart.
        """
        target = int(min_batches)
        if target <= 0 or timeout_seconds < 0:
            raise ValueError("Invalid ready queue target/deadline")
        queue = getattr(self.iterator, "_data_queue", None)
        if queue is None or not callable(getattr(queue, "qsize", None)):
            raise RuntimeError("Ready queue qsize is unavailable; cannot verify RAM readiness")
        capacity = int(self.training_cfg.get("num_workers", 4))*int(self.training_cfg.get("dataloader_prefetch_factor", self.training_cfg.get("prefetch_factor", 4)))
        remaining = len(self._tagged_sampler)
        if target > capacity:
            raise ValueError(f"Ready queue target {target} exceeds available prefetch capacity {capacity}")
        target = min(target, remaining)
        errors = sum(error is not None for _, _, error in self._warm_buffer)
        valid = len(self._warm_buffer)-errors
        seen = {_request_tag(tag) for tag, _, _ in self._warm_buffer}
        error_limit = int(self.training_cfg.get("ready_max_prefill_errors", 64))
        if error_limit <= 0:
            raise ValueError("ready_max_prefill_errors must be positive")
        started = time.monotonic()
        exhausted = False
        while True:
            try:
                queued = int(queue.qsize())
            except (NotImplementedError, OSError, ValueError) as exc:
                raise RuntimeError("Ready queue qsize failed; cannot verify RAM readiness") from exc
            elapsed = time.monotonic()-started
            if valid >= target or exhausted or elapsed >= timeout_seconds:
                return {"ready": valid >= target, "ready_batches": valid,
                        "valid_ready_batches": valid, "error_batches": errors,
                        "buffered_batches": len(self._warm_buffer), "exhausted": exhausted,
                        "target_batches": target, "elapsed_seconds": elapsed,
                        "prefetch_capacity": capacity}
            if queued:
                try:
                    raw = next(self.iterator)
                except StopIteration:
                    exhausted = True
                    continue
                tag, batch, error = raw
                tag = _request_tag(tag)
                if tag[0] != self.cursor.epoch or not 0 <= tag[1] < self.cursor.length or self.cursor.contains(tag[1]) or tag in seen:
                    raise RuntimeError(f"Invalid or duplicate ready prefill request: {tag}")
                if error is not None:
                    if error["category"] == "executor_bug":
                        raise TaggedDataError(tag, error)
                    errors += 1
                    if errors > error_limit:
                        raise RuntimeError(f"Ready prefill exceeded {error_limit} failed requests; no valid RAM watermark")
                elif not isinstance(batch, dict) or not batch:
                    raise TypeError("Ready prefill collate_fn must return a nonempty batch mapping")
                else:
                    valid += 1
                seen.add(tag)
                self._warm_buffer.append(raw)
                # Removing one completion permits the DataLoader to dispatch
                # another request. Neither action advances the training cursor.
                continue
            time.sleep(min(.05, max(0., timeout_seconds-elapsed)))

    def close(self):
        iterator, loader = self.iterator, self.loader
        self.iterator = self.loader = None
        self._warm_buffer.clear()
        if iterator is None:
            return
        workers = list(getattr(iterator, "_workers", ()))
        if not workers:
            return
        import psutil
        from torch.utils.data._utils import signal_handling
        iterator._shutdown = True
        if getattr(iterator, "_worker_pids_set", False):
            signal_handling._remove_worker_pids(id(iterator))
            iterator._worker_pids_set = False
        for worker in workers:
            try:
                for child in reversed(psutil.Process(worker.pid).children(recursive=True)):
                    child.kill()
            except psutil.NoSuchProcess:
                pass
        for worker in workers:
            if worker.is_alive():
                worker.kill()
        deadline = time.monotonic()+5.
        for worker in workers:
            worker.join(timeout=max(0., deadline-time.monotonic()))
        for queue in [*iterator._index_queues, iterator._worker_result_queue]:
            queue.cancel_join_thread()
            queue.close()
        loader._iterator = None
        alive = [worker.pid for worker in workers if worker.is_alive()]
        if alive:
            raise RuntimeError(f"ReadyDataStream workers did not exit: {alive}")

    def restart(self):
        self.close()
        self._start()
        self.logger.warning("ReadyDataStream workers rebuilt: epoch=%d frontier=%d ranges=%s",
            self.cursor.epoch, self.cursor.frontier, self.cursor.ranges)


def gather_data_state(stream, pending_batch=None, unconsumed_batches=()):
    """Call on every rank at checkpoints, before rank-zero pure-weight metadata."""
    if not getattr(stream, "ready_first", False):
        if unconsumed_batches:
            raise ValueError("Legacy stream cannot undo an external ready buffer")
        return stream.state_dict(pending_batch=bool(pending_batch))
    local = stream.state_dict(pending_batch, unconsumed_batches)
    world = dist.get_world_size() if dist.is_initialized() else 1
    cursors = [None]*world
    if dist.is_initialized():
        dist.all_gather_object(cursors, local)
    else:
        cursors[0] = local
    return {"version": 2, "stream_kind": "ready_first_distributed",
            "world_size": world, "rank_cursors": cursors}


def select_data_state(saved, expected_signature, rank, world_size, expected_epoch_length=None):
    """Select an exact local cursor; None means main must warn and estimate."""
    if saved is None:
        return None
    if not isinstance(saved, dict):
        raise ValueError("Malformed saved ready-data state")
    state = saved
    if "rank_cursors" in state:
        cursors = state["rank_cursors"]
        if not isinstance(cursors, (list, tuple)):
            raise ValueError("Malformed rank cursor list")
        if not isinstance(state.get("world_size"), int) or state["world_size"] <= 0:
            raise ValueError("Malformed saved ready world size")
        if state.get("world_size") != world_size:
            return None
        if len(cursors) != world_size or not 0 <= rank < len(cursors):
            raise ValueError("Rank cursor count does not match saved world size")
        state = cursors[rank]
    if not isinstance(state, dict):
        raise ValueError("Malformed per-rank ready-data state")
    if state.get("stream_kind") != "ready_first" or state.get("version") != 2:
        return None
    if not isinstance(state.get("sampler_signature"), str):
        raise ValueError("Missing ready-data sampler signature")
    if state.get("sampler_signature") != expected_signature:
        return None
    try:
        length = int(state["epoch_length"])
        if length <= 0:
            raise ValueError("Nonpositive ready-data epoch length")
        if expected_epoch_length is not None and length != expected_epoch_length:
            return None
        ConsumptionCursor(state["epoch"], length, state["batch_offset"], state.get("consumed_ranges", ()))
        last_epoch = int(state["epoch"])
        for cursor in state.get("future_epochs", ()):
            if int(cursor["epoch"]) != last_epoch+1:
                raise ValueError("Missing, unsorted or duplicate future epoch")
            ConsumptionCursor(cursor["epoch"], length, cursor["batch_offset"], cursor.get("consumed_ranges", ()))
            last_epoch = int(cursor["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Corrupt ready-data cursor: {exc}") from exc
    return state
