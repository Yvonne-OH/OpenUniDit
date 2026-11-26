"""Rank-local accounting of TAR samples actually received by training.

Call ``record_batch`` exactly once at the training consumption boundary, never
from a dataset worker or a prefetch callback. Counts describe URI occurrences
(including repeats), not unique samples or successful optimizer updates.
Locality describes the TAR-only *consumption* sequence, not physical I/O order.
Only ``save`` and ``load`` access the filesystem; neither opens a source TAR.
"""

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import uuid
from collections.abc import Mapping


TAR_URIS_KEY = "__tar_uris__"
_VERSION = 1
_POSITION = re.compile(r"[0-9]+:[0-9]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COUNTERS = ("consumed_samples", "batches", "last_offset",
             "adjacent_pairs", "increasing_offset_pairs")


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _tar_location(value):
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise TypeError("TAR metadata entries must be strings, paths, or None")
    if not value.startswith("tar://"):
        return None
    path, separator, position = value[6:].rpartition("/")
    if not path or not separator or not _POSITION.fullmatch(position):
        raise ValueError(f"Malformed TAR URI: {value!r}")
    offset, _size = position.split(":")
    # Preserve the original TAR path; do not resolve or stat source files.
    return path, int(offset)


def _atomic_write(path, content):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class TarUsageTracker:
    """One mutable tracker per training rank; not intended for worker sharing."""

    def __init__(self):
        self.consumed_samples = 0
        self.batches = 0
        self._last_tar_path = None
        self._tars = {}

    def record_batch(self, batch):
        """Count this consumed batch and return its number of TAR entries.

        ``__tar_uris__``, when present, is authoritative (even when empty).
        Otherwise use ``image_paths``. Accept a scalar path or a flat list/tuple;
        non-TAR paths and None are ignored. Invalid TAR URIs raise before any
        counters change. Each TAR's batch count increments at most once per call.
        Calling this twice counts two consumptions; it does not deduplicate.
        """
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        values = batch[TAR_URIS_KEY] if TAR_URIS_KEY in batch else batch.get("image_paths")
        if values is None:
            return 0
        if isinstance(values, (str, os.PathLike)):
            values = (values,)
        if not isinstance(values, (list, tuple)):
            raise TypeError("TAR metadata must be a scalar path or flat list/tuple")
        locations = [_tar_location(value) for value in values]
        locations = [location for location in locations if location is not None]
        if not locations:
            return 0
        seen = set()
        for path, offset in locations:
            counters = self._tars.setdefault(path, {
                "consumed_samples": 0, "batches": 0, "last_offset": 0,
                "adjacent_pairs": 0, "increasing_offset_pairs": 0,
            })
            if self._last_tar_path == path:
                counters["adjacent_pairs"] += 1
                counters["increasing_offset_pairs"] += int(offset > counters["last_offset"])
            counters["consumed_samples"] += 1
            counters["last_offset"] = offset
            if path not in seen:
                counters["batches"] += 1
                seen.add(path)
            self._last_tar_path = path
        self.consumed_samples += len(locations)
        self.batches += 1
        return len(locations)

    def state_dict(self):
        """Return independent, JSON-compatible counters without any file I/O.

        ``batches`` counts batches containing TAR entries. Per-TAR batch counts
        overlap when a batch contains several TARs and must not be summed as a
        global batch count. Adjacency ignores intervening non-TAR entries.
        """
        return {
            "version": _VERSION,
            "consumed_samples": self.consumed_samples,
            "batches": self.batches,
            "last_tar_path": self._last_tar_path,
            "tars": {path: dict(counts) for path, counts in sorted(self._tars.items())},
        }

    def save(self, output_dir, step, rank):
        """Atomically write unique step/rank JSON and TSV checkpoint sidecars.

        Return a small reference suitable for the main checkpoint. JSON is the
        authoritative restore state; TSV is a derived human-readable report.
        Unique suffixes prevent separate saves/processes overwriting each other.
        """
        _nonnegative_int(step, "step")
        _nonnegative_int(rank, "rank")
        directory = Path(output_dir).expanduser().absolute()
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"tar-usage.step-{step:012d}.rank-{rank:06d}.{uuid.uuid4().hex}"
        json_path = directory / f"{stem}.json"
        tsv_path = directory / f"{stem}.tsv"
        payload = self.state_dict()
        payload.update(step=step, rank=rank)
        content = (json.dumps(payload, ensure_ascii=True, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        report = io.StringIO(newline="")
        writer = csv.writer(report, delimiter="\t", lineterminator="\n")
        writer.writerow(("step", "rank", "tar_path", *_COUNTERS, "increasing_offset_ratio"))
        for path, counts in payload["tars"].items():
            pairs = counts["adjacent_pairs"]
            ratio = counts["increasing_offset_pairs"] / pairs if pairs else ""
            writer.writerow((step, rank, path, *(counts[key] for key in _COUNTERS), ratio))
        try:
            _atomic_write(tsv_path, report.getvalue().encode("utf-8"))
            _atomic_write(json_path, content)
        except BaseException:
            for path in (json_path, tsv_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise
        return {
            "path": str(json_path), "sha256": hashlib.sha256(content).hexdigest(),
            "tsv_path": str(tsv_path), "step": step, "rank": rank,
        }

    @classmethod
    def load(cls, reference):
        """Verify the checkpoint reference and restore all consumption counters.

        Missing, corrupt, or incompatible state raises; never silently reset
        history. Restoring must be paired with the same checkpoint's data cursor.
        """
        if not isinstance(reference, Mapping):
            raise TypeError("reference must be a mapping returned by save")
        expected_hash = reference.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            raise ValueError("reference.sha256 must contain 64 lowercase hex digits")
        path = reference.get("path")
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("reference.path is required")
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("TAR usage snapshot SHA-256 mismatch")
        payload = json.loads(content)
        if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != _VERSION:
            raise ValueError("Unsupported TAR usage snapshot version")
        for key in ("step", "rank", "consumed_samples", "batches"):
            _nonnegative_int(payload.get(key), key)
        for key in ("step", "rank"):
            if key in reference:
                _nonnegative_int(reference[key], f"reference.{key}")
                if reference[key] != payload[key]:
                    raise ValueError(f"TAR usage snapshot {key} mismatch")
        tars = payload.get("tars")
        if not isinstance(tars, dict):
            raise ValueError("Snapshot tars must be a mapping")
        for tar_path, counts in tars.items():
            if not isinstance(tar_path, str) or not tar_path or not isinstance(counts, dict):
                raise ValueError("Invalid TAR path/counters in snapshot")
            for key in _COUNTERS:
                _nonnegative_int(counts.get(key), f"{tar_path}.{key}")
            if not 1 <= counts["batches"] <= counts["consumed_samples"]:
                raise ValueError("Invalid per-TAR batch/sample counts")
            if not 0 <= counts["increasing_offset_pairs"] <= counts["adjacent_pairs"] < counts["consumed_samples"]:
                raise ValueError("Invalid per-TAR adjacency counts")
        total = payload["consumed_samples"]
        batches = payload["batches"]
        if sum(row["consumed_samples"] for row in tars.values()) != total:
            raise ValueError("TAR usage sample totals disagree")
        if (not tars and (total != 0 or batches != 0 or payload.get("last_tar_path") is not None)):
            raise ValueError("Invalid empty TAR usage snapshot")
        if tars:
            if not max(row["batches"] for row in tars.values()) <= batches <= min(total, sum(row["batches"] for row in tars.values())):
                raise ValueError("TAR usage batch totals disagree")
            if payload.get("last_tar_path") not in tars:
                raise ValueError("Last TAR path is absent from snapshot")
        tracker = cls()
        tracker.consumed_samples = total
        tracker.batches = batches
        tracker._last_tar_path = payload.get("last_tar_path")
        tracker._tars = {tar_path: {key: counts[key] for key in _COUNTERS}
                         for tar_path, counts in tars.items()}
        return tracker
