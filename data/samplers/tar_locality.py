"""Deterministic, bounded-block TAR locality without changing a pool's multiset.

Readers return integer (source_id, shard_id, byte_offset) columns.  Only a
bounded key chunk is materialized as Python tuples by the legacy callback.
This module does not open TAR files or change epoch/sample quotas.
"""
from __future__ import annotations

import heapq
import numpy as np


def _key_columns(indices, read_keys_fn, read_columns_fn, chunk_size):
    columns = None
    for start in range(0, len(indices), chunk_size):
        part = indices[start:start + chunk_size]
        if read_columns_fn is not None:
            values = read_columns_fn(part)
            if len(values) != 3:
                raise ValueError("read_columns_fn must return three integer columns")
            values = tuple(np.asarray(value) for value in values)
        else:
            keys = np.asarray(read_keys_fn(part))
            if keys.shape != (len(part), 3):
                raise ValueError("read_keys_fn must return one (source, shard, offset) key per index")
            values = tuple(keys[:, col] for col in range(3))
        for value in values:
            if value.shape != (len(part),) or value.dtype.kind not in "iu":
                raise ValueError("TAR locality keys must be one-dimensional integer columns")
        if columns is None:
            columns = tuple(np.empty(len(indices), dtype=value.dtype) for value in values)
        for dest, value in zip(columns, values):
            if dest.dtype != value.dtype:
                raise ValueError("TAR key column dtypes must be consistent between chunks")
            dest[start:start + len(part)] = value
    return columns


def reorder_for_tar_locality(indices, read_keys_fn=None, rank_batch_size=1,
                             world_size=1, block_samples=128, seed=0, *,
                             read_columns_fn=None, key_chunk_size=65536):
    """Return each input occurrence once, with long same-shard runs per rank.

    Shard-local blocks are shuffled uniformly, then assigned to the currently
    shortest rank lane.  Each lane retains each block's increasing offsets.
    Equal-length prefixes of the lanes are interleaved in rank_batch_size
    chunks, matching the global-batch/rank slicing used by BucketBatchSampler.
    Uneven lane tails are appended without padding or dropping any occurrence.
    Those tails contain fewer than world_size * (block_samples + batch_size)
    items; only that final region loses the rank-lane locality guarantee.

    This is not a uniform random sample permutation: correlations within a
    bounded shard block are intentional.  The input multiset is unchanged.
    """
    for name, value in (("rank_batch_size", rank_batch_size), ("world_size", world_size),
                        ("block_samples", block_samples), ("key_chunk_size", key_chunk_size)):
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rank_batch_size, world_size, block_samples, key_chunk_size = (
        int(rank_batch_size), int(world_size), int(block_samples), int(key_chunk_size))
    if read_columns_fn is None and read_keys_fn is None:
        raise ValueError("TAR locality requires read_keys_fn or read_columns_fn")
    if len(indices) == 0:
        return []
    items = np.asarray(indices)
    if items.ndim != 1 or items.dtype.kind not in "iu":
        raise ValueError("indices must be a one-dimensional integer sequence")
    source, shard, offset = _key_columns(items, read_keys_fn, read_columns_fn, key_chunk_size)
    order = np.lexsort((offset, shard, source))
    source, shard = source[order], shard[order]
    boundaries = np.flatnonzero((source[1:] != source[:-1]) | (shard[1:] != shard[:-1])) + 1
    boundaries = np.concatenate(([0], boundaries, [len(items)]))
    del source, shard, offset
    # Only block metadata is held as Python objects, never one tuple per item.
    blocks = [(start, min(start + block_samples, int(end)))
              for begin, end in zip(boundaries[:-1], boundaries[1:])
              for start in range(int(begin), int(end), block_samples)]
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(blocks))
    lane_blocks = [[] for _ in range(world_size)]
    heap = [(0, rank) for rank in range(world_size)]
    for block_id in permutation:
        length, rank = heapq.heappop(heap)
        begin, end = blocks[int(block_id)]
        lane_blocks[rank].append((begin, end))
        heapq.heappush(heap, (length + end - begin, rank))
    lane_lengths = [0] * world_size
    for length, rank in heap:
        lane_lengths[rank] = length
    common = (min(lane_lengths) // rank_batch_size) * rank_batch_size
    result = np.empty_like(items)
    main = result[:common * world_size].reshape(-1, world_size, rank_batch_size)
    tail = common * world_size
    for rank, (assigned, length) in enumerate(zip(lane_blocks, lane_lengths)):
        lane = np.empty(length, dtype=items.dtype)
        pos = 0
        for begin, end in assigned:
            lane[pos:pos + end - begin] = items[order[begin:end]]
            pos += end - begin
        if common:
            main[:, rank, :] = lane[:common].reshape(-1, rank_batch_size)
        result[tail:tail + length - common] = lane[common:]
        tail += length - common
    assert tail == len(items)
    return result.tolist()
