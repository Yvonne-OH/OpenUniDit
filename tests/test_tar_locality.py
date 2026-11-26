"""CPU-only coverage, rank locality, and sampler migration regression checks."""
import ast
from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.samplers.tar_locality import reorder_for_tar_locality


def _load_sampler():
    try:
        from data.samplers.bucket_sampler import BucketBatchSampler
        return BucketBatchSampler
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
    # The algorithm never calls its torch Sampler base.  Let the same CPU-only
    # tests run on development hosts without installing the training runtime.
    path = ROOT / "data/samplers/bucket_sampler.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "torch.utils.data")]
    class Sampler:
        def __class_getitem__(cls, _):
            return cls
    namespace = {"Sampler": Sampler, "__name__": "tar_locality_sampler_test"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["BucketBatchSampler"]


BucketBatchSampler = _load_sampler()


def columns(indices):
    ids = np.asarray(indices, dtype=np.int64)
    return (ids // 8192).astype(np.uint16), (ids // 128).astype(np.uint32), (ids % 128) * 4096


def keys(indices):
    return list(zip(*(column.tolist() for column in columns(indices))))


class TarLocalityTests(unittest.TestCase):
    def test_preserves_every_occurrence_including_irregular_tails(self):
        rng = np.random.default_rng(99)
        for n in (0, 1, 7, 19, 127, 301, 1231):
            indices = rng.integers(0, 16000, n).tolist()
            for ranks in (1, 2, 9, 72):
                actual = reorder_for_tar_locality(indices, keys, 6, ranks, 35, 44)
                self.assertEqual(Counter(actual), Counter(indices), (n, ranks))

    def test_column_and_legacy_callbacks_match_and_are_bounded(self):
        ids = np.random.default_rng(8).permutation(1009)
        calls = []
        def vector(part):
            calls.append(len(part))
            return columns(part)
        vector_result = reorder_for_tar_locality(ids, rank_batch_size=6, world_size=9,
            block_samples=64, seed=11, read_columns_fn=vector, key_chunk_size=37)
        legacy_result = reorder_for_tar_locality(ids, keys, 6, 9, 64, 11, key_chunk_size=41)
        self.assertEqual(vector_result, legacy_result)
        self.assertEqual(sum(calls), len(ids))
        self.assertLessEqual(max(calls), 37)

    def test_seed_replay_and_seed_change(self):
        ids = list(range(8192))
        a = reorder_for_tar_locality(ids, keys, 8, 8, 64, 5)
        self.assertEqual(a, reorder_for_tar_locality(ids, keys, 8, 8, 64, 5))
        self.assertNotEqual(a, reorder_for_tar_locality(ids, keys, 8, 8, 64, 6))

    def test_each_rank_reads_whole_sorted_blocks(self):
        ids = np.random.default_rng(21).permutation(32768)
        result = np.asarray(reorder_for_tar_locality(ids, keys, 8, 8, 64, 9))
        global_batches = result.reshape(-1, 8, 8)
        for rank in range(8):
            stream = global_batches[:, rank, :].reshape(-1)
            for block in stream.reshape(-1, 64):
                self.assertEqual(len(set((block // 128).tolist())), 1)
                self.assertTrue(np.all(np.diff(block) > 0))

    def test_input_is_not_mutated_and_source_is_part_of_key(self):
        ids = list(range(24))
        saved = ids.copy()
        def source_keys(part):
            a = np.asarray(part)
            return a // 8, np.zeros(len(a), dtype=np.int64), a % 8
        out = reorder_for_tar_locality(ids, world_size=1, block_samples=8,
                                       read_columns_fn=source_keys)
        self.assertEqual(ids, saved)
        for block in np.asarray(out).reshape(-1, 8):
            self.assertEqual(len(set((block // 8).tolist())), 1)

    def test_invalid_keys_and_settings_fail(self):
        for kwargs in ({"world_size": 0}, {"block_samples": -1}, {"rank_batch_size": 1.5},
                       {"key_chunk_size": True}):
            with self.assertRaises(ValueError):
                reorder_for_tar_locality([1, 2], keys, **kwargs)
        with self.assertRaises(ValueError):
            reorder_for_tar_locality([1, 2], lambda _: [(0, 0, 1)])
        with self.assertRaises(ValueError):
            reorder_for_tar_locality([1, 2], lambda _: [(0, 0, 1.2), (0, 0, 2.3)])


class SamplerTransitionTests(unittest.TestCase):
    def sampler(self, rank=0, world=2, policy=None, epoch=0):
        sampler = BucketBatchSampler({"512_square": list(range(4096))}, 8,
            rank=rank, world_size=world, seed=42, drop_last=True, sort_keys_fn=keys)
        if policy is not None:
            sampler.configure_tar_locality(policy, read_columns_fn=columns)
        sampler.set_epoch(epoch)
        return sampler

    def test_disabled_policy_keeps_legacy_sequence(self):
        sampler = self.sampler()
        expected = list(sampler)
        sampler.configure_tar_locality(None)
        self.assertEqual(list(sampler), expected)
        # Fixed legacy first batch catches accidental changes to RNG consumption.
        self.assertEqual(expected[0], [49, 114, 380, 1188, 1273, 1830, 2675, 3964])

    def test_transition_preserves_prefix_and_global_suffix_multiset(self):
        boundary = 19
        policy = {"version": 1, "block_samples": 64,
                  "activation_epoch": 0, "activation_batch": boundary}
        old_suffix, new_suffix = [], []
        all_new = []
        for rank in range(2):
            old = list(self.sampler(rank))
            new = list(self.sampler(rank, policy=policy))
            self.assertEqual(new[:boundary], old[:boundary])
            self.assertEqual(len(old), len(new))
            old_suffix.extend(x for batch in old[boundary:] for x in batch)
            new_suffix.extend(x for batch in new[boundary:] for x in batch)
            all_new.extend(x for batch in new for x in batch)
        self.assertEqual(Counter(old_suffix), Counter(new_suffix))
        self.assertEqual(Counter(all_new), Counter(range(4096)))

    def test_resume_is_exact_suffix_and_epoch_activation_is_explicit(self):
        policy = {"block_samples": 64, "activation_epoch": 2, "activation_batch": 9}
        self.assertEqual(list(self.sampler(policy=policy, epoch=1)), list(self.sampler(epoch=1)))
        for epoch in (2, 3):
            full = list(self.sampler(policy=policy, epoch=epoch))
            for start in (0, 5, 9, 27, len(full)):
                sampler = self.sampler(policy=policy, epoch=epoch)
                sampler.set_start_batch(start, epoch=epoch)
                self.assertEqual(list(sampler), full[start:])
        self.assertNotEqual(list(self.sampler(policy=policy, epoch=2)),
                            list(self.sampler(policy=policy, epoch=3)))

    def test_activation_gives_each_rank_long_local_runs(self):
        boundary = 19
        for rank in range(8):
            sampler = BucketBatchSampler({"512_square": list(range(32768))}, 8,
                rank=rank, world_size=8, seed=42, drop_last=True, sort_keys_fn=keys)
            sampler.configure_tar_locality({"block_samples": 64, "activation_batch": boundary}, columns)
            suffix = np.asarray(list(sampler)[boundary:]).reshape(-1)
            shards = suffix // 128
            cuts = np.flatnonzero(shards[1:] != shards[:-1]) + 1
            lengths = np.diff(np.r_[0, cuts, len(shards)])
            self.assertGreaterEqual(float(np.median(lengths)), 48)

    def test_multibucket_counts_and_prefix_are_preserved(self):
        pools = {"256_square": list(range(1024)), "512_square": list(range(1024, 2048))}
        def make(rank, enabled):
            sampler = BucketBatchSampler(pools, 8, rank=rank, world_size=2,
                seed=17, drop_last=True, sort_keys_fn=keys)
            if enabled:
                sampler.configure_tar_locality({"block_samples": 64, "activation_batch": 11}, columns)
            return list(sampler)
        old, new = [], []
        for rank in range(2):
            before, after = make(rank, False), make(rank, True)
            self.assertEqual(before[:11], after[:11])
            old.extend(x for batch in before for x in batch)
            new.extend(x for batch in after for x in batch)
        self.assertEqual(Counter(old), Counter(new))

    def test_batch_sort_uses_one_callback_and_preserves_donor_targets(self):
        calls = []
        sampler = self.sampler()
        def counted(indices):
            calls.append(list(indices))
            return keys(indices)
        sampler.sort_keys_fn = counted
        batch = [(30, "256_square"), 3, (19, "256_square")]
        self.assertEqual(sampler._sort_batch(batch), [3, (19, "256_square"), (30, "256_square")])
        self.assertEqual(calls, [[30, 3, 19]])


if __name__ == "__main__":
    unittest.main()
