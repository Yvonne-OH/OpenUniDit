import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.samplers.bucket_sampler import BucketBatchSampler
from optimizations.ready_data_stream import sampler_signature, ReadyDataStream, select_data_state
from optimizations.tar_locality_state import locality_activation, saved_tar_order


def sampler(rank, policy=None):
    obj = BucketBatchSampler({'x': list(range(1024))}, 4, seed=42, rank=rank,
                             world_size=2, drop_last=True,
                             sort_keys_fn=lambda ids: [(0, i//128, i%128) for i in ids])
    if policy:
        obj.configure_tar_locality(policy)
    return obj


class ResumeTests(unittest.TestCase):
    def test_activation_accounts_for_out_of_order_and_future(self):
        saved = {'rank_cursors': [
            {'epoch': 0, 'batch_offset': 5, 'consumed_ranges': [[8, 10]]},
            {'epoch': 0, 'batch_offset': 7, 'future_epochs': [
                {'epoch': 1, 'batch_offset': 2, 'consumed_ranges': [[4, 6]]}]}]}
        self.assertEqual(locality_activation(saved, 0, 5, 128), (1, 6))
        self.assertEqual(locality_activation(None, 0, 128, 128), (1, 0))

    def test_policy_disagreement_is_rejected(self):
        with self.assertRaises(ValueError):
            saved_tar_order({'rank_cursors': [{'tar_locality_order': {'version': 1}}, {}]})

    def test_unconsumed_suffix_and_consumed_holes_keep_identity(self):
        states = [{'epoch': 0, 'batch_offset': 11, 'consumed_ranges': [[14, 17]]},
                  {'epoch': 0, 'batch_offset': 13, 'consumed_ranges': [[19, 21]]}]
        epoch, activation = locality_activation({'rank_cursors': states}, 0, 11, 128)
        policy = dict(version=1, block_samples=32, activation_epoch=epoch, activation_batch=activation)
        old_global, new_global = [], []
        for rank in range(2):
            old = list(sampler(rank))
            new_sampler = sampler(rank, policy)
            new = list(new_sampler)
            self.assertEqual(old[:activation], new[:activation])
            old_global.extend(v for batch in old[activation:] for v in batch)
            new_global.extend(v for batch in new[activation:] for v in batch)
            # Subsequent checkpoint resumes the same transitioned permutation.
            new_sampler.set_start_batch(activation+9, epoch=0)
            self.assertEqual(list(new_sampler), new[activation+9:])
        self.assertEqual(sorted(old_global), sorted(new_global))

    def test_training_and_ready_signatures_match(self):
        tree = ast.parse((ROOT/'scripts/train/train_giga_exp.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sampler_signature')
        ns = dict(hashlib=hashlib, json=json)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<signature>', 'exec'), ns)
        obj = sampler(0)
        legacy = sampler_signature(obj)
        self.assertEqual(legacy, ns['sampler_signature'](obj))
        obj.configure_tar_locality(dict(version=1, block_samples=32, activation_epoch=0, activation_batch=21))
        self.assertNotEqual(legacy, sampler_signature(obj))
        self.assertEqual(sampler_signature(obj), ns['sampler_signature'](obj))

    def test_new_cursor_keeps_policy_and_usage_reference(self):
        from optimizations.ready_data_stream import ConsumptionCursor
        obj = sampler(0, dict(version=1, block_samples=32, activation_epoch=0, activation_batch=21))
        stream = ReadyDataStream.__new__(ReadyDataStream)
        stream.cursor = ConsumptionCursor(0, len(obj), 30, [[32, 34]])
        stream._future = {}
        stream.sampler, stream.signature = obj, sampler_signature(obj)
        stream.tar_usage_reference = {'path': '/example/immutable-counts.json'}
        state = stream.state_dict()
        self.assertEqual(saved_tar_order(state), obj.tar_locality_order)
        self.assertEqual(select_data_state(state, stream.signature, 0, 2, len(obj)), state)
        self.assertEqual(state['tar_usage_reference'], stream.tar_usage_reference)


if __name__ == '__main__':
    unittest.main()
