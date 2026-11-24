"""Exact request accounting with real out-of-order workers and restarts."""
import json
import logging
import random
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import optimizations
optimizations.__path__ = [str(ROOT/'optimizations'), *list(optimizations.__path__)]
from optimizations.ready_data_stream import (ConsumptionCursor, ReadyDataStream,
    REQUEST_KEY, TaggedDataError, gather_data_state, sampler_signature, select_data_state)


class Sampler:
    seed=42;world_size=1;rank=0;batch_size=1
    def __init__(self, length):
        self.length=length;self.start=0;self.epoch=0
    def __len__(self):return self.length
    def set_start_batch(self, offset, epoch=0):self.start=offset;self.epoch=epoch
    def __iter__(self):
        for index in range(self.start,self.length):yield [index]


class Dataset:
    def __init__(self,length=12,slow=None,bad=None):
        self.length,self.slow,self.bad=length,slow or {},bad or {}
    def __len__(self):return self.length
    def __getitem__(self,index):
        time.sleep(self.slow.get(index,0))
        if index in self.bad:
            exc=ValueError('bad request '+str(index));exc.category=self.bad[index];raise exc
        return index


def collate(items):return {'values':items}


class ReadyStreamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.logger=logging.getLogger('ready-data-tests')
        self.streams=[]
    def tearDown(self):
        for stream in self.streams:stream.close()
    def stream(self,data=None,state=None,workers=0,timeout=5):
        data=data or Dataset()
        stream=ReadyDataStream(data,Sampler(len(data)),
            {'num_workers':workers,'prefetch_factor':2,'dataloader_timeout_seconds':timeout},
            self.logger,resume_state=state,collate_fn=collate)
        self.streams.append(stream);return stream

    def test_interval_cursor_and_exact_pending_undo(self):
        cursor=ConsumptionCursor(0,1000000)
        for value in [2,3,1]:cursor.consume(value)
        self.assertEqual(cursor.frontier,0);self.assertEqual(cursor.ranges,[[1,4]])
        cursor.consume(0)
        self.assertEqual(cursor.frontier,4);self.assertEqual(cursor.ranges,[])
        cursor.unconsume(1)
        self.assertEqual(cursor.frontier,1);self.assertEqual(cursor.ranges,[[2,4]])
        cursor.consume(1);self.assertEqual(cursor.frontier,4)
        with self.assertRaises(ValueError):cursor.consume(1)

    def test_cursor_random_consumption_undo_and_serialization(self):
        rng=random.Random(42);length=60;consumed=set();cursor=ConsumptionCursor(3,length)
        for _ in range(500):
            if consumed and rng.random()<.3:
                value=rng.choice(sorted(consumed));consumed.remove(value);cursor.unconsume(value)
            elif len(consumed)<length:
                value=rng.choice(sorted(set(range(length))-consumed));consumed.add(value);cursor.consume(value)
            self.assertEqual({value for value in range(length) if cursor.contains(value)},consumed)
            expected=next((value for value in range(length) if value not in consumed),length)
            self.assertEqual(cursor.frontier,expected)
            state=cursor.state_dict()
            cursor=ConsumptionCursor(state['epoch'],length,state['batch_offset'],state['consumed_ranges'])

    def test_actual_workers_ready_first_and_unconsumed_prefetch_replayed(self):
        stream=self.stream(Dataset(12,{0:.6}),workers=2)
        before=stream.state_dict()
        ready=stream.wait_ready(2,4)
        self.assertTrue(ready['ready'])
        self.assertEqual(stream.state_dict(),before)
        first=stream.next_batch()
        self.assertNotEqual(first[REQUEST_KEY],(0,0))
        self.assertEqual(stream.batch_offset,first[REQUEST_KEY][1]+1)
        self.assertEqual(stream.epoch,first[REQUEST_KEY][0])
        consumed={first[REQUEST_KEY][1]}
        second=stream.next_batch();consumed.add(second[REQUEST_KEY][1])
        state=json.loads(json.dumps(stream.state_dict()))
        stream.close()
        resumed=self.stream(Dataset(12),state=state)
        replayed=[resumed.next_batch()[REQUEST_KEY][1] for _ in range(12-len(consumed))]
        self.assertEqual(set(replayed),set(range(12))-consumed)
        self.assertEqual(len(replayed),len(set(replayed)))

    def test_deferred_batch_replays_in_ram_and_across_restart(self):
        stream = self.stream(Dataset(4))
        batch = stream.next_batch()
        stream.defer_batch(batch)
        self.assertEqual(stream.state_dict()["batch_offset"], 0)
        self.assertIs(stream.next_batch(), batch)
        stream.defer_batch(batch)
        stream.restart()
        self.assertEqual([stream.next_batch()[REQUEST_KEY] for _ in range(4)],
                         [(0, i) for i in range(4)])

    def test_pending_tag_undo_after_frontier_jump(self):
        stream=self.stream(Dataset(5))
        # Inject real tagged completions out of order to make frontier jump.
        stream.iterator=iter([((0,1),{'values':[1]},None),((0,2),{'values':[2]},None),
                              ((0,0),{'values':[0]},None)])
        stream.next_batch();stream.next_batch();pending=stream.next_batch()
        self.assertEqual(stream.cursor.frontier,3)
        state=stream.state_dict(pending)
        self.assertEqual(state['batch_offset'],0);self.assertEqual(state['consumed_ranges'],[[1,3]])
        self.assertEqual(stream.cursor.frontier,3)
        resumed=self.stream(Dataset(5),state=state)
        self.assertEqual([resumed.next_batch()['values'][0] for _ in range(3)],[0,3,4])

    def test_pending_older_epoch_and_future_consumed_requests(self):
        stream=self.stream(Dataset(2))
        stream.next_batch();pending=stream.next_batch()
        later=stream.next_batch()
        self.assertEqual(later[REQUEST_KEY],(1,0))
        state=stream.state_dict(pending)
        self.assertEqual(state['epoch'],0);self.assertEqual(state['batch_offset'],1)
        self.assertEqual(state['future_epochs'][0]['batch_offset'],1)
        resumed=self.stream(Dataset(2),state=state)
        self.assertEqual(resumed.next_batch()[REQUEST_KEY],(0,1))
        self.assertEqual(resumed.next_batch()[REQUEST_KEY],(1,1))
        self.assertEqual(resumed.next_batch()[REQUEST_KEY],(2,0))

    def test_tagged_worker_error_is_consumed_and_fatal_category_preserved(self):
        for category in ['corrupt','executor_bug']:
            stream=self.stream(Dataset(3,bad={0:category}),workers=1)
            with self.assertRaises(TaggedDataError) as caught:stream.next_batch()
            self.assertEqual(caught.exception.category,category)
            self.assertEqual(caught.exception.request_tag,(0,0))
            self.assertEqual(stream.state_dict()['batch_offset'],1)
            self.assertEqual(stream.next_batch()[REQUEST_KEY],(0,1))
            stream.close()

    def test_actual_timeout_rebuild_replays_unknown_but_not_consumed(self):
        stream=self.stream(Dataset(4,{0:2.,2:2.}),workers=2,timeout=.5)
        first=stream.next_batch()
        self.assertEqual(first[REQUEST_KEY],(0,1))
        # Request3 can also be ready; consume all completions before the timeout.
        consumed={1}
        while True:
            try:consumed.add(stream.next_batch()[REQUEST_KEY][1])
            except RuntimeError as exc:
                self.assertIn('DataLoader timed out',str(exc));break
        before=stream.state_dict()
        stream.data.slow={}
        stream.restart()
        self.assertEqual(stream.state_dict(),before)
        remaining=[stream.next_batch()[REQUEST_KEY][1] for _ in range(4-len(consumed))]
        self.assertEqual(set(remaining),set(range(4))-consumed)

    def test_per_rank_gather_selection_and_legacy_default(self):
        stream=self.stream(Dataset(4))
        stream.next_batch();local=stream.state_dict()
        remote={**local,'batch_offset':0,'consumed_ranges':[[2,3]]}
        def gather(output,value):output[:]=[value,remote]
        with patch('optimizations.ready_data_stream.dist.is_initialized',return_value=True),\
             patch('optimizations.ready_data_stream.dist.get_world_size',return_value=2),\
             patch('optimizations.ready_data_stream.dist.all_gather_object',side_effect=gather):
            state=gather_data_state(stream)
        self.assertEqual(state['world_size'],2)
        self.assertEqual(select_data_state(state,stream.signature,0,2),local)
        self.assertEqual(select_data_state(state,stream.signature,1,2),remote)
        self.assertIsNone(select_data_state(state,stream.signature,0,8))
        self.assertIsNone(select_data_state(state,'changed',0,2))
        self.assertIsNone(select_data_state(state,stream.signature,0,2,expected_epoch_length=99))
        for broken in [dict(local,batch_offset=-1),dict(local,batch_offset=10),
                       dict(local,consumed_ranges=[[2,4],[3,4]])]:
            with self.assertRaises(ValueError):select_data_state(broken,stream.signature,0,1)
        class Legacy:
            def state_dict(self,pending_batch=False):return {'old':True,'pending':pending_batch}
        self.assertEqual(gather_data_state(Legacy()),{'old':True,'pending':False})

    def test_ready_qsize_unavailable_is_explicit(self):
        stream=self.stream()
        with self.assertRaisesRegex(RuntimeError,'qsize is unavailable'):stream.wait_ready(1,0)
        class BadQueue:
            def qsize(self):raise NotImplementedError
        class Iterator:
            _data_queue=BadQueue()
        stream.iterator=Iterator()
        stream.training_cfg['num_workers']=1
        with self.assertRaisesRegex(RuntimeError,'qsize failed'):stream.wait_ready(1,0)

    def test_ready_watermark_near_epoch_end_and_prefetch_config_alias(self):
        sampler=Sampler(6)
        stream=ReadyDataStream(Dataset(6),sampler,{'num_workers':2,
            'dataloader_prefetch_factor':3,'prefetch_factor':1},self.logger,
            batch_offset=4,collate_fn=collate)
        self.streams.append(stream)
        result=stream.wait_ready(6,4)
        self.assertTrue(result['ready'])
        self.assertEqual(result['prefetch_capacity'],6)
        self.assertEqual(result['target_batches'],2)
        self.assertEqual(stream.state_dict()['batch_offset'],4)
        self.assertEqual({stream.next_batch()[REQUEST_KEY][1] for _ in range(2)},{4,5})
        with self.assertRaisesRegex(ValueError,'exceeds available prefetch'):stream.wait_ready(7,0)

    def test_all_error_completions_do_not_pass_ready_watermark(self):
        stream=self.stream(Dataset(5,bad={index:'corrupt' for index in range(5)}),workers=2)
        state=stream.state_dict()
        result=stream.wait_ready(2,.3)
        self.assertFalse(result['ready'])
        self.assertEqual(result['valid_ready_batches'],0)
        self.assertGreater(result['error_batches'],0)
        self.assertEqual(stream.state_dict(),state)
        self.assertIsNone(stream.last_request)
        with self.assertRaises(TaggedDataError):stream.next_batch()
        self.assertEqual(stream.state_dict()['batch_offset']+sum(end-start for start,end in stream.cursor.ranges),1)

    def test_mixed_errors_count_only_valid_payloads_and_keep_error_for_skip(self):
        stream=self.stream(Dataset(8,bad={0:'corrupt',1:'invalid_input'}),workers=1)
        state=stream.state_dict()
        result=stream.wait_ready(2,4)
        self.assertTrue(result['ready'])
        self.assertEqual(result['valid_ready_batches'],2)
        self.assertEqual(result['error_batches'],2)
        self.assertEqual(result['buffered_batches'],4)
        self.assertEqual(stream.state_dict(),state)
        self.assertIsNone(stream.last_request)
        for expected in [0,1]:
            with self.assertRaises(TaggedDataError) as caught:stream.next_batch()
            self.assertEqual(caught.exception.request_tag,(0,expected))
        self.assertEqual(stream.next_batch()[REQUEST_KEY],(0,2))

    def test_prefill_restart_replays_all_unconsumed_warm_payloads(self):
        stream=self.stream(Dataset(6),workers=2)
        state=stream.state_dict()
        self.assertTrue(stream.wait_ready(3,4)['ready'])
        self.assertEqual(len(stream._warm_buffer),3)
        self.assertEqual(stream.state_dict(),state)
        stream.restart()
        self.assertFalse(stream._warm_buffer)
        self.assertEqual(stream.state_dict(),state)
        received=[stream.next_batch()[REQUEST_KEY][1] for _ in range(6)]
        self.assertEqual(set(received),set(range(6)))
        self.assertEqual(len(received),len(set(received)))

    def test_prefill_fatal_error_and_bad_request_limit_are_explicit(self):
        fatal=self.stream(Dataset(3,bad={0:'executor_bug'}),workers=1)
        with self.assertRaises(TaggedDataError) as caught:fatal.wait_ready(1,4)
        self.assertEqual(caught.exception.category,'executor_bug')
        self.assertEqual(fatal.state_dict()['batch_offset'],0)
        limited=self.stream(Dataset(5,bad={index:'corrupt' for index in range(5)}),workers=1)
        limited.training_cfg['ready_max_prefill_errors']=2
        with self.assertRaisesRegex(RuntimeError,'exceeded 2 failed requests'):limited.wait_ready(1,4)
        self.assertEqual(limited.state_dict()['batch_offset'],0)


if __name__=='__main__':unittest.main()
