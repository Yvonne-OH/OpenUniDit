"""Real two-rank Gloo checkpoint metadata round trip with different ready order."""
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import torch
import torch.distributed as dist
from test_ready_data_stream import Dataset,Sampler,collate
from optimizations.ready_data_stream import ReadyDataStream,REQUEST_KEY,gather_data_state,select_data_state


def main():
    torch.set_num_threads(1);dist.init_process_group('gloo')
    rank=dist.get_rank();assert dist.get_world_size()==2
    streams=[]
    try:
        sampler=Sampler(6);sampler.rank=rank;sampler.world_size=2
        stream=ReadyDataStream(Dataset(6),sampler,{'num_workers':0},logging.getLogger('ready-gloo'),collate_fn=collate)
        streams.append(stream)
        order=[1,3] if rank==0 else [2,0]
        stream.iterator=iter([((0,offset),{'values':[offset]},None) for offset in order])
        stream.next_batch();pending=stream.next_batch()
        state=gather_data_state(stream,pending)
        payload=json.loads(json.dumps(state))
        assert payload['rank_cursors'][0]!=payload['rank_cursors'][1]
        gathered=[None,None];dist.all_gather_object(gathered,payload)
        assert gathered[0]==gathered[1]
        selected=select_data_state(payload,stream.signature,rank,2,expected_epoch_length=6)
        restored_sampler=Sampler(6);restored_sampler.rank=rank;restored_sampler.world_size=2
        restored=ReadyDataStream(Dataset(6),restored_sampler,{'num_workers':0},logging.getLogger('ready-gloo'),
            resume_state=selected,signature=stream.signature,collate_fn=collate)
        streams.append(restored)
        requests=[restored.next_batch()[REQUEST_KEY][1] for _ in range(5)]
        assert set(requests)==set(range(6))-{order[0]}
        assert len(requests)==len(set(requests))
        assert restored.state_dict()['batch_offset']==6
        if rank==0:print('PASS real 2-rank ready cursor gather/select/pending undo/replay; metadata bytes=',len(json.dumps(payload)),flush=True)
    finally:
        for stream in streams:stream.close()
        dist.destroy_process_group()


if __name__=='__main__':main()
