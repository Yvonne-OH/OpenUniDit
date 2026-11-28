"""GPU distributed equivalence: uneven batches, GAS, clipping, exact resume, eval.
Run: torchrun --standalone --nproc-per-node=4 tests/test_fsdp2_backend_distributed.py
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
import torch.distributed as dist
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from optimizations.distributed_backend import FSDP2Engine, backend_name
from optimizations.training_profiler import merge_intervals, duration, overlap

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(16, 32),torch.nn.SiLU(),torch.nn.Linear(32,16)) for _ in range(2)])
        self.out = torch.nn.Linear(16, 32)
        self.activation_checkpointing = True
    def forward(self,x):
        from torch.utils.checkpoint import checkpoint
        for b in self.blocks:
            x = x + (checkpoint(b,x,use_reentrant=False) if self.training and self.activation_checkpointing else b(x))
        return self.out(x)

def batch(rank, micro, step):
    gen=torch.Generator(device='cuda').manual_seed(4000+step*100+rank*7+micro)
    b=rank%2+1+micro
    return torch.randn(b,16,device='cuda',generator=gen),torch.randn(b,32,device='cuda',generator=gen)

def update(engine, reference, optimizer, step):
    world=dist.get_world_size();rank=dist.get_rank()
    count=sum(batch(r,m,step)[0].shape[0] for r in range(world) for m in range(2))
    for m in range(2):
        x,y=batch(rank,m,step)
        loss=(engine(x)-y).square().mean(1).sum()*(2*world/count)
        engine.backward(loss);engine.step()
        for r in range(world):
            x,y=batch(r,m,step)
            ((reference(x)-y).square().mean(1).sum()/count).backward()
    norm=torch.nn.utils.clip_grad_norm_(reference.parameters(),.1,foreach=False)
    torch.testing.assert_close(engine.get_global_grad_norm(),norm,rtol=2e-5,atol=2e-6)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    full=engine.full_weights()
    if rank==0:
        for key,value in reference.state_dict().items():torch.testing.assert_close(full[key],value.cpu(),rtol=2e-5,atol=2e-6)

def main():
    torch.set_num_threads(1);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size()
    assert backend_name({})=='deepspeed'
    assert duration([(0,4),(2,6)])==6 and overlap([(0,4)],[(2,8)])==2
    paths=[tempfile.mkdtemp(prefix='giga-fsdp2-test-',dir=os.environ.get('GIGA_TEST_SHARED_TMP')) if rank==0 else None];dist.broadcast_object_list(paths,src=0)
    for name in ('fsdp2','hsdp'):
        for sync, reshard in ((True,True),(False,True),(True,False),(False,False)):
            cfg={'distributed':{'backend':name,'fsdp2':{'shard_size':int(os.environ.get('GIGA_TEST_SHARD_SIZE',2)),'gradient_clipping':.1,'sync_each_micro':sync,'reshard_after_forward':reshard}},
                 'training':{'gradient_accumulation_steps':2,'learning_rate':.003,'muon':{'enabled':False},'adam':{'weight_decay':.005}}}
            torch.manual_seed(42);model=Model().cuda();reference=copy.deepcopy(model)
            optimizer=torch.optim.AdamW(reference.parameters(),lr=.003,betas=(.9,.95),weight_decay=.005,foreach=False)
            engine=FSDP2Engine(model,cfg,torch.float32)
            for step in range(2):update(engine,reference,optimizer,step)
            engine.save_checkpoint(paths[0],f'{name}-{sync}-{reshard}',{'step':2,'samples_seen':123,'data_cursor':{'batch':4}})
            full_before=engine.full_weights()
            with engine.validation_context(),torch.no_grad():
                engine.eval()
                # Unequal forward counts must not invoke collectives/deadlock.
                for i in range(rank+1):
                    x,_=batch(rank,0,9)
                    torch.testing.assert_close(engine(x),reference.eval()(x),rtol=2e-5,atol=2e-6)
                engine.train();reference.train()
            for p in engine.module.parameters():
                with torch.no_grad():p.add_(1)
            _,meta=engine.load_checkpoint(paths[0],f'{name}-{sync}-{reshard}')
            assert meta['data_cursor']=={'batch':4} and engine.global_steps==2
            update(engine,reference,optimizer,2)
            # Pure-weight import must be collective-safe and maintain FP32 local shards.
            weights={k:v.to(torch.bfloat16) for k,v in reference.cpu().state_dict().items()}
            expected_weights={k:v.float().cpu().clone() for k,v in weights.items()}
            engine.load_full_weights(weights);reference.cuda()
            restored=engine.full_weights()
            if rank==0:
                for k,v in expected_weights.items():torch.testing.assert_close(restored[k],v,rtol=0,atol=0)
            assert all(p.dtype==torch.float32 for p in engine.module.parameters())
            if rank==0:print('PASS',name,'sync_each_micro',sync,'reshard_after_forward',reshard,flush=True)
            del engine,reference,optimizer,model;torch.cuda.empty_cache();dist.barrier()
    if rank==0:print('PASS FSDP2/HSDP numerical, clipping, accumulation, resume and uneven validation',flush=True)
    dist.destroy_process_group()
if __name__=='__main__':main()
