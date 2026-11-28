"""CPU two-rank test of the production while loop, ready data and checkpoints.

Run with torchrun --standalone --nproc-per-node=2. ToyEngine replaces only
DeepSpeed hardware, using real DDP and AdamW. This does not replace DS/GPU
validation. No production dataset, GPU, tracker service or output is used.
"""
from __future__ import annotations
import ast
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid
from contextlib import ExitStack, nullcontext
from contextlib import nullcontext as validation_context
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import optimizations
optimizations.__path__=[str(ROOT/'optimizations'),*list(optimizations.__path__)]
from optimizations.ready_data_stream import (ReadyDataStream, REQUEST_KEY,
    select_data_state)
from optimizations.mhc_monitoring import TrainingMonitor
from optimizations.sigma_monitoring import SigmaMonitor

SOURCE = Path(os.environ.get('GIGA_TRAIN_SOURCE', ROOT/'scripts/train/train_giga_exp.py'))
TREE = ast.parse(SOURCE.read_text())


class DataError(RuntimeError):
    category = 'corrupt'


class MixedSampler:
    seed=42; batch_size=1; bucket_batch_sizes={'small':2}; weights=None
    randomize_order_on_start=True; max_repeat_per_epoch=None
    def __init__(self, rank, length=20):
        self.rank, self.world_size, self.length = rank, 2, length
        self.start_batch, self.epoch = 0, 0
    def __len__(self): return self.length
    def set_start_batch(self, offset, epoch=0):
        self.start_batch, self.epoch = offset, epoch
    def __iter__(self):
        for offset in range(self.start_batch, self.length):
            yield [offset*2+index for index in range(1+(offset//2)%2)]


class DelayedDataset:
    def __init__(self, rank): self.rank=rank
    def __len__(self): return 40
    def __getitem__(self, index):
        offset, item = divmod(index, 2)
        delay = .008
        if item == 0 and self.rank == 0 and offset == 0:
            delay += 1.
        elif item == 0 and self.rank == 1 and offset == 1:
            delay += .08
        time.sleep(delay)
        if self.rank == 1 and offset == 4:
            raise DataError('one rank-local corrupt request')
        return {'pixel_values': np.ones((1,1,1,1), np.float32),
                'caption': f'rank{self.rank}-request{offset}-sample{item}'}


def collate(items):
    return {'pixel_values': np.stack([item['pixel_values'] for item in items]),
            'captions': [item['caption'] for item in items]}


class ToyModel(torch.nn.Module):
    patch_size=(1,1,1)
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.1))
    def forward(self, value, timestep, text, mask): return value*self.weight
    def semantic_metadata(self): return {'cpu_ready_loop_test': True}


class ToyEngine:
    def __init__(self, model, optimizer, rank):
        self.module, self.optimizer, self.rank = model, optimizer, rank
        self.ddp = DistributedDataParallel(model)
        self.global_steps = self.micro_steps = 0
        self.forward_losses, self.checkpoints = [], []
        self.text_ids = []
        self.giga_ready_pipeline_enabled = True
        self._norm=0.
        self.expected_grad_sum = 0.
        self.sample_count = 0
    def __call__(self, value, timestep, text, mask):
        self.text_ids.append(text[:, 0, 0].tolist())
        prediction=self.ddp(value,timestep,text,mask)
        # Sigma is fixed at .75; the identity VAE returns unit latents.
        noise=(value.detach().float()-.25)/.75
        self.forward_losses.append((prediction.detach().float()-(noise-1)).square().flatten(1).mean(1).tolist())
        self.expected_grad_sum += float((2 * value * (prediction.detach() - (noise - 1))).sum())
        self.sample_count += len(value)
        return prediction
    def backward(self, loss): (loss/2).backward()
    def step(self):
        self.micro_steps += 1
        if self.micro_steps%2 == 0:
            reference = torch.tensor([self.expected_grad_sum, self.sample_count])
            dist.all_reduce(reference)
            torch.testing.assert_close(self.module.weight.grad, reference[0] / reference[1])
            self.expected_grad_sum = 0.
            self.sample_count = 0
            self._norm=float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in self.module.parameters()])))
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
            self.global_steps += 1
    def get_global_grad_norm(self): return self._norm
    def save_checkpoint(self, directory, tag, client_state):
        path=Path(directory)/tag;path.mkdir(parents=True,exist_ok=True)
        torch.save({'module':self.module.state_dict(),'optimizer':self.optimizer.state_dict(),
                    'client_state':client_state},path/f'rank-{self.rank}.pt')
        self.checkpoints.append(json.loads(json.dumps(client_state)))


class TokenBatch:
    def __init__(self,prompts):
        self.prompts=prompts;self.attention_mask=torch.ones(len(prompts),2,dtype=torch.bool)
    def to(self,device):return self


class Tokenizer:
    eos_token='<EMPTY>'
    def __init__(self):self.prompts=[]
    def __call__(self,prompts,**kwargs):
        self.prompts.append(list(prompts));return TokenBatch(prompts)


class InspectMonitor:
    def __init__(self,cfg,model):
        self.actual=TrainingMonitor(cfg,'cpu',model,mhc_enabled=False)
        self.mhc_enabled=self.actual.mhc_enabled
        self.updates=[]
    def begin_update(self,step):return self.actual.begin_update(step)
    def phase(self,name):return self.actual.phase(name)
    def record_micro(self,*args,**kwargs):return self.actual.record_micro(*args,**kwargs)
    def end_update(self,step,**kwargs):
        self.updates.append((step,dict(kwargs)))
        return self.actual.end_update(step,**kwargs)


def namespace(rank):
    names={'_safe_log','_data_error_category','next_training_batch','sample_loss_scale','set_lr','sampler_signature',
           'atomic_write','save_torch_weights','save_pure_weights','_read_checkpoint_metadata',
           '_semantic_metadata','restore_data_cursor','checkpoint_distributed'}
    ns=dict(torch=torch,dist=dist,time=time,logging=logging,Path=Path,json=json,re=re,
        os=os,uuid=uuid,hashlib=hashlib,math=math,validation_context=validation_context,nullcontext=nullcontext,_dist_rank=lambda:rank)
    nodes=[node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=30))
    rank,world=dist.get_rank(),dist.get_world_size();assert world==2
    started=time.monotonic();streams=[];output=None
    try:
        locations=[tempfile.mkdtemp(prefix='ready-train-loop-') if rank==0 else None]
        dist.broadcast_object_list(locations,src=0);output=Path(locations[0])
        ns=namespace(rank);sampler=MixedSampler(rank)
        model=ToyModel();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.005)
        engine=ToyEngine(model,optimizer,rank);tokenizer=Tokenizer()
        settings={'num_workers':2,'dataloader_prefetch_factor':2,'dataloader_timeout_seconds':3}
        stream=ReadyDataStream(DelayedDataset(rank),sampler,settings,logging.getLogger('ready-loop'),
            signature=ns['sampler_signature'](sampler),collate_fn=collate)
        streams.append(stream)
        cfg={'seed':42,'training':{'learning_rate':.01,'lr_warmup_steps':2,'total_steps':4,
            'cfg_dropout':.5,'cpu_cfg_dropout':True,'checkpointing_steps':2,'ready_data_pipeline':True},
            'scheduler':{'shift':3},'validation':{'enabled':False},
            'monitoring':{'enabled':True,'every_n_steps':2,
                          'sigma':{'enabled':True,'every_n_steps':2,'tensor_every_n_steps':1}}}
        monitor=InspectMonitor(cfg,model);payloads=[];valid=[];requested=[];lr_steps=[]
        actual_next,actual_lr=ns['next_training_batch'],ns['set_lr']
        def tracked_next(data_stream,device,logger,step,micro,pending=None):
            batch=actual_next(data_stream,device,logger,step,micro,pending)
            requested.append((tuple(data_stream.last_request),batch is None))
            if batch is not None:
                valid.append({'step':step,'tag':tuple(batch[REQUEST_KEY]),'captions':list(batch['captions'])})
            return batch
        def tracked_lr(*args):lr_steps.append(args[2]);return actual_lr(*args)
        ns.update(next_training_batch=tracked_next,set_lr=tracked_lr,engine=engine,
            ds_optimizer=optimizer,client_optimizer=optimizer,cfg=cfg,monitor=monitor,
            sigma_monitor=SigmaMonitor(cfg,'cpu'), tar_usage=None,
            device=torch.device('cpu'),dtype=torch.float32,rank=rank,step=0,max_steps=4,
            grad_accum=2,SIGMA_BUCKETS=10,samples_seen=0,pending_batch=None,
            consecutive_batch_skips=0,max_consecutive_batch_skips=4,data_stream=stream,
            vae=SimpleNamespace(encode=lambda pixels:SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda:pixels))),
            latents_mean=0.,latents_std_inv=1.,text_encoder=None,tokenizer=tokenizer,tc={},
            chat_format_prompts=lambda tok,prompts,flag,enable_thinking=True:prompts,
            encode_text=lambda enc,inputs,layer:torch.tensor(
                [sum(p.encode()) for p in inputs.prompts], dtype=torch.float32)[:, None, None].expand(-1, 2, 4),
            progress=SimpleNamespace(update=lambda *a:None,set_postfix=lambda **kwargs:None),
            log_every=1,logger=logging.getLogger('ready-loop'),wandb_run=None,swanlab_run=True,
            swanlab=SimpleNamespace(log=lambda payload,step:payloads.append((step,dict(payload)))),
            sigma_bucket_ema=[float('nan')]*10,checkpoint_dir=output/'checkpoints/state',output_dir=output,
            prune_state_checkpoints=lambda *a:None)
        train_main=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        loop=next(node for node in train_main.body if isinstance(node,ast.While))
        helpers = [node for node in train_main.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"encode_captions", "fetch_microbatch"}]
        lookahead = "--text-lookahead" in sys.argv
        train_stream, encode_stream = Mock(), Mock()
        from collections import deque
        ns.update(lookahead=None, throughput_window=deque())
        ns["text_stream"] = encode_stream if lookahead else None
        original_rand=torch.rand
        def fixed_sigma(*shape,**kwargs):
            if kwargs.get('generator') is not None:return original_rand(*shape,**kwargs)
            return torch.full(shape,.5,**kwargs)
        with ExitStack() as patches:
            patches.enter_context(patch.object(torch, "rand", side_effect=fixed_sigma))
            if lookahead:
                patches.enter_context(patch.object(torch.cuda, "stream", return_value=nullcontext()))
                patches.enter_context(patch.object(torch.cuda, "current_stream", return_value=train_stream))
                patches.enter_context(patch.object(torch.Tensor, "record_stream", autospec=True))
            exec(compile(ast.Module(body=[*helpers, loop], type_ignores=[]), str(SOURCE), "exec"), ns)
        if lookahead:
            assert train_stream.wait_stream.call_count >= 4
        assert len(tokenizer.prompts) == len(valid)
        assert engine.text_ids == [[sum(p.encode()) for p in prompts] for prompts in tokenizer.prompts]
        assert engine.global_steps==4 and engine.micro_steps==8
        assert len(valid)==8 and len(requested)==9 and sum(skipped for _,skipped in requested)==1,requested
        assert lr_steps==[1,2,3,4] and [entry['step'] for entry in engine.checkpoints]==[2,4]
        assert int(optimizer.state[model.weight]['step'])==4
        unique_request_count = 8 if rank == 0 else 9
        assert len(set(tag for tag,_ in requested)) == unique_request_count, requested
        if rank == 0:
            failed_index = next(i for i, (_, skipped) in enumerate(requested) if skipped)
            assert requested[failed_index][0] == requested[failed_index+1][0]
        assert len({entry['tag'] for entry in valid}) == 8
        assert [tag[1] for tag,_ in requested]!=list(range(9)),requested
        assert {len(entry['captions']) for entry in valid}=={1,2},valid
        for record,prompts in zip(valid,tokenizer.prompts):
            epoch,offset=record['tag'];seed=42+rank*1_000_003+epoch*1_000_000_007+offset+1
            dropped=original_rand(len(prompts),generator=torch.Generator(device='cpu').manual_seed(seed))<.5
            expected=[tokenizer.eos_token if bool(drop) else caption for drop,caption in zip(dropped,record['captions'])]
            assert prompts==expected,(record,prompts,expected)
        assert sum(entry['skipped_batches'] for _,entry in monitor.updates)==1
        assert sum(entry['data_requests'] for _,entry in monitor.updates)==9
        local=[]
        for update in range(1,5):
            items=range(2 * (update - 1), 2 * update)
            local.append({'samples':sum(len(valid[i]['captions']) for i in items),
                          'loss_sum':sum(sum(engine.forward_losses[i]) for i in items)})
        all_expected=[None]*world;dist.all_gather_object(all_expected,local)
        samples=0
        for update in range(4):
            count=sum(values[update]['samples'] for values in all_expected);samples+=count
            if rank==0:
                step,payload=payloads[update];assert step==update+1
                assert payload['train/samples']==samples
                assert abs(payload['train/sigma_mean']-.75)<1e-7
                loss=sum(values[update]['loss_sum'] for values in all_expected)/count
                assert math.isclose(payload['train/loss'],loss,rel_tol=2e-6,abs_tol=2e-6),(payload,loss)
                if step % 2 == 0:
                    window_count=sum(values[j]['samples'] for values in all_expected for j in (update-1,update))
                    window_sum=sum(values[j]['loss_sum'] for values in all_expected for j in (update-1,update))
                    assert payload['diagnostics/samples']==window_count
                    assert payload['diagnostics/all/sigma_0.7-0.8/samples']==window_count
                    assert math.isclose(payload['diagnostics/all/sigma_0.7-0.8/loss_mean'],
                                        window_sum/window_count,rel_tol=2e-6)
        assert ns['samples_seen']==samples
        saved=engine.checkpoints[-1]['data_cursor']
        assert saved['world_size']==2 and saved['stream_kind']=='ready_first_distributed'
        assert saved['rank_cursors'][0]!=saved['rank_cursors'][1],saved
        dist.barrier()
        metadata=ns['_read_checkpoint_metadata'](output/'checkpoints/weights/latest.pt')
        assert metadata['data_cursor']==saved and metadata['samples_seen']==samples
        selected=select_data_state(saved,stream.signature,rank,world,expected_epoch_length=len(sampler))
        engine.giga_data_cursor=saved
        epoch,offset=ns['restore_data_cursor'](engine,sampler,4,2,logging.getLogger('ready-resume'))
        assert engine.giga_ready_resume_state==selected
        stream.close()
        restored=ReadyDataStream(DelayedDataset(rank),MixedSampler(rank),settings,
            logging.getLogger('ready-resume'),epoch=epoch,batch_offset=offset,
            resume_state=selected,signature=stream.signature,collate_fn=collate)
        streams.append(restored)
        consumed={tag[1] for tag,_ in requested};remaining=set(range(len(sampler)))-consumed
        replay=[]
        for index in range(len(remaining)):
            if index==2:restored.restart()
            replay.append(restored.next_batch()[REQUEST_KEY][1])
        assert len(replay)==len(set(replay)) and set(replay)==remaining,(consumed,replay)
        assert restored.state_dict()['batch_offset']==len(sampler)
        weights=[None]*world;dist.all_gather_object(weights,float(model.weight.detach()))
        assert weights[0]==weights[1]
        elapsed=time.monotonic()-started
        dist.barrier()
        if rank==0:
            print('PASS CPU real main while + DDP/AdamW: 4 updates / 8 valid micro, one global skip; '
                  'bulk samples/loss/sigma, CPU CFG dropout, atomic checkpoint and exact per-rank ready resume/restart; '
                  f'elapsed={elapsed:.3f}s. This is not DS/GPU validation.',flush=True)
    finally:
        for stream in streams:stream.close()
        if rank==0 and output is not None:shutil.rmtree(output,ignore_errors=True)
        dist.destroy_process_group()


if __name__=='__main__':main()
