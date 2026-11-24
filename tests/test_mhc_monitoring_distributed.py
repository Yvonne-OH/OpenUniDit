"""Two-rank Gloo integration of enabled monitoring with actual mHC forward/backward."""
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'thirdparty')]
import torch
import torch.distributed as dist
from optimizations.mhc_monitoring import TrainingMonitor
from models.giga_transformer_mhc_transfer import GigaExpTransformerMHC


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank=dist.get_rank();assert dist.get_world_size()==2
    torch.manual_seed(42+rank)
    spec={'dim':32,'ffn_dim':64,'layers':3,'heads':2,'text_dim':16,
        'in_channels':4,'out_channels':4,'rope_axes_dims':[4,4,4,4],
        'adaln_type':'basis','adaln_rank':8,'text_trans':{'enabled':True,'layers':1,'heads':2},
        'activation_checkpointing':True}
    model=GigaExpTransformerMHC(**spec,n_streams=4,sinkhorn_iters=10).train()
    with torch.no_grad():
        for block in model.blocks:
            block.mhc_attn.mapping.bias[:4].add_(.3*rank)
            block.mhc_ffn.mapping.bias[:4].add_(.2*rank)
    cfg={'model':spec,'monitoring':{'enabled':True,'every_n_steps':2,'phase_timings':True,
                                  'peak_tflops_per_gpu':989,'per_layer':[0,2]}}
    # Different host clocks prove that rates use the slowest rank's duration.
    seconds=1.+rank;clock=iter([0.,seconds,seconds])
    monitor=TrainingMonitor(cfg,'cpu',model,True,lambda:next(clock))
    expected_samples=expected_text=expected_flops=0.
    expected_mean_sum=expected_mean_weight=0.
    local_row_max=0.;local_res_min=math.inf
    for update in (1,2):
        monitor.begin_update(update);model.zero_grad(set_to_none=True)
        for micro in range(2):
            batch=1+(rank+micro)%2
            lengths=[3+rank] if batch==1 else [5,2+rank]
            mask=torch.zeros(batch,5,dtype=torch.bool)
            for i,length in enumerate(lengths):mask[i,:length]=True
            with monitor.phase('dit_forward'):
                output=model(torch.randn(batch,4,1,2,2),torch.full((batch,),400.),torch.randn(batch,5,16),mask)
            with monitor.phase('backward'):output.square().mean().backward()
            if update==2:
                for key,value in model.mhc_stats().items():
                    if key.endswith('/pre_mean'):
                        expected_mean_sum+=float(value)*batch*9;expected_mean_weight+=batch*9
                    if key.endswith('/sinkhorn_row_error'):local_row_max=max(local_row_max,float(value))
                    if key.endswith('/res_min'):local_res_min=min(local_res_min,float(value))
            monitor.record_micro(batch,4,mask)
            expected_samples+=batch;expected_text+=sum(lengths)
            # Independent explicit coefficients for this tiny architecture:
            # 3 dense blocks + 6 mHC mappings; one refiner block; input/output projections.
            expected_flops+=batch*(9*258048+5*63488+4*1280)
            expected_flops+=1152*sum((4+n)**2 for n in lengths)+384*sum(n*n for n in lengths)
        payload=monitor.end_update(update,grad_norm=1.+rank,skipped_batches=update,
            data_wait_seconds=.1*update*(rank+1),data_wait_max_seconds=.07*update*(rank+1),
            data_read_seconds=.05*update*(rank+1),data_read_max_seconds=.04*update*(rank+1),
            data_requests=update+rank)
        if update==1:assert payload=={}
    sums=torch.tensor([expected_samples,expected_text,expected_flops,expected_mean_sum,expected_mean_weight],dtype=torch.float64)
    dist.all_reduce(sums)
    row=torch.tensor([local_row_max],dtype=torch.float64);dist.all_reduce(row,op=dist.ReduceOp.MAX)
    minimum=torch.tensor([local_res_min],dtype=torch.float64);dist.all_reduce(minimum,op=dist.ReduceOp.MIN)
    assert math.isclose(payload['performance/samples_per_sec'],float(sums[0])/2,rel_tol=1e-12)
    assert math.isclose(payload['performance/text_tokens_per_sec'],float(sums[1])/2,rel_tol=1e-12)
    assert math.isclose(payload['performance/mfu_dit_estimate'],float(sums[2])/(2*2*989e12),rel_tol=1e-12)
    assert math.isclose(payload['mhc/all/pre_mean'],float(sums[3]/sums[4]),rel_tol=1e-12)
    assert payload['mhc/all/sinkhorn_row_error']==float(row[0])
    assert payload['mhc/all/res_min']==float(minimum[0])
    assert payload['performance/skipped_batches_window']==3
    assert payload['train/grad_norm']==2
    assert payload['data/requests_global_sum']==8
    assert payload['data/requests_mean_per_rank']==4
    for name,total,slowest,longest in [('fetch_and_sync',.9,.6,.28),('read_wait',.45,.3,.16)]:
        assert math.isclose(payload[f'data/{name}_seconds_global_sum'],total,rel_tol=1e-12)
        assert math.isclose(payload[f'data/{name}_seconds_mean_per_rank'],total/2,rel_tol=1e-12)
        assert math.isclose(payload[f'data/{name}_seconds_slowest_rank'],slowest,rel_tol=1e-12)
        assert math.isclose(payload[f'data/{name}_seconds_max_request'],longest,rel_tol=1e-12)
        assert math.isclose(payload[f'data/{name}_wall_fraction_per_rank'],total/4,rel_tol=1e-12)
    assert payload['mhc/all/nonfinite_count']==0
    assert 'mhc/blocks/0/mhc_attn/pre_mean' in payload
    assert 'mhc/blocks/2/mhc_attn/pre_mean' in payload
    assert 'mhc/blocks/1/mhc_attn/pre_mean' not in payload
    assert not any(key.startswith('performance/gpu_phase_ms/') for key in payload)
    assert all(math.isfinite(value) for value in payload.values())
    gathered=[None,None];dist.all_gather_object(gathered,payload)
    assert gathered[0]==gathered[1]
    if rank==0:
        print('PASS 2-rank enabled mHC monitoring: weighted means, extrema, shared skips, MFU, selected layers, checkpointed forward/backward; metrics=',len(payload),flush=True)
        print(json.dumps({key:payload[key] for key in ['performance/samples_per_sec','mhc/all/pre_mean','mhc/all/sinkhorn_row_error','performance/mfu_dit_estimate','performance/skipped_batches_window']},indent=2),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
