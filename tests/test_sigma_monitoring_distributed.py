"""Exact two-rank reference with uneven batches and disjoint sigma coverage."""
from pathlib import Path
import sys
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from optimizations.sigma_monitoring import SigmaMonitor


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank=dist.get_rank();assert dist.get_world_size()==2
    cfg={'dataset':{'buckets':['256_square','512_square']},
         'monitoring':{'enabled':True,'sigma':{'enabled':True,'every_n_steps':2,'tensor_every_n_steps':1}}}
    monitor=SigmaMonitor(cfg,'cpu')
    for step in (1,2):
        monitor.begin_update(step)
        sigma=torch.tensor([.05]) if rank==0 else torch.tensor([.05,.05,.95])
        losses=torch.tensor([1.]) if rank==0 else torch.tensor([3.,5.,7.])
        n=len(sigma);shape=(n,1,1,1,1)
        p=losses.sqrt().reshape(shape)
        monitor.record_micro(sigma,losses,torch.ones(n,3),[bool(rank)]*n,
                             (256,256) if rank==0 else (512,512),
                             prediction=p,latents=torch.zeros(shape),noise=torch.zeros(shape))
        result=monitor.end_update(step)
        if step==1:assert result=={}
    low='diagnostics/all/sigma_0.0-0.1/'
    high='diagnostics/all/sigma_0.9-1.0/'
    assert result[low+'samples']==6 and result[high+'samples']==2
    assert result[low+'loss_mean']==3 and result[high+'loss_mean']==7
    assert result[low+'loss_contribution']==2.25 and result[high+'loss_contribution']==1.75
    assert result['diagnostics/resolution/256x256/samples']==2
    assert result['diagnostics/resolution/512x512/samples']==6
    assert result['diagnostics/cfg/dropped/fraction']==.75
    assert math.isclose(result['diagnostics/output/sigma_0.0-0.1/pred_rms'],math.sqrt(3),rel_tol=1e-6)
    assert all(math.isfinite(v) for v in result.values())
    gathered=[None,None];dist.all_gather_object(gathered,result)
    assert gathered[0]==gathered[1]
    if rank==0:print('PASS sigma 2-rank: unequal counts, disjoint bins, strata, moments, identical payloads',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
