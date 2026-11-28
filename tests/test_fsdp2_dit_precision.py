"""Real DiT BF16/FP32-RoPE AC regression against replicated BF16 autograd."""
import copy,os,sys
from pathlib import Path
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'thirdparty')]
from models.giga_transformer import GigaExpTransformer
from optimizations.distributed_backend import FSDP2Engine

def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(1);dist.init_process_group('nccl')
    for backend in ['fsdp2','hsdp']:
        torch.manual_seed(123)
        model=GigaExpTransformer(in_channels=48,out_channels=48,dim=256,ffn_dim=768,heads=2,layers=3,text_dim=512,
            rope_axes_dims=[16,48,48,16],patch_size=[1,1,1],adaln_type='basis',adaln_rank=64,final_image_only=True,
            text_trans={'enabled':True,'layers':2,'heads':2},activation_checkpointing=True,activation_checkpointing_ratio=.65).cuda().bfloat16()
        reference=copy.deepcopy(model)
        cfg={'distributed':{'backend':backend,'fsdp2':{'shard_size':int(os.environ.get('GIGA_TEST_SHARD_SIZE',2))}},'training':{'muon':{'enabled':False},'gradient_accumulation_steps':1}}
        engine=FSDP2Engine(model,cfg,torch.bfloat16)
        x=torch.randn(2,48,1,8,6,device='cuda',dtype=torch.bfloat16)
        text=torch.randn(2,13,512,device='cuda',dtype=torch.bfloat16)
        sigma=torch.tensor([.2,.8],device='cuda',dtype=torch.float32)*1000
        mask=torch.ones(2,13,device='cuda',dtype=torch.bool);mask[0,7:]=False
        actual=engine(x,sigma,text,mask);expected=reference(x,sigma,text,mask)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.float().square().mean().backward();expected.float().square().mean().backward()
        error=denom=0.
        for (name,p),(other,q) in zip(engine.module.named_parameters(),reference.named_parameters()):
            assert name==other
            if q.grad is None:assert p.grad is None;continue
            grad=p.grad.full_tensor()
            error+=(grad.float()-q.grad.float()).square().sum().item();denom+=q.grad.float().square().sum().item()
        relative=(error/max(denom,1e-30))**.5
        assert relative<.02,relative
        if dist.get_rank()==0:print('PASS BF16 DiT AC',backend,'output identical, gradient relative L2',relative,flush=True)
        del engine,model,reference;torch.cuda.empty_cache();dist.barrier()
    dist.destroy_process_group()
if __name__=='__main__':main()
