"""Bounded backend A/B with the actual DiT and optional frozen Qwen/VAE.

Synthetic repeated inputs exclude PFS and data workers. Separate invocations
use identical seeds/checkpoint/shapes/GAS/AC. Timing excludes initialization,
checkpoint and profiling; --profile is a separate diagnostic run.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'thirdparty')]
import torch
import torch.distributed as dist
import yaml
from optimizations.distributed_backend import FSDP2Engine
from optimizations.training_profiler import TrainingProfiler
from models.model_factory import build_diffusion_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--backend',choices=['deepspeed','fsdp2','hsdp'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--weights');p.add_argument('--tiny',action='store_true')
    p.add_argument('--dit-only',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--accum',type=int,default=2)
    p.add_argument('--text-tokens',type=int,default=128);p.add_argument('--width',type=int,default=512);p.add_argument('--height',type=int,default=512)
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--updates',type=int,default=20)
    p.add_argument('--keep-forward-weights',action='store_true');p.add_argument('--ac',type=float);p.add_argument('--shard-size',type=int);p.add_argument('--defer-sync',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(1);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_float32_matmul_precision('high')
    import deepspeed
    deepspeed.init_distributed(dist_backend='nccl')
    rank,world=dist.get_rank(),dist.get_world_size();device=torch.device('cuda',torch.cuda.current_device())
    cfg=yaml.safe_load(Path(a.config).read_text());cfg['distributed']['backend']=a.backend
    cfg['distributed']['fsdp2']={'shard_size':a.shard_size or int(os.environ['LOCAL_WORLD_SIZE']), 'sync_each_micro':not a.defer_sync,'reduce_dtype':'fp32','reshard_after_forward':not a.keep_forward_weights}
    train=cfg['training'];train.update(gradient_accumulation_steps=a.accum,batch_size_per_device=a.batch,torch_compile=False,regional_compile=False)
    if train.get('muon',{}).get('enabled',True):raise ValueError('AdamW only')
    cfg.setdefault('monitoring',{})['profiler']={'enabled':a.profile,'wait_steps':a.warmup,'warmup_steps':1,'active_steps':min(3,a.updates-1),'ranks':'all'}
    if a.ac is not None:cfg['model']['activation_checkpointing_ratio']=a.ac
    if a.tiny:
        cfg['model'].update(dim=512,ffn_dim=1536,heads=4,layers=4,text_dim=512,text_trans={'enabled':True,'layers':2,'heads':4})
    if not a.tiny and not a.weights:raise ValueError('Full-size A/B requires a fixed --weights checkpoint')
    torch.manual_seed(42)
    model=build_diffusion_model(cfg).to(device=device,dtype=torch.bfloat16)
    spec=importlib.util.spec_from_file_location('giga_backend_benchmark_helpers',ROOT/'scripts/train/train_giga_exp.py')
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    model.giga_text_encoder_metadata=helper.text_encoder_metadata(cfg["components"]["text_encoder"])
    if a.weights:helper.load_init_weights(model,a.weights,strict=True)
    # The production DeepSpeed optimizer wrapper and settings are unchanged.
    if a.backend=='deepspeed':
        optimizer=helper.build_optimizer(model,train)
        ds_path=Path(cfg['distributed']['config']);ds_path=ds_path if ds_path.is_absolute() else ROOT/ds_path
        ds_cfg=json.loads(ds_path.read_text());ds_cfg.update(train_micro_batch_size_per_gpu=a.batch,gradient_accumulation_steps=a.accum,train_batch_size=a.batch*a.accum*world,steps_per_print=1000000)
        engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,model_parameters=model.parameters(),config=ds_cfg)
    else:engine=FSDP2Engine(model,cfg,torch.bfloat16)
    engine.train()
    torch.manual_seed(100+rank)
    banks=[]
    for micro in range(a.accum):
        latent=torch.randn(a.batch,48,1,a.height//16,a.width//16,device=device)
        noise=torch.randn_like(latent);sigma=torch.rand(a.batch,device=device);sigma=3*sigma/(1+2*sigma)
        features=torch.randn(a.batch,a.text_tokens,cfg['model']['text_dim'],device=device,dtype=torch.bfloat16)
        mask=torch.ones(a.batch,a.text_tokens,device=device,dtype=torch.bool)
        mask[:,-max(1,a.text_tokens//5):]=False
        banks.append((latent,noise,sigma,features,mask))
    encoder=vae=None
    if not (a.dit_only or a.tiny):
        from models.text_encoding import build_tokenizer,build_text_encoder,chat_format_prompts,encode_text
        from diffusers import AutoencoderKLWan
        tc=cfg['components']['text_encoder'];dc=cfg['components']['wan_vae']
        tokenizer=build_tokenizer(tc);encoder=build_text_encoder(tc,torch.bfloat16,device).eval().requires_grad_(False)
        vae=AutoencoderKLWan.from_pretrained(dc['pretrained_path'],subfolder=dc.get('subfolder','vae'),torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval().requires_grad_(False)
        mean,invstd=helper.wan_latent_stats(vae,device)
        captions=chat_format_prompts(tokenizer,['A red apple on a white table. '*a.text_tokens]*a.batch,tc.get('use_chat_template'),tc.get('enable_thinking',True))
        inputs=tokenizer(captions,return_tensors='pt',truncation=True,padding='max_length',max_length=a.text_tokens).to(device)
        pixels=torch.randn(a.batch,3,1,a.height,a.width,device=device,dtype=torch.bfloat16)
    out=Path(a.output);prof=TrainingProfiler(cfg,out.parent,engine.module)
    losses=[];norms=[];events=[]
    def update(index):
        prof.begin_update(index+1)
        for latent,noise,sigma,features,mask in banks:
            if encoder is not None:
                with torch.no_grad(),prof.phase('text_encode'):
                    features=encode_text(encoder,inputs,tc.get('layer'));mask=inputs.attention_mask.bool()
                with torch.no_grad(),prof.phase('vae'):
                    latent=(vae.encode(pixels).latent_dist.mean.float()-mean)*invstd
            scale=sigma[:,None,None,None,None];xt=((1-scale)*latent+scale*noise).to(torch.bfloat16)
            with prof.phase('dit_forward'):
                pred=engine(xt,sigma*1000,features,mask)
                loss=(pred.float()-(noise-latent)).square().mean()
            with prof.phase('backward'):engine.backward(loss)
            with prof.phase('optimizer'):engine.step()
            losses.append(loss.detach())
        norms.append(engine.get_global_grad_norm())
        prof.end_update()
    for index in range(a.warmup):update(index)
    dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for index in range(a.warmup,a.warmup+a.updates):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();update(index);end.record();events.append((begin,end))
    torch.cuda.synchronize();wall=time.perf_counter()-start;prof.close()
    local={'rank':rank,'host':socket.gethostname(),'wall_s':wall,'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30,
           'update_ms':[b.elapsed_time(e) for b,e in events],'losses':[float(x) for x in losses], 'grad_norms':[float(x) for x in norms], 'global_steps':engine.global_steps}
    rows=[None]*world;dist.all_gather_object(rows,local)
    if rank==0:
        wall=max(r['wall_s'] for r in rows)
        result={'backend':a.backend,'world_size':world,'hosts':sorted(set(r['host'] for r in rows)), 'batch':a.batch,'accum':a.accum,'model':cfg['model'],
                'scope':'DiT only' if a.tiny or a.dit_only else 'DiT + frozen Qwen/VAE, synthetic inputs',
                'source_checkpoint':a.weights,'shape':[a.height,a.width,a.text_tokens], 'profile_enabled':a.profile,
                'seconds_per_update':wall/a.updates,'samples_per_second':a.batch*a.accum*world*a.updates/wall,
                'max_allocated_gib':max(r['allocated_gib'] for r in rows),'ranks':rows,
                'backend_options':cfg['distributed'].get('fsdp2',{}), 'code_revision':os.environ.get('GIGA_CODE_REVISION'),
                'torch_version':torch.__version__, 'cuda_version':torch.version.cuda, 'deepspeed_version':deepspeed.__version__, 'gpu_name':torch.cuda.get_device_name(),
                'communication_env':{k:os.environ.get(k) for k in ['NCCL_IB_DISABLE','CUDA_DEVICE_MAX_CONNECTIONS','NCCL_DEBUG']}}
        result['finite']=all(torch.isfinite(torch.tensor(r['losses']+r['grad_norms'])).all().item() for r in rows)
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
        print('BACKEND_BENCHMARK',json.dumps({k:v for k,v in result.items() if k not in ['model','ranks']}),flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
