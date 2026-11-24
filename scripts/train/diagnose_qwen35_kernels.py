"""GPU-only numerical diagnosis; writes no training checkpoints or model updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'thirdparty')]


def main():
    query=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if query.stdout.strip():raise RuntimeError('GPU 0 is busy; no diagnostic started')
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    import yaml
    from models.text_encoding import build_tokenizer,build_text_encoder,chat_format_prompts
    from optimizations.qwen35_fast import resolve_kernels,compatible_dispatch
    modeling,chunk,conv=resolve_kernels()
    cfg=yaml.safe_load((ROOT/'configs/stage_1_5b_qwen35_4b_fast.yaml').read_text())
    tc=cfg['components']['text_encoder'];device=torch.device('cuda:0')
    tokenizer=build_tokenizer(tc);encoder=build_text_encoder(tc,torch.bfloat16,device)
    ref_chunk,ref_conv=modeling._giga_reference_chunk,modeling._giga_reference_conv
    fast_chunk,fast_conv=compatible_dispatch(chunk),compatible_dispatch(conv)
    def pre_normalized(q,k,v,**kwargs):
        if kwargs.get('use_qk_l2norm_in_kernel'):
            q=modeling.l2norm(q,dim=-1,eps=1e-6)
            k=modeling.l2norm(k,dim=-1,eps=1e-6)
            kwargs['use_qk_l2norm_in_kernel']=False
        return fast_chunk(q,k,v,**kwargs)
    modes={'reference':(ref_chunk,ref_conv),'conv_only':(ref_chunk,fast_conv),
           'gdn_only':(fast_chunk,ref_conv),'both':(fast_chunk,fast_conv),
           'pre_norm_both':(pre_normalized,fast_conv)}
    cases={'normal_en':['A realistic portrait of a woman wearing a red coat.'],
           'normal_mixed':['A realistic portrait of a woman wearing a red coat.','白色桌子上并排摆放三个红苹果。'],
           'long_padded':['A detailed photograph of '+(' a'*500),'A red apple.'],
           'raw_repeat':['A detailed photograph of '+(' a'*128)]}
    results={}
    with torch.no_grad():
        for name,prompts in cases.items():
            texts=prompts if name=='raw_repeat' else chat_format_prompts(tokenizer,prompts,True,False)
            inputs=tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=512).to(device)
            mask=inputs.attention_mask.bool();reference=None;rows={}
            for mode,(core,convolution) in modes.items():
                modeling.torch_chunk_gated_delta_rule,modeling.causal_conv1d_fn=core,convolution
                torch.cuda.synchronize();start=time.perf_counter()
                out=encoder(**inputs,use_cache=False,return_dict=True,output_hidden_states=True)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                hidden=out.hidden_states
                if mode=='reference':reference=tuple(t.clone() for t in hidden)
                errors=[]
                for a,b in zip(hidden,reference):
                    a,b=a[mask].float(),b[mask].float()
                    errors.append(float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)))
                a,b=out.last_hidden_state[mask].float(),reference[-1][mask].float()
                rows[mode]={'relative_rms':errors[-1],'layer_errors':errors,
                    'cosine':float(torch.nn.functional.cosine_similarity(a,b).mean()),'wall_ms_cold':elapsed}
                print('KERNEL_DIAGNOSTIC',name,mode,rows[mode]['relative_rms'],rows[mode]['cosine'],elapsed,flush=True)
            results[name]={'shape':list(inputs.input_ids.shape),'modes':rows}
    modeling.torch_chunk_gated_delta_rule,modeling.causal_conv1d_fn=ref_chunk,ref_conv
    target=Path(cfg['output_dir'])/'performance/kernel_diagnostic.json'
    target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(results,indent=2)+'\n')
    print('DIAGNOSTIC_SAVED',target,flush=True)


if __name__=='__main__':main()
