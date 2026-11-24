"""Real-encoder A/B benchmark; refuses to contend with any existing GPU process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'thirdparty')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/stage_1_5b_qwen35_4b_fast.yaml')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Check before importing CUDA libraries or allocating model weights.
    try:
        query=subprocess.run(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    except (OSError,subprocess.CalledProcessError) as exc:
        raise SystemExit('GPU inventory unavailable; no benchmark started. Use an idle GPU instance.') from exc
    if query.stdout.strip():raise RuntimeError('Selected GPU already has a compute process; no benchmark started')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    import yaml
    from models.text_encoding import build_tokenizer,build_text_encoder,encode_text
    from optimizations.qwen35_fast import kernel_mode,final_features,kernel_self_test
    torch.set_num_threads(1)
    device=torch.device('cuda:0')
    cfg=yaml.safe_load(Path(args.config).read_text());tc=cfg['components']['text_encoder']
    kernel_check=kernel_self_test(device)
    tokenizer=build_tokenizer(tc)
    encoder=build_text_encoder(tc,torch.bfloat16,device)
    results=[]
    # Short/long prompts and unequal right padding cover both production batch sizes.
    with torch.no_grad():
        for batch,length in [(1,128),(2,128),(1,512),(2,2048)]:
            texts=['A detailed photograph of '+(' a'*length)]
            if batch==2:texts.append('A red apple.')
            inputs=tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=length).to(device)
            outputs={};timings={}
            for mode in ('reference','fast'):
                function=encode_text if mode=='reference' else final_features
                with kernel_mode(mode):
                    for _ in range(2):function(encoder,inputs,-1)
                    torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin=time.perf_counter();start.record()
                    for _ in range(8):output=function(encoder,inputs,-1)
                    end.record();torch.cuda.synchronize()
                    timings[mode]={'wall_ms':(time.perf_counter()-begin)*1000/8,'cuda_ms':start.elapsed_time(end)/8}
                    outputs[mode]=output.float()
            mask=inputs.attention_mask.bool()
            reference,fast=outputs['reference'][mask],outputs['fast'][mask]
            error=float((fast-reference).square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-8))
            assert torch.isfinite(fast).all() and error<0.05, f'Encoder feature relative error {error}'
            row={'batch':batch,'tokens':inputs.input_ids.shape[1],'relative_rms':error,'timings':timings,
                 'speedup':timings['reference']['wall_ms']/timings['fast']['wall_ms']}
            print('ENCODER_BENCHMARK',json.dumps(row),flush=True);results.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'kernel_check':kernel_check,'encoder_benchmarks':results},indent=2)+'\n')


if __name__=='__main__':main()
