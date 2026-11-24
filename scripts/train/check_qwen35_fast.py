"""CPU-only preflight: validates dependencies and a complete source/own checkpoint."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import os
import sys


def complete_snapshot(path):
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if size>100_000_000:raise ValueError('Invalid header size')
        header=json.loads(f.read(size))
    end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
    if path.stat().st_size!=8+size+end:raise ValueError('Incomplete safetensors file')
    return header.get('__metadata__',{})


def main():
    import yaml
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    prefix=Path(sys.prefix).resolve()
    if prefix.name!='gwp_qwen35_fast':raise RuntimeError(f'Activate the isolated fast environment; got {prefix}')
    dependencies={}
    for package in ('fla-core','causal-conv1d'):
        distribution=importlib.metadata.distribution(package)
        location=Path(distribution.locate_file('')).resolve()
        if not location.is_relative_to(prefix):raise RuntimeError(f'{package} must be installed in the isolated environment')
        dependencies[package]=distribution.version
    ds=json.loads(Path(cfg['distributed']['config']).read_text())
    timer=ds['timers']['throughput']
    assert timer['enabled'] is False and timer['synchronized'] is False
    assert cfg['components']['text_encoder']['layer']==-1
    assert cfg['training']['resume_mode']=='auto'
    output=Path(cfg['output_dir'])
    own=output/'checkpoints/weights'
    source=own if list(own.glob('step-*.safetensors')) or list(own.glob('step-*.pt')) else Path(cfg['training']['bootstrap_weights_from'])
    candidates=sorted(source.glob('step-*.safetensors'),key=lambda p:int(p.stem.split('-')[1]),reverse=True)
    for snapshot in candidates:
        try:
            metadata=complete_snapshot(snapshot)
            break
        except (ValueError, OSError, KeyError, json.JSONDecodeError):continue
    else:raise RuntimeError(f'No complete source snapshot in {source}')
    assert (output/'dataset_index_cache').is_dir(), 'Dataset cache not prepared'
    if os.environ.get('GIGA_CHECK_ONLY') != '1':
        count=int(os.environ.get('NPROC_PER_NODE',os.environ.get('GPUS','8')))
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        devices=visible.split(',')[:count] if visible else [str(i) for i in range(count)]
        if len(devices)!=count:raise RuntimeError('Insufficient visible GPUs for the requested process count')
        for device in devices:
            try:
                query=subprocess.run(['nvidia-smi','-i',device,'--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            except (OSError,subprocess.CalledProcessError) as exc:
                raise RuntimeError('GPU inventory unavailable; no training started') from exc
            if query.stdout.strip():raise RuntimeError(f'GPU {device} already has a compute process; no training started')
    print('FAST_PREFLIGHT_OK',json.dumps({'dependencies':dependencies,'checkpoint':str(snapshot),
          'source_step':metadata.get('step',snapshot.stem),'has_actual_cursor':bool(metadata.get('data_cursor')),
          'output':str(output)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
