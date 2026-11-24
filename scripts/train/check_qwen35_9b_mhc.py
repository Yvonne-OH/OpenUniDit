"""CPU-only asset/config preflight for the Qwen3.5-9B mHC branch."""
import argparse
import json
from pathlib import Path
import struct
import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());tc=cfg['components']['text_encoder']
    root=Path(tc['pretrained_path']);manifest=json.loads((root/'download_manifest.json').read_text())
    assert manifest['revision']==tc['revision'], 'Model revision mismatch'
    for name,size in manifest['files'].items():
        p=root/name
        assert p.is_file() and (size is None or p.stat().st_size==size), f'Incomplete model file: {name}'
    hf=json.loads((root/'config.json').read_text())
    assert hf['text_config']['hidden_size']==cfg['model']['text_dim']==4096
    assert tc['layer']==-1 and cfg['mhc']['enabled'] is True
    assert cfg['training']['resume_mode']=='auto' and cfg['training']['init_from_strict'] is True
    output=Path(cfg['output_dir']);origin=json.loads((output/'initialization.json').read_text())
    snapshot=Path(origin['initial_checkpoint'])
    with snapshot.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];assert n<100_000_000
        header=json.loads(f.read(n))
    end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
    assert snapshot.stat().st_size==8+n+end==origin['checkpoint_bytes']
    assert header['text_in.weight']['shape']==[cfg['model']['dim'],2560]
    assert (output/'dataset_index_cache').is_dir()
    print('QWEN35_9B_MHC_READY',json.dumps({'source_step':origin['source_step'],'source':str(snapshot),
        'output':str(output),'mhc':cfg['mhc'],'model_revision':manifest['revision'],
        'text_projection':'2560 -> 4096: only text_in.weight is reinitialized',
        'progress':'New adaptation branch begins at step 0; later auto resumes its own weights.'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
