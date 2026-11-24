"""Optimized entry point; delegates training/cursor/checkpoint logic to the original trainer."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'thirdparty')]


def main():
    import torch
    import yaml
    from scripts.train import train_giga_exp as training
    if '--help' in sys.argv or '-h' in sys.argv:
        training.main()
        return
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args,_=parser.parse_known_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    tc=cfg['components']['text_encoder']
    if tc['type'] != 'qwen3_5' or tc['layer'] != -1:
        raise ValueError('This entry point requires Qwen3.5 text features at layer=-1')
    local_rank=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local_rank)
    from optimizations.qwen35_fast import install_training_optimizations, install_resume_bootstrap
    report=install_training_optimizations(training,torch.device('cuda',local_rank))
    if cfg['training'].get('bootstrap_weights_from'):
        install_resume_bootstrap(training,cfg['training']['bootstrap_weights_from'])
    import json
    report_dir=Path(cfg['output_dir'])/'performance'
    report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/f'kernels_rank{os.environ.get("RANK",0)}.json').write_text(json.dumps(report,indent=2)+'\n')
    training.main()


if __name__ == '__main__':
    main()
