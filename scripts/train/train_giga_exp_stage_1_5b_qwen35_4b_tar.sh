#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh
cd "$ROOT"
export CONFIG="${CONFIG:-${1:-configs/stage_1_5b_qwen35_4b_tar.yaml}}"
export MASTER_PORT="${MASTER_PORT:-29535}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fail before torchrun if the pinned encoder or the copied initialization is missing.
python - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path
import yaml
from safetensors import safe_open

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
tc = cfg['components']['text_encoder']
root = Path(tc['pretrained_path'])
manifest = json.loads((root / 'download_manifest.json').read_text())
assert manifest['revision'] == tc['revision'], 'encoder revision mismatch'
for name, size in manifest['files'].items():
    p = root / name
    assert p.is_file() and (size is None or p.stat().st_size == size), f'incomplete model file: {p}'
output = Path(cfg['output_dir'])
origin = json.loads((output / 'initialization.json').read_text())
initial = Path(origin['initial_checkpoint'])
assert initial.is_file() and initial.stat().st_size == origin['checkpoint_bytes']
with safe_open(initial, framework='pt', device='cpu') as handle:
    assert len(list(handle.keys())) > 0
assert (output / 'dataset_index_cache').is_dir(), 'dataset cache not prepared'
print(f"Qwen3.5-4B ready: {manifest['revision']}; DiT source step={origin['source_step']}; output={output}", flush=True)
print('首次启动继承上述 DiT 权重，新分支从 step=0 warmup；后续 auto 恢复本分支权重。', flush=True)
PY

if [[ "${GIGA_CHECK_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec bash scripts/train/train_giga_exp_stage_1_5b_tar.sh "$CONFIG"
