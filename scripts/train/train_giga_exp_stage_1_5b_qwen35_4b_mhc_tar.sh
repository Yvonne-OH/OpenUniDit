#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
gwp=/home/jovyan/pfs/users/zhanqian.wu/env/gwp
# The platform's conda subprocess can print plugin banners into its activation output.
# Reuse an outer gwp setup, including when the wrapper selected it through PATH.
if [[ "${CONDA_PREFIX:-}" != "$gwp" && "$(command -v python || true)" != "$gwp/bin/python" ]]; then
  source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh || {
    echo "Failed to initialize gwp" >&2
    exit 1
  }
fi
export PATH="$gwp/bin:$PATH"
export PYTHON_EXEC="$gwp/bin/python"
config="${CONFIG:-${1:-configs/stage_1_5b_qwen35_4b_mhc_tar.yaml}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export YISU_IMAGE_TIMEOUT_SECONDS=48 YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=120 YISU_IMAGE_PERSISTENT_DECODER=0

# 与 stage_1_5b_tar.sh 一致：百度注入的 RANK 表示节点编号，WORLD_SIZE 表示节点数。
gpus=${NPROC_PER_NODE:-${GPUS:-8}}
rank=${RANK:-${NODE_RANK:-}}
if [[ -n "$rank" ]]; then
  nodes=${WORLD_SIZE:-${NNODES:-1}}
else
  nodes=1
  rank=0
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
port=${MASTER_PORT:-29543}

# CPU preflight also works on a login node. Ignore platform banners when reading output_dir.
output=$("$PYTHON_EXEC" - "$config" <<'PY' | grep '^/'
import json
import os
import re
import sys
from pathlib import Path
from safetensors import SafetensorError, safe_open
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
tc = cfg["components"]["text_encoder"]
encoder = Path(tc["pretrained_path"])
manifest = json.loads((encoder / "download_manifest.json").read_text())
assert manifest["revision"] == tc["revision"], "4B encoder revision mismatch"
for name, size in manifest["files"].items():
    path = encoder / name
    assert path.is_file() and (size is None or path.stat().st_size == size), path
width = json.loads((encoder / "config.json").read_text())["text_config"]["hidden_size"]
assert cfg["model"]["text_dim"] == width == 2560
training = cfg["training"]
pin = Path(training["init_from"])
validation = Path(training["init_validation_dir"])

def check_dense(path):
    with safe_open(path, framework="numpy") as weights:
        if weights.get_slice("text_in.weight").get_shape() != [cfg["model"]["dim"], width]:
            raise ValueError(f"Unexpected text projection in {path}")

if pin.is_symlink():
    snapshot = pin.resolve(strict=True)
    check_dense(snapshot)
else:
    steps = sorted((int(p.name[5:]), p) for p in validation.iterdir()
                   if re.fullmatch(r"step-[0-9]+", p.name) and p.is_dir())
    for step, directory in reversed(steps):
        candidate = validation.parent / "checkpoints" / "weights" / (directory.name + ".safetensors")
        if not (directory / "rank0.json").is_file() or not candidate.is_file():
            continue
        try:
            check_dense(candidate)
        except (FileNotFoundError, SafetensorError) as exc:
            print(f"Skipping unreadable checkpoint {candidate}: {exc}", file=sys.stderr)
            continue
        snapshot = candidate
        break
    else:
        raise RuntimeError(f"No validated step with a readable dense checkpoint under {validation.parent}")
    if os.environ.get("GIGA_CHECK_ONLY") != "1" and os.environ.get("GIGA_DRY_RUN") != "1":
        pin.parent.mkdir(parents=True, exist_ok=True)
        try:
            pin.symlink_to(snapshot)
        except FileExistsError:
            if not pin.is_symlink():
                raise RuntimeError(f"Expected a checkpoint symlink at {pin}")
            snapshot = pin.resolve(strict=True)
            check_dense(snapshot)
print(f"mHC initialization from validated dense checkpoint: {snapshot}", file=sys.stderr)
cache = Path(cfg["dataset"]["index_cache_dir"]).resolve(strict=True)
assert all(any(cache.glob(f"signature_{kind}_*.json")) for kind in ("tar", "t2i")), cache
out = Path(cfg["output_dir"])
assert out.is_absolute()
# The trainer passes output_dir/dataset_index_cache directly to the dataset.
# Concurrent node starts can all attempt the same atomic symlink creation.
out.mkdir(parents=True, exist_ok=True)
link = out / "dataset_index_cache"
try:
    link.symlink_to(cache, target_is_directory=True)
except FileExistsError:
    if link.resolve(strict=True) != cache:
        raise RuntimeError(f"Index cache points to a different dataset: {link}")
print(out)
PY
)
if [[ "${GIGA_CHECK_ONLY:-0}" == 1 ]]; then
  echo "4B encoder, dense initialization and frozen index found: $output"
  exit 0
fi

export GIGA_NODE_RANK="$rank"
export GIGA_RUN_LOG_DIR="$output/logs/${GIGA_RUN_ID:-multi_${MASTER_ADDR}_${port}}"
cmd=("$PYTHON_EXEC" -m torch.distributed.run --nnodes "$nodes" --nproc_per_node "$gpus" --node_rank "$rank"
     --master_addr "$MASTER_ADDR" --master_port "$port"
     scripts/train/train_giga_exp.py --config "$config")
if [[ "${GIGA_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')
libcuda="/usr/lib/x86_64-linux-gnu/libcuda.so.$driver"
if [[ -f "$libcuda" ]]; then
  export LD_PRELOAD="$libcuda${LD_PRELOAD:+:$LD_PRELOAD}"
fi
# Keep Triton compilation traffic off shared PFS.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/giga_qwen35_4b_mhc_${UID}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/giga_qwen35_4b_mhc_${UID}/inductor}"
mkdir -p "$GIGA_RUN_LOG_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
printf 'Launching %s node(s), %s GPU(s)/node, node rank %s with %s\n' "$nodes" "$gpus" "$rank" "$PYTHON_EXEC"
"${cmd[@]}" 2>&1 | tee -a "$GIGA_RUN_LOG_DIR/console.node-$rank.log"
