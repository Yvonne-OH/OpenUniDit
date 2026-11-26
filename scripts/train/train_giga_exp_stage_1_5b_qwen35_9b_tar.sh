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
config="${CONFIG:-${1:-configs/stage_1_5b_qwen35_9b_tar.yaml}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.6}"
export YISU_IMAGE_TIMEOUT_SECONDS=48 YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=120
export YISU_IMAGE_PERSISTENT_DECODER="${YISU_IMAGE_PERSISTENT_DECODER:-1}"
export YISU_TAR_MAX_OPEN_FILES="${YISU_TAR_MAX_OPEN_FILES:-8}"
# Decoder IPC stays on node-local storage, never the shared training output dir.
export YISU_IMAGE_TMP_DIR="${YISU_IMAGE_TMP_DIR:-/tmp/giga-image-decode-${UID}}"
mkdir -p "$YISU_IMAGE_TMP_DIR"

# 与 stage_1_5b_qwen35_4b_mhc_tar.sh 一致：百度注入的 RANK 表示节点编号，WORLD_SIZE 表示节点数。
gpus=${NPROC_PER_NODE:-${GPUS:-8}}
rank=${RANK:-${NODE_RANK:-}}
if [[ -n "$rank" ]]; then
  nodes=${WORLD_SIZE:-${NNODES:-1}}
else
  nodes=1
  rank=0
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
port=${MASTER_PORT:-29545}

# CPU preflight also works on a login node. Ignore platform banners when reading output_dir.
output=$("$PYTHON_EXEC" - "$config" <<'PY' | grep '^/'
import json
import sys
from pathlib import Path
from safetensors import safe_open
import torch
from models.model_factory import build_diffusion_model
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
tc = cfg["components"]["text_encoder"]
encoder = Path(tc["pretrained_path"])
manifest = json.loads((encoder / "download_manifest.json").read_text())
assert manifest["revision"] == tc["revision"], "9B encoder revision mismatch"
for name, size in manifest["files"].items():
    path = encoder / name
    assert path.is_file() and (size is None or path.stat().st_size == size), path
width = json.loads((encoder / "config.json").read_text())["text_config"]["hidden_size"]
assert cfg["model"]["text_dim"] == width == 4096
training = cfg["training"]

# The dense warm-start loader preserves uniformly shaped tensors. Check its only exception.
# init_from is optional: once checkpoints/weights has a snapshot, it can be removed from the
# config and resume_mode: auto takes over.
init = training.get("init_from")
if init:
    init = Path(init)
    with torch.device("meta"):
        expected = build_diffusion_model(cfg).state_dict()
    with safe_open(str(init), framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(expected):
            raise ValueError(f"Checkpoint tensor names differ from the target dense model: {init}")
        different = {k: weights.get_slice(k).get_shape() for k, p in expected.items()
                     if weights.get_slice(k).get_shape() != list(p.shape)}
        if different != {"text_in.weight": [cfg["model"]["dim"], 2560]}:
            raise ValueError(f"Unexpected warm-start shape changes: {different}")
        print(f"Warm start {init.name}: inherit {len(expected)-1} tensors; initialize "
              f"text_in.weight {list(expected['text_in.weight'].shape)}", file=sys.stderr)

cache = Path(cfg["dataset"]["index_cache_dir"]).resolve(strict=True)
assert all(any(cache.glob(f"signature_{kind}_*.json")) for kind in ("tar", "t2i")), cache
out = Path(cfg["output_dir"])
assert out.is_absolute()
out.mkdir(parents=True, exist_ok=True)
# 缓存应已完整复制到 output_dir/dataset_index_cache (训练进程按此路径读取)。
local_cache = out / "dataset_index_cache"
if not local_cache.is_dir():
    raise RuntimeError(f"Missing copied index cache at {local_cache}")
print(out)
PY
)
if [[ "${GIGA_CHECK_ONLY:-0}" == 1 ]]; then
  echo "9B encoder, dense initialization and copied index cache found: $output"
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

# Use the same SwanLab account as the 4B run when the environment has no explicit key.
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  SWANLAB_API_KEY=$("$PYTHON_EXEC" - <<'AUTH'
from pathlib import Path
import yaml
reference = Path("configs/stage_1_5b_qwen35_4b_tar.yaml")
print(yaml.safe_load(reference.read_text())["logging"]["swanlab"].get("api_key") or "")
AUTH
)
  export SWANLAB_API_KEY
fi
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')
libcuda="/usr/lib/x86_64-linux-gnu/libcuda.so.$driver"
if [[ -f "$libcuda" ]]; then
  export LD_PRELOAD="$libcuda${LD_PRELOAD:+:$LD_PRELOAD}"
fi
# Keep Triton compilation traffic off shared PFS.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/giga_qwen35_9b_tar_${UID}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/giga_qwen35_9b_tar_${UID}/inductor}"
mkdir -p "$GIGA_RUN_LOG_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
printf 'Launching %s node(s), %s GPU(s)/node, node rank %s with %s\n' "$nodes" "$gpus" "$rank" "$PYTHON_EXEC"
"${cmd[@]}" 2>&1 | tee -a "$GIGA_RUN_LOG_DIR/console.node-$rank.log"
