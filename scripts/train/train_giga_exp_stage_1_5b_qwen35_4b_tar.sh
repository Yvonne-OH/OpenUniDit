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
config="${CONFIG:-${1:-configs/stage_1_5b_qwen35_4b_tar.yaml}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.6}"
export YISU_IMAGE_TIMEOUT_SECONDS=48 YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=120 YISU_IMAGE_PERSISTENT_DECODER=0

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
port=${MASTER_PORT:-29544}

# CPU preflight also works on a login node. Ignore platform banners when reading output_dir.
output=$("$PYTHON_EXEC" - "$config" <<'PY' | grep '^/'
import json
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

def check_dense(path):
    with safe_open(path, framework="numpy") as weights:
        if weights.get_slice("text_in.weight").get_shape() != [cfg["model"]["dim"], width]:
            raise ValueError(f"Unexpected text projection in {path}")

# 非 mHC: init_from 可选 —— 设置时校验 dense 权重文件; 未设置时依赖 resume_mode=auto
# 从 output 目录自动继承最新 checkpoint。
init_from = training.get("init_from")
if init_from:
    init_from = Path(init_from)
    assert init_from.is_file(), f"init_from 必须是已存在的权重文件: {init_from}"
    try:
        check_dense(init_from)
    except (FileNotFoundError, SafetensorError) as exc:
        raise RuntimeError(f"init_from checkpoint unreadable: {exc}") from exc
    print(f"Non-mHC initialization from dense checkpoint: {init_from}", file=sys.stderr)
else:
    print("init_from 未设置; 依赖 resume_mode=auto 从 output 目录继承最新 checkpoint", file=sys.stderr)

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
  echo "4B encoder, dense initialization and copied index cache found: $output"
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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/giga_qwen35_4b_tar_${UID}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/giga_qwen35_4b_tar_${UID}/inductor}"
mkdir -p "$GIGA_RUN_LOG_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
printf 'Launching %s node(s), %s GPU(s)/node, node rank %s with %s\n' "$nodes" "$gpus" "$rank" "$PYTHON_EXEC"
"${cmd[@]}" 2>&1 | tee -a "$GIGA_RUN_LOG_DIR/console.node-$rank.log"
