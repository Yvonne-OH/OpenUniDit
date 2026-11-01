#!/usr/bin/env bash
# mHC 对比实验独立启动脚本: 只改 CONFIG 与训练入口, 其余环境逻辑与
# train_giga_exp_stage_1.sh 保持一致 (不触碰主分支脚本)。
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
cd "$ROOT"

# CUDA 803 修复: 预加载与驱动同版本的 libcuda
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}"
if [[ -f "$LIBCUDA" && ":${LD_PRELOAD:-}:" != *":$(basename "$LIBCUDA"):*" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

# 与主分支一致的数据加载超时预算
export YISU_IMAGE_TIMEOUT_SECONDS=48
export YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=120
export YISU_IMAGE_PERSISTENT_DECODER=0
export NCCL_DEBUG=WARN

# mHC 第一版 eager: 不开 compile, 避免 Sinkhorn 循环在 inductor 下引入未知数值问题
unset TORCH_LOGS || true

CONFIG="${CONFIG:-${1:-configs/mhc_2stream.yaml}}"
GPUS="${NPROC_PER_NODE:-${GPUS:-8}}"

EXTERNAL_RANK="${RANK:-${NODE_RANK:-}}"
EXTERNAL_WORLD_SIZE="${WORLD_SIZE:-}"
if [[ -n "$EXTERNAL_RANK" ]]; then
  WORLD_SIZE="${EXTERNAL_WORLD_SIZE:-1}"
  RANK="$EXTERNAL_RANK"
else
  WORLD_SIZE=1
  RANK=0
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"   # 与主分支 29500 错开, 允许并行跑对比实验

# 平台注入的 "Module imported successfully" 行会污染命令替换, 只保留 / 开头的 output_dir
OUTPUT_DIR="$(python - "$CONFIG" <<'PY' | grep '^/'
import sys, yaml
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_absolute():
    path = Path.cwd() / path
print(yaml.safe_load(path.read_text())["output_dir"])
PY
)"
if [[ -n "${GIGA_RUN_ID:-}" ]]; then
  RUN_ID="$GIGA_RUN_ID"
elif (( WORLD_SIZE > 1 )); then
  RUN_ID="multi_${MASTER_ADDR}_${MASTER_PORT}_n${WORLD_SIZE}_$(basename "$CONFIG" .yaml)"
else
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
RUN_LOG_DIR="$OUTPUT_DIR/logs/$RUN_ID"
mkdir -p "$RUN_LOG_DIR"
export GIGA_RUN_LOG_DIR="$RUN_LOG_DIR"
export GIGA_NODE_RANK="$RANK"

CMD=(torchrun \
  --nproc_per_node "$GPUS" \
  --nnodes "$WORLD_SIZE" \
  --node_rank "$RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  scripts/train/train_giga_exp_mhc.py \
  --config "$CONFIG")

printf 'command: %q ' "${CMD[@]}" | tee "$RUN_LOG_DIR/command.txt"
printf '\n' | tee -a "$RUN_LOG_DIR/command.txt"

set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG_DIR/console.log"
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
