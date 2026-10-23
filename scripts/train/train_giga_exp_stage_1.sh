#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
cd "$ROOT"

# 修复 CUDA Error 803: ldconfig 优先命中 cuda-12.8/compat 的旧版 libcuda (570.86.10),
# 与驱动 (nvidia-smi 所报版本) 不匹配; 预加载 /usr/lib 下与驱动同版本的 libcuda。
# 不用 head -1: pipefail 下 head 提前关管道会让 nvidia-smi 偶发 SIGPIPE(141) 直接杀死脚本
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}"
if [[ -f "$LIBCUDA" && ":${LD_PRELOAD:-}:" != *":$(basename "$LIBCUDA"):*" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

# 图像样本加载超时预算: DecodeBudget 将样本预算均分给最多 3 个候选 (48/3=16s/次),
# 避免多机 192 路并发冷缓存启动时, Lustre 首读 ~11s 的样本被判超时换图。
# 单次尝试有效超时 = min(YISU_IMAGE_TIMEOUT_SECONDS, 样本预算/剩余候选数)。
export YISU_IMAGE_TIMEOUT_SECONDS=48
export YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=48
# 常驻图像解码子进程: 免每样本 fork+import+lmdb env open; 故障自动回退一次性隔离路径
export YISU_IMAGE_PERSISTENT_DECODER=1

CONFIG="${CONFIG:-${1:-configs/stage_1.yaml}}"
GPUS="${NPROC_PER_NODE:-${GPUS:-8}}"

# 单机多卡: 设置 NPROC_PER_NODE 或 GPUS。
# 多机多卡: 每台机器设置 WORLD_SIZE=机器数、RANK=机器编号、MASTER_ADDR、MASTER_PORT、NPROC_PER_NODE。
# 注意: shell 启动时的 WORLD_SIZE 表示机器数；torchrun 启动 worker 后会自行注入全局进程 WORLD_SIZE。
EXTERNAL_RANK="${RANK:-${NODE_RANK:-}}"
EXTERNAL_WORLD_SIZE="${WORLD_SIZE:-}"
if [[ -n "$EXTERNAL_RANK" ]]; then
  WORLD_SIZE="${EXTERNAL_WORLD_SIZE:-1}"
  RANK="$EXTERNAL_RANK"
else
  # 避免继承上层残留的 WORLD_SIZE=8 将单机误判为 8 台机器。
  WORLD_SIZE=1
  RANK=0
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

OUTPUT_DIR="$(python - "$CONFIG" <<'PY'
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
  scripts/train/train_giga_exp.py \
  --config "$CONFIG")

printf 'command: %q ' "${CMD[@]}" | tee "$RUN_LOG_DIR/command.txt"
printf '\n' | tee -a "$RUN_LOG_DIR/command.txt"

# Keep the launcher output in this run's log directory while preserving exit status.
set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG_DIR/console.log"
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
