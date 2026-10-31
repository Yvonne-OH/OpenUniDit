#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
cd "$ROOT"

CONFIG="${CONFIG:-${1:-configs/stage_1_2.yaml}}"
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

OUTPUT_DIR="$(python - "$CONFIG" <<'PY' | grep '^/'
# 平台注入的 ftelasticagent 钩子会往 stdout 打 "Module imported successfully: ...",
# 命令替换会把它一起捕获; 只保留以 / 开头的行 = yaml 里的 output_dir。
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
