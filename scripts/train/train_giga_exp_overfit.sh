#!/usr/bin/env bash
# 过拟合实验启动器: configs/stage_1_overfit.yaml (monet_coyo 2k 条小数据 + 纯 FM, 无 KD)。
# 结构与 train_giga_exp_stage_1.sh 一致, 默认 8 卡; 单卡调试时 GPUS=1 bash scripts/train/train_giga_exp_overfit.sh。
# 用法:
#   nohup bash scripts/train/train_giga_exp_overfit.sh > /tmp/overfit_launch.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=7 GPUS=1 bash scripts/train/train_giga_exp_overfit.sh
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

# 图像样本加载超时预算: DecodeBudget 将样本预算均分给最多 3 个候选,
# 避免多机 192 路并发冷缓存启动时, Lustre 首读 ~11s 的样本被判超时换图。
export YISU_IMAGE_TIMEOUT_SECONDS=48
export YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=120
export YISU_IMAGE_PERSISTENT_DECODER=0

# NCCL 只在 WARN 及以上时输出: 单卡跑不到 NCCL, 多卡时保留异常可见性
export NCCL_DEBUG=WARN

CONFIG="${CONFIG:-${1:-configs/stage_1_overfit.yaml}}"
GPUS="${NPROC_PER_NODE:-${GPUS:-8}}"

WORLD_SIZE=1
RANK=0
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29555}"

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
  RUN_ID="${GIGA_RUN_ID}_$(date +%Y%m%d_%H%M%S)"
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
