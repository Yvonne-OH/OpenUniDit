#!/usr/bin/env bash
# 8 卡对比实验启动脚本: 单机 8 卡对半分, 同时从 0 训练 mHC 与 baseline。
#   - GPU 0-3: mHC 4-stream   (configs/mhc_from0.yaml    -> scripts/train/train_giga_exp_mhc.sh)
#   - GPU 4-7: baseline 28L   (configs/stage_1_from0.yaml -> scripts/train/train_giga_exp_stage_1.sh)
# 两份 yaml 数据源一致 (仅 5 个源), lr/调度/seed 全部对齐, 仅模型结构不同。
# 停止: kill $(cat /tmp/giga_cmp_from0.pids)  或分别 kill 两个 PID。
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
cd "$ROOT"

MHC_CONFIG="${MHC_CONFIG:-configs/mhc_from0.yaml}"
BASE_CONFIG="${BASE_CONFIG:-configs/stage_1_from0.yaml}"
GPUS_PER_RUN="${GPUS_PER_RUN:-4}"

# 两路各 4 卡, 端口错开 (29511/29500 已与各自脚本默认一致, 这里显式固定防环境残留)
export MASTER_ADDR=127.0.0.1
RUN_ID_PREFIX="${RUN_ID_PREFIX:-cmp_from0_$(date +%Y%m%d_%H%M%S)}"

# mHC: GPU 0-3
(
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export NPROC_PER_NODE="$GPUS_PER_RUN"
  export MASTER_PORT=29511
  export GIGA_RUN_ID="${RUN_ID_PREFIX}_mhc"
  exec bash scripts/train/train_giga_exp_mhc.sh "$MHC_CONFIG"
) &
MHC_PID=$!

# baseline: GPU 4-7
(
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  export NPROC_PER_NODE="$GPUS_PER_RUN"
  export MASTER_PORT=29500
  export GIGA_RUN_ID="${RUN_ID_PREFIX}_baseline"
  exec bash scripts/train/train_giga_exp_stage_1.sh "$BASE_CONFIG"
) &
BASE_PID=$!

echo "$MHC_PID $BASE_PID" > /tmp/giga_cmp_from0.pids
echo "[cmp] mHC pid=$MHC_PID (GPU 0-3, port 29511), baseline pid=$BASE_PID (GPU 4-7, port 29500)"
echo "[cmp] pids saved to /tmp/giga_cmp_from0.pids"

STATUS_MHC=0
STATUS_BASE=0
wait "$MHC_PID" || STATUS_MHC=$?
wait "$BASE_PID" || STATUS_BASE=$?

echo "[cmp] finished: mHC exit=$STATUS_MHC, baseline exit=$STATUS_BASE"
# 任一路非 0 即以非 0 退出, 便于外层感知失败
if (( STATUS_MHC != 0 || STATUS_BASE != 0 )); then
  exit 1
fi
exit 0
