#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh
cd "$ROOT"
export CONFIG="${CONFIG:-${1:-configs/stage_1_5b_qwen35_9b_mhc_tar.yaml}}"
export MASTER_PORT="${MASTER_PORT:-29539}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

# Local model and frozen source checkpoint must be complete before torchrun.
python scripts/train/check_qwen35_9b_mhc.py --config "$CONFIG"

if [[ "${GIGA_CHECK_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec bash scripts/train/train_giga_exp_stage_1_5b_tar.sh "$CONFIG"
