#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/pfs/users/zhanqian.wu/ckpt/flux-3-image-klein-base}"
OUTPUT="${OUTPUT:-${ROOT}/outputs/output.png}"
PROMPT="${PROMPT:-A cinematic photograph of a red fox standing in a snowy pine forest at golden hour, detailed winter fur, warm rim light, shallow depth of field.}"

export PYTHONPATH="${ROOT}/diffusers/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python "${ROOT}/infer.py" "${PROMPT}" \
  --model-path "${MODEL_PATH}" \
  --output "${OUTPUT}" \
  --height "${HEIGHT:-1024}" \
  --width "${WIDTH:-1024}" \
  --steps "${STEPS:-50}" \
  --guidance-scale "${GUIDANCE_SCALE:-3.5}" \
  --seed "${SEED:-0}" \
  "$@"
