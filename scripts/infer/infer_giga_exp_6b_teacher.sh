#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
cd "$ROOT"

# 与 train_giga_exp_stage_1.sh 相同的 libcuda 预加载修复 (CUDA Error 803)
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}"
if [[ -f "$LIBCUDA" && ":${LD_PRELOAD:-}:" != *":$(basename "$LIBCUDA"):*" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

OUT_DIR="${OUT_DIR:-$ROOT/outputs/teacher_6b_infer}"
mkdir -p "$OUT_DIR"

PROMPT_ARGS=()
# 未显式传 --prompt 时注入默认 sanity check prompts (3 条不同风格)
if ! printf '%s\n' "$@" | grep -q -- '--prompt'; then
  PROMPT_ARGS=(
    --prompt "a majestic snow leopard resting on a rocky cliff, golden hour light, wildlife photography"
    --prompt "a bustling night market in Taipei, neon signs, steam rising from food stalls, cinematic"
    --prompt "watercolor painting of a small red sailboat on a calm lake surrounded by autumn mountains"
  )
fi

python scripts/infer/infer_giga_exp_6b_teacher.py \
  --output "$OUT_DIR/output.png" \
  "${PROMPT_ARGS[@]}" \
  "$@"
