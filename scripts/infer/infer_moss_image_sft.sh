#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 直接在这里修改推理参数。
PROMPT='Satirical documentary illustration, open-plan office of a modern tech company, rows of workstations stretching like farm ridges. Multiple programmers sit at desks, tiredly staring at computer screens and writing code. Desks are cluttered with requirement documents, coffee mugs and error popups. A manager stands beside the workstation in the aisle, leaning forward with an angry expression, loudly reprimanding the programmer while clutching progress reports. No physical violence in the scene, only conflict under heavy work pressure. Harsh overhead office lighting, stuffy and oppressive atmosphere. Muted earth tones mixed with cool grays, realistic painting style, tense and heavy mood. Rich details of laptops, sticky notes and water cups, wide composition with depth, satirical workplace metaphor illustration, high detail, 8k'
MODEL_PATH="/home/jovyan/pfs/users/zhanqian.wu/ckpt/MOSS-Image-SFT"
OUTPUT="${ROOT}/outputs/moss_image_plantation.png"
HEIGHT=1024
WIDTH=1024
STEPS=50
GUIDANCE_SCALE=3.0
SEED=42
NEGATIVE_PROMPT=''
NUM_IMAGES=1
MAX_SEQUENCE_LENGTH=4096
CPU_OFFLOAD=false

ARGS=(
  "${ROOT}/infer_moss_image.py"
  "${PROMPT}"
  --model-path "${MODEL_PATH}"
  --output "${OUTPUT}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --steps "${STEPS}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --negative-prompt "${NEGATIVE_PROMPT}"
  --num-images "${NUM_IMAGES}"
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}"
  --seed "${SEED}"
)

if [[ "${CPU_OFFLOAD}" == true ]]; then
  ARGS+=(--cpu-offload)
fi

exec python "${ARGS[@]}"
