#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 输入: 下面两种方式可同时使用 (结果按 PROMPTS 在前、文件在后拼接) ──
# 1) 直接在 bash 里写 prompt 列表 (数组, 每个元素一条)
PROMPTS=(
  'A fluffy orange cat sitting on a windowsill, warm morning light, shallow depth of field, photorealistic.'
  'A young woman in a red blazer, business suit, modern office background, professional portrait, natural light.'
  'Snow-capped mountain range at sunrise, golden alpenglow, alpine lake reflection, wide landscape, cinematic.'
  'A golden retriever puppy running through autumn leaves in a park, motion blur, warm afternoon sun, photorealistic.'
  'Bustling Asian night market street food stall, steaming dumplings, neon signs, bokeh lights, documentary photo.'
  'Modern glass skyscraper facade at blue hour, geometric reflections, low angle shot, architectural photography.'
  'A vintage 1970s motorcycle parked on a desert road, chrome details, sunset backlight, film photography style.'
  'Still life of fresh fruits in a ceramic bowl on rustic wooden table, soft window light, Dutch masters style oil painting.'
  'Cozy wooden cabin in snowy pine forest at dusk, warm light from windows, falling snow, atmospheric.'
  'Surfer riding a turquoise wave at sunset, spray in the air, tropical beach, action sports photography.'
  'Book cover design, minimal typography with the title STORIES OF THE SEA, deep blue ocean waves illustration, flat design.'
  'An astronaut floating in front of a glowing nebula, detailed spacesuit reflections, science fiction concept art.'
  'Assorted sushi platter on black slate, salmon nigiri and tuna rolls, wasabi and ginger, studio food photography.'
  'A red tractor plowing golden wheat fields at dawn, dust in the air, rural farmland, wide shot.'
  'Jazz quartet performing on a dim stage, double bass and saxophone, warm spotlight, smoky atmosphere.'
  'Luxury perfume bottle on silk fabric, crystal glass reflections, studio product photography, elegant lighting.'
  'Children book illustration of a small fox wearing a scarf walking through a birch forest, soft watercolor style.'
  'Soccer player striking the ball mid-air in a packed stadium at night, floodlights, dynamic sports photo.'
  'Grand old library interior with tall wooden bookshelves, spiral staircase, reading lamps, warm tones.'
  'Colorful hot air balloons floating over Cappadocia rock valleys at sunrise, pastel sky, travel photography.'
)
# 2) 或指向文本文件 (每行一条 prompt, 空行忽略); 留空则不启用
PROMPT_FILE=""

# ── 输出与权重 (支持环境变量覆盖, 如 CHECKPOINT=xxx.safetensors bash infer_giga_exp.sh) ──
OUTPUT_DIR="${OUTPUT_DIR:-}"   # 空 = outputs/pretrain_v2/vis_token_exp/<ckpt 名>/，下设 shift_* 子目录
CHECKPOINT="${CHECKPOINT:-}"   # 空 = 自动取 stage_1 output_dir/checkpoints/weights 下最新 step-*
CONFIG="${CONFIG:-${ROOT}/configs/stage_1.yaml}"

# ── 采样参数 (默认与训练验证一致) ──
HEIGHT=512      # 必须是 32 的倍数; 默认横屏 192x128 (WxH, 对应训练桶 192x128)
WIDTH=512
STEPS=40        # 训练验证用 20; 质量优先可加到 40+
GUIDANCE_SCALE=3.0
NEGATIVE_PROMPT="blurry, distorted, deformed, low quality"
SHIFTS=(1 3 6 9) # 同一权重、prompt 和 seed，对比四组采样 shift
SEED=42         # 所有 prompt 固定同一噪声种子
BATCH_SIZE="${BATCH_SIZE:-4}"   # 越大越快(结果与 batch 大小无关)
GPU=""          # 空 = 自动用最后一张 GPU (如 8 卡机为 7)

# 四组共用启动时选定的 checkpoint，避免训练期间新权重写入导致对比不一致。
CHECKPOINT="$(python - "$ROOT" "$CONFIG" "$CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import yaml
root, config, checkpoint = sys.argv[1:]
config = Path(config)
if not config.is_absolute():
    config = Path(root) / config
cfg = yaml.safe_load(config.read_text())
target = Path(checkpoint) if checkpoint else Path(cfg["output_dir"]) / "checkpoints" / "weights"
if target.is_dir():
    candidates = list(target.glob("step-*.safetensors")) + list(target.glob("step-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"目录里没有 step-* 权重文件: {target}")
    target = max(candidates, key=lambda p: int(p.stem.split("-")[1]))
if not target.is_file():
    raise FileNotFoundError(f"权重不存在: {target}")
print(target.resolve())
PY
)"
if [[ -z "$OUTPUT_DIR" ]]; then
  CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
  OUTPUT_DIR="$ROOT/outputs/pretrain_v2/vis_token_exp/${CHECKPOINT_NAME%.*}"
fi

ARGS=(
  "${SCRIPT_DIR}/infer_giga_exp.py"
  --config "${CONFIG}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --steps "${STEPS}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --negative-prompt "${NEGATIVE_PROMPT}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
)
for prompt in "${PROMPTS[@]}"; do
  ARGS+=(--prompt "${prompt}")
done
if [[ -n "${PROMPT_FILE}" ]]; then
  ARGS+=(--prompt-file "${PROMPT_FILE}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -z "${GPU}" ]]; then
  GPU=$(( $(nvidia-smi -L | wc -l) - 1 ))
fi
for SHIFT in "${SHIFTS[@]}"; do
  SHIFT_OUTPUT_DIR="${OUTPUT_DIR}/shift_${SHIFT}"
  printf '\n开始推理 shift=%s → %s\n' "$SHIFT" "$SHIFT_OUTPUT_DIR"
  CUDA_VISIBLE_DEVICES="${GPU}" python "${ARGS[@]}" \
    --shift "$SHIFT" --output-dir "$SHIFT_OUTPUT_DIR"
done
printf '\n四组 shift 推理完成 → %s\n' "$OUTPUT_DIR"
