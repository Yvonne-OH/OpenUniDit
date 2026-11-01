#!/usr/bin/env bash
# KD 推理启动器: prompt / 输出约定完全继承 infer_giga_exp.sh (20 条标准 prompt,
# outputs/pretrain_v2/vis_token_exp/<ckpt 名>/shift_X_cfg_Y/), 输出目录加 _kd 后缀
# (同 infer_giga_exp_bos.sh 的 _5090 模式)。CFGS/SHIFTS 与 infer_giga_exp.sh 对齐 (3 6 9 12 × 3 6), STEPS=50。
# GPU 自动挑显存占用最低的一张, 训练/挖矿在跑时不抢卡。
# 用法:
#   bash scripts/infer/infer_giga_exp_kd.sh                     # 最新 step-* + 20 条默认 prompt
#   CHECKPOINT=/path/step-00235000.safetensors bash scripts/infer/infer_giga_exp_kd.sh
#   CFGS=(3 6) SHIFTS=(6) bash scripts/infer/infer_giga_exp_kd.sh   # 环境变量覆盖组合
#   GPU=6 bash scripts/infer/infer_giga_exp_kd.sh               # 显式指定 GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── conda 环境 (推理依赖: torch / transformers / diffusers) ──
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh

# 修复 CUDA Error 803: ldconfig 会命中 cuda-12.8/compat 的旧 libcuda, 预加载与驱动同版本的库
# (不用 head -1: pipefail 下 head 提前关管道会让 nvidia-smi 偶发 SIGPIPE 直接杀死脚本)
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}"
if [[ -f "$LIBCUDA" && ":${LD_PRELOAD:-}:" != *":$(basename "$LIBCUDA"):*" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

# ── 输入: 与 infer_giga_exp.sh 同一组 20 条标准 prompt (两种方式可同时使用) ──
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

# ── 输出与权重 (支持环境变量覆盖, 如 CHECKPOINT=xxx.safetensors bash infer_giga_exp_kd.sh) ──
OUTPUT_DIR="${OUTPUT_DIR:-}"   # 空 = outputs/pretrain_v2/vis_token_exp/<ckpt 名>_kd/，下设 shift_* 子目录
CHECKPOINT="${CHECKPOINT:-}"   # 空 = 自动取 config output_dir/checkpoints/weights 下最新 step-*
CONFIG="${CONFIG:-${ROOT}/configs/stage_1_kd_v3_traj.yaml}"

# ── 采样参数: CFGS/SHIFTS/SEED 等组合约定与 infer_giga_exp.sh 一致; 仅 STEPS 取 KD 标准 50 ──
HEIGHT=512      # 必须是 32 的倍数
WIDTH=512
STEPS=50        # KD 标准步数
CFGS=(3 6 9 12) # guidance scale 组合, 与 SHIFTS 笛卡尔积
NEGATIVE_PROMPT="<|im_end|>"   # 与训练一致: EOS 无条件分支
SHIFTS=(3 6)    # 同一权重、prompt 和 seed, 对比 shift × cfg 组合
SEED=42         # 所有 prompt 固定同一噪声种子
BATCH_SIZE="${BATCH_SIZE:-4}"   # 越大越快(结果与 batch 大小无关)
GPU=""          # 空 = 自动挑显存占用最低的 GPU (训练/挖矿在跑时不抢卡)

# 共用启动时选定的 checkpoint，避免训练期间新权重写入导致对比不一致。
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
  OUTPUT_DIR="$ROOT/outputs/pretrain_v2/vis_token_exp/${CHECKPOINT_NAME%.*}_kd"
fi

ARGS=(
  "${SCRIPT_DIR}/infer_giga_exp_kd.py"
  --config "${CONFIG}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --steps "${STEPS}"
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

if [[ -z "${GPU}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # 自动选卡: 显存占用最低的一张 (nvidia-smi CSV: index, memory.used)
  GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | sed -n '1p' | cut -d',' -f1)"
fi

for SHIFT in "${SHIFTS[@]}"; do
  for CFG in "${CFGS[@]}"; do
    COMBO_OUTPUT_DIR="${OUTPUT_DIR}/shift_${SHIFT}_cfg_${CFG}"
    printf '\n开始推理 shift=%s cfg=%s → %s\n' "$SHIFT" "$CFG" "$COMBO_OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES="${GPU}" python "${ARGS[@]}" \
      --shift "$SHIFT" --guidance-scale "$CFG" --output-dir "$COMBO_OUTPUT_DIR"
  done
done
printf '\nshift×cfg 共 %d 组推理完成 → %s\n' \
  "$(( ${#SHIFTS[@]} * ${#CFGS[@]} ))" "$OUTPUT_DIR"
