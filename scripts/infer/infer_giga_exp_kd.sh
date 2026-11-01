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

# ── 输入: 短/中/长三组 prompt, 题目相同、长度不同 (对比 caption 长度敏感性) ──
# short ≈ 20 词 (沿用原 20 条标准 prompt), mid ≈ 50 词, long ≈ 100 词
PROMPTS_SHORT=(
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
PROMPTS_MID=(
  'A fluffy orange cat with bright amber eyes sits on a sun-warmed windowsill, golden morning light streaming through sheer linen curtains behind it, a leafy houseplant softly blurred in the foreground, shallow depth of field isolating the face, ultra-detailed fur, photorealistic close-up on an 85mm lens.'
  'A confident young woman wearing a tailored red blazer over a white silk blouse stands in a modern glass office, floor-to-ceiling windows behind her revealing a soft-focus city skyline, diffused daylight from the left sculpting her features, shallow depth of field, professional corporate portrait, natural skin texture.'
  'A vast snow-capped mountain range at sunrise, golden alpenglow igniting the highest peaks while deep blue shadow fills the valleys below, a mirror-still alpine lake in the foreground reflecting the ridgeline, thin mist drifting over the water, wide cinematic landscape, 16mm lens, high dynamic range.'
  'A joyful golden retriever puppy bounding through deep drifts of fallen autumn leaves in a quiet park, leaves kicked into the air mid-stride, slight motion blur on the paws, low afternoon sunlight backlighting the scene in warm amber tones, breath visible in the crisp air, photorealistic action shot.'
  'A bustling Asian night market street food stall with bamboo steamers releasing clouds of steam, a vendor cooking over open flame, strings of red and cyan neon signs receding down the narrow street, crowds dissolved into warm bokeh, lens flare from a street lamp, documentary photography, 35mm.'
  'A modern glass skyscraper facade photographed from directly below at blue hour, thousands of lit windows forming a strict geometric grid, mirror-like reflections of the deep blue sky sliding across the curved glass panels, crisp architectural lines converging toward the sky, tripod long exposure, architectural photography.'
  'A restored vintage 1970s motorcycle parked at an angle on an empty desert road, polished chrome catching the low sunset backlight, a long shadow stretching across cracked asphalt, heat shimmer on the horizon, distant mesas in dark silhouette, warm faded film grain, 1970s film photography aesthetic.'
  'A still life of fresh pears, figs and grapes arranged in a hand-thrown ceramic bowl on a rustic wooden table, soft directional window light from the left casting gentle shadows, wrinkled linen cloth beneath the bowl, dew glistening on the fruit skin, Dutch masters oil painting style, chiaroscuro.'
  'A cozy wooden cabin half-buried in snow in a dense pine forest at blue dusk, warm amber light glowing from every window and spilling onto the drifts, thick flakes drifting down through the fading light, chimney smoke rising into the darkening sky, quiet atmospheric winter scene.'
  'A lone surfer carving a bottom turn on a glassy turquoise wave at sunset, a fan of spray exploding off the rail and backlit by the low orange sun, tropical beach and palm trees soft in the background, water droplets suspended mid-air, action sports photography, 400mm telephoto.'
  'A minimalist book cover design for STORIES OF THE SEA, the title set in an elegant thin serif typeface across the upper third, flat illustrations of deep blue ocean waves with white foam curls filling the lower two thirds, generous negative space, muted navy and cream palette, flat graphic design.'
  'An astronaut in a detailed white EVA suit floating weightless in front of a vast glowing nebula, violet and teal gas clouds studded with bright newborn stars, soft rim light from a distant sun outlining the suit, the nebula reflected in the gold visor, cinematic science fiction concept art.'
  'An assorted sushi platter arranged on a black slate board, glossy salmon nigiri with individually visible rice grains, neatly rolled tuna maki dusted with sesame seeds, a mound of wasabi and pale pink pickled ginger beside ceramic chopstick rests, dramatic side lighting, dark background, studio food photography.'
  'A red tractor plowing a vast golden wheat field at dawn, a trail of dust hanging in the crisp morning air behind it, the low sun cresting the horizon and gilding the dust cloud, neat furrows receding toward a distant farmhouse, wide establishing shot, rural documentary style.'
  'A jazz quartet performing on a dim club stage, a double bassist leaning into his instrument, a saxophonist mid-solo at the microphone, drummer and pianist softer behind them, a single warm spotlight cutting through artificial smoke, brass glinting, deep shadows, smoky intimate atmosphere.'
  'A faceted crystal perfume bottle standing on draped cream silk fabric, refracted caustic light patterns dancing across the silk folds, amber liquid glowing inside the glass against a dark gradient background, subtle reflection on the polished black surface below, single softbox from the upper left, luxury product photography.'
  'A children book illustration of a small russet fox wearing a hand-knitted red scarf, walking along a winding dirt path through a birch forest of slender white trunks, soft golden leaves drifting down around it, gentle watercolor washes with visible paper texture, warm inviting palette, storybook composition.'
  'A soccer player frozen mid-air in an overhead bicycle kick striking the ball, a floodlit stadium packed with spectators behind him, thousands of camera flashes sparkling through the stands like stars, blades of grass kicked up around his body, dramatic night lighting, dynamic sports photography.'
  'The interior of a grand old library, towering dark wooden bookshelves rising four stories to a vaulted ceiling, a wrought-iron spiral staircase winding up between the stacks, brass reading lamps casting warm pools of light over green-shaded desks, dust motes drifting through shafts of light, warm antique tones.'
  'Dozens of colorful hot air balloons drifting at different altitudes over the surreal rock valleys of Cappadocia at sunrise, fairy chimney formations glowing pink and gold in the first light, long soft shadows across the valley floor, gradient pastel sky from peach to pale blue, aerial travel photography.'
)
PROMPTS_LONG=(
  'A fluffy orange cat with bright amber eyes sits perfectly still on a sun-warmed wooden windowsill, thick golden morning light streaming through sheer linen curtains and striping the wall behind it, dust motes drifting lazily through the sunbeam, a leafy monstera blurred in the near foreground and the rooftops of an old European town soft in the background, whiskers backlit into a fine halo, individual guard hairs catching the light across its chest, shallow depth of field isolating the calm half-lidded face, photorealistic close-up portrait, shot on an 85mm prime lens at f/1.8, natural color grade.'
  'A confident young woman in her late twenties wearing a tailored red blazer over a white silk blouse, one hand resting lightly on the back of a designer chair, standing in a corner office with floor-to-ceiling windows that reveal a soft-focus city skyline at midday, diffused daylight from a large window on the left gently sculpting her cheekbones, shallow depth of field melting the skyline into clean bokeh, natural skin texture with fine detail in the eyes, a subtle confident smile, professional corporate headshot, photorealistic, 85mm lens at f/2.0, neutral color grading.'
  'A vast snow-capped mountain range photographed at sunrise, the first golden alpenglow igniting the highest summits while the deep valleys below still rest in cold blue shadow, a mirror-still alpine lake filling the foreground and reflecting the ridgeline with perfect symmetry, thin ribbons of mist drifting across the water surface, scattered boulders and a lone pine breaking the shoreline on the right, crisp cold air with exceptional visibility, layered ridges fading into the distance, wide cinematic landscape composition, 16mm lens, high dynamic range, polarized sky, ultra-detailed.'
  'A joyful golden retriever puppy bounding through deep drifts of fallen maple and oak leaves in a quiet park, captured mid-stride with leaves and dust kicked into the air around its paws, slight motion blur on the flying leaves while the face stays sharp, low afternoon sunlight backlighting the whole scene in warm amber tones, the puppy breath visible in the crisp autumn air, ears flopping and tongue out, dappled shadows from bare trees striping the ground, photorealistic action shot, fast shutter, 300mm telephoto compression, rich fall color palette.'
  'A bustling Asian night market street food stall at peak hours, a vendor lifting the lid from a tower of bamboo steamers to release a billowing cloud of steam backlit by hanging bulbs, woks flaring over open flame beside her, strings of red and cyan neon signs receding down the narrow street into compressed bokeh, queues of customers dissolved into warm silhouettes, glistening braised meats and dumplings in the foreground catching the light, lens flare from a street lamp in the upper corner, documentary photography, handheld 35mm, authentic candid energy.'
  'A modern glass skyscraper facade photographed from directly below at blue hour, thousands of individually lit windows forming a strict geometric grid that converges dramatically toward the sky, mirror-like reflections of the deep blue evening sky and a few thin clouds sliding across the curved glass panels, steel mullions catching a faint warm glow from the sun just below the horizon, crisp architectural lines with zero distortion, long exposure from a tripod keeping every edge perfectly sharp, architectural photography, tilt-shift lens, cool blue palette with amber window light.'
  'A restored vintage 1970s motorcycle parked at an angle on an empty desert highway, polished chrome fuel tank and spoked wheels catching the low sunset backlight, a long precise shadow stretching across cracked asphalt toward the camera, heat shimmer warping the horizon line, distant red mesas standing in dark silhouette against a gradient orange sky, scattered desert scrub and a weathered mile marker beside the road, warm faded film grain across the frame, 1970s film photography aesthetic, Kodachrome palette, 50mm lens, eye-level composition, cinematic and nostalgic.'
  'A still life of fresh pears, figs and dark grapes arranged casually in a hand-thrown ceramic bowl on a rustic oak table, soft directional window light from the left casting long gentle shadows across the wood grain, a wrinkled linen cloth bunched beneath the bowl, dew glistening on the fruit skins, a silver knife and a half-peeled pear resting to one side, the dark muted background falling away into deep shadow, rich earthy palette of greens, purples and browns, Dutch masters oil painting style, museum-quality chiaroscuro, hyper-detailed textures.'
  'A cozy wooden cabin half-buried in deep snow in a dense pine forest at blue dusk, warm amber light glowing from every window and spilling long golden rectangles across the untouched drifts, thick snowflakes drifting down through the fading indigo light, a curl of chimney smoke rising into the darkening sky, a line of footprints leading from the edge of the frame to the front door, icicles hanging from the eaves and firewood stacked under the porch, the forest fading into soft darkness behind, quiet atmospheric winter scene, cinematic composition, photorealistic.'
  'A lone surfer carving a hard bottom turn on a glassy turquoise wave at sunset, a wide fan of spray exploding off the rail and glowing amber as it catches the low orange sun, water droplets suspended mid-air like sparks, the tropical beach, palm trees and a distant headland soft in the background, the wave open face showing clean laminar flow and a thin lip of foam, spray blowing offshore in the wind, low telephoto perspective compressing the action, action sports photography, 400mm lens, fast shutter, frozen motion, saturated tropical color grade.'
  'A minimalist book cover design for STORIES OF THE SEA, the title set in an elegant thin serif typeface widely tracked across the upper third of the cover, the author name in small capitals at the base, a flat illustration of deep blue ocean waves with stylized white foam curls filling the lower two thirds, generous negative space around every element, a muted two-color palette of navy and cream with a single coral accent line, subtle paper grain texture over the whole design, flat graphic design, front view, centered composition, print-ready aesthetic.'
  'An astronaut in a detailed white EVA suit floating weightless in front of a vast glowing nebula, violet and teal gas clouds studded with brilliant newborn stars filling the entire background, soft rim light from a distant sun outlining the suit inflated layers and backpack rails, the swirling nebula mirrored in the curved gold visor, a faint tether curling away into the darkness, instrument panels glowing softly at the chest, tiny ice crystals drifting off the glove, cinematic science fiction concept art, epic scale, moody and contemplative atmosphere.'
  'An assorted sushi platter arranged with precision on a black slate board, glossy salmon nigiri with individually visible rice grains and marbled flesh, neatly rolled tuna maki dusted with sesame seeds, a dragon roll topped with avocado slices, a sculpted mound of wasabi and pale pink pickled ginger curled like petals, soy sauce in a small ceramic dish and wooden chopsticks resting on a holder, dramatic side lighting raking across the board to emphasize texture, deep black background, faint steam suggesting freshness, studio food photography, 45-degree angle.'
  'A red tractor plowing a vast golden wheat field at dawn, a long trail of dust hanging in the crisp morning air behind the machine and glowing amber as the low sun crests the horizon directly behind it, neat dark furrows receding in perspective toward a distant farmhouse and a line of bare trees, crows scattered across the freshly turned soil, the driver silhouetted in the cab, dew silvering the stubble at the field edges, wide establishing shot, rural farmland documentary style, 35mm, golden hour haze, photorealistic.'
  'A jazz quartet performing on a dim club stage, a double bassist leaning deeply into his instrument on the left, a saxophonist mid-solo at a vintage microphone with cheeks drawn, drummer and pianist softer in the background, a single warm spotlight cutting a cone through artificial haze, brass instruments glinting with amber highlights, audience tables reduced to dark silhouettes and candle flames in the foreground, cigarette smoke curling through the beam, deep pools of shadow swallowing the brick walls, smoky intimate atmosphere, available-light photojournalism, high ISO grain.'
  'A faceted crystal perfume bottle standing on softly draped cream silk fabric, refracted caustic light patterns dancing across the silk folds in bright webbed arcs, amber liquid glowing warmly inside the thick glass against a dark gradient background falling from charcoal to black, a subtle mirrored reflection on the polished black surface below, the bottle cap resting beside it, a single large softbox from the upper left defining the crystal edges, faint golden particles drifting in the backlight, luxury product photography, macro detail, elegant and minimal.'
  'A children book illustration of a small russet fox wearing a hand-knitted red scarf walking along a winding dirt path through a birch forest, slender white trunks with black peeling markings rising on both sides, soft golden leaves drifting down around the fox and carpeting the path, a few toadstools and mossy stones beside the way, gentle watercolor washes bleeding softly into visible paper texture, fine ink linework defining the fur and foliage, warm inviting palette of ochre, cream and soft red, storybook composition with the fox small and central.'
  'A soccer player frozen mid-air in an overhead bicycle kick striking the ball, his body arched perfectly horizontal above the manicured pitch, a floodlit stadium packed with ninety thousand spectators rising behind him, thousands of camera flashes sparkling through the stands like a field of stars, blades of grass and divots kicked up around his trailing boot, the ball compressing against his striking foot, dramatic stadium lighting raking down against the night sky, shallow depth of field isolating the athlete, dynamic sports photography, 300mm lens, frozen motion.'
  'The interior of a grand old library, towering dark wooden bookshelves rising four stories to a vaulted coffered ceiling, wrought-iron spiral staircases winding up between the stacks on brass rails, thousands of leather-bound volumes with gilt lettering, brass reading lamps casting warm pools of light over green-shaded desks, long shafts of dusty light slanting from tall arched windows, a marble bust and a globe on the mezzanine, dust motes drifting slowly through the beams, worn parquet floor reflecting the lamplight, warm antique tones, photorealistic architectural interior.'
  'Dozens of colorful hot air balloons in stripes, checks and swallowtail patterns drifting at many different altitudes over the surreal rock valleys of Cappadocia at sunrise, fairy chimney formations glowing pink and gold in the first horizontal light, long soft shadows stretching west across the valley floor, a wicker basket and fluttering burner flame visible on the nearest balloon, thin haze layering the receding ridges, a gradient pastel sky from peach at the horizon to pale blue above, travel photography, aerial perspective, serene and majestic, ultra-detailed.'
)

# ── 输出与权重 (支持环境变量覆盖, 如 CHECKPOINT=xxx.safetensors bash infer_giga_exp_kd.sh) ──
OUTPUT_DIR="${OUTPUT_DIR:-}"   # 空 = outputs/pretrain_v2/vis_token_exp/<ckpt 名>_kd/，下设 shift_* 子目录
CHECKPOINT="${CHECKPOINT:-}"   # 空 = 自动取 config output_dir/checkpoints/weights 下最新 step-*
CONFIG="${CONFIG:-${ROOT}/configs/stage_1_kd_v3_traj.yaml}"

# ── 采样参数: 固定 shift=6, 只扫 CFG, 对比 short/mid/long 三组 prompt 长度; STEPS 取 KD 标准 50 ──
HEIGHT=512      # 必须是 32 的倍数
WIDTH=512
STEPS=50        # KD 标准步数
CFGS=(3 6 9 12) # guidance scale 组合
NEGATIVE_PROMPT="<|im_end|>"   # 与训练一致: EOS 无条件分支
SHIFTS=(6)      # 固定 shift=6
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

# 公共参数 (prompt 按长度分组在循环内追加)
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
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -z "${GPU}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # 自动选卡: 显存占用最低的一张 (nvidia-smi CSV: index, memory.used)
  GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | sed -n '1p' | cut -d',' -f1)"
fi

for LEN in short mid long; do
  case "$LEN" in
    short) LEN_PROMPTS=("${PROMPTS_SHORT[@]}") ;;
    mid)   LEN_PROMPTS=("${PROMPTS_MID[@]}") ;;
    long)  LEN_PROMPTS=("${PROMPTS_LONG[@]}") ;;
  esac
  LEN_ARGS=("${ARGS[@]}")
  for prompt in "${LEN_PROMPTS[@]}"; do
    LEN_ARGS+=(--prompt "${prompt}")
  done
  for SHIFT in "${SHIFTS[@]}"; do
    for CFG in "${CFGS[@]}"; do
      COMBO_OUTPUT_DIR="${OUTPUT_DIR}/${LEN}/shift_${SHIFT}_cfg_${CFG}"
      printf '\n开始推理 prompt=%s shift=%s cfg=%s → %s\n' "$LEN" "$SHIFT" "$CFG" "$COMBO_OUTPUT_DIR"
      CUDA_VISIBLE_DEVICES="${GPU}" python "${LEN_ARGS[@]}" \
        --shift "$SHIFT" --guidance-scale "$CFG" --output-dir "$COMBO_OUTPUT_DIR"
    done
  done
done
printf '\nshort/mid/long × shift×cfg 共 %d 组推理完成 → %s\n' \
  "$(( 3 * ${#SHIFTS[@]} * ${#CFGS[@]} ))" "$OUTPUT_DIR"
