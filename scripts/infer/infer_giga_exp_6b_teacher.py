"""GWP 6B teacher 推理验证脚本 (蒸馏前的 sanity check)。

用途: 确认 gigaworld stage1 6B checkpoint 在本仓库的加载与前向全部正常,
生成图片肉眼核验 teacher 质量。仅文本条件 T2I (与 KD 训练条件形式一致)。

用法:
    python scripts/infer/infer_giga_exp_6b_teacher.py \
        --prompt "a cat sitting on a windowsill, photo" \
        --prompt "a red sports car on a mountain road at sunset" \
        --output /home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/teacher_infer/out.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from pipeline.giga_pipeline_6b_teacher import GigaPipeline6BTeacher, clean_prompt_6b_teacher

# 默认路径对齐 configs/stage_1.yaml components 与 teacher checkpoint
DEFAULT_TEACHER_CHECKPOINT = "/home/jovyan/pfs/users/zhanqian.wu/exp/stage1_wan_image_qwen3_vl_mid_train_256/checkpoints/checkpoint-44000"
DEFAULT_QWEN3_VL_PATH = "/mnt/nfs/models/huggingface/Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_WAN_VAE_PATH = "/mnt/nfs/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, disfigured, "
    "bad anatomy, extra limbs, malformed hands, text, watermark, logo, artifacts"
)
# Z-Image/sft 仓库推理约定 (ZImagePipeline.scheduler: shift 6.0; validation: 30 步 CFG 2.5)
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 2.5
DEFAULT_SHIFT = 6.0


def main():
    parser = argparse.ArgumentParser(description="GWP 6B teacher T2I inference (KD sanity check)")
    parser.add_argument("--checkpoint", default=DEFAULT_TEACHER_CHECKPOINT, help="teacher checkpoint (目录或 model_weights.pt)")
    parser.add_argument("--qwen3-vl-path", default=DEFAULT_QWEN3_VL_PATH)
    parser.add_argument("--wan-vae-path", default=DEFAULT_WAN_VAE_PATH)
    parser.add_argument("--prompt", action="append", required=True, help="文本 prompt (可重复)")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=1024, help="输出高 (必须被 32 整除)")
    parser.add_argument("--width", type=int, default=1024, help="输出宽 (必须被 32 整除)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT, help="推理 sigma schedule shift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=1, help="每条 prompt 生成张数")
    parser.add_argument("--output", default="outputs/teacher_6b_infer/output.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-layer", type=int, default=-2)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🎯 teacher T2I | prompts={len(args.prompt)} | {args.width}x{args.height} | "
          f"steps={args.steps} | cfg={args.guidance} | shift={args.shift} | seed={args.seed}")
    print(f"📦 checkpoint: {args.checkpoint}")

    t0 = time.time()
    pipe = GigaPipeline6BTeacher.from_pretrained(
        teacher_checkpoint=args.checkpoint,
        qwen3_vl_path=args.qwen3_vl_path,
        vae_path=args.wan_vae_path,
        attention_backend="auto",
        text_layer=args.text_layer,
        scheduler_shift=args.shift,
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"✅ 组件加载完成 ({time.time() - t0:.1f}s) | attention: "
          f"{__import__('models.giga_transformer_6b_teacher', fromlist=['x']).GLOBAL_ATTENTION_BACKEND}")

    rng = random.Random(args.seed)
    records = []
    for pi, raw_prompt in enumerate(args.prompt):
        prompt = clean_prompt_6b_teacher(raw_prompt)
        for index in range(max(1, args.num_images)):
            image_seed = args.seed if index == 0 else rng.randrange(1, 2**31 - 1)
            suffix = "" if len(args.prompt) == 1 and args.num_images == 1 else f"_p{pi}_s{image_seed}"
            out_file = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
            t0 = time.time()
            images = pipe(
                [prompt], height=args.height, width=args.width, steps=args.steps,
                guidance_scale=args.guidance, seed=image_seed,
                negative_prompt=args.negative_prompt, device=device,
            )
            infer_time = time.time() - t0
            images[0].save(out_file, quality=95)
            print(f"💾 [p{pi}/{index}] seed={image_seed} {infer_time:.1f}s → {out_file}")
            records.append({
                "prompt_index": pi, "image_index": index, "seed": image_seed,
                "prompt": prompt, "negative_prompt": args.negative_prompt,
                "height": args.height, "width": args.width,
                "steps": args.steps, "guidance_scale": args.guidance, "shift": args.shift,
                "checkpoint": args.checkpoint, "output_path": str(out_file),
                "seconds": round(infer_time, 2),
            })

    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(records[0] if len(records) == 1 else records, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"✅ 完成 {len(records)} 张 | 元数据: {meta_path}")


if __name__ == "__main__":
    main()
