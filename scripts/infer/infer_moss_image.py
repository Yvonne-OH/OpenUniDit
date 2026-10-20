#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/home/jovyan/pfs/users/zhanqian.wu/ckpt/MOSS-Image-SFT")
sys.path.insert(0, str(ROOT / "diffusers" / "src"))
sys.path.insert(0, str(DEFAULT_MODEL))

from pipeline_moss_image import MOSSImagePipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="MOSS-Image-SFT 本地推理")
    parser.add_argument("prompt", help="文生图提示词，也支持 MOSS-Image 的 JSON layout prompt")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "moss_image.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument(
        "--uncond-skip-layers",
        type=int,
        nargs="*",
        default=[22],
        help="CFG negative branch 跳过的 DiT block；传 --uncond-skip-layers 不带值可关闭",
    )
    parser.add_argument("--cpu-offload", action="store_true", help="显存不足时启用 sequential CPU offload")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-Image-SFT 推理需要 CUDA GPU。")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{args.model_path}")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("height 和 width 必须是正的 16 的倍数。")

    pipe = MOSSImagePipeline.from_components_pretrained(
        transformer_path=args.model_path / "transformer",
        text_encoder_path=args.model_path / "text_encoder",
        vae_path=args.model_path / "vae",
        torch_dtype=torch.bfloat16,
        vae_dtype=torch.float32,
        local_files_only=True,
    )
    pipe.tokenizer = AutoTokenizer.from_pretrained(
        args.model_path / "tokenizer",
        local_files_only=True,
    )
    if pipe.tokenizer.chat_template is None:
        template_path = args.model_path / "tokenizer" / "chat_template.jinja"
        pipe.tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        pipe.to("cuda")

    images = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images,
        max_sequence_length=args.max_sequence_length,
        uncond_skip_layers=args.uncond_skip_layers,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = args.output.stem, args.output.suffix or ".png"
    for index, image in enumerate(images):
        output = args.output if len(images) == 1 else args.output.with_name(f"{stem}_{index:02d}{suffix}")
        image.save(output)
        print(output)


if __name__ == "__main__":
    main()
