#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/home/jovyan/pfs/users/zhanqian.wu/ckpt/flux-3-image-klein-base")
sys.path.insert(0, str(ROOT / "diffusers" / "src"))

from diffusers import ModularPipeline  # noqa: E402
from diffusers.modular_pipelines import Flux3AutoBlocks  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.3 Image Klein Base Diffusers 推理")
    parser.add_argument("prompt", help="完整场景描述；启用 prompt upsampler 时可传简短提示词")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "output.png")
    parser.add_argument("--image", type=Path, action="append", default=[], help="参考图，可重复传入，最多 10 张")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--cpu-offload", action="store_true", help="显存不足时自动将组件卸载到 CPU")
    parser.add_argument(
        "--use-prompt-upsampler",
        action="store_true",
        help="加载 modular_model_index.json 指定的独立 PUP 模型（需要已下载或可联网）",
    )
    parser.add_argument("--allow-download", action="store_true", help="允许从 Hugging Face 补充缺失文件")
    return parser.parse_args()


def missing_shards(model_path):
    missing = []
    for index_path in model_path.glob("**/*.safetensors.index.json"):
        index = json.loads(index_path.read_text())
        for filename in sorted(set(index.get("weight_map", {}).values())):
            shard = index_path.parent / filename
            if not shard.is_file():
                missing.append(shard)
    return missing


def load_pipeline(args):
    if args.use_prompt_upsampler:
        pipe = ModularPipeline.from_pretrained(
            args.model_path,
            workflow="image_conditioned" if args.image else "text2image",
            local_files_only=not args.allow_download,
        )
    else:
        blocks = Flux3AutoBlocks().get_workflow("image_conditioned" if args.image else "text2image")
        blocks.sub_blocks.pop("prompt_upsampler")
        pipe = blocks.init_pipeline(args.model_path)

    pipe.load_components(
        dtype={"vae": torch.float32, "default": torch.bfloat16},
        local_files_only=not args.allow_download,
    )
    pipe.guider = pipe.guider.new(guidance_scale=args.guidance_scale)

    if args.cpu_offload:
        pipe.enable_auto_cpu_offload(device="cuda")
    else:
        pipe.to("cuda")
    return pipe


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU，FLUX.3 Klein Base 不支持本脚本的 CPU 推理模式。")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{args.model_path}")
    if len(args.image) > 10:
        raise ValueError("最多支持 10 张参考图。")
    if args.height % 16 or args.width % 16:
        raise ValueError("height 和 width 必须是 16 的倍数。")

    missing = missing_shards(args.model_path)
    if missing and not args.allow_download:
        files = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"checkpoint 仍未下载完整，缺少以下分片：\n{files}")

    references = [Image.open(path).convert("RGB") for path in args.image]
    pipe = load_pipeline(args)
    call_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "num_images_per_prompt": args.num_images,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
        "output": "images",
    }
    if references:
        call_kwargs["image"] = references

    images = pipe(**call_kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = args.output.stem, args.output.suffix or ".png"
    for index, image in enumerate(images):
        output = args.output if len(images) == 1 else args.output.with_name(f"{stem}_{index:02d}{suffix}")
        image.save(output)
        print(output)


if __name__ == "__main__":
    main()
