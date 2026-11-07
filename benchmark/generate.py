#!/usr/bin/env python3
"""benchmark.generate: 对评测 prompt 集批量生成图片 (复用 scripts/infer/infer_giga_exp.py 管线).

输出结构:
    <output-root>/<bench_name>/<ckpt_tag>/<id>_seed{seed}.png
    <output-root>/<bench_name>/<ckpt_tag>/manifest.json

说明:
- 复用方式: scripts/ 目录不是 python 包 (无 __init__.py), 故用 importlib 按文件路径加载
  infer_giga_exp 模块, 复用其 ckpt 加载 (兼容 AdaLN basis 迁移)、Qwen3-VL 编码、flow ODE
  采样 sample_batch、VAE 解码与工具函数; 不走 subprocess (其 CLI 只支持纯文本 prompt, 无法
  携带 jsonl 的 id/meta).
- 默认需要 GPU (--device cuda); CPU 仅用于极小规模调试 (40 步 ODE 会非常慢).

示例:
    python -m benchmark.generate --bench hpsv2 --num-seeds 1 --limit 4 \
        --checkpoint outputs/.../checkpoints/weights/step-025000.safetensors \
        --output-root outputs/bench
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# FA3 (flash_attn_3) 位于 thirdparty: 必须在 import models.* 之前注入 (与 infer_giga_exp.py 一致)
_THIRDPARTY = str(ROOT / "thirdparty")
if Path(_THIRDPARTY).is_dir() and _THIRDPARTY not in sys.path:
    sys.path.insert(0, _THIRDPARTY)


def _load_infer_module():
    """按文件路径加载 scripts/infer/infer_giga_exp.py (scripts 不是 python 包)."""
    path = ROOT / "scripts" / "infer" / "infer_giga_exp.py"
    spec = importlib.util.spec_from_file_location("infer_giga_exp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["infer_giga_exp"] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    ap = argparse.ArgumentParser(description="评测集批量生成")
    ap.add_argument("--bench", required=True,
                    choices=["geneval", "heim", "hpsv2", "qwen_image_bench"],
                    help="评测集名称 (对应 benchmark/prompts/<bench>_prompts.jsonl)")
    ap.add_argument("--prompts-jsonl", type=Path, default=None,
                    help="自定义 prompts jsonl; 缺省用 benchmark/prompts/<bench>_prompts.jsonl")
    ap.add_argument("--config", default="configs/stage_1.yaml")
    ap.add_argument("--checkpoint", type=Path, required=True, help="权重文件或含 step-* 的目录")
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "bench")
    ap.add_argument("--num-seeds", type=int, default=1, help="每个 prompt 生成几个 seed")
    ap.add_argument("--seed-base", type=int, default=42, help="起始 seed, 第 k 个 seed = seed_base + k")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 条 prompt")
    ap.add_argument("--ids", default=None, help="只生成这些 id (逗号分隔), 冒烟测试用")
    ap.add_argument("--height", type=int, default=128, help="像素高, 须为 32 的倍数")
    ap.add_argument("--width", type=int, default=192, help="像素宽, 须为 32 的倍数")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=None, help="缺省取 scheduler.validation_shift")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--negative-prompt", default="<|im_end|>")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已存在的输出文件 (断点续跑)")
    return ap.parse_args()


def load_prompts(path: Path, bench: str, limit=None, ids=None):
    """读 jsonl → [{"id","benchmark","prompt","meta"}]; 支持 limit / ids 过滤."""
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        wanted = {x.strip() for x in ids.split(",") if x.strip()}
        items = [it for it in items if it["id"] in wanted]
    if limit:
        items = items[:limit]
    for it in items:
        it.setdefault("benchmark", bench)
    return items


def main():
    args = parse_args()
    infer = _load_infer_module()
    import torch
    import yaml
    from PIL import Image
    from safetensors.torch import load_file
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            infer.log_error("需要 CUDA GPU (或用 --device cpu 调试)")
            raise SystemExit(2)
        if device.index is None:  # 缺省用最后一张可见 GPU (与 infer 脚本一致)
            index = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else max(torch.cuda.device_count() - 1, 0)
            device = torch.device(f"cuda:{index}")

    prompts_path = args.prompts_jsonl or Path(__file__).parent / "prompts" / f"{args.bench}_prompts.jsonl"
    items = load_prompts(prompts_path, args.bench, limit=args.limit, ids=args.ids)
    if not items:
        infer.log_error(f"没有可用的 prompt: {prompts_path}")
        raise SystemExit(2)

    cfg_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    checkpoint, step = infer.resolve_checkpoint(args.checkpoint, cfg["output_dir"])
    ckpt_tag = checkpoint.stem
    out_dir = args.output_root / args.bench / ckpt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tc, dc = cfg["components"]["text_encoder"], cfg["components"]["wan_vae"]
    dtype = torch.bfloat16 if cfg["training"].get("mixed_precision", "bf16") == "bf16" else torch.float16
    shift = args.shift if args.shift is not None else float(
        cfg["scheduler"].get("validation_shift") or cfg["scheduler"].get("shift") or 1.0)

    print(infer.paint(infer.Style.BOLD, "⚙️  评测生成设置"), flush=True)
    print(infer.render_settings_table([
        ("bench", f"{args.bench} ({len(items)} prompts x {args.num_seeds} seeds)"),
        ("prompts_jsonl", str(prompts_path)),
        ("checkpoint", f"{checkpoint.name} (step {step})"),
        ("size", f"{args.height} x {args.width}"),
        ("steps", str(args.steps)),
        ("guidance_scale", str(args.guidance_scale)),
        ("shift", str(shift)),
        ("seeds", f"{args.seed_base}..{args.seed_base + args.num_seeds - 1}"),
        ("batch_size", str(args.batch_size)),
        ("device", str(device)),
        ("output_dir", str(out_dir)),
    ]), flush=True)

    infer.log_info(f"🧩 加载组件: encoder={tc['pretrained_path']} · vae={dc['pretrained_path']}")
    tokenizer = AutoTokenizer.from_pretrained(tc["tokenizer_path"], local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        tc["pretrained_path"], torch_dtype=dtype, local_files_only=True).eval().to(device)
    vae = AutoencoderKLWan.from_pretrained(
        dc["pretrained_path"], subfolder=dc.get("subfolder", "vae"),
        torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    model_cls = infer.GigaExpTransformer
    if cfg.get("mhc", {}).get("enabled") or "n_streams" in cfg["model"]:
        from models.giga_transformer_mhc import GigaExpTransformerMHC
        model_cls = GigaExpTransformerMHC
    model = model_cls(**cfg["model"]).to(device, dtype=dtype)
    state = load_file(str(checkpoint)) if checkpoint.suffix == ".safetensors" \
        else torch.load(str(checkpoint), map_location="cpu")
    infer.load_compatible_state_dict(model, state)
    infer.log_ok(f"📦 模型就绪: {checkpoint.name} (step {step})")

    z_dim = vae.config.z_dim
    compress = vae.config.scale_factor_spatial
    mean, std_inv = infer.wan_latent_stats(vae, device)
    layer = tc.get("layer")
    max_length = tc.get("max_length", 512)

    # 待生成任务列表: (item, seed); 已存在且 --skip-existing 时跳过
    jobs = [(it, args.seed_base + k) for it in items for k in range(args.num_seeds)]
    if args.skip_existing:
        jobs = [(it, s) for it, s in jobs
                if not (out_dir / f"{it['id']}_seed{s}.png").exists()]
    manifest = {
        "benchmark": args.bench, "prompts_jsonl": str(prompts_path), "num_prompts": len(items),
        "checkpoint": str(checkpoint), "step": step, "config": str(args.config),
        "height": args.height, "width": args.width, "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale, "shift": shift,
        "seed_base": args.seed_base, "num_seeds": args.num_seeds,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "images": [],
    }
    n_batches = math.ceil(len(jobs) / args.batch_size)
    t0 = time.time()
    for bi in range(n_batches):
        batch = jobs[bi * args.batch_size:(bi + 1) * args.batch_size]
        batch_prompts = [it["prompt"] for it, _ in batch]
        text, mask, neg, neg_mask = infer.encode_cfg_prompts(
            tokenizer, encoder, batch_prompts, args.negative_prompt, device, layer, max_length)
        # 每个 job 独立种子生成噪声 (与 infer 脚本相同约定, 结果与 batch_size 无关)
        noises = []
        for it, seed in batch:
            generator = torch.Generator(device=device).manual_seed(seed)
            noises.append(torch.randn(1, z_dim, 1, args.height // compress, args.width // compress,
                                      device=device, dtype=torch.float32, generator=generator))
        x = torch.cat(noises, 0)
        x = infer.sample_batch(model, x, text, mask, neg, neg_mask,
                               steps=args.steps, guidance=args.guidance_scale,
                               shift=shift, dtype=dtype)
        with torch.no_grad():
            images = vae.decode(x / std_inv + mean).sample
        for local_i, (it, seed) in enumerate(batch):
            path = out_dir / f"{it['id']}_seed{seed}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(infer.to_uint8(images[local_i])).save(path)
            manifest["images"].append({
                "file": path.name, "id": it["id"], "benchmark": it["benchmark"],
                "seed": seed, "prompt": it["prompt"], "meta": it.get("meta", {}),
            })
            infer.log_image(f"{path.name} · {it['prompt'][:64]}")
        done = min((bi + 1) * args.batch_size, len(jobs))
        infer.log_ok(f"进度 {done}/{len(jobs)} · 已用时 {time.time() - t0:.0f}s")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    infer.log_ok(f"🎉 完成: {len(jobs)} 张图 → {out_dir} (manifest 已写入)")


if __name__ == "__main__":
    main()
