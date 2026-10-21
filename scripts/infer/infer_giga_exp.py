#!/usr/bin/env python3
"""Giga-Exp stage_1 T2I flow-matching 推理: 多 prompt (list) → flow ODE 采样 → Wan VAE 解码 → PNG.

- 模型/文本编码器/VAE/σ-shift schedule 全部对齐训练脚本 train_giga_exp.py 的验证逻辑
- 权重: --checkpoint 指定文件或目录; 缺省自动取 output_dir/checkpoints/weights 最新 step-*
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from diffusers import AutoencoderKLWan
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from models.giga_transformer import GigaExpTransformer

try:
    from tqdm import tqdm, trange
except ImportError:  # 缺 tqdm 时退化为无进度条
    tqdm = trange = None

# Wan VAE 空间压缩 16 × patch 2 ⇒ 像素宽高须被 32 整除
PIXEL_MULTIPLE = 32


# ── 彩色 log (非 TTY / NO_COLOR 时自动退化为纯文本) ──
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class Style:
    RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, GRAY = (
        "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[90m")


def paint(color: str, text: str) -> str:
    return f"{color}{text}{Style.RESET}" if USE_COLOR else str(text)


def log_info(msg):
    print(paint(Style.CYAN, f"ℹ️  {msg}"), flush=True)


def log_ok(msg):
    print(paint(Style.GREEN, f"✅ {msg}"), flush=True)


def log_warn(msg):
    print(paint(Style.YELLOW, f"⚠️  {msg}"), flush=True)


def log_error(msg):
    print(paint(Style.RED, f"❌ {msg}"), flush=True)


def log_image(msg):
    print(paint(Style.MAGENTA, f"🖼️  {msg}"), flush=True)


def render_settings_table(rows):
    """两列 box-drawing 设置表 (单元格里不放 emoji, 保证对齐)."""
    key_w = max(len(key) for key, _ in rows)
    val_w = max(len(value) for _, value in rows)
    top = f"┌{'─' * (key_w + 2)}┬{'─' * (val_w + 2)}┐"
    mid = f"├{'─' * (key_w + 2)}┼{'─' * (val_w + 2)}┤"
    bot = f"└{'─' * (key_w + 2)}┴{'─' * (val_w + 2)}┘"
    lines = [top]
    for i, (key, value) in enumerate(rows):
        if i:
            lines.append(mid)
        lines.append(f"│ {paint(Style.GRAY, key.ljust(key_w))} │ {value.ljust(val_w)} │")
    lines.append(bot)
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Giga-Exp stage_1 T2I 推理")
    parser.add_argument("--prompt", action="append", default=[], help="单条 prompt, 可重复传入构成 list")
    parser.add_argument("--prompt-file", type=Path, default=None, help="prompt 文件, 每行一条, 空行忽略")
    parser.add_argument("--config", default="configs/stage_1.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="权重文件或含 step-* 的目录; 缺省取 output_dir/checkpoints/weights 最新")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="缺省 outputs/pretrain/vis_token_exp/<checkpoint 名>")
    parser.add_argument("--height", type=int, default=128, help="输出像素高, 必须是 32 的倍数 (默认 128, 横屏 192x128)")
    parser.add_argument("--width", type=int, default=192, help="输出像素宽, 必须是 32 的倍数 (默认 192, 横屏 192x128)")
    parser.add_argument("--steps", type=int, default=40, help="采样步数 (训练验证用 20)")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=None, help="σ schedule shift; 缺省取 scheduler.validation_shift")
    parser.add_argument("--seed", type=int, default=42, help="固定噪声种子, 所有 prompt 共用")
    parser.add_argument("--batch-size", type=int, default=4, help="list input 的分批大小")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="cuda 缺省自动选最后一张可见 GPU; cpu 用于调试")
    return parser.parse_args()


def collect_prompts(args):
    prompts = [str(p) for p in args.prompt]
    if args.prompt_file is not None:
        prompts += [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    if not prompts:
        log_error("没有可推理的 prompt: 用 --prompt 重复传入, 或 --prompt-file 指定文件")
        raise SystemExit(2)
    return prompts


def resolve_checkpoint(path, output_dir):
    """返回 (权重文件路径, step): 文件/目录均可; 目录取最新 step-*."""
    target = Path(path) if path is not None else Path(output_dir) / "checkpoints" / "weights"
    if target.is_dir():
        candidates = sorted(target.glob("step-*.safetensors")) + sorted(target.glob("step-*.pt"))
        if not candidates:
            raise FileNotFoundError(f"目录里没有 step-* 权重文件: {target}")
        target = max(candidates, key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    if not target.is_file():
        raise FileNotFoundError(f"权重不存在: {target}")
    match = re.search(r"step-(\d+)", target.name)
    return target, int(match.group(1)) if match else 0


def load_compatible_state_dict(model, state):
    """只加载名称和 shape 都匹配的参数, 其余保留初始化值; 含 patchify 迁移 (对齐训练脚本)."""
    current = model.state_dict()
    matched = {key: value for key, value in state.items()
               if key in current and current[key].shape == value.shape}
    patch_volume = math.prod(model.patch_size)
    if patch_volume > 1:
        for key in ("patch_in.weight", "out.weight", "out.bias"):
            if key not in state or key not in current or key in matched:
                continue
            value = state[key]
            if key == "patch_in.weight" and value.ndim == 2 and current[key].shape == (value.shape[0], value.shape[1] * patch_volume):
                matched[key] = value.repeat(1, patch_volume) / patch_volume
            elif key == "out.weight" and value.ndim == 2 and current[key].shape == (value.shape[0] * patch_volume, value.shape[1]):
                matched[key] = value.repeat(patch_volume, 1)
            elif key == "out.bias" and value.ndim == 1 and current[key].shape == (value.shape[0] * patch_volume,):
                matched[key] = value.repeat(patch_volume)
    result = model.load_state_dict(matched, strict=False)
    print(f"权重加载: matched={len(matched)}, reinitialized={len(result.missing_keys)}, "
          f"ignored={len(state) - len(matched)}", flush=True)


def wan_latent_stats(vae, device):
    """Wan 官方约定: z_norm = (z_vae - mean) / std, decode 前 z_vae = z_norm * std + mean."""
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, vae.config.z_dim, 1, 1, 1).float()
    std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=device)).view(1, vae.config.z_dim, 1, 1, 1).float()
    return mean, std_inv


def encode_cfg_prompts(tokenizer, encoder, prompts, device, layer, max_length):
    """正样本 + eos 负样本一次性编码, 统一 padding 长度后拆分 (对齐训练验证逻辑, 供 CFG cat)."""
    filled = [p.strip() if p and p.strip() else tokenizer.eos_token for p in prompts]
    inputs = tokenizer(filled + [tokenizer.eos_token] * len(filled),
                       return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    out = encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=layer is not None)
    text_all = out.hidden_states[layer] if layer is not None else out.last_hidden_state
    mask_all = inputs.attention_mask.bool()
    n = len(filled)
    return text_all[:n], mask_all[:n], text_all[n:], mask_all[n:]


@torch.no_grad()
def sample_batch(model, x, text, mask, neg, neg_mask, *, steps, guidance, shift, dtype):
    """flow ODE: σ 从 1(纯噪声) → 0(数据), schedule 与训练分布一致 (shift 变换), 步长取相邻 σ 差."""
    def shift_sigma(u):
        return shift * u / (1.0 + (shift - 1.0) * u)

    model.eval()
    step_iter = trange(steps, desc="  ⏳ ODE", leave=False, unit="step", colour="#FFD75F") if trange else range(steps)
    for i in step_iter:
        sigma = shift_sigma(1.0 - i / steps)
        next_sigma = shift_sigma(1.0 - (i + 1) / steps)
        if trange:
            step_iter.set_postfix_str(f"σ={sigma:.3f}")
        timestep = torch.full((x.shape[0] * 2,), sigma * 1000.0, device=x.device)
        v_all = model(
            torch.cat((x, x), 0).to(dtype),
            timestep,
            torch.cat((neg, text), 0),
            torch.cat((neg_mask, mask), 0),
        ).float()
        v_uncond, v_cond = v_all.chunk(2, 0)
        v = v_uncond + guidance * (v_cond - v_uncond)
        x = x - v * (sigma - next_sigma)
    return x


def to_uint8(latent_image):  # [C,T=1,H,W] → HWC uint8
    frame = latent_image[:, 0].detach().float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
    return ((frame + 1) * 127.5).round().astype("uint8")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        # 缺省用最后一张可见 GPU; 外部已设 CUDA_VISIBLE_DEVICES 时用其第 0 张
        index = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else max(torch.cuda.device_count() - 1, 0)
        device = torch.device(f"cuda:{index}")
    if device.type == "cuda" and not torch.cuda.is_available():
        log_error("需要 CUDA GPU (或用 --device cpu 调试)")
        raise SystemExit(2)
    if args.height % PIXEL_MULTIPLE or args.width % PIXEL_MULTIPLE:
        log_error(f"height/width 必须是 {PIXEL_MULTIPLE} 的倍数, 当前 {args.height}x{args.width}")
        raise SystemExit(2)
    cfg = yaml.safe_load((Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config).read_text())
    prompts = collect_prompts(args)
    checkpoint, step = resolve_checkpoint(args.checkpoint, cfg["output_dir"])
    # 缺省输出: outputs/pretrain/vis_token_exp/<ckpt 名>/
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / "outputs" / "pretrain" / "vis_token_exp" / checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    tc, dc = cfg["components"]["text_encoder"], cfg["components"]["wan_vae"]
    dtype = torch.bfloat16 if cfg["training"].get("mixed_precision", "bf16") == "bf16" else torch.float16
    shift = args.shift if args.shift is not None else float(cfg["scheduler"].get("validation_shift") or cfg["scheduler"].get("shift") or 1.0)

    print(paint(Style.BOLD, "⚙️  推理设置"), flush=True)
    print(render_settings_table([
        ("config", str(args.config)),
        ("checkpoint", f"{checkpoint.name} (step {step})"),
        ("size", f"{args.height} x {args.width}"),
        ("steps", str(args.steps)),
        ("guidance_scale", str(args.guidance_scale)),
        ("shift", str(shift)),
        ("seed", str(args.seed)),
        ("batch_size", str(args.batch_size)),
        ("prompts", str(len(prompts))),
        ("device", str(device)),
        ("output_dir", str(output_dir)),
    ]), flush=True)

    log_info(f"🧩 加载组件: encoder={tc['pretrained_path']} · vae={dc['pretrained_path']}")
    tokenizer = AutoTokenizer.from_pretrained(tc["tokenizer_path"], local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        tc["pretrained_path"], torch_dtype=dtype, local_files_only=True).eval().to(device)
    vae = AutoencoderKLWan.from_pretrained(
        dc["pretrained_path"], subfolder=dc.get("subfolder", "vae"),
        torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    model = GigaExpTransformer(**cfg["model"]).to(device, dtype=dtype)
    state = load_file(str(checkpoint)) if checkpoint.suffix == ".safetensors" \
        else torch.load(str(checkpoint), map_location="cpu")
    load_compatible_state_dict(model, state)
    log_ok(f"📦 模型就绪: {checkpoint.name} (step {step})")

    z_dim = vae.config.z_dim
    compress = vae.config.scale_factor_spatial
    mean, std_inv = wan_latent_stats(vae, device)
    layer = tc.get("layer")
    max_length = tc.get("max_length", 512)
    manifest = {
        "checkpoint": str(checkpoint), "step": step, "config": str(args.config),
        "height": args.height, "width": args.width, "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale, "shift": shift, "seed": args.seed,
        "images": [],
    }
    total_batches = math.ceil(len(prompts) / args.batch_size)
    batch_range = range(0, len(prompts), args.batch_size)
    batch_iter = (tqdm(batch_range, total=total_batches, desc="🎨 采样", unit="batch", colour="#5FD7FF")
                  if tqdm else batch_range)
    for start in batch_iter:
        batch_index = list(range(start, min(start + args.batch_size, len(prompts))))
        batch_prompts = prompts[start:start + args.batch_size]
        text, mask, neg, neg_mask = encode_cfg_prompts(tokenizer, encoder, batch_prompts, device, layer, max_length)
        # 所有 prompt 固定同一 seed (逐条独立生成器, 结果与 batch_size 无关, 可复现)
        noises = []
        for _ in batch_index:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noises.append(torch.randn(1, z_dim, 1, args.height // compress, args.width // compress,
                                      device=device, dtype=torch.float32, generator=generator))
        x = torch.cat(noises, 0)
        x = sample_batch(model, x, text, mask, neg, neg_mask,
                         steps=args.steps, guidance=args.guidance_scale, shift=shift, dtype=dtype)
        with torch.no_grad():  # VAE 解码不需要梯度
            images = vae.decode(x / std_inv + mean).sample
        for local_i, idx in enumerate(batch_index):
            path = output_dir / f"{idx:03d}_seed{args.seed}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(to_uint8(images[local_i])).save(path)
            manifest["images"].append({"file": path.name, "index": idx, "seed": args.seed, "prompt": batch_prompts[local_i]})
            log_image(f"{path.name} · {batch_prompts[local_i][:64]}")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_ok(f"🎉 完成: {len(prompts)} 张图 → {output_dir}")


if __name__ == "__main__":
    main()
