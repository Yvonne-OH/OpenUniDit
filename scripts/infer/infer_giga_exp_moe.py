#!/usr/bin/env python3
"""Giga-Exp stage_1 T2I flow-matching MoE 推理: 多 prompt (list) → flow ODE 采样 → Wan VAE 解码 → PNG.

- 基于 infer_giga_exp.py: 参数约定/文本编码/采样/保存逻辑完全一致, 仅增加 MoE 差异
- 模型构造传入训练 yaml 的 model.moe 配置 (含 layers 键, GigaExpTransformer 内部按层分发 MoE/dense);
  纯 torch bf16 推理 (无 DeepSpeed), 构造/搬设备后 cast_router_fp32() 保持 router 权重 fp32
- 路由固定 phase C 纯学习路由: moe_routing_step=10**9 (远超 phase_b_end_step), 无强制均衡/jitter;
  eval 模式下 MoEFFN 不产生随机路由, aux loss 仅暂存不参与输出 (推理无副作用)
- 权重: --checkpoint 指定文件或目录; 缺省取 BOS 下载缓存最新 step-* (配套 infer_giga_exp_moe_bos.sh);
  dense ckpt 自动做 dense→MoE 迁移 (每层 FFN 拷贝到全部 experts, router gate 随机初始化)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# FA3 (flash_attn_3) 位于 thirdparty: 必须在 import models.* 之前注入,
# 否则 giga_transformer_mhc 顶层 `from models.giga_transformer import _fa3_func` 会失败
# (注入方式与 train_giga_exp_mhc.py 一致)
_THIRDPARTY = str(ROOT / "thirdparty")
if Path(_THIRDPARTY).is_dir() and _THIRDPARTY not in sys.path:
    sys.path.insert(0, _THIRDPARTY)

import torch
import yaml
from diffusers import AutoencoderKLWan
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from models.giga_transformer import (GigaExpTransformer, apply_adaln_basis_migration,
                                     apply_mod_adaln_migration)
from models.moe import apply_dense_to_moe_migration

try:
    from tqdm import tqdm, trange
except ImportError:  # 缺 tqdm 时退化为无进度条
    tqdm = trange = None

# 推理路由固定 phase C (纯学习路由): routing_step 拉到远超 phase_b_end_step,
# schedule.at() 恒返回 (phase C, learned_probability=1.0, phase_c_jitter);
# eval 模式下 jitter 与 balanced-random 路由均不生效, 与 MoE 训练收敛后的路由行为一致
MOE_ROUTING_STEP = 10 ** 9


def _read_weight_metadata(path):
    """读取权重文件内嵌的 semantic metadata (safetensors 头 / .pt 旁路 .json); 无则返回 None。"""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw = (handle.metadata() or {}).get("giga_semantic_metadata")
        return json.loads(raw) if raw else None
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8")).get("giga_semantic_metadata")


def _report_patchify_reinit(model, state, current, matched, metadata):
    """patchify 配置变化 → patchify 入口/出口层无法继承时红字警告 (推理侧)。

    兼容语义: shape 匹配的参数全部正常继承, patchify 入口/出口
    (patch_in.weight / out.weight / out.bias) 保持重新随机初始化。"""
    dropped = [key for key in ("patch_in.weight", "out.weight", "out.bias")
               if key in state and key in current and key not in matched]
    if not dropped:
        return
    new_ps = list(model.patch_size)
    old_ps = metadata.get("patch_size") if isinstance(metadata, dict) else None
    if old_ps is not None:
        origin = f"checkpoint patch_size={list(old_ps)} → 当前 patch_size={new_ps}"
    else:
        # 旧格式 checkpoint 无 semantic metadata: 由 patch_in 输入维反推 patch volume
        weight = state.get("patch_in.weight")
        inferred = weight.shape[1] // model.in_channels if weight is not None and weight.ndim == 2 \
            and model.in_channels and weight.shape[1] % model.in_channels == 0 else None
        origin = (f"checkpoint 未记录 patch_size, 由 patch_in 输入维推断 patch volume={inferred} "
                  f"→ 当前 patch_size={new_ps}")
    message = (f"🚨 patchify 不兼容: {origin}。\n"
               f"    patchify 入口/出口层无法继承 (shape 不匹配): {', '.join(dropped)}, "
               f"共 {sum(state[k].numel() for k in dropped) / 1e6:.2f}M 参数保持重新随机初始化;\n"
               f"    其余 {len(matched)}/{len(state)} 项参数全部正常继承。\n"
               f"    patchify 层未经训练 (随机初始化), 推理质量会显著劣化, 属预期现象。")
    print(f"\033[1;31m{message}\033[0m", flush=True)


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
    parser = argparse.ArgumentParser(description="Giga-Exp stage_1 T2I MoE 推理")
    parser.add_argument("--prompt", action="append", default=[], help="单条 prompt, 可重复传入构成 list")
    parser.add_argument("--prompt-file", type=Path, default=None, help="prompt 文件, 每行一条, 空行忽略")
    parser.add_argument("--config", default="configs/stage_1_xl_moe.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="权重文件或含 step-* 的目录; 缺省取 CHECKPOINT 环境变量或 BOS 下载缓存 "
                             "(BOS_LOCAL_CKPT_DIR, 默认 outputs/bos_ckpt/giga_exp_image/stage_1_xl_moe) 最新 step-*")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="缺省 outputs/pretrain_xl_moe/<checkpoint 名>")
    parser.add_argument("--height", type=int, default=128, help="输出像素高, 必须是 32 的倍数 (默认 128, 横屏 192x128)")
    parser.add_argument("--width", type=int, default=192, help="输出像素宽, 必须是 32 的倍数 (默认 192, 横屏 192x128)")
    parser.add_argument("--steps", type=int, default=40, help="采样步数 (训练验证用 20)")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=None, help="σ schedule shift; 缺省取 scheduler.validation_shift")
    parser.add_argument("--seed", type=int, default=42, help="固定噪声种子, 所有 prompt 共用")
    parser.add_argument("--negative-prompt", default="<|im_end|>", help="自然语言 negative prompt; 缺省使用 EOS 无条件分支")
    parser.add_argument("--batch-size", type=int, default=4, help="list input 的分批大小")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="cuda 缺省自动选最后一张可见 GPU; cpu 用于调试")
    # ── attention profiling (旁路 hook, 不改变 forward 数值) ──
    parser.add_argument("--attn-profile", action="store_true",
                        help="推理时同步做 attention profiling, 结果存 <output>/attn_profile/batch_*/")
    parser.add_argument("--attn-profile-layers", default=None,
                        help="只统计这些层: 逗号分隔 (如 0,7,14,21,27) 或 'all'; 缺省每隔 5 层 (0,5,10,...)")
    parser.add_argument("--attn-profile-branch", default="cond", choices=["cond", "uncond", "both"],
                        help="统计哪个 CFG 分支 (batch 后半 = cond); 默认 cond")
    parser.add_argument("--attn-profile-stride", type=int, default=1,
                        help="每 N 个 ODE step 采样一次 (默认 1 = 每步都采)")
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


def load_moe_compatible_state_dict(model, state, metadata=None):
    """MoE 版兼容加载: 只加载名称和 shape 都匹配的参数, 其余保留初始化值.

    与 infer_giga_exp.py 的 load_compatible_state_dict 一致 (AdaLN 结构迁移 + patchify 兼容),
    额外增加 dense→MoE 热启动迁移 (对齐训练脚本 train_giga_exp.py):
    - ckpt 缺 experts/router 键 (dense ckpt) → apply_dense_to_moe_migration 把 MoE 层的
      dense FFN (blocks.{i}.ff.{0,2}.weight) 拷贝到该层全部 experts (强制均衡初始化),
      router gate 保持随机初始化; dense 层 (moe.layers 跳过的层) 键名直接匹配;
    - ckpt 已含 experts 键 (MoE checkpoint) → 直接加载, 不做迁移。"""
    current = model.state_dict()
    basis = apply_adaln_basis_migration(current, state)
    if basis["source"] is None:
        migration = apply_mod_adaln_migration(current, state)
        if migration["copied"]:
            print(f"mod-AdaLN 迁移: {len(migration['copied'])} 层旧 mod.* 已拷贝到 mod_img/mod_txt", flush=True)
        if migration["missing"]:
            print(f"⚠️ mod-AdaLN: {len(migration['missing'])} 层 ckpt 缺失调制权重, 保持随机初始化", flush=True)
        if migration["orphan_dual"]:
            print(f"⚠️ mod-AdaLN: 模型为 shared 但 ckpt 是 dual-branch, {len(migration['orphan_dual'])} 层保持随机初始化", flush=True)
    elif basis["converted"]:
        print(f"\033[1;31m⚠️ adaln-basis 迁移: 旧 {basis['source']} 全秩 ckpt 已低秩转换 "
              f"(rank={basis['rank']}, 捕获能量 {basis['energy']:.1%}, {len(basis['layers'])} 层)\033[0m", flush=True)
    else:
        print("adaln-basis: ckpt 已是共享基结构, 直接加载", flush=True)
    # dense→MoE 迁移 (models/moe.py, 与训练侧同一实现): 就地改写 state 后再匹配
    upcycle = apply_dense_to_moe_migration(current, state)
    if upcycle["model_moe"] and upcycle["already_moe"]:
        print("moe-upcycle: ckpt 已含 experts/router 权重 (MoE checkpoint), 直接加载", flush=True)
    elif upcycle["model_moe"] and upcycle["upcycled"]:
        print(f"\033[1;33m⚠️ moe-upcycle: dense ckpt 的每层 FFN 已拷贝到全部 experts "
              f"(强制均衡初始化, {len(upcycle['upcycled'])} 层); router gate 保持随机初始化, "
              f"路由无训练痕迹, 建议使用 MoE 训练产出的权重\033[0m", flush=True)
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
    _report_patchify_reinit(model, state, current, matched, metadata)
    result = model.load_state_dict(matched, strict=False)
    print(f"权重加载: matched={len(matched)}, reinitialized={len(result.missing_keys)}, "
          f"ignored={len(state) - len(matched)}", flush=True)


def wan_latent_stats(vae, device):
    """Wan 官方约定: z_norm = (z_vae - mean) / std, decode 前 z_vae = z_norm * std + mean."""
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, vae.config.z_dim, 1, 1, 1).float()
    std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=device)).view(1, vae.config.z_dim, 1, 1, 1).float()
    return mean, std_inv


def encode_cfg_prompts(tokenizer, encoder, prompts, negative_prompt, device, layer, max_length):
    """正样本 + negative prompt 一次性编码, 统一 padding 长度后拆分 (供 CFG cat)."""
    filled = [p.strip() if p and p.strip() else tokenizer.eos_token for p in prompts]
    negative = negative_prompt.strip() if negative_prompt and negative_prompt.strip() else tokenizer.eos_token
    inputs = tokenizer(filled + [negative] * len(filled),
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
    cfg = yaml.safe_load((Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config).read_text())
    # Wan VAE 空间压缩 16 × 模型空间 patch ⇒ 像素宽高须被 16·max(ph,pw) 整除 (patch 111 → 16)
    pixel_multiple = 16 * max(int(p) for p in cfg["model"].get("patch_size", [1, 2, 2])[1:])
    if args.height % pixel_multiple or args.width % pixel_multiple:
        log_error(f"height/width 必须是 {pixel_multiple} 的倍数 (patch_size={cfg['model'].get('patch_size')}), "
                  f"当前 {args.height}x{args.width}")
        raise SystemExit(2)
    prompts = collect_prompts(args)
    # 权重来源优先级: --checkpoint > CHECKPOINT 环境变量 > BOS 下载缓存最新 step-*
    # (缓存目录 = infer_giga_exp_moe_bos.sh 的 LOCAL_CKPT_DIR, 可用 BOS_LOCAL_CKPT_DIR 覆盖;
    #  目录/文件均不存在时报清晰 FileNotFoundError, 对齐 tiny_bos 脚本惯例)
    default_ckpt_dir = Path(os.environ.get("BOS_LOCAL_CKPT_DIR")
                            or ROOT / "outputs" / "bos_ckpt" / "giga_exp_image" / "stage_1_xl_moe")
    checkpoint, step = resolve_checkpoint(args.checkpoint or os.environ.get("CHECKPOINT") or default_ckpt_dir, None)
    # 缺省输出: outputs/pretrain_xl_moe/<ckpt 名>/
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / "outputs" / "pretrain_xl_moe" / checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    tc, dc = cfg["components"]["text_encoder"], cfg["components"]["wan_vae"]
    dtype = torch.bfloat16 if cfg["training"].get("mixed_precision", "bf16") == "bf16" else torch.float16
    shift = args.shift if args.shift is not None else float(cfg["scheduler"].get("validation_shift") or cfg["scheduler"].get("shift") or 1.0)
    moe_cfg = cfg["model"].get("moe") or {}
    moe_desc = (f"{moe_cfg.get('num_experts', 4)} experts top-{moe_cfg.get('top_k', 2)}, "
                f"layers={moe_cfg.get('layers', '全部层')}") if moe_cfg else "dense (无 MoE)"

    print(paint(Style.BOLD, "⚙️  推理设置"), flush=True)
    print(render_settings_table([
        ("config", str(args.config)),
        ("moe", moe_desc),
        ("checkpoint", f"{checkpoint.name} (step {step})"),
        ("size", f"{args.height} x {args.width}"),
        ("steps", str(args.steps)),
        ("guidance_scale", str(args.guidance_scale)),
        ("negative_prompt", args.negative_prompt or "<|im_end|>"),
        ("shift", str(shift)),
        ("seed", str(args.seed)),
        ("batch_size", str(args.batch_size)),
        ("prompts", str(len(prompts))),
        ("device", str(device)),
        ("output_dir", str(output_dir)),
        ("attn_profile", str(args.attn_profile)),
    ]), flush=True)

    log_info(f"🧩 加载组件: encoder={tc['pretrained_path']} · vae={dc['pretrained_path']}")
    tokenizer = AutoTokenizer.from_pretrained(tc["tokenizer_path"], local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        tc["pretrained_path"], torch_dtype=dtype, local_files_only=True).eval().to(device)
    vae = AutoencoderKLWan.from_pretrained(
        dc["pretrained_path"], subfolder=dc.get("subfolder", "vae"),
        torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    # 构造时原样传入 model.moe (含 layers 键, GigaExpTransformer 内部剥掉 layers 后按层分发,
    # 缺省层全 MoE / {first,last} 跳过首尾 / list 显式层号); 推理端不做 mHC 分发
    model = GigaExpTransformer(**cfg["model"]).to(device, dtype=dtype)
    if getattr(model, "has_moe", False):
        # router 全 fp32 计算域: .to(bf16) 会把 gate 一并降精度, 构造/搬设备后统一回浮
        # (MoEFFN.forward 本身也会把权重与输入上浮 fp32, 两条路径兼容)
        model.cast_router_fp32()
        # 路由固定 phase C 纯学习路由: 远超 phase_b_end_step, 无强制均衡/jitter;
        # eval 模式无随机路由, aux loss 仅暂存 last_moe_aux/moe_aux_losses, 不参与输出 (无副作用);
        # 不挂 _moe_stats_collector (denoising-step 统计钩子缺省关闭, 零开销)
        model.set_moe_routing_step(MOE_ROUTING_STEP)
        log_info(f"🧠 MoE: {len(model.moe_layer_ids)}/{model.layers} 层 MoE "
                 f"(experts={moe_cfg.get('num_experts', 4)}, top_k={moe_cfg.get('top_k', 2)}), "
                 f"routing_step={MOE_ROUTING_STEP} → phase C 纯学习路由, router=fp32")
    state = load_file(str(checkpoint)) if checkpoint.suffix == ".safetensors" \
        else torch.load(str(checkpoint), map_location="cpu")
    load_moe_compatible_state_dict(model, state, _read_weight_metadata(checkpoint))
    log_ok(f"📦 模型就绪: {checkpoint.name} (step {step})")

    z_dim = vae.config.z_dim
    compress = vae.config.scale_factor_spatial
    mean, std_inv = wan_latent_stats(vae, device)
    layer = tc.get("layer")
    max_length = tc.get("max_length", 512)
    # ── attention profiling 初始化 (attn_profile=None 时零开销) ──
    prof_kwargs = None
    if args.attn_profile:
        from tools.attn_profile import DiTAttentionProfiler
        spec = (args.attn_profile_layers or "").strip()
        if spec.lower() == "all":
            prof_layers = None                       # 全部层
        elif spec:
            prof_layers = [int(x) for x in spec.split(",") if x.strip()]
        else:
            prof_layers = list(range(0, len(model.blocks), 5))   # 缺省每隔 5 层
        ps_t, ps_h, ps_w = model.patch_size
        prof_grid = (ps_t, (args.height // compress) // ps_h, (args.width // compress) // ps_w)
        prof_kwargs = dict(layers=prof_layers, branch=args.attn_profile_branch,
                           stride=args.attn_profile_stride, grid=prof_grid,
                           meta={"checkpoint": f"{checkpoint.name} (step {step})",
                                 "size": f"{args.height}x{args.width}",
                                 "steps": args.steps, "guidance": args.guidance_scale,
                                 "shift": shift, "seed": args.seed})
        log_info(f"📊 attention profiling: layers={'all' if prof_layers is None else prof_layers}, "
                 f"branch={args.attn_profile_branch}, stride={args.attn_profile_stride}")
    manifest = {
        "checkpoint": str(checkpoint), "step": step, "config": str(args.config),
        "height": args.height, "width": args.width, "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale, "negative_prompt": args.negative_prompt or tokenizer.eos_token,
        "shift": shift, "seed": args.seed,
        "images": [],
    }
    total_batches = math.ceil(len(prompts) / args.batch_size)
    batch_range = range(0, len(prompts), args.batch_size)
    batch_iter = (tqdm(batch_range, total=total_batches, desc="🎨 采样", unit="batch", colour="#5FD7FF")
                  if tqdm else batch_range)
    for start in batch_iter:
        batch_index = list(range(start, min(start + args.batch_size, len(prompts))))
        batch_prompts = prompts[start:start + args.batch_size]
        text, mask, neg, neg_mask = encode_cfg_prompts(
            tokenizer, encoder, batch_prompts, args.negative_prompt, device, layer, max_length)
        # 所有 prompt 固定同一 seed (逐条独立生成器, 结果与 batch_size 无关, 可复现)
        noises = []
        for _ in batch_index:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noises.append(torch.randn(1, z_dim, 1, args.height // compress, args.width // compress,
                                      device=device, dtype=torch.float32, generator=generator))
        x = torch.cat(noises, 0)
        # 每个 batch 独立 profiler (统计状态按 batch 隔离, 逐 prompt 可对比)
        batch_profiler = None
        if prof_kwargs is not None:
            batch_profiler = DiTAttentionProfiler(model, **prof_kwargs)
        with (batch_profiler if batch_profiler is not None else contextlib.nullcontext()):
            x = sample_batch(model, x, text, mask, neg, neg_mask,
                             steps=args.steps, guidance=args.guidance_scale, shift=shift, dtype=dtype)
        if batch_profiler is not None:
            prof_dir = output_dir / "attn_profile" / f"batch_{start:03d}"
            prof_summary = batch_profiler.save(prof_dir)
            print(DiTAttentionProfiler.render_summary(prof_summary, max_steps=4), flush=True)
            log_ok(f"📊 attention profiling 已保存 → {prof_dir}")
            manifest.setdefault("attn_profile_dirs", []).append(str(prof_dir))
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
