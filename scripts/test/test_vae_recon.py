"""VAE 重建质量与易用性对比: encode → decode 往返 (raw latents, 不做归一化)。

候选 (ckpt/vae_bench/, 每个 = diffusers vae 子目录):
  qwen_image_21  AutoencoderKLQwenImage21  Qwen-Image 2.x 世代 VAE (16ch)
  flux3_klein    AutoencoderKLFlux2        FLUX.3 klein VAE (32ch)
  moss_flux2     AutoencoderKLFlux2        MOSS-Image 自带, FLUX.2 架构 (32ch)
  z_image        AutoencoderKL             Z-Image 自带, Qwen-Image v1/FLUX.1 风格 (16ch)
  wan22_video    AutoencoderKLWan          Wan2.2 TI2V 视频 VAE, 图像模式 T=1 (48ch, 16x 空间压缩)

指标: PSNR / SSIM (torch 实现, 无 skimage 依赖) / encode+decode 耗时 / 显存峰值 / latent 形状与压缩率。
分辨率: 512², 1024², 768x512 (非方), 500x500 (奇数尺寸, 记录是否报错)。
输出: <out>/summary.json + summary.md + grid_<image>.png (首行原图, 之后每个 VAE 一行)。

用法: CUDA_VISIBLE_DEVICES=0 python scripts/test/test_vae_recon.py [--gpu 0] [--vae qwen_image_21 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# /opt/conda 的 peft→transformers 链与 hub 2.4 不兼容; VAE 测试用不到 peft (diffusers 0.42
# 只有 hooks/layerwise_casting 一处 eager import peft.helpers), 打桩绕过, 真环境可用时不打桩。
import contextlib
import sys
import types as _types

try:
    import peft  # noqa: F401
except Exception:
    import importlib.machinery
    _peft = _types.ModuleType("peft")
    _peft.__path__ = []
    _peft.__spec__ = importlib.machinery.ModuleSpec("peft", None)
    _peft.__version__ = "0.0.0"
    _peft_helpers = _types.ModuleType("peft.helpers")
    _peft_helpers.__spec__ = importlib.machinery.ModuleSpec("peft.helpers", None)
    _peft_helpers.disable_input_dtype_casting = contextlib.nullcontext
    _peft.helpers = _peft_helpers
    # diffusers hooks/layerwise_casting 见 is_peft_available()=True 就 eager import 下面两个,
    # 一并补齐 (BaseTunerLayer 只被 isinstance/类型注解使用, 用 nn.Module 顶替)
    _peft_tuners = _types.ModuleType("peft.tuners")
    _peft_tuners.__path__ = []
    _peft_tuners_utils = _types.ModuleType("peft.tuners.tuners_utils")
    _peft_tuners_utils.BaseTunerLayer = torch.nn.Module
    _peft_utils_mod = _types.ModuleType("peft.utils")
    _peft_utils_mod.__path__ = []
    sys.modules.update({
        "peft": _peft, "peft.helpers": _peft_helpers,
        "peft.tuners": _peft_tuners, "peft.tuners.tuners_utils": _peft_tuners_utils,
        "peft.utils": _peft_utils_mod,
    })

ROOT = Path(__file__).resolve().parents[2]
CKPT_BENCH = Path("/home/jovyan/pfs/users/zhanqian.wu/ckpt/vae_bench")
DEFAULT_IMAGES = [
    "/mnt/nfs/data/public/qwenimage_t2i/images/green_checker.png",   # 硬边缘 + 高频棋盘
    "/mnt/nfs/data/public/qwenimage_t2i/images/blue_stripes.png",    # 细条纹
    "/mnt/nfs/models/HunyuanImage-3-Instruct/assets/robot.png",      # 细节渲染
    "/mnt/nfs/models/HunyuanImage-3-Instruct/assets/wechat.png",     # 文字 / 图形界面
    "/home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/stage_1_kd_28layers_v3_traj/vis/step-00235000/000_seed42.png",  # 自然照片
]
RESOLUTIONS = [(512, 512), (1024, 1024), (768, 512), (500, 500)]  # (W, H)

VAES = {
    "qwen_image_21": CKPT_BENCH / "qwen_image_21",
    "flux3_klein": CKPT_BENCH / "flux3_klein",
    "moss_flux2": CKPT_BENCH / "moss_flux2",
    "z_image": CKPT_BENCH / "z_image",
    "wan22_video": CKPT_BENCH / "wan22_video",
}


def load_vae(path: Path, device):
    import diffusers
    cfg = json.load(open(path / "config.json"))
    cls_name = cfg.get("_class_name", "AutoencoderKL")
    fallbacks = {"AutoencoderKLQwenImage21": "AutoencoderKLQwenImage"}
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        cls = getattr(diffusers, fallbacks.get(cls_name, "AutoencoderKL"))
    vae = cls.from_pretrained(path, torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    # AutoencoderKLWan / AutoencoderKLQwenImage21 走 [B,C,T,H,W] 5D 输入, 其余 [B,C,H,W]
    is_5d = "Wan" in cls_name or "QwenImage21" in cls_name
    return vae, cls_name, is_5d


def preprocess(img: Image.Image, w: int, h: int, is_5d: bool, qwen21: bool = False) -> torch.Tensor:
    """PIL → [-1,1] 张量: 5D 模型 [B,C,T=1,H,W], 其余 [B,C,H,W]。
    QwenImage21 (NFS 权重) 是 RGB+condition 4 通道变体, 第 4 通道置 0。"""
    x = torch.from_numpy(__import__("numpy").asarray(img.resize((w, h), Image.LANCZOS))).float() / 127.5 - 1.0
    x = x.permute(2, 0, 1)[None]
    if qwen21:
        x = torch.cat([x, torch.zeros(1, 1, h, w)], 1)
    return x[:, :, None] if is_5d else x


@torch.no_grad()
def roundtrip(vae, x: torch.Tensor, is_5d: bool):
    """encode→decode 往返, 返回 (recon_rgb, x_rgb, latent)。"""
    enc = vae.encode(x)
    latent = getattr(enc, "latent_dist", enc)
    latent = latent.mode() if hasattr(latent, "mode") else latent.sample()
    recon = vae.decode(latent)
    recon = recon.sample if hasattr(recon, "sample") else recon[0]
    if is_5d:
        recon = recon[:, :, 0]
    x_rgb = x[:, :, 0] if (is_5d and x.dim() == 5) else x  # 5D 输入的 RGB 参照也要挤掉 T 维
    if recon.shape[1] == 4 and x_rgb.shape[1] == 4:  # QwenImage21: RGB+condition(0), 指标只比 RGB
        recon, x_rgb = recon[:, :3], x_rgb[:, :3]
    elif recon.shape[1] != x_rgb.shape[1]:
        recon = recon[:, : x_rgb.shape[1]]
    return recon.clamp(-1, 1), x_rgb.clamp(-1, 1), latent


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """a/b: [-1,1], 峰值 2.0。"""
    import math
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse < 1e-9 else 10 * math.log10(4.0 / mse)


def gaussian_kernel(ks: int = 11, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(ks) - ks // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    return (k / k.sum())[None, None]


_K = gaussian_kernel()


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """a/b: [1,3,H,W] in [-1,1]。标准 SSIM 参数 (11x11, K1=0.01, K2=0.03, L=2)。"""
    L = 2.0
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    k = _K.to(a.device).expand(3, 1, 11, 11)
    mu_a = F.conv2d(a, k, groups=3, padding=5)
    mu_b = F.conv2d(b, k, groups=3, padding=5)
    var_a = F.conv2d(a * a, k, groups=3, padding=5) - mu_a ** 2
    var_b = F.conv2d(b * b, k, groups=3, padding=5) - mu_b ** 2
    cov = F.conv2d(a * b, k, groups=3, padding=5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))
    return s.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", action="append", default=[], help="只测指定 VAE (可多次), 默认全部")
    ap.add_argument("--images", nargs="*", default=None, help="覆盖测试图片列表")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = args.vae or list(VAES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("/home/jovyan/pfs/users/zhanqian.wu/exp/vae_bench") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = args.images if args.images is not None else [p for p in DEFAULT_IMAGES if Path(p).exists()]
    srcs = [Image.open(p).convert("RGB") for p in img_paths]
    print(f"测试 {len(names)} 个 VAE × {len(srcs)} 张图 × {len(RESOLUTIONS)} 分辨率 → {out_dir}")

    rows, grids = [], {}
    for name in names:
        path = VAES[name]
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        try:
            vae, cls_name, is_5d = load_vae(path, device)
        except Exception as e:
            rows.append({"vae": name, "error": f"加载失败: {e}"})
            print(f"❌ {name} 加载失败: {e}")
            continue
        load_s = time.time() - t0
        cfg = vae.config
        norm = {}
        for k in ("scaling_factor", "shift_factor", "z_dim", "latent_channels", "block_out_channels"):
            if getattr(cfg, k, None) is not None:
                norm[k] = getattr(cfg, k)
        if getattr(cfg, "latents_mean", None):
            norm["latents_mean/std"] = f"{len(cfg.latents_mean)}ch per-channel"
        rec = {"vae": name, "class": cls_name, "load_s": round(load_s, 1),
               "path": str(path), "norm_cfg": norm, "res": {}, "imgs": {}}
        print(f"▶ {name} ({cls_name}) norm={norm}")

        # 分辨率兼容性 + 速度 (512² 单图计时)
        qwen21 = "QwenImage21" in cls_name
        for w, h in RESOLUTIONS:
            x = preprocess(srcs[0], w, h, is_5d, qwen21).to(device)
            try:
                torch.cuda.synchronize(); t1 = time.time()
                rec_lat, _, lat = roundtrip(vae, x, is_5d)
                torch.cuda.synchronize()
                dt = time.time() - t1
                rec["res"][f"{w}x{h}"] = {
                    "ok": True, "sec": round(dt, 2),
                    "latent": list(lat.shape[-3:]),
                    "compress": round((w * h) / (lat.shape[-2] * lat.shape[-1]), 1),
                }
                print(f"   {w}x{h} ✓ {dt:.2f}s latent={tuple(lat.shape[-3:])}")
            except Exception as e:
                rec["res"][f"{w}x{h}"] = {"ok": False, "error": str(e)[:160]}
                print(f"   {w}x{h} ✗ {str(e)[:80]}")

        # 逐图质量指标 (512²)
        vram = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        for tag, src, p in zip(["img0", "img1", "img2", "img3", "img4"], srcs, img_paths):
            x = preprocess(src, 512, 512, is_5d, qwen21).to(device)
            try:
                rec_lat, x_rgb, lat = roundtrip(vae, x, is_5d)
                p_val = psnr(rec_lat, x_rgb)
                s_val = ssim(rec_lat, x_rgb)
                rec["imgs"][tag] = {"src": Path(p).name, "psnr": round(p_val, 2), "ssim": round(s_val, 4)}
                grids.setdefault(tag, {"src_img": src, "rows": []})["rows"].append((name, rec_lat))
                print(f"   {tag} {Path(p).name}: PSNR={p_val:.2f} SSIM={s_val:.4f}")
            except Exception as e:
                rec["imgs"][tag] = {"src": Path(p).name, "error": str(e)[:160]}
                print(f"   {tag} ✗ {str(e)[:100]}")
        rec["vram_gib"] = round(vram, 2)
        rows.append(rec)
        del vae
        torch.cuda.empty_cache()

    # 汇总
    summary = {r["vae"]: r for r in rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["vae", "class", "psnr_img0..4", "ssim_img0..4", "512sec", "1024sec", "vram_gib", "errors"])
        for r in rows:
            if "error" in r:
                wcsv.writerow([r["vae"], "", "", "", "", "", "", r["error"]])
                continue
            p = [r["imgs"].get(t, {}).get("psnr", "ERR") for t in ["img0", "img1", "img2", "img3", "img4"]]
            s = [r["imgs"].get(t, {}).get("ssim", "ERR") for t in ["img0", "img1", "img2", "img3", "img4"]]
            errs = sum(1 for v in r["res"].values() if not v.get("ok"))
            wcsv.writerow([r["vae"], r["class"], *p, *s,
                           r["res"].get("512x512", {}).get("sec"), r["res"].get("1024x1024", {}).get("sec"),
                           r["vram_gib"], errs or ""])

    # 拼图: 每张测试图一个 grid, 行 = 原图 + 各 VAE
    import numpy as np
    for tag, g in grids.items():
        tiles = [np.asarray(g["src_img"].resize((512, 512), Image.LANCZOS))]
        for name, rec_lat in g["rows"]:
            arr = ((rec_lat[0].float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype("uint8")
            tiles.append(arr)
        grid = np.concatenate(tiles, axis=0)
        Image.fromarray(grid).save(out_dir / f"grid_{tag}.png")

    print(f"\n✅ 完成 → {out_dir}/summary.csv / summary.json / grid_*.png")


if __name__ == "__main__":
    main()
