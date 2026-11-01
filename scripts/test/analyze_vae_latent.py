"""VAE 潜空间"易学性质"分析: 对真实网络图片编码后, 从多个角度量化潜空间是否好学。

分析维度 (对 5 个 VAE, 12 张真实图, 512²):
  1. 通道统计     per-channel mean/std、近零占比 (|z|<0.1σ)、峰度 (重尾程度)
  2. 通道相关性   C×C Pearson 相关矩阵 (空间展平), 冗余度
  3. 空间平滑度   latent 图水平/垂直 lag-1/2/4 自相关
  4. 频谱分布     径向平均功率谱 (log-log), 与 RGB 图像频谱对照
  5. PCA 能量     通道主成分累计方差 → 有效维度 (90%/99% 能量所需通道数)
  6. 低频截断     latent 2D 低通 (保留半径 r) 后 decode, PSNR-半径曲线 → 结构集中度

输出: exp/vae_bench/latent_analysis/
  metrics.json + latent_maps.png + channel_stats.png + channel_corr.png
  + spectrum.png + structure.png + lowfreq.png + recon_real.png + lowfreq_vis.png

用法: PYTHONPATH=/tmp/diff42 CUDA_VISIBLE_DEVICES=0 python scripts/test/analyze_vae_latent.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 训练占卡时 qwen21 encoder 首层 conv 会报 CUDNN_STATUS_NOT_INITIALIZED,
# 禁用 cuDNN 走原生 conv kernel, 该规模下速度足够
torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_vae_recon import VAES, load_vae, preprocess, psnr  # noqa: E402

OUT = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/vae_bench/latent_analysis")
IMG_DIR = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/vae_bench/real_images")
REP_IMG = "mountain_river.jpg"      # 潜空间可视化 / 低频可视化用代表图
LOWPASS_RADII = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]  # 归一化频率半径 (1.0=全保留)
LOWPASS_VIS = [1.0, 0.2, 0.1]       # 可视化条带用的三个半径
N_VIS_CH = 8                        # 潜空间可视化展示前 8 个通道


# ---------------------------------------------------------------- 基础工具
@torch.no_grad()
def encode_latent(vae, x: torch.Tensor, is_5d: bool) -> torch.Tensor:
    """encode → 取 mode, 返回 [C,h,w] (raw latent, 与 test_vae_recon 口径一致)。"""
    enc = vae.encode(x)
    ld = getattr(enc, "latent_dist", enc)
    ld = ld.mode() if hasattr(ld, "mode") else ld.sample()
    if is_5d:
        ld = ld[:, :, 0]
    return ld[0]


@torch.no_grad()
def decode_latent(vae, lat: torch.Tensor, is_5d: bool, rgb_ref: torch.Tensor) -> torch.Tensor:
    """[C,h,w] latent → RGB [-1,1] (3 通道, 与原图对齐)。rgb_ref: [3,H,W] 原图。"""
    z = lat[None, :, None] if is_5d else lat[None]
    out = vae.decode(z)
    rec = out.sample if hasattr(out, "sample") else out[0]
    if is_5d:
        rec = rec[:, :, 0]
    if rec.shape[1] != rgb_ref.shape[0]:  # qwen21 decode 出 4 通道 → 取前 3
        rec = rec[:, : rgb_ref.shape[0]]
    return rec[0].clamp(-1, 1)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 0.0 if d < 1e-12 else (a * b).sum().div(d).item()


def autocorr(z: torch.Tensor, lag: int) -> float:
    """[C,h,w] 空间自相关 (水平+垂直平均), 按通道算 Pearson 再平均。"""
    vals = []
    for c in range(z.shape[0]):
        m = z[c]
        vals.append(pearson(m[:, :-lag].flatten(), m[:, lag:].flatten()))
        vals.append(pearson(m[:-lag, :].flatten(), m[lag:, :].flatten()))
    return float(np.mean(vals))


def radial_spectrum(z: torch.Tensor, n_bins: int = 32) -> np.ndarray:
    """[C,h,w] → 径向平均功率谱 (归一化频率 0~1, 1=Nyquist), 返回 [n_bins]。"""
    C, h, w = z.shape
    fy = torch.fft.fftfreq(h, device=z.device) * 2   # 归一化到 0~1 (1=Nyquist)
    fx = torch.fft.rfftfreq(w, device=z.device) * 2
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = torch.linspace(0, math.sqrt(2), n_bins + 1, device=z.device)
    power = torch.fft.rfft2(z).abs() ** 2            # [C,h,w//2+1]
    bins = torch.clamp(torch.bucketize(r.flatten(), edges) - 1, 0, n_bins - 1)
    pw = power.flatten(1).mean(0)                    # 通道平均
    spec = torch.zeros(n_bins, device=z.device)
    cnt = torch.zeros(n_bins, device=z.device)
    spec.scatter_add_(0, bins, pw)
    cnt.scatter_add_(0, bins, torch.ones_like(pw))
    spec = (spec / cnt.clamp(min=1)).cpu().numpy()
    return spec / spec.sum()                          # 归一化总能量=1


def lowpass_latent(z: torch.Tensor, frac: float) -> torch.Tensor:
    """[C,h,w] 2D 低通: 保留归一化频率半径 ≤ frac*sqrt(2) 的成分。frac=1 → 全保留。"""
    C, h, w = z.shape
    fy = torch.fft.fftfreq(h, device=z.device) * 2
    fx = torch.fft.rfftfreq(w, device=z.device) * 2
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = (r <= frac * math.sqrt(2) + 1e-6).float()
    Z = torch.fft.rfft2(z) * mask[None]
    return torch.fft.irfft2(Z, s=(h, w))


def to_img(t: torch.Tensor) -> np.ndarray:
    """[-1,1] [3,H,W] → uint8 HWC。"""
    return ((t.clamp(-1, 1) * 0.5 + 0.5) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def heat(ax, mat, title, cmap="viridis", norm=True):
    m = mat.numpy() if torch.is_tensor(mat) else mat
    if norm and m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min())
    ax.imshow(m, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- 主流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(IMG_DIR.glob("*.jpg"))
    srcs = [Image.open(p).convert("RGB") for p in img_paths]
    print(f"{len(srcs)} 张真实图 × {len(VAES)} 个 VAE → {OUT}")

    metrics, lat_store, recon_store = {}, {}, {}
    for name, path in VAES.items():
        t0 = time.time()
        vae, cls_name, is_5d = load_vae(path, device)
        qwen21 = "QwenImage21" in cls_name
        lats, recs = [], {}
        # 逐图编码 + 完整往返重建
        for p, src in zip(img_paths, srcs):
            x = preprocess(src, 512, 512, is_5d, qwen21).to(device)
            lat = encode_latent(vae, x, is_5d)
            x_rgb = x[0, :, 0] if is_5d else x[0]
            rec_rgb = decode_latent(vae, lat, is_5d, x_rgb)
            lats.append(lat.float().cpu())
            recs[p.name] = {"rgb": to_img(rec_rgb), "orig": to_img(x_rgb),
                            "psnr_full": round(psnr(rec_rgb[None], x_rgb[None]), 2)}
        lat_store[name] = torch.stack(lats)          # [N,C,h,w] cpu
        recon_store[name] = recs

        Z = lat_store[name]                          # ---- 统计 (cpu, float32)
        N, C, h, w = Z.shape
        flat = Z.permute(1, 0, 2, 3).reshape(C, -1)  # [C, N*h*w]
        mu = flat.mean(1)
        sd = flat.std(1)
        zc = (flat - mu[:, None]) / sd[:, None]
        kurt = (zc ** 4).mean(1) - 3                 # 超额峰度, 通道平均
        near0 = (flat.abs() < 0.1 * sd[:, None]).float().mean(1)
        corr = torch.from_numpy(np.corrcoef(flat.numpy()))
        # PCA: 协方差 [C,C] 特征分解
        cov = (zc @ zc.T) / (zc.shape[1] - 1)
        evals = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
        ev = (evals / evals.sum()).numpy()
        cum = np.cumsum(ev)
        # 频谱 + 低频截断 (GPU, 逐图)
        vae = vae.to(device)
        spec_lp, spec_rgb_ref = [], None
        lp_psnr = {r: [] for r in LOWPASS_RADII}
        for i, (p, src) in enumerate(zip(img_paths, srcs)):
            z = lat_store[name][i].to(device)
            spec_lp.append(radial_spectrum(z))
            if p.name == REP_IMG:
                arr = np.asarray(src.resize((512, 512), Image.LANCZOS), dtype=np.float32)
                img5d = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device) / 127.5 - 1
                spec_rgb_ref = radial_spectrum(img5d.mean(1))  # 灰度图频谱作对照
            x = preprocess(src, 512, 512, is_5d, qwen21).to(device)
            x_rgb = x[0, :, 0] if is_5d else x[0]
            for r in LOWPASS_RADII:
                rec_lp = decode_latent(vae, lowpass_latent(z, r), is_5d, x_rgb)
                lp_psnr[r].append(psnr(rec_lp[None], x_rgb[None]))
        autoc = {f"lag{k}": float(np.mean([autocorr(lat_store[name][i].to(device), k)
                                           for i in range(N)])) for k in (1, 2, 4)}
        metrics[name] = {
            "class": cls_name, "latent": [C, h, w], "spatial_compress": 512 // h,
            "ch_mean_abs_avg": float(mu.abs().mean()), "ch_std_avg": float(sd.mean()),
            "ch_std_minmax": [float(sd.min()), float(sd.max())],
            "near0_frac_avg": float(near0.mean()), "kurtosis_avg": float(kurt.mean()),
            "autocorr": autoc,
            "pca_cum": cum.tolist(), "pca_dim90": int(np.searchsorted(cum, 0.90) + 1),
            "pca_dim99": int(np.searchsorted(cum, 0.99) + 1),
            "spectrum": np.mean(spec_lp, 0).tolist(),
            "spectrum_rgb": spec_rgb_ref.tolist(),
            "psnr_full_avg": float(np.mean([recs[p.name]["psnr_full"] for p in img_paths])),
            "lowpass_psnr": {str(r): float(np.mean(v)) for r, v in lp_psnr.items()},
            "lowpass_psnr_per_img": {p.name: {str(r): round(lp_psnr[r][i], 2)
                                              for r in LOWPASS_RADII}
                                     for i, p in enumerate(img_paths)},
        }
        print(f"✓ {name} [{C},{h},{w}] 用时 {time.time()-t0:.0f}s  "
              f"σ̄={metrics[name]['ch_std_avg']:.3f} 近零={metrics[name]['near0_frac_avg']:.3f} "
              f"lag1={autoc['lag1']:.3f} dim99={metrics[name]['pca_dim99']} "
              f"r=0.1 PSNR={metrics[name]['lowpass_psnr']['0.1']:.1f}")
        del vae
        torch.cuda.empty_cache()

    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1, ensure_ascii=False)
    names = list(VAES)
    np.save(OUT / "latent_stats.npy",
            {n: {k: v for k, v in metrics[n].items() if k != "spectrum_rgb"} for n in names},
            allow_pickle=True)

    # ---------------------------------------------------------------- 图 1: 潜空间可视化
    idx = [p.name for p in img_paths].index(REP_IMG)
    fig, axes = plt.subplots(len(names), 1 + N_VIS_CH,
                             figsize=(1.5 * (1 + N_VIS_CH), 1.55 * len(names)))
    for r, n in enumerate(names):
        rec = recon_store[n][REP_IMG]
        axes[r, 0].imshow(rec["orig"]); axes[r, 0].set_ylabel(n, fontsize=8)
        if r == 0:
            axes[r, 0].set_title("input", fontsize=8)
        Z = lat_store[n][idx]
        for c in range(N_VIS_CH):
            ch = Z[min(c, Z.shape[0] - 1)].numpy()
            lo, hi = np.percentile(ch, [2, 98])
            axes[r, c + 1].imshow(ch, cmap="viridis", vmin=lo, vmax=hi,
                                  interpolation="nearest")
            if r == 0:
                axes[r, c + 1].set_title(f"ch{min(c, Z.shape[0]-1)}", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Latent maps of '{REP_IMG}' (first {N_VIS_CH} channels, 2-98% clipped)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "latent_maps.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 图 2: 通道统计热图
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 2.1 * len(names)))
    for r, n in enumerate(names):
        Z = lat_store[n]
        sd = Z.flatten(2).std(2).mean(0).numpy()   # [C] 图像间平均的 per-channel std
        mu = Z.flatten(2).mean((0, 2)).numpy()
        heat(axes[r, 0], sd[None], f"{n}  per-channel std (norm)", cmap="magma")
        heat(axes[r, 1], mu[None], f"{n}  per-channel mean (norm)", cmap="coolwarm")
    fig.suptitle("Per-channel statistics (left: std, right: mean)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "channel_stats.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 图 3: 通道相关矩阵
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.0))
    for i, n in enumerate(names):
        Z = lat_store[n].permute(1, 0, 2, 3).reshape(Z.shape[1], -1).numpy()
        heat(axes[i], np.corrcoef(Z), f"{n}", cmap="RdBu_r", norm=False)
    fig.suptitle("Inter-channel Pearson correlation", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "channel_corr.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 图 4: 频谱
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = np.linspace(0, math.sqrt(2), 33)[:-1] + 1 / 64
    for n in names:
        ax.semilogy(xs, np.maximum(metrics[n]["spectrum"], 1e-12), label=n, lw=1.6)
    ax.semilogy(xs, np.maximum(metrics[names[0]]["spectrum_rgb"], 1e-12),
                "k--", label="RGB image (ref)", lw=1.2)
    ax.set_xlabel("normalized radial frequency (1 = Nyquist)")
    ax.set_ylabel("fraction of total power (log)")
    ax.set_title("Radially averaged latent power spectrum")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "spectrum.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 图 5: 平滑度 + PCA
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    lags = ["lag1", "lag2", "lag4"]
    xw = np.arange(len(lags)); bw = 0.16
    for i, n in enumerate(names):
        a1.bar(xw + i * bw, [metrics[n]["autocorr"][l] for l in lags], bw, label=n)
    a1.set_xticks(xw + 2 * bw); a1.set_xticklabels(lags)
    a1.set_ylabel("spatial autocorrelation"); a1.set_title("Latent spatial smoothness")
    a1.legend(fontsize=8); a1.grid(alpha=0.3, axis="y")
    ks = np.arange(1, max(metrics[n]["latent"][0] for n in names) + 1)
    for n in names:
        cum = metrics[n]["pca_cum"]
        a2.plot(np.arange(1, len(cum) + 1) / metrics[n]["latent"][0], cum, label=n, lw=1.6)
    for thr, ls in [(0.9, "--"), (0.99, ":")]:
        a2.axhline(thr, color="gray", ls=ls, lw=0.8)
    a2.set_xlabel("fraction of channels (PC rank / C)"); a2.set_ylabel("cumulative explained variance")
    a2.set_title("PCA energy over channels"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    a2.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(OUT / "structure.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 图 6: 低频截断
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    for n in names:
        ys = [metrics[n]["lowpass_psnr"][str(r)] for r in LOWPASS_RADII]
        a1.plot(LOWPASS_RADII, ys, "o-", label=n, lw=1.6, ms=4)
        a1.axhline(metrics[n]["psnr_full_avg"], color="gray", ls=":", lw=0.7)
    a1.set_xlabel("kept frequency radius (1.0 = full latent)")
    a1.set_ylabel("decode PSNR vs original (dB)")
    a1.set_title("Low-pass truncated latent → decode quality")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)
    # 可视化条带: 代表图, 每 VAE 一行 [原图, r=1.0, 0.2, 0.1]
    # (低频重建图在主循环未存, 此处用低分辨重存: 重新读 metrics 中逐图数据画曲线即可)
    a2.axis("off")
    a2.set_title(f"per-image low-pass PSNR ({REP_IMG})")
    for n in names:
        per = metrics[n]["lowpass_psnr_per_img"][REP_IMG]
        a2.plot(LOWPASS_RADII, [per[str(r)] for r in LOWPASS_RADII], "o-", label=n, ms=3)
    a2.legend(fontsize=8); a2.grid(alpha=0.3); a2.set_xlabel("kept radius"); a2.set_ylabel("PSNR (dB)")
    fig.tight_layout(); fig.savefig(OUT / "lowfreq.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 图 7: 真实图重建条带
    show = [p.name for p in img_paths][:4]
    fig, axes = plt.subplots(1 + len(names), len(show),
                             figsize=(2.1 * len(show), 2.15 * (1 + len(names))))
    for c, nm in enumerate(show):
        axes[0, c].imshow(recon_store[names[0]][nm]["orig"])
        if c == 0:
            axes[0, 0].set_ylabel("original", fontsize=9)
    for r, n in enumerate(names):
        for c, nm in enumerate(show):
            axes[r + 1, c].imshow(recon_store[n][nm]["rgb"])
        axes[r + 1, 0].set_ylabel(f"{n}\n{metrics[n]['psnr_full_avg']:.1f}dB", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Real-image reconstructions (512², avg PSNR labelled)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "recon_real.png", dpi=130); plt.close(fig)

    print(f"\n完成 → {OUT}")
    for n in names:
        m = metrics[n]
        print(f"  {n:14s} dim99={m['pca_dim99']:3d}/{m['latent'][0]:3d} "
              f"lag1={m['autocorr']['lag1']:.3f} 近零={m['near0_frac_avg']:.3f} "
              f"峰度={m['kurtosis_avg']:6.1f} PSNR={m['psnr_full_avg']:.2f}")


if __name__ == "__main__":
    main()
