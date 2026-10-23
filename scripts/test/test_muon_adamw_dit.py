"""Muon+AdamW 优化器验证 — 真·MiniDiT 版 (flow matching v-pred, CPU, 4 组并行).

模型: MiniDiT (patch embed + sin-cos pos emb + timestep AdaLN-Zero blocks + MHSA + unpatchify out 头)
任务: 小型 flow matching 去噪 — x_t=(1-t)*x0+t*noise, 目标 v=noise-x0 (与 Giga 训练目标同构)
对比: AdamW(1e-3/3e-3) vs Muon+AdamW(muon 3x/1x), 4 组进程级并行训练, 另按 t 分桶评估.

用法:
  python scripts/test/test_muon_adamw_dit.py            # 分组检查 + 并行收敛对比 + 图表
"""
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "train"))

import torch
import torch.nn.functional as F
from torch import nn

BLUE, GREEN, RED, YELLOW = "34", "32", "31", "33"
OUT_DIR = Path("/home/jovyan/pfs/users/zhanqian.wu/output/muon_adamw_bench")
STEPS = 1800


def cprint(msg, color=BLUE):
    print(f"\033[1;{color}m{msg}\033[0m", flush=True)


# ─────────────────────────── MiniDiT (真 DiT 结构) ───────────────────────────

def timestep_embedding(t, freq_dim=64):
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t[:, None].float() * freqs[None] * 1000.0  # timestep = σ×1000, 与训练一致
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_dim=64):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


class DiTBlock(nn.Module):
    """AdaLN-Zero 调制 + MHSA + MLP (DiT 标准结构)."""

    def __init__(self, dim, heads, mlp_ratio=3):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[1].weight)  # AdaLN-Zero: gate 初始为 0
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        b, n, _ = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, -1)
        x = x + gate1[:, None] * self.proj(attn)
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.fc2(F.silu(self.fc1(h)))


def pos_embed_2d(grid, dim):
    """固定 2D sin-cos 位置编码 [grid*grid, dim]."""
    d4 = dim // 4
    pos = torch.arange(grid).float()
    freqs = torch.exp(-math.log(10000) * torch.arange(d4) / d4)
    ex = torch.cat([torch.cos(pos[:, None] * freqs[None]), torch.sin(pos[:, None] * freqs[None])], -1)
    ey = torch.cat([torch.cos(pos[:, None] * freqs[None]), torch.sin(pos[:, None] * freqs[None])], -1)
    emb = torch.cat([ex.repeat(grid, 1), ey.repeat_interleave(grid, 0)], -1)  # [g*g, 2*d4]
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class MiniDiT(nn.Module):
    def __init__(self, in_ch=16, grid=8, dim=128, depth=6, heads=4, mlp_ratio=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.grid = grid
        self.in_proj = nn.Linear(in_ch, dim)
        self.register_buffer("pos", pos_embed_2d(grid, dim))
        self.t_embed = TimestepEmbedder(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, in_ch)  # "out." 前缀 -> AdamW
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        h = self.in_proj(x) + self.pos[None]
        c = self.t_embed(t)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]
        return self.out(h)


# ─────────────────────────── flow matching 任务 ───────────────────────────

def gaussian_kernel(k=5, sigma=1.2):
    ax = torch.arange(k).float() - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None]
    return kernel / kernel.sum()


def make_data(n, seed, in_ch=16, grid=8):
    """结构化 x0: 平滑低频 latent (多通道高斯平滑场, std≈0.8), 模拟真实图像 latent 的频谱."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, in_ch, grid, grid, generator=g)
    kernel = gaussian_kernel()[None, None].expand(in_ch, 1, -1, -1)
    smooth = F.conv2d(base, kernel, padding=2, groups=in_ch)
    x0 = smooth / smooth.std() * 0.8
    return x0.flatten(2).transpose(1, 2)  # [n, grid*grid, in_ch]


def sample_batch(x0_all, batch, generator):
    n = x0_all.shape[0]
    idx = torch.randint(0, n, (batch,), generator=generator)
    t = torch.rand(batch, generator=generator)
    return x0_all[idx], t


def vpred_loss(model, x0, t, generator):
    noise = torch.randn(x0.shape, generator=generator)
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * noise
    v = noise - x0
    return F.mse_loss(model(x_t, t), v)


# ─────────────────────────── 测试 ───────────────────────────

def group_check():
    from train_giga_exp import build_optimizer
    cprint("▶️ 分组检查: MiniDiT (out.*/1D -> AdamW, 其余 2D -> Muon)")
    model = MiniDiT()
    cfg = {"learning_rate": 1e-3,
           "muon": {"enabled": True, "learning_rate": "3x", "momentum": 0.95, "weight_decay": 0.0},
           "adam": {"betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0}}
    opt = build_optimizer(model, cfg)
    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in opt.adam.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        expect_muon = not name.startswith("out.") and p.ndim == 2
        assert (id(p) in muon_ids) == expect_muon and (id(p) in adam_ids) == (not expect_muon), f"分组错误: {name}"
    n_muon = sum(p.numel() for g in opt.muon.param_groups for p in g["params"])
    n_adam = sum(p.numel() for g in opt.adam.param_groups for p in g["params"])
    total = sum(p.numel() for p in model.parameters())
    cprint(f"  ✅ 总参数 {total/1e6:.2f}M | Muon: {n_muon/1e6:.2f}M (矩阵) | AdamW: {n_adam/1e3:.0f}K (out 头/1D)")
    assert n_muon + n_adam == total
    cprint("  ✅ 分组正确 (out 头与全部 1D 走 AdamW, 其余 2D 矩阵走 Muon, 无遗漏)", GREEN)


def run_group(job):
    """单组训练 (子进程执行); 全部 seed 与串行版一致, 结果确定性复现."""
    name, muon_lr, base_lr, steps, threads = job
    torch.set_num_threads(threads)

    torch.manual_seed(42)
    x0_train = make_data(8192, seed=100)
    x0_eval = make_data(1024, seed=200)
    eval_t = torch.linspace(0, 1, 64).repeat(16)  # 1024 个固定 t 均匀覆盖 [0,1]
    x0_e = x0_eval[:1024]
    eval_gen = torch.Generator().manual_seed(999)

    model = MiniDiT(seed=7)  # 四组同一起点
    if muon_lr is None:
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
    else:
        from train_giga_exp import build_optimizer
        cfg = {"learning_rate": base_lr,
               "muon": {"enabled": True, "learning_rate": muon_lr, "momentum": 0.95, "weight_decay": 0.0},
               "adam": {"betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0}}
        opt = build_optimizer(model, cfg)

    losses = []
    g = torch.Generator().manual_seed(0)  # 各组相同的数据顺序
    for step in range(steps):
        x0, t = sample_batch(x0_train, 256, g)
        loss = vpred_loss(model, x0, t, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == steps - 1:
            with torch.no_grad():
                eval_loss = vpred_loss(model, x0_e, eval_t, eval_gen).item()
            losses.append((step, eval_loss))
        if step % 450 == 0:
            print(f"    [{name}] step={step}/{steps} eval={eval_loss:.4f}", flush=True)

    with torch.no_grad():
        noise_e = torch.randn(x0_e.shape, generator=eval_gen)  # x_t 与 v-target 用同一份噪声
        errs = (model((1 - eval_t[:, None, None]) * x0_e + eval_t[:, None, None] * noise_e,
                      eval_t) - (noise_e - x0_e)).pow(2).mean((1, 2))
    t_bucket = [errs[(eval_t >= lo) & (eval_t < hi)].mean().item()
                for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.0)]]
    return name, losses, t_bucket


def convergence_test(steps=STEPS):
    import multiprocessing
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from concurrent.futures import ProcessPoolExecutor, as_completed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = [
        ("AdamW lr=1e-3", None, 1e-3),
        ("AdamW lr=3e-3", None, 3e-3),
        ("Muon+AdamW (muon 3x)", "3x", 1e-3),
        ("Muon+AdamW (muon 1x)", "1x", 1e-3),
    ]
    threads = max(4, (os.cpu_count() or 8) // len(configs))
    cprint(f"▶️ MiniDiT flow-matching 收敛对比 ({steps} 步, batch=256, eval 1024 样本, "
           f"4 组并行 × {threads} 线程)")

    results, t_bucket_final = {}, {}
    with ProcessPoolExecutor(max_workers=len(configs),
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        jobs = [(n, mlr, blr, steps, threads) for n, mlr, blr in configs]
        futures = [ex.submit(run_group, job) for job in jobs]
        for fut in as_completed(futures):
            name, losses, t_bucket = fut.result()
            results[name] = losses
            t_bucket_final[name] = t_bucket
            cprint(f"  ✅ {name}: eval {losses[0][1]:.4f} -> {losses[-1][1]:.5f}", GREEN)

    # 先落盘再画图: 训练数据不因画图异常丢失
    finals = {n: ls[-1][1] for n, ls in results.items()}
    (OUT_DIR / "curves_dit.json").write_text(json.dumps({"curves": results, "t_bucket": t_bucket_final}))
    best = min(finals.values())
    adam_base = finals["AdamW lr=1e-3"]

    # 图 1: 收敛曲线
    plt.figure(figsize=(9, 5.5))
    for name, ls in results.items():
        xs, ys = zip(*ls)
        plt.plot(xs, ys, label=f"{name} (final={ys[-1]:.4f})", linewidth=1.8)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("eval v-pred MSE (log)")
    plt.title("Muon+AdamW vs AdamW — MiniDiT flow matching (AdaLN-Zero, 1.65M, CPU)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "convergence_dit.png", dpi=130)

    # 图 2: t 分桶去噪误差 (最终模型)
    plt.figure(figsize=(9, 5))
    buckets = ["t∈[0,.25)\n低噪", "t∈[.25,.5)", "t∈[.5,.75)", "t∈[.75,1]\n高噪"]
    width = 0.2
    xs = range(4)
    for i, (name, _, _) in enumerate(configs):
        plt.bar([x + (i - 1.5) * width for x in xs], t_bucket_final[name], width, label=name)
    plt.yscale("log")
    plt.xticks(xs, buckets)
    plt.ylabel("v-pred MSE (log)")
    plt.title("去噪误差按噪声水平 t 分桶 (步数结束时)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "t_bucket_dit.png", dpi=130)

    cprint("\n📊 汇总 (eval v-pred MSE)", BLUE)
    cprint(f"{'优化器':<30}{'final':>10}{'vs AdamW-1e-3':>16}")
    for name, _, _ in configs:
        mark = " 🏆" if finals[name] == best else ""
        cprint(f"{name:<30}{finals[name]:>10.5f}{(finals[name]/adam_base-1)*100:>+15.1f}%{mark}",
               GREEN if finals[name] == best else "0")
    cprint(f"\n🖼️  图: {OUT_DIR/'convergence_dit.png'} | {OUT_DIR/'t_bucket_dit.png'}", YELLOW)


def main():
    cprint("=" * 72)
    cprint("Muon+AdamW 验证 — MiniDiT (真 DiT: AdaLN-Zero + MHSA + v-pred flow matching)")
    cprint("=" * 72)
    group_check()
    convergence_test()
    cprint("\n🎉 MiniDiT 验证完成", GREEN)


if __name__ == "__main__":
    main()
