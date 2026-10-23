"""Muon+AdamW (MuonAdamW) 优化器验证: 小模型 CPU 测试.

Part 1 数值准确性:
  A. ZeRO 分片路径 (_step_partitioned_muon, torchrun 2 rank) vs 原生 torch.optim.Muon
  B. AdamW 分支 vs torch.optim.AdamW 对照
  C. build_optimizer 参数分组 (out.*/1D -> AdamW, 其余 2D -> Muon)
  D. set_lr warmup/cosine 按 _lr_scale 缩放
Part 2 收敛速度:
  TinyDiT 合成回归, Muon+AdamW (3x / 1x) vs AdamW (1e-3 / 3e-3), loss 曲线图.

用法:
  python scripts/test/test_muon_adamw.py            # 全部
  python scripts/test/test_muon_adamw.py --shard-test   # 仅分片数值测试 (torchrun 调用)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "train"))

import torch
import torch.nn.functional as F
from torch import nn

BLUE, GREEN, RED, YELLOW = "34", "32", "31", "33"


def cprint(msg, color=BLUE):
    print(f"\033[1;{color}m{msg}\033[0m", flush=True)


# ─────────────────────────── 小模型 (模拟 Giga 命名) ───────────────────────────

class TinyBlock(nn.Module):
    def __init__(self, dim, ffn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class TinyDiT(nn.Module):
    def __init__(self, dim=256, ffn=1024, depth=6, in_dim=64, out_dim=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.in_proj = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([TinyBlock(dim, ffn) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)  # "out." 前缀 -> AdamW

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.layers:
            h = blk(h)
        return self.out(self.norm_out(h))


OPT_CFG = {
    "learning_rate": 1e-3,
    "muon": {"enabled": True, "learning_rate": "3x", "momentum": 0.95, "weight_decay": 0.0},
    "adam": {"betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0},
}


def build_tiny_optimizer(model, muon_lr="3x", base_lr=1e-3):
    from train_giga_exp import build_optimizer
    cfg = dict(OPT_CFG)
    cfg["learning_rate"] = base_lr
    cfg["muon"] = {**OPT_CFG["muon"], "learning_rate": muon_lr}
    return build_optimizer(model, cfg)


# ─────────────────────────── Part 1: 数值准确性 ───────────────────────────

def shard_numeric_test():
    """torchrun 2 rank: _step_partitioned_muon (ZeRO 模拟) vs 原生 torch.optim.Muon."""
    import torch.distributed as dist
    from train_giga_exp import MuonAdamW

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    shapes = [(64, 32), (32, 64), (128, 64), (64, 128)]
    total = sum(a * b for a, b in shapes)
    shard_numel = total // world
    flat_grad = torch.randn(total)

    # ── 路径 1: 分片 (模拟 DeepSpeed ZeRO 把 2D 参数替换为 1D flat shard) ──
    torch.manual_seed(123)
    flat_param = torch.randn(total)
    shard = flat_param[rank * shard_numel:(rank + 1) * shard_numel].clone().requires_grad_(True)
    placeholders = [torch.zeros(s) for s in shapes]
    wrapper = MuonAdamW(placeholders, [], OPT_CFG["muon"], OPT_CFG["adam"], 1e-3)
    wrapper.muon.param_groups[0]["params"] = [shard]
    wrapper.muon.state.clear()
    for _ in range(5):
        shard.grad = flat_grad[rank * shard_numel:(rank + 1) * shard_numel].clone()
        wrapper.step()
    parts = [torch.empty(shard_numel) for _ in range(world)]
    dist.all_gather(parts, shard.detach())
    flat_dist = torch.cat(parts)

    # ── 路径 2: 原生 torch.optim.Muon (ground truth) ──
    mats, off = [], 0
    for s in shapes:
        mats.append(flat_param[off:off + s[0] * s[1]].view(s).clone().requires_grad_(True))
        off += s[0] * s[1]
    ref = torch.optim.Muon(mats, lr=3e-3, momentum=0.95, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
    for _ in range(5):
        ref.zero_grad()
        off = 0
        for m in mats:
            m.grad = flat_grad[off:off + m.numel()].view(m.shape).clone()
            off += m.numel()
        ref.step()
    flat_ref = torch.cat([m.flatten() for m in mats])

    if rank == 0:
        diff = (flat_dist - flat_ref).abs().max().item()
        rel = diff / flat_ref.abs().max().item()
        cprint(f"  分片 vs 原生 Muon: max_abs_diff={diff:.3e} (rel={rel:.3e})", GREEN if diff < 1e-5 else RED)
        assert diff < 1e-5, f"分片路径数值不一致: {diff}"
        cprint("  ✅ Part 1-A: ZeRO 分片路径 == 原生 torch.optim.Muon (5 步)", GREEN)
    dist.barrier()
    dist.destroy_process_group()


def single_numeric_tests():
    from train_giga_exp import MuonAdamW, set_lr

    # ── B: AdamW 分支 vs torch.optim.AdamW ──
    cprint("▶️ Part 1-B: AdamW 分支 vs torch.optim.AdamW")
    torch.manual_seed(7)
    ref_params = [torch.randn(48, requires_grad=True), torch.randn(96, 32, requires_grad=True)]
    our_params = [p.detach().clone().requires_grad_(True) for p in ref_params]
    ref = torch.optim.AdamW(ref_params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    wrapper = MuonAdamW([], our_params, OPT_CFG["muon"], OPT_CFG["adam"], 1e-3)
    torch.manual_seed(8)
    grads = [torch.randn_like(p) for p in ref_params]
    for _ in range(5):
        for i, p in enumerate(ref_params):
            p.grad = grads[i].clone()
        for i, p in enumerate(our_params):
            p.grad = grads[i].clone()
        ref.step()
        wrapper.step()
    diff = max((a - b).abs().max().item() for a, b in zip(ref_params, our_params))
    cprint(f"  max_abs_diff={diff:.3e}", GREEN if diff == 0.0 or diff < 1e-7 else RED)
    assert diff < 1e-7, "AdamW 分支与 torch.optim.AdamW 不一致"
    cprint("  ✅ AdamW 分支逐元素一致", GREEN)

    # ── C: build_optimizer 分组 ──
    cprint("▶️ Part 1-C: build_optimizer 参数分组 (out.*/1D -> AdamW, 其余 2D -> Muon)")
    model = TinyDiT()
    opt = build_tiny_optimizer(model)
    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in opt.adam.param_groups for p in g["params"]}
    muon_count = adam_count = 0
    for name, p in model.named_parameters():
        expect_muon = not name.startswith("out.") and p.ndim == 2
        if expect_muon:
            assert id(p) in muon_ids and id(p) not in adam_ids, f"{name} 应在 Muon 组"
            muon_count += 1
        else:
            assert id(p) in adam_ids and id(p) not in muon_ids, f"{name} 应在 AdamW 组"
            adam_count += 1
    n_muon_el = sum(p.numel() for p in (p for g in opt.muon.param_groups for p in g["params"]))
    n_adam_el = sum(p.numel() for p in (p for g in opt.adam.param_groups for p in g["params"]))
    cprint(f"  Muon 组: {muon_count} 参数 ({n_muon_el:,} 元素) | AdamW 组: {adam_count} 参数 ({n_adam_el:,} 元素)")
    cprint(f"  muon lr={opt.muon.param_groups[0]['lr']:.1e} (3x), adam lr={opt.adam.param_groups[0]['lr']:.1e}", GREEN)
    cprint("  ✅ 分组正确", GREEN)

    # ── D: set_lr 缩放 ──
    cprint("▶️ Part 1-D: set_lr warmup/cosine 按 _lr_scale 缩放")
    train_cfg = {"learning_rate": 8e-5, "lr_warmup_steps": 10, "total_steps": 100}
    fake_ds = type("FakeDS", (), {"param_groups": [dict(g) for g in opt.param_groups]})()
    lr_now = set_lr(opt, fake_ds, 5, train_cfg)  # warmup 中点: 8e-5 * 0.5 = 4e-5
    assert abs(lr_now - 4e-5) < 1e-12
    muon_lr = opt.muon.param_groups[0]["lr"]
    adam_lr = opt.adam.param_groups[0]["lr"]
    assert abs(muon_lr - 4e-5 * 3) < 1e-12, f"muon lr={muon_lr}"
    assert abs(adam_lr - 4e-5) < 1e-12, f"adam lr={adam_lr}"
    lr_end = set_lr(opt, fake_ds, 200, train_cfg)  # cosine 末端 -> 0
    assert lr_end < 1e-9
    cprint(f"  step=5: base=4e-5, muon={muon_lr:.2e} (3x), adam={adam_lr:.2e}; 末端 lr={lr_end:.1e} -> 0", GREEN)
    cprint("  ✅ set_lr 双写 (wrapper + DeepSpeed groups) 且按 _lr_scale 缩放", GREEN)


# ─────────────────────────── Part 2: 收敛速度 ───────────────────────────

def convergence_test(steps=1200, out_dir=REPO.parent.parent / "output" / "muon_adamw_bench"):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    torch.manual_seed(42)
    in_dim = out_dim = 64
    n_train, n_eval = 8192, 1024
    teacher = TinyDiT(seed=999)
    with torch.no_grad():
        x_all = torch.randn(n_train + n_eval, in_dim)
        y_all = teacher(x_all) + 0.01 * torch.randn(n_train + n_eval, out_dim)
    x_eval, y_eval = x_all[n_train:], y_all[n_train:]

    configs = [
        ("AdamW lr=1e-3", dict(muon_lr=None, base_lr=1e-3)),
        ("AdamW lr=3e-3", dict(muon_lr=None, base_lr=3e-3)),
        ("Muon+AdamW (adam 1e-3, muon 3x)", dict(muon_lr="3x", base_lr=1e-3)),
        ("Muon+AdamW (adam 1e-3, muon 1x)", dict(muon_lr="1x", base_lr=1e-3)),
    ]
    batch = 256
    results = {}
    cprint(f"▶️ Part 2: 收敛对比 TinyDiT(6 层, dim=256, ffn=1024) 合成回归, {steps} 步, batch={batch}")
    for name, kw in configs:
        model = TinyDiT(seed=7).to(device)  # 四组同一起点
        opt = (torch.optim.AdamW(model.parameters(), lr=kw["base_lr"], betas=(0.9, 0.95), eps=1e-8)
               if kw["muon_lr"] is None else build_tiny_optimizer(model, kw["muon_lr"], kw["base_lr"]))
        losses = []
        g = torch.Generator().manual_seed(0)  # 各组相同的数据顺序
        pbar = tqdm(range(steps), desc=f"  {name}", ncols=100, leave=True)
        for step in pbar:
            idx = torch.randint(0, n_train, (batch,), generator=g)
            loss = F.mse_loss(model(x_all[idx]), y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == steps - 1:
                with torch.no_grad():
                    eval_loss = F.mse_loss(model(x_eval), y_eval).item()
                losses.append((step, eval_loss))
                pbar.set_postfix(eval_loss=f"{eval_loss:.4f}")
        results[name] = losses
        cprint(f"  ✅ {name}: 初始 eval={losses[0][1]:.4f} -> 最终 eval={losses[-1][1]:.4f}", GREEN)

    # 曲线图
    plt.figure(figsize=(9, 5.5))
    for name, losses in results.items():
        xs, ys = zip(*losses)
        plt.plot(xs, ys, label=f"{name} (final={ys[-1]:.4f})", linewidth=1.8)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("eval MSE (log)")
    plt.title("Muon+AdamW vs AdamW convergence (TinyDiT synthetic regression, CPU)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    png = out_dir / "convergence.png"
    plt.savefig(png, dpi=130)

    # 达到阈值步数对比表
    finals = {n: ls[-1][1] for n, ls in results.items()}
    best = min(finals.values())
    target = finals["AdamW lr=1e-3"] * 0.5  # 相对 AdamW 基准减半
    cprint("\n📊 汇总 (eval MSE)", BLUE)
    cprint(f"{'优化器':<38}{'final':>10}{'步数(达到 0.5×AdamW-1e-3 终值)':>32}")
    cprint(f"  目标 loss = {target:.4f}")
    for name, ls in results.items():
        reach = next((s for s, v in ls if v <= target), None)
        mark = "🏆" if finals[name] == best else "  "
        cprint(f"{mark}{name:<38}{finals[name]:>10.4f}{str(reach):>32}", GREEN if finals[name] == best else "0")
    (out_dir / "curves.json").write_text(json.dumps(results))
    cprint(f"\n🖼️  曲线图: {png}", YELLOW)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-test", action="store_true")
    parser.add_argument("--skip-convergence", action="store_true")
    args = parser.parse_args()

    if args.shard_test:
        shard_numeric_test()
        return

    cprint("=" * 72)
    cprint("Muon+AdamW 优化器验证 (小模型, CPU)")
    cprint("=" * 72)

    cprint("\n▶️ Part 1-A: ZeRO 分片路径 (_step_partitioned_muon) torchrun 2 rank")
    cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node=2", str(Path(__file__).resolve()), "--shard-test"]
    proc = subprocess.run(cmd, cwd=str(REPO))
    assert proc.returncode == 0, "Part 1-A 分片数值测试失败"
    cprint("✅ Part 1-A 完成", GREEN)

    single_numeric_tests()

    if not args.skip_convergence:
        convergence_test()

    cprint("\n" + "=" * 72)
    cprint("🎉 全部验证通过: MuonAdamW 数值正确, 见上方收敛对比", GREEN)
    cprint("=" * 72)


if __name__ == "__main__":
    main()
