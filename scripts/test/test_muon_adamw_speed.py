"""Muon+AdamW 训练速度基准 — MiniDiT wall-clock (CPU, 2 rank 模拟 ZeRO-2 分片).

对比每步耗时分解 (fwd+bwd / optimizer.step):
  1. AdamW 全参数
  2. Muon+AdamW 直接路径 (2D 参数, 无 ZeRO)
  3. Muon+AdamW ZeRO 分片路径 (1D flat shard + all_gather, 模拟正式训练 DeepSpeed 路径)

用法:
  torchrun --nproc_per_node=2 scripts/test/test_muon_adamw_speed.py
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "test"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

import torch
import torch.distributed as dist

from test_muon_adamw_dit import MiniDiT, make_data, vpred_loss, sample_batch

BLUE, GREEN, YELLOW = "34", "32", "33"
OUT_DIR = Path("/home/jovyan/pfs/users/zhanqian.wu/output/muon_adamw_bench")
WARMUP, TIMED, BATCH = 12, 40, 256


def cprint(msg, color=BLUE):
    print(f"\033[1;{color}m{msg}\033[0m", flush=True)


def build_cfg():
    return {"learning_rate": 1e-3,
            "muon": {"enabled": True, "learning_rate": "3x", "momentum": 0.95, "weight_decay": 0.0},
            "adam": {"betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0}}


def shard_optimizer(model, opt):
    """模拟 DeepSpeed ZeRO-2: 2D 参数 → 1D flat shard (按 world 均分), 重建 muon group."""
    matrix = [p for n, p in model.named_parameters() if not n.startswith("out.") and p.ndim == 2]
    world = dist.get_world_size()
    rank = dist.get_rank()
    total = sum(p.numel() for p in matrix)
    assert total % world == 0, f"2D 参数量 {total} 无法被 world={world} 整除"
    n = total // world
    shard = torch.nn.Parameter(torch.zeros(n))
    opt.muon.param_groups[0]["params"] = [shard]
    opt.muon.state.clear()
    return matrix, shard, n


def main():
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "48")))
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()

    torch.manual_seed(42)
    x0_train = make_data(4096, seed=100)
    g = torch.Generator().manual_seed(0)
    model = MiniDiT(seed=7)

    configs = [("AdamW", "adam"), ("Muon+AdamW (直接 2D)", "muon"), ("Muon+AdamW (ZeRO 分片)", "shard")]
    results = {}
    for name, kind in configs:
        dist.barrier()
        model.zero_grad(set_to_none=True)
        if kind == "adam":
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
            matrix = shard = None
        else:
            from train_giga_exp import build_optimizer
            opt = build_optimizer(model, build_cfg())
            if kind == "shard":
                matrix, shard, shard_n = shard_optimizer(model, opt)
            else:
                matrix = shard = None

        t_fb = t_opt = t_flat = 0.0
        count = 0
        for step in range(WARMUP + TIMED):
            x0, t = sample_batch(x0_train, BATCH, g)
            t0 = time.perf_counter()
            loss = vpred_loss(model, x0, t, g)
            loss.backward()
            t1 = time.perf_counter()
            if kind == "shard":
                # 模拟 ZeRO contiguous flat grad 组装 (直写场景无 cat 开销, 单列不计入 step)
                flat = torch.cat([p.grad.reshape(-1) for p in matrix])
                shard.grad = flat.narrow(0, rank * shard_n, shard_n).clone()
                t2 = time.perf_counter()
                opt.step()  # 检测 1D shard → _step_partitioned_muon (all_gather × 2 + NS)
                t3 = time.perf_counter()
                t_flat += t2 - t1
                t_opt += t3 - t2
            else:
                t2 = time.perf_counter()
                opt.step()
                t3 = time.perf_counter()
                t_opt += t3 - t2
            t_fb += t1 - t0
            if step >= WARMUP:
                count += 1
            opt.zero_grad(set_to_none=True)

        total = (t_fb + t_opt) / count * 1000
        fb = t_fb / count * 1000
        opt_ms = t_opt / count * 1000
        flat_ms = t_flat / count * 1000
        results[name] = {"step_total_ms": round(total, 1), "fwd_bwd_ms": round(fb, 1),
                         "opt_step_ms": round(opt_ms, 1), "flat_prep_ms": round(flat_ms, 1),
                         "opt_ratio": round(opt_ms / total * 100, 1)}
        if rank == 0:
            cprint(f"  ✅ {name}: 每步 {total:.1f} ms (fwd+bwd {fb:.1f} | step {opt_ms:.1f} "
                   f"{'| flat组装 %.1f' % flat_ms if kind == 'shard' else ''} | opt 占比 {opt_ms / total * 100:.0f}%)", GREEN)

    dist.barrier()
    if rank == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "speed_dit.json").write_text(json.dumps(results, indent=2))
        cprint("\n📊 汇总 (MiniDiT 1.65M, batch=256, CPU 48 线程 × 2 rank)", BLUE)
        cprint(f"{'配置':<28}{'每步总耗时':>12}{'fwd+bwd':>10}{'opt.step':>10}{'opt占比':>9}")
        base = results["AdamW"]["step_total_ms"]
        for name, _ in configs:
            r = results[name]
            cprint(f"{name:<28}{r['step_total_ms']:>9.1f}ms{r['fwd_bwd_ms']:>8.1f}ms"
                   f"{r['opt_step_ms']:>8.1f}ms{r['opt_ratio']:>8.1f}%", GREEN)
        cprint(f"\n💾 {OUT_DIR / 'speed_dit.json'}", YELLOW)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
