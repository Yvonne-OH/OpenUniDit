"""Teacher Trajectory Replay 缓存生成 (离线, 一次性)。

对每条训练 caption: 随机噪声 x1 → teacher (GWP 6B, 冻结) 以正式推理口径做 50-step
CFG Euler rollout (shift=6.0, guidance=3.5 ≡ 标准 CFG s=2.5) → 得到真实 ODE 轨迹状态
{x_0^T ... x_50^T} → 每条轨迹随机抽 4 个 2-step 区间 (覆盖高/中高/中低/低噪声段),
保存 (x_k^T, x_{k+2}^T, σ_k, σ_{k+1}, σ_{k+2}, caption)。

关键点:
  - 状态来自 teacher 速度场的真实数值积分, 不是 FM 线性插值 —— 训练分布 = 推理轨迹分布
  - CFG 组合与 kd_trajectory_diag.py 完全一致: v = v_uncond + g·(v_cond - v_uncond)
    (teacher 原始输出 = z0 - noise, 积分取负对齐 student 约定, 与 diag 同)
  - caption 用 teacher chat-template 约定编码; 缓存只存 caption 字符串, 训练时用共享
    Qwen3-VL 编码器现算 hidden (teacher/student 条件逐位一致)
  - 多卡: --rank/--world 分片, 各写各的 index_rank{r}.json; 断点重跑自动续 (按已有 shard 行数)
  - 噪声种子 = base_seed + 全局轨迹序号 → 重跑/多卡结果可复现

用法 (8 卡并行, 共 10000 条):
  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python scripts/cache/gen_teacher_traj_cache.py \
        --config configs/stage_1_kd_v3_traj.yaml --rank $i --world 8 &
  done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.lmdb_dataset import GigaLMDBDataset
from pipeline.giga_pipeline_6b_teacher import build_gwp6b_teacher, encode_prompts_6b_teacher

# 4 个区间 (50-step schedule 的 step 索引, 由 30-step 方案等比放大 5/3): 高/中高/中低/低噪声
SEGMENT_REGIONS = [(3, 10), (13, 20), (27, 33), (40, 45)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage_1_kd_v3_traj.yaml")
    p.add_argument("--out-dir", default=None, help="缺省取 config kd.traj.cache_dir")
    p.add_argument("--num-trajectories", type=int, default=10000, help="全局总轨迹数")
    p.add_argument("--batch-size", type=int, default=4, help="每次同时 rollout 的轨迹数 (×2 CFG 分支)")
    p.add_argument("--shard-size", type=int, default=250, help="每个 shard 收纳的轨迹数")
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--world", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--seed", type=int, default=20260101, help="全局噪声种子基 (每轨迹 + 全局序号)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text() if cfg_path.is_absolute() else (ROOT / cfg_path).read_text())
    traj_cfg = (cfg.get("kd") or {}).get("traj") or {}
    teacher_cfg = cfg["teacher"]
    tc = cfg["components"]["text_encoder"]
    out_dir = Path(args.out_dir or traj_cfg.get("cache_dir") or (ROOT / "outputs" / "teacher_traj_cache"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    steps = int(traj_cfg.get("cache_steps", 50))
    shift = float(traj_cfg.get("cache_shift", 6.0))
    guidance = float(traj_cfg.get("guidance", 3.5))
    layer = int(teacher_cfg.get("layer", -2))
    max_cond = int(teacher_cfg.get("max_condition_tokens", 4064))

    def shift_sigma(u):
        return shift * u / (1.0 + (shift - 1.0) * u)

    sigmas = [shift_sigma(1.0 - i / steps) for i in range(steps + 1)]

    # ---- 模型: teacher + 共享 Qwen3-VL 编码器 (不载 VAE —— 缓存只有 latent) ----
    from transformers import AutoModel, AutoProcessor
    vl = AutoModel.from_pretrained(tc["pretrained_path"], torch_dtype=dtype,
                                   local_files_only=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(tc["tokenizer_path"], local_files_only=True)
    teacher, teacher_ckpt = build_gwp6b_teacher(teacher_cfg, device, dtype)
    print(f"🧑‍🏫 teacher: {teacher_ckpt} | {steps} steps, shift={shift}, cfg={guidance}", flush=True)

    # ---- caption 源: 训练数据集 records (与 FM 分支同分布) ----
    data = GigaLMDBDataset(cfg["dataset"], seed=int(cfg.get("seed", 42)))
    captions = [str(r.get("caption", "")).strip() for r in data.records]
    captions = [c for c in captions if c]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(captions), generator=g).tolist()
    captions = [captions[i] for i in perm]
    per_rank = (args.num_trajectories + args.world - 1) // args.world
    lo = args.rank * per_rank
    hi = min(lo + per_rank, args.num_trajectories)
    my_caps = [captions[i % len(captions)] for i in range(lo, hi)] if lo < args.num_trajectories else []
    print(f"📝 rank {args.rank}/{args.world}: {len(my_caps)} 条轨迹 "
          f"(全局 {args.num_trajectories}, caption 池 {len(captions)})", flush=True)

    # ---- 续跑: 已有 shard 行数 / 每轨迹区间数 = 已完成轨迹数 ----
    seg_per_traj = len(SEGMENT_REGIONS)
    done_trajs = 0
    shard_seq = 0
    for d in sorted(out_dir.glob(f"shard_r{args.rank}_*")):
        sig_fp = d / "sigmas.npy"
        if not sig_fp.exists():
            continue
        rows = int(np.load(sig_fp, mmap_mode="r").shape[0])
        if rows > 0:
            done_trajs += rows // seg_per_traj
            shard_seq += 1
    if done_trajs:
        print(f"⏩ 续跑: 已有 {done_trajs} 条轨迹, 从第 {done_trajs + 1} 条继续", flush=True)
    todo = my_caps[done_trajs:]

    t_patch = int(teacher_cfg.get("patch_size", 2))
    t_fpatch = int(teacher_cfg.get("f_patch_size", 1))
    res = int(cfg["dataset"]["resolution"])
    z_shape = (int(teacher_cfg["in_channels"]), 1, res // 8, res // 8)

    rows_xs, rows_xe, rows_sg, rows_cap = [], [], [], []
    shard_rows_total = 0

    def flush_shard():
        nonlocal shard_seq, rows_xs, rows_xe, rows_sg, rows_cap, shard_rows_total
        if not rows_xs:
            return
        shard_dir = out_dir / f"shard_r{args.rank}_{shard_seq:05d}"
        shard_dir.mkdir(exist_ok=True)
        np.save(shard_dir / "x_start.npy", np.stack(rows_xs).astype(np.float16))
        np.save(shard_dir / "x_end.npy", np.stack(rows_xe).astype(np.float16))
        np.save(shard_dir / "sigmas.npy", np.stack(rows_sg).astype(np.float32))
        (shard_dir / "captions.json").write_text(json.dumps(rows_cap, ensure_ascii=False))
        index_fp = out_dir / f"index_rank{args.rank}.json"
        index = json.loads(index_fp.read_text()) if index_fp.exists() else {"meta": {}, "shards": []}
        index["meta"] = {
            "teacher_checkpoint": str(teacher_ckpt), "steps": steps, "shift": shift,
            "guidance": guidance, "sigma_schedule": [round(s, 6) for s in sigmas],
            "segment_regions": SEGMENT_REGIONS, "seed_base": args.seed,
            "latent_shape": list(rows_xs[0].shape),
        }
        index["shards"] = [s for s in index["shards"] if s["dir"] != str(shard_dir)]
        index["shards"].append({"dir": str(shard_dir), "rows": len(rows_xs)})
        tmp = index_fp.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1))
        tmp.replace(index_fp)
        print(f"💾 shard {shard_dir.name}: {len(rows_xs)} rows", flush=True)
        shard_seq += 1
        rows_xs, rows_xe, rows_sg, rows_cap = [], [], [], []
        shard_rows_total = 0

    t0 = time.time()
    for start in range(0, len(todo), args.batch_size):
        batch_caps = todo[start:start + args.batch_size]
        n = len(batch_caps)
        caps = encode_prompts_6b_teacher(processor, vl, batch_caps + [""] * n, device,
                                         layer=layer, max_condition_tokens=max_cond)
        caps = [c.to(dtype) for c in caps]
        cond_caps, uncond_cap = caps[:n], caps[n]
        # 噪声种子 = base_seed + 全局轨迹序号 (续跑/多卡可复现)
        seeds = [args.seed + done_trajs + start + j for j in range(n)]
        x = torch.stack([
            torch.randn(*z_shape, generator=torch.Generator("cpu").manual_seed(s)).to(device)
            for s in seeds])
        states = [list(x[j:j + 1]) for j in range(n)]  # states[j][i] = σ_i 时刻状态
        for i in range(steps):
            sig, sig_next = sigmas[i], sigmas[i + 1]
            t = torch.full((n,), 1.0 - sig, device=device, dtype=dtype)
            xx = list(torch.stack([s[-1] for s in states]).to(dtype).unbind(0))
            # cond 与 uncond 分支拼成一个 batch 前向 (共享同一状态 x)
            out = teacher(x=xx + xx, t=torch.cat([t, t]),
                          cap_feats=cond_caps + [uncond_cap] * n,
                          patch_size=t_patch, f_patch_size=t_fpatch)
            mo = out.sample if hasattr(out, "sample") else out[0]
            if isinstance(mo, list):
                mo = torch.stack(mo)
            f_cond, f_un = mo[:n].float(), mo[n:].float()
            vel = -(f_un + guidance * (f_cond - f_un))  # student 约定: noise - z0
            x_next = torch.stack([s[-1] for s in states]) + (sig_next - sig) * vel
            for j in range(n):
                states[j].append(x_next[j])
        # 每条轨迹抽 4 个 2-step 区间
        rng = torch.Generator().manual_seed(seeds[0] + 777)
        sts = [s[0] for s in states]
        for j in range(n):
            for region_lo, region_hi in SEGMENT_REGIONS:
                k = int(torch.randint(region_lo, region_hi - 1, (1,), generator=rng).item())
                rows_xs.append(sts[k].cpu().numpy())
                rows_xe.append(sts[k + 2].cpu().numpy())
                rows_sg.append(np.array([sigmas[k], sigmas[k + 1], sigmas[k + 2]], dtype=np.float32))
                rows_cap.append(batch_caps[j])
            shard_rows_total += seg_per_traj
        done_trajs += n
        if (start // args.batch_size) % 5 == 0:
            rate = (start + n) / max(time.time() - t0, 1e-6)
            eta_h = (len(todo) - start - n) / max(rate, 1e-6) / 3600
            print(f"  rank{args.rank} traj {done_trajs} | {rate:.2f} traj/s | ETA {eta_h:.1f} h", flush=True)
        if shard_rows_total >= args.shard_size * seg_per_traj:
            flush_shard()
    flush_shard()
    print(f"✅ rank{args.rank} 完成: 共 {done_trajs} 条轨迹 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
