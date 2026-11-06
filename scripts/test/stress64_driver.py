#!/usr/bin/env python3
"""64 卡压测多轮驱动 + 新旧对比聚合报告。

用法:
  python scripts/test/stress64_driver.py --runs 3          # 每后端 3 轮, 交替顺序执行
  python scripts/test/stress64_driver.py --aggregate-only  # 只聚合已有结果
结果: /tmp/stress64_<backend>_run<i>/  与 /tmp/stress64_report.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path("/tmp")


def run_one(backend: str, idx: int, ranks: int, load_ranks: int, batches: int) -> Path:
    outdir = RUNS / f"stress64_{backend}_run{idx}"
    cmd = [sys.executable, "scripts/test/stress_dataloader_64ranks.py",
           "--backend", backend, "--ranks", str(ranks), "--batches", str(batches),
           "--load-ranks", str(load_ranks), "--out", str(outdir)]
    print(f"\n### {'=' * 70}\n### run{idx}: {backend} (load_ranks={load_ranks})\n### {'=' * 70}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return outdir


def pstat(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"


def aggregate(outs: dict[str, list[Path]], report_path: Path) -> None:
    lines = ["# 64 卡数据链路压测: mmap 列式索引 vs legacy pickle 索引\n",
             "> 单节点 2TB RAM / 192 核; Phase B = 64 rank × 4 worker 全速取数, 每 rank 15 batch (bs=8)。\n"]
    for backend, dirs in outs.items():
        lines.append(f"\n## backend = {backend} ({len(dirs)} 轮)\n")
        for d in dirs:
            try:
                s = json.loads((d / "summary.json").read_text())
            except Exception as exc:
                lines.append(f"- `{d.name}`: 无 summary ({exc})")
                continue
            a, b = s["phase_a"], s["phase_b"]
            lines.append(
                f"- `{d.name}`\n"
                f"  - 冷启动 (Phase A): ok {a['ok']}/{a['n_procs']}, load p50 **{pstat(a['load_s_p50'])}s**, "
                f"max {pstat(a['load_s_max'])}s, 峰值 RSS {pstat(a['peak_rss_gib_max'])}GiB/rank\n"
                f"  - 稳态 (Phase B): batch {b['total_ok_batches']}/{b['total_ok_batches'] + b['total_skipped_batches']}, "
                f"样本 {b['total_samples']:,}, 吞吐 **{b['throughput_samples_per_s']:.1f} samples/s**, "
                f"batch p50/p90/p99 = {pstat(b['batch_time_p50'])}/{pstat(b['batch_time_p90'])}/{pstat(b['batch_time_p99'])}s\n"
                f"  - 内存: 峰值 ΣRSS **{pstat(b['peak_rss_gib'])}GiB** / ΣPss {pstat(b['peak_pss_gib'])}GiB, "
                f"父进程索引加载 {pstat(b['parent_load_s'])}s\n")
    # 汇总均值
    lines.append("\n## 多轮均值对比\n")
    lines.append("| 指标 | " + " | ".join(f"{k} (n={len(v)})" for k, v in outs.items()) + " |")
    lines.append("|---|" + "---|" * len(outs))

    def avg(vals):
        xs = [x for x in vals if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    def collect(backend, key, phase="phase_b"):
        rs = []
        for d in outs[backend]:
            try:
                rs.append(json.loads((d / "summary.json").read_text())[phase][key])
            except Exception:
                rs.append(None)
        return rs

    for label, key, phase, fmt in [
            ("冷启动 load p50 (s)", "load_s_p50", "phase_a", "{:.1f}"),
            ("冷启动 峰值 RSS/rank (GiB)", "peak_rss_gib_max", "phase_a", "{:.2f}"),
            ("稳态吞吐 (samples/s)", "throughput_samples_per_s", "phase_b", "{:.1f}"),
            ("batch p90 (s)", "batch_time_p90", "phase_b", "{:.1f}"),
            ("batch p99 (s)", "batch_time_p99", "phase_b", "{:.1f}"),
            ("峰值 ΣRSS (GiB)", "peak_rss_gib", "phase_b", "{:.0f}"),
            ("峰值 ΣPss (GiB)", "peak_pss_gib", "phase_b", "{:.0f}")]:
        row = f"| {label} |"
        for backend in outs:
            v = avg(collect(backend, key, phase))
            row += f" {fmt.format(v)} |"
        lines.append(row)
    lines.append("\n> 说明: 各轮按 mmap→legacy 交替执行以中和 LMDB/NFS 页缓存预热偏差; "
                 "legacy 冷启动只能并发 8 进程 (64 进程全开约 2.5TB 内存, 单节点必然 OOM), "
                 "64 rank 生产含义为每 rank 独立加载。\n")
    report_path.write_text("\n".join(lines))
    print(f"报告: {report_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--batches", type=int, default=15)
    p.add_argument("--ranks", type=int, default=64)
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args()

    # mmap 已有 run1 (/tmp/stress64_mmap), legacy 已有 run1 (/tmp/stress64_legacy)
    existing = {"mmap": [RUNS / "stress64_mmap"], "legacy": [RUNS / "stress64_legacy"]}
    outs = {"mmap": [], "legacy": []}
    if not args.aggregate_only:
        # 从 run2 开始交替执行剩余轮次
        for i in range(2, args.runs + 1):
            for backend, load_ranks in (("mmap", args.ranks), ("legacy", 8)):
                d = run_one(backend, i, args.ranks, load_ranks, args.batches)
                outs[backend].append(d)
    for backend, dirs in existing.items():
        outs[backend] = [d for d in dirs if (d / "summary.json").exists()] + outs[backend]
    aggregate(outs, RUNS / "stress64_report.md")


if __name__ == "__main__":
    main()
