#!/usr/bin/env python3
"""48 卡数据链路压力测试: 在单机 1:1 模拟 n6×8 卡训练作业的 dataloader 并发。

真实 48 卡作业的数据侧负载 = 48 rank × 4 dataloader worker = 192 路并发读。
本脚本在单机 (192 核, 挂同一份 PFS) 上完整复刻:

  - GigaLMDBDataset (与 train_giga_exp.py 相同的构建参数与 index_cache_dir)
  - BucketBatchSampler(rank=r, world_size=48) 分片 + 高分桶借图
  - DataLoader(num_workers=4, prefetch_factor=4, persistent_workers=True, batch_size=8)
  - 与训练循环一致的坏数据跳步语义 (category 分类, 连跳 64 上限)

内存设计: 23GB 索引只在父进程加载一次, 48 个 rank 进程 fork 共享 (COW)。
不做 GPU 计算 (压测对象是数据链路本身, 无 GPU 消费节流, 比 real job 更狠)。

用法:
  python scripts/test/stress_dataloader_48gpus.py                # 默认 48 rank × 15 batch
  python scripts/test/stress_dataloader_48gpus.py --ranks 2 --batches 2   # 冒烟
结果输出: /tmp/stress48_<ts>/ (rank*.json + rank*.log + summary.json)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/stage_1.yaml")
    p.add_argument("--ranks", type=int, default=48, help="模拟的 rank 数 (真实作业=48)")
    p.add_argument("--workers", type=int, default=4, help="每 rank dataloader worker 数")
    p.add_argument("--batches", type=int, default=15, help="每 rank 成功取 batch 数上限")
    p.add_argument("--deadline-s", type=int, default=1800, help="每个 rank 的墙钟预算 (秒)")
    p.add_argument("--pace", type=float, default=0.0, help="每 batch 后模拟 GPU step 耗时 (秒), 0=不限速全速压")
    p.add_argument("--timeout", type=float, default=48, help="YISU_IMAGE_TIMEOUT_SECONDS (单次尝试上限)")
    p.add_argument("--sample-timeout", type=float, default=120, help="YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS (样本预算)")
    p.add_argument("--persistent", type=int, default=0, choices=[0, 1], help="YISU_IMAGE_PERSISTENT_DECODER")
    p.add_argument("--out", default=None, help="结果输出目录 (默认 /tmp/stress48_<ts>)")
    return p.parse_args()


def categorize(exc) -> str:
    """与 train_giga_exp.py::_data_error_category 相同的归类逻辑。"""
    category = getattr(exc, "category", None)
    if category:
        return str(category)
    try:
        from data.datasets.video_decode import media_error_category
        return media_error_category(exc)
    except Exception:
        return "executor_bug"


def run_rank(rank: int, cfg: dict, args, outdir: Path, deadline_abs: float):
    """子进程入口: 复刻训练侧一个 rank 的 dataloader 消费循环。"""
    logfile = open(outdir / f"rank{rank:02d}.log", "w")

    def emit(msg):
        stamp = time.strftime("%H:%M:%S")
        logfile.write(f"[{stamp} rank{rank:02d}] {msg}\n")
        logfile.flush()

    os.environ["RANK"] = str(rank)
    try:
        from torch.utils.data import DataLoader
        from data.lmdb_dataset import GigaLMDBDataset  # noqa: F401  (parent 已构建, 此处仅对齐)
        from data.samplers.bucket_sampler import BucketBatchSampler
        from data.datasets.image_database import collate_image_batch

        sampler = BucketBatchSampler(
            data.bucket_indices, int(cfg["training"]["batch_size_per_device"]),
            weights=cfg["dataset"].get("bucket_sampling_weights"), seed=int(cfg.get("seed", 42)),
            rank=rank, world_size=args.ranks, randomize_order_on_start=True, drop_last=True)
        loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                            prefetch_factor=4, persistent_workers=True, collate_fn=collate_image_batch)

        stats = {
            "rank": rank, "ok_batches": 0, "skipped_batches": 0, "samples": 0,
            "categories": {}, "batch_times": [], "consecutive_max": 0,
            "skip_reasons": Counter(), "exit_reason": "",
        }
        cur_wait = {"t0": None, "batch": None}

        def heartbeat():
            while True:
                time.sleep(60)
                w = cur_wait["t0"]
                waiting = f" cur_batch_wait={time.monotonic() - w:.0f}s" if w else ""
                emit(f"heartbeat ok={stats['ok_batches']} skip={stats['skipped_batches']}{waiting}")

        threading.Thread(target=heartbeat, daemon=True).start()

        def flush_stats():
            (outdir / f"rank{rank:02d}.json").write_text(json.dumps({**stats, "categories": dict(stats["categories"]),
                                                                     "skip_reasons": dict(stats["skip_reasons"]),
                                                                     "batch_times": stats["batch_times"][:2000]}))

        iterator = iter(loader)
        consecutive = 0
        t_start = time.monotonic()
        while stats["ok_batches"] < args.batches:
            if time.time() >= deadline_abs:
                stats["exit_reason"] = "deadline"
                break
            cur_wait["t0"] = time.monotonic()
            cur_wait["batch"] = stats["ok_batches"] + stats["skipped_batches"] + 1
            t0 = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                sampler.set_epoch(sampler.epoch + 1)
                iterator = iter(loader)
                continue
            except Exception as exc:  # 与训练循环相同的跳步语义
                category = categorize(exc)
                stats["skipped_batches"] += 1
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                stats["skip_reasons"][f"{type(exc).__name__}: {str(exc)[:160]}"] += 1
                consecutive += 1
                stats["consecutive_max"] = max(stats["consecutive_max"], consecutive)
                wait = time.monotonic() - t0
                emit(f"skip batch#{cur_wait['batch']} wait={wait:.1f}s category={category} "
                     f"consecutive={consecutive}: {type(exc).__name__}: {str(exc)[:200]}")
                if category == "executor_bug" or consecutive > 64:
                    stats["exit_reason"] = f"fatal:{type(exc).__name__}"
                    flush_stats()
                    emit("FATAL executor_bug/连跳超限, 终止本 rank")
                    os._exit(3)
                flush_stats()
                continue
            consecutive = 0
            stats["consecutive_max"] = max(stats["consecutive_max"], 0)
            wait = time.monotonic() - t0
            stats["batch_times"].append(round(wait, 3))
            stats["ok_batches"] += 1
            stats["samples"] += len(batch["pixel_values"])
            cur_wait["t0"] = None
            del batch
            flush_stats()
            if args.pace > 0:
                time.sleep(args.pace)
        else:
            stats["exit_reason"] = "done"
        stats["wall_s"] = round(time.monotonic() - t_start, 1)
        flush_stats()
        bt = stats["batch_times"]
        emit(f"done exit={stats['exit_reason']} ok={stats['ok_batches']} skip={stats['skipped_batches']} "
             f"samples={stats['samples']} wall={stats['wall_s']}s "
             f"batch_time_avg={sum(bt) / len(bt):.1f}s" if bt else f"done ok=0 wall={stats['wall_s']}s")
    except Exception as exc:
        try:
            emit(f"CRASH {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc(file=logfile)
        except Exception:
            pass
        os._exit(2)
    finally:
        try:
            logfile.close()
        except Exception:
            pass


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def main():
    global data  # fork 共享: rank 子进程通过模块级 data 访问 23GB 索引
    args = parse_args()
    outdir = Path(args.out or f"/tmp/stress48_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"结果目录: {outdir}")

    # 与启动脚本一致的环境变量 (修复后默认值), 子进程/decode 子进程全部继承
    os.environ["YISU_IMAGE_TIMEOUT_SECONDS"] = str(args.timeout)
    os.environ["YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS"] = str(args.sample_timeout)
    os.environ["YISU_IMAGE_PERSISTENT_DECODER"] = str(args.persistent)
    print(f"env: TIMEOUT={args.timeout} SAMPLE_TIMEOUT={args.sample_timeout} PERSISTENT_DECODER={args.persistent}")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    print(f"config={args.config} ranks={args.ranks} workers={args.workers} "
          f"batches/rank={args.batches} deadline={args.deadline_s}s pace={args.pace}s")

    t0 = time.time()
    from data.lmdb_dataset import GigaLMDBDataset
    data = GigaLMDBDataset(cfg["dataset"], seed=int(cfg.get("seed", 42)),
                           index_cache_dir=str(Path(cfg["output_dir"]) / "dataset_index_cache"))
    print(f"dataset ready: records={len(data.records):,} "
          f"buckets={ {k: len(v) for k, v in data.bucket_indices.items()} } load={time.time() - t0:.0f}s", flush=True)

    # fork 前置检查: 父进程无 CUDA、无线程 (DataLoader/decode 全部在子进程内)
    import multiprocessing as mp
    assert mp.get_start_method() == "fork", "必须使用 fork 以共享 23GB 索引"

    deadline_abs = time.time() + args.deadline_s
    procs = []
    for r in range(args.ranks):
        p = mp.Process(target=run_rank, args=(r, cfg, args, outdir, deadline_abs), daemon=False)
        p.start()
        procs.append(p)
    print(f"已拉起 {args.ranks} 个 rank 进程 (各 {args.workers} worker, 共 {args.ranks * args.workers} 路并发), "
          f"deadline={args.deadline_s}s, 等待完成...", flush=True)

    grace = args.deadline_s + 150
    t_start = time.time()
    while any(p.is_alive() for p in procs):
        time.sleep(10)
        alive = sum(1 for p in procs if p.is_alive())
        if time.time() - t_start > grace:
            print(f"⚠️ 超过宽限期 {grace}s, 强制清理 {alive} 个存活 rank", flush=True)
            for p in procs:
                if p.is_alive():
                    p.kill()
            break
        if (int(time.time() - t_start)) % 60 < 10:
            print(f"⏳ {alive}/{args.ranks} rank 存活, elapsed={time.time() - t_start:.0f}s", flush=True)

    exitcodes = [p.exitcode for p in procs]
    crashed = [i for i, c in enumerate(exitcodes) if c not in (0, None)]
    if crashed:
        print(f"⚠️ 非 0 退出的 rank: {crashed}", flush=True)

    # ─── 汇总 ───
    rank_stats = []
    for r in range(args.ranks):
        f = outdir / f"rank{r:02d}.json"
        if f.exists():
            rank_stats.append(json.loads(f.read_text()))
        else:
            rank_stats.append({"rank": r, "ok_batches": 0, "skipped_batches": 0, "samples": 0,
                               "categories": {}, "batch_times": [], "exit_reason": "no_json(被杀/崩溃)"})

    all_times = [t for s in rank_stats for t in s.get("batch_times", [])]
    total_ok = sum(s["ok_batches"] for s in rank_stats)
    total_skip = sum(s["skipped_batches"] for s in rank_stats)
    total_samples = sum(s["samples"] for s in rank_stats)
    cats = Counter()
    for s in rank_stats:
        cats.update(s.get("categories", {}))
    # attempt_failed 计数 (TimeoutError 单次尝试失败, stdout 必打)
    attempt_failed = 0
    attempt_timeout = 0
    for r in range(args.ranks):
        lf = outdir / f"rank{r:02d}.log"
        if not lf.exists():
            continue
        try:
            text = lf.read_text()
        except OSError:
            continue
        attempt_failed += text.count("stage=attempt_failed")
        attempt_timeout += text.count("TimeoutError")

    wall = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"48卡数据链路压力测试汇总  (ranks={args.ranks} workers={args.workers} 墙钟={wall:.0f}s)")
    print("=" * 78)
    print(f"batch 成功/跳过: {total_ok} / {total_skip}  (跳过率 {total_skip / max(1, total_ok + total_skip) * 100:.1f}%)")
    print(f"样本总数: {total_samples:,}  聚合吞吐: {total_samples / max(1, wall):.1f} samples/s")
    if all_times:
        print(f"batch 取数耗时: p50={percentile(all_times, .5):.1f}s p90={percentile(all_times, .9):.1f}s "
              f"p99={percentile(all_times, .99):.1f}s max={max(all_times):.1f}s (n={len(all_times)})")
    print(f"跳过异常分类: {dict(cats)}")
    print(f"单次尝试失败 (attempt_failed): {attempt_failed} 次, 其中 TimeoutError: {attempt_timeout} 次")
    print(f"rank 退出原因: {Counter(s['exit_reason'] for s in rank_stats)}")
    summary = {"args": vars(args), "wall_s": wall, "total_ok_batches": total_ok,
               "total_skipped_batches": total_skip, "total_samples": total_samples,
               "throughput_samples_per_s": total_samples / max(1, wall),
               "batch_time_p50": percentile(all_times, .5), "batch_time_p90": percentile(all_times, .9),
               "batch_time_p99": percentile(all_times, .99), "batch_time_max": max(all_times) if all_times else 0,
               "categories": dict(cats), "attempt_failed": attempt_failed,
               "attempt_timeout": attempt_timeout, "exitcodes": exitcodes,
               "rank_stats": [{k: v for k, v in s.items() if k != "batch_times"} for s in rank_stats]}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细: {outdir}/summary.json  rank 日志: {outdir}/rank*.log")


if __name__ == "__main__":
    # 全局数据集: fork 共享给所有 rank 子进程
    data = None
    main()
