#!/usr/bin/env python3
"""64 卡数据链路压力测试: 新旧索引后端 (mmap 列式 vs legacy pickle) 对比。

真实 64 卡作业 = 64 rank × 4 dataloader worker = 256 路并发读。本脚本在单机
(2TB RAM / 192 核) 上分两个阶段复刻:

  Phase A 冷启动: N 个独立进程各自构建 GigaLMDBDataset (与 torchrun 每 rank
    独立加载索引一致), 测每 rank 加载秒数与峰值 RSS。
    - mmap 后端: 64 rank 直接跑 (每 rank 只建映射, 内存可承受)。
    - legacy 后端: 每 rank 反序列化 ~20GB pickle, 64 rank 全量会逼近内存上限,
      用 --load-ranks 控制实际并发 (结果按线性外推说明趋势)。

  Phase B 稳态读取: 父进程加载一次索引后 fork 64 个 rank 进程 (与生产
    dataloader worker 的 fork 共享一致), 各 rank 经 BucketBatchSampler 分片 +
    DataLoader(num_workers=4, prefetch_factor=4, persistent_workers=True) 消费
    N 个 batch。父进程周期采样 ΣRSS/ΣPss: mmap 共享页应表现为 Pss 总量远小于
    RSS 总量, legacy 则因 COW 两者接近。

用法:
  python scripts/test/stress_dataloader_64ranks.py --backend mmap  --batches 15
  python scripts/test/stress_dataloader_64ranks.py --backend legacy --load-ranks 8 --batches 15
  python scripts/test/stress_dataloader_64ranks.py --backend mmap --ranks 2 --batches 2   # 冒烟
结果输出: /tmp/stress64_<backend>_<ts>/ (rank*.json + rank*.log + load*.json + summary.json)
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEM_AVAILABLE_FLOOR_GIB = 120  # 内存护栏: 低于此值强制清理, 保护节点


def gib_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/stage_1.yaml")
    p.add_argument("--backend", default="mmap", choices=["mmap", "legacy"],
                   help="索引后端: mmap=列式 mmap 共享 (新), legacy=pickle 整份反序列化 (旧)")
    p.add_argument("--ranks", type=int, default=64, help="Phase B 模拟的 rank 数")
    p.add_argument("--load-ranks", type=int, default=64,
                   help="Phase A 独立构建索引的进程数 (legacy 建议 8, 避免内存超限)")
    p.add_argument("--workers", type=int, default=4, help="每 rank dataloader worker 数")
    p.add_argument("--batches", type=int, default=15, help="每 rank 成功取 batch 数上限")
    p.add_argument("--batch-size", type=int, default=None, help="覆盖 config 的 batch_size_per_device (如 32卡 bs=4/8 对比)")
    p.add_argument("--deadline-s", type=int, default=1800, help="每阶段墙钟预算 (秒)")
    p.add_argument("--pace", type=float, default=0.0, help="每 batch 后模拟 GPU step 耗时 (秒), 0=全速压")
    p.add_argument("--timeout", type=float, default=48, help="YISU_IMAGE_TIMEOUT_SECONDS (单次尝试上限)")
    p.add_argument("--sample-timeout", type=float, default=120, help="YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS (样本预算)")
    p.add_argument("--persistent", type=int, default=0, choices=[0, 1], help="YISU_IMAGE_PERSISTENT_DECODER")
    p.add_argument("--out", default=None, help="结果输出目录 (默认 /tmp/stress64_<backend>_<ts>)")
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


def proc_stat(pid: int):
    """读单个进程的 RSS/Pss KiB (smaps_rollup), 进程消失返回 None。"""
    try:
        rss = pss = None
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Rss:"):
                rss = int(line.split()[1])
            elif line.startswith("Pss:"):
                pss = int(line.split()[1])
        return rss, pss
    except OSError:
        return None


# ─────────────────────────── Phase A: 冷启动加载 ───────────────────────────

def run_load(proc_id: int, cfg: dict, args, outdir: Path, deadline_abs: float):
    logfile = open(outdir / f"load{proc_id:02d}.log", "w")
    try:
        os.environ["RANK"] = str(proc_id)
        t0 = time.monotonic()
        import yaml as _yaml  # noqa: F401  (保证与训练侧一致的导入顺序)
        from data.lmdb_dataset import GigaLMDBDataset
        ds = GigaLMDBDataset(cfg["dataset"], seed=42,
                             index_cache_dir=str(Path(cfg["output_dir"]) / "dataset_index_cache"))
        load_s = time.monotonic() - t0
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        n_records = len(ds.records)
        result = {"proc": proc_id, "load_s": round(load_s, 1),
                  "peak_rss_gib": round(peak_rss_kib / 1024**2, 2),
                  "records": n_records}
        (outdir / f"load{proc_id:02d}.json").write_text(json.dumps(result))
        print(f"[load{proc_id:02d}] backend={args.backend} load={load_s:.1f}s "
              f"peak_rss={result['peak_rss_gib']:.1f}GiB records={n_records:,}", flush=True)
    except Exception as exc:
        (outdir / f"load{proc_id:02d}.json").write_text(json.dumps({"proc": proc_id, "error": repr(exc)}))
        print(f"[load{proc_id:02d}] CRASH {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc(file=logfile)
        os._exit(2)
    finally:
        logfile.close()
    os._exit(0)


def phase_a(cfg: dict, args, outdir: Path) -> dict:
    n = args.load_ranks
    print(f"\n{'=' * 78}\nPhase A 冷启动: {n} 个独立进程各自构建索引 (backend={args.backend})\n{'=' * 78}", flush=True)
    deadline_abs = time.time() + args.deadline_s
    t0 = time.monotonic()
    import multiprocessing as mp
    procs = [mp.Process(target=run_load, args=(i, cfg, args, outdir, deadline_abs)) for i in range(n)]
    for p in procs:
        p.start()
    killed = False
    while any(p.is_alive() for p in procs):
        time.sleep(5)
        avail = gib_available()
        if avail < MEM_AVAILABLE_FLOOR_GIB and not killed:
            print(f"⚠️ 内存护栏触发 (MemAvailable={avail:.0f}GiB), 清理全部加载进程", flush=True)
            killed = True
            for p in procs:
                if p.is_alive():
                    p.kill()
        if killed and not any(p.is_alive() for p in procs):
            break
    wall = time.monotonic() - t0
    results = []
    for i in range(n):
        f = outdir / f"load{i:02d}.json"
        if f.exists():
            results.append(json.loads(f.read_text()))
        else:
            results.append({"proc": i, "error": "no_json(被杀/崩溃)"})
    ok = [r for r in results if "load_s" in r]
    times = sorted(r["load_s"] for r in ok)
    peak = [r["peak_rss_gib"] for r in ok]
    summary = {
        "backend": args.backend, "n_procs": n, "ok": len(ok),
        "wall_s": round(wall, 1), "load_s_p50": times[len(times) // 2] if times else None,
        "load_s_min": times[0] if times else None, "load_s_max": times[-1] if times else None,
        "peak_rss_gib_max": max(peak) if peak else None,
        "estimated_total_gib": round(sum(peak), 1) if peak else None,
        "oom_guard_triggered": killed, "results": results,
    }
    (outdir / "phase_a.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Phase A 完成: ok={len(ok)}/{n} wall={wall:.0f}s "
          f"load p50={summary['load_s_p50']}s max={summary['load_s_max']}s "
          f"peak_rss_max={summary['peak_rss_gib_max']}GiB "
          f"n_proc 总内存≈{summary['estimated_total_gib']}GiB", flush=True)
    return summary


# ─────────────────────────── Phase B: 稳态读取 ───────────────────────────

def run_rank(rank: int, cfg: dict, args, outdir: Path, deadline_abs: float, mem_stop: dict):
    logfile = open(outdir / f"rank{rank:02d}.log", "w")

    def emit(msg):
        stamp = time.strftime("%H:%M:%S")
        logfile.write(f"[{stamp} rank{rank:02d}] {msg}\n")
        logfile.flush()

    try:
        os.environ["RANK"] = str(rank)
        from torch.utils.data import DataLoader
        from data.samplers.bucket_sampler import BucketBatchSampler
        from data.datasets.image_database import collate_image_batch

        sampler = BucketBatchSampler(
            data.bucket_indices, int(args.batch_size or cfg["training"]["batch_size_per_device"]),
            weights=cfg["dataset"].get("bucket_sampling_weights"), seed=42,
            rank=rank, world_size=args.ranks, randomize_order_on_start=True, drop_last=True)
        loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                            prefetch_factor=4, persistent_workers=True, collate_fn=collate_image_batch)

        stats = {"rank": rank, "ok_batches": 0, "skipped_batches": 0, "samples": 0,
                 "categories": {}, "batch_times": [], "exit_reason": ""}
        cur_wait = {"t0": None}

        def heartbeat():
            while True:
                time.sleep(60)
                w = cur_wait["t0"]
                waiting = f" cur_batch_wait={time.monotonic() - w:.0f}s" if w else ""
                emit(f"heartbeat ok={stats['ok_batches']} skip={stats['skipped_batches']}{waiting}")

        threading.Thread(target=heartbeat, daemon=True).start()

        def flush_stats():
            (outdir / f"rank{rank:02d}.json").write_text(json.dumps(
                {**stats, "categories": dict(stats["categories"]),
                 "batch_times": stats["batch_times"][:2000]}))

        iterator = iter(loader)
        consecutive = 0
        t_start = time.monotonic()
        while stats["ok_batches"] < args.batches:
            if time.time() >= deadline_abs:
                stats["exit_reason"] = "deadline"
                break
            if mem_stop["flag"]:
                stats["exit_reason"] = "oom_guard"
                break
            cur_wait["t0"] = time.monotonic()
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
                consecutive += 1
                wait = time.monotonic() - t0
                emit(f"skip batch wait={wait:.1f}s category={category} consecutive={consecutive}: "
                     f"{type(exc).__name__}: {str(exc)[:200]}")
                if category == "executor_bug" or consecutive > 64:
                    stats["exit_reason"] = f"fatal:{type(exc).__name__}"
                    flush_stats()
                    emit("FATAL executor_bug/连跳超限, 终止本 rank")
                    break  # 不用 os._exit: 跳过 finally 会把 persistent worker 变孤儿
                flush_stats()
                continue
            consecutive = 0
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
            # persistent_workers=True 的 worker 是 daemon 进程, 依赖 atexit 回收;
            # os._exit 跳过 atexit → worker 孤儿化 (fork 场景实测 1670 个残留), 必须显式关停。
            iterator = None
            loader = None
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            logfile.close()
        except Exception:
            pass
    os._exit(0)


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def phase_b(cfg: dict, args, outdir: Path) -> dict:
    global data
    print(f"\n{'=' * 78}\nPhase B 稳态读取: 父进程加载索引 → fork {args.ranks} rank "
          f"(×{args.workers} worker)\n{'=' * 78}", flush=True)
    t_load0 = time.monotonic()
    from data.lmdb_dataset import GigaLMDBDataset
    data = GigaLMDBDataset(cfg["dataset"], seed=42,
                           index_cache_dir=str(Path(cfg["output_dir"]) / "dataset_index_cache"))
    parent_load_s = time.monotonic() - t_load0
    parent_rss = proc_stat(os.getpid())
    print(f"dataset ready: records={len(data.records):,} "
          f"parent_load={parent_load_s:.0f}s parent_rss={parent_rss[0] / 1024**2:.1f}GiB "
          f"parent_pss={parent_rss[1] / 1024**2:.1f}GiB", flush=True)

    import multiprocessing as mp
    assert mp.get_start_method() == "fork", "必须使用 fork 以共享索引"
    deadline_abs = time.time() + args.deadline_s
    mem_stop = {"flag": False}
    procs = [mp.Process(target=run_rank, args=(r, cfg, args, outdir, deadline_abs, mem_stop)) for r in range(args.ranks)]
    t0 = time.monotonic()
    for p in procs:
        p.start()
    print(f"已拉起 {args.ranks} 个 rank 进程 (共 {args.ranks * args.workers} 路并发), 等待完成...", flush=True)

    samples = []
    killed = False
    grace = args.deadline_s + 300
    while any(p.is_alive() for p in procs):
        time.sleep(15)
        rss_total = pss_total = 0
        alive = 0
        for p in procs:
            if p.is_alive():
                st = proc_stat(p.pid)
                if st:
                    alive += 1
                    rss_total += st[0]
                    pss_total += st[1]
        avail = gib_available()
        samples.append({"t": round(time.monotonic() - t0), "alive": alive,
                        "rss_gib": round(rss_total / 1024**2, 1),
                        "pss_gib": round(pss_total / 1024**2, 1),
                        "avail_gib": round(avail)})
        print(f"⏳ {alive}/{args.ranks} rank 存活 elapsed={samples[-1]['t']}s "
              f"ΣRSS={samples[-1]['rss_gib']}GiB ΣPss={samples[-1]['pss_gib']}GiB "
              f"MemAvailable={samples[-1]['avail_gib']}GiB", flush=True)
        if avail < MEM_AVAILABLE_FLOOR_GIB and not killed:
            print("⚠️ 内存护栏触发, 停止新取数并清理", flush=True)
            mem_stop["flag"] = True
        if time.monotonic() - t0 > grace:
            print(f"⚠️ 超过宽限期 {grace}s, 强制清理", flush=True)
            killed = True
            for p in procs:
                if p.is_alive():
                    p.kill()
        if killed and not any(p.is_alive() for p in procs):
            break

    wall = time.monotonic() - t0
    exitcodes = [p.exitcode for p in procs]
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
    peak_rss = max((s["rss_gib"] for s in samples), default=0)
    peak_pss = max((s["pss_gib"] for s in samples), default=0)
    final_rss = samples[-1]["rss_gib"] if samples else 0
    final_pss = samples[-1]["pss_gib"] if samples else 0
    print("\n" + "=" * 78)
    print(f"Phase B 汇总 (backend={args.backend} ranks={args.ranks} workers={args.workers} 墙钟={wall:.0f}s)")
    print("=" * 78)
    print(f"batch 成功/跳过: {total_ok} / {total_skip}  (跳过率 {total_skip / max(1, total_ok + total_skip) * 100:.1f}%)")
    print(f"样本总数: {total_samples:,}  聚合吞吐: {total_samples / max(1, wall):.1f} samples/s")
    if all_times:
        print(f"batch 取数耗时: p50={percentile(all_times, .5):.1f}s p90={percentile(all_times, .9):.1f}s "
              f"p99={percentile(all_times, .99):.1f}s max={max(all_times):.1f}s (n={len(all_times)})")
    print(f"内存: 峰值 ΣRSS={peak_rss}GiB ΣPss={peak_pss}GiB | 稳态 ΣRSS={final_rss}GiB "
          f"ΣPss={final_pss}GiB (Pss≪RSS → 页共享生效)")
    print(f"跳过异常分类: {dict(cats)}")
    print(f"rank 退出原因: {Counter(s['exit_reason'] for s in rank_stats)}")
    summary = {"backend": args.backend, "parent_load_s": round(parent_load_s, 1),
               "ranks": args.ranks, "workers": args.workers, "wall_s": round(wall, 1),
               "total_ok_batches": total_ok, "total_skipped_batches": total_skip,
               "total_samples": total_samples, "throughput_samples_per_s": total_samples / max(1, wall),
               "batch_time_p50": percentile(all_times, .5), "batch_time_p90": percentile(all_times, .9),
               "batch_time_p99": percentile(all_times, .99), "batch_time_max": max(all_times) if all_times else 0,
               "peak_rss_gib": peak_rss, "peak_pss_gib": peak_pss,
               "final_rss_gib": final_rss, "final_pss_gib": final_pss,
               "categories": dict(cats), "oom_guard_triggered": mem_stop["flag"],
               "exitcodes": exitcodes, "rss_samples": samples[-10:],
               "rank_stats": [{k: v for k, v in s.items() if k != "batch_times"} for s in rank_stats]}
    (outdir / "phase_b.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细: {outdir}/phase_b.json  rank 日志: {outdir}/rank*.log", flush=True)
    return summary


def main():
    args = parse_args()
    outdir = Path(args.out or f"/tmp/stress64_{args.backend}_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"结果目录: {outdir}")

    os.environ["GIGA_INDEX_BACKEND"] = args.backend
    os.environ["YISU_IMAGE_TIMEOUT_SECONDS"] = str(args.timeout)
    os.environ["YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS"] = str(args.sample_timeout)
    os.environ["YISU_IMAGE_PERSISTENT_DECODER"] = str(args.persistent)
    print(f"env: backend={args.backend} TIMEOUT={args.timeout} SAMPLE_TIMEOUT={args.sample_timeout}")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    print(f"config={args.config} ranks={args.ranks} workers={args.workers} "
          f"batches/rank={args.batches} load_ranks={args.load_ranks}")

    phase_a_summary = phase_a(cfg, args, outdir)
    phase_b_summary = phase_b(cfg, args, outdir)

    report = {"args": vars(args), "phase_a": phase_a_summary, "phase_b": phase_b_summary}
    (outdir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n✅ 全部完成: {outdir}/summary.json")


if __name__ == "__main__":
    data = None  # Phase B: 父进程加载, fork 共享给所有 rank 子进程
    main()
