#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_read_stress.py — 训练级读取压力测试: PFS LMDB vs NFS TAR

模拟训练 dataloader 的读取模式:
  - 多源混合 (与 stage_1_xl 白名单一致)
  - 分辨率 bucket 加权采样 (256_square/256x192/192x256/512_square/512x384/384x512)
  - 每样本: 随机取一行 → 读原始图片字节 → PIL 解码 RGB → resize 到 bucket (cover 近似)

两个后端:
  pfs : unified_images_lmdb 的 mdb 分片 (复用训练的 mmap 列式索引缓存)
  nfs : unified_images_tar 的 tar 分片 seek+read (采样表来自 tar 侧 parquet images[] 列)

并发线程数梯度 32/64/128/256, 每档固定时长, 输出 samples/s、MB/s、延迟分位数。

用法:
  python3 tools/bench_read_stress.py --backend both --seconds 20
  python3 tools/bench_read_stress.py --backend pfs --concurrency 64 --seconds 30
  python3 tools/bench_read_stress.py --backend nfs --sources FLUX-Reason-6M JourneyDB
  python3 tools/bench_read_stress.py --backend nfs --rescan      # 重建 NFS 采样表缓存
  python3 tools/bench_read_stress.py --backend nfs --no-decode   # 只测 IO 不解码
"""

import argparse
import glob
import io
import json
import os
import struct
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PFS_LMDB_ROOT = "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_images_lmdb"
NFS_TAR_ROOT = "/mnt/nfs/data/zhanqian.wu/unified_images_tar"
INDEX_CACHE_DIR = "/home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/stage_1_xl/dataset_index_cache"
NFS_SCAN_CACHE = "/tmp/bench_read_nfs_index.npz"   # NFS 采样表缓存 (可删除重建)

DEFAULT_BUCKETS = ["256_square", "256x192", "192x256", "512_square", "512x384", "384x512"]
DEFAULT_WEIGHTS = {
    "256_square": 0.14, "256x192": 0.08, "192x256": 0.08,
    "512_square": 0.315, "512x384": 0.1925, "384x512": 0.1925,
}

_LMDB_FIELD_SINGLE = ("img_bytes", "image_bytes", "img", "jpg", "data")


def parse_bucket(bucket: str) -> tuple[int, int]:
    if bucket.endswith("_square"):
        size = int(bucket.split("_", 1)[0])
        return size, size
    width, height = bucket.split("x", 1)
    return int(width), int(height)


def assign_buckets(w, h, buckets, chunk=1_000_000):
    """cover 最近桶 (log 宽高差最小), 分块算避免 n×6 大矩阵; w/h 缺失归 512_square。"""
    bw = np.array([parse_bucket(b)[0] for b in buckets], np.float64)
    bh = np.array([parse_bucket(b)[1] for b in buckets], np.float64)
    out = np.empty(len(w), np.int8)
    fallback = buckets.index("512_square")
    for s in range(0, len(w), chunk):
        e = min(s + chunk, len(w))
        ww = np.maximum(w[s:e].astype(np.float64), 1.0)
        hh = np.maximum(h[s:e].astype(np.float64), 1.0)
        err = np.abs(np.log(bw[None, :] / ww[:, None])) + np.abs(np.log(bh[None, :] / hh[:, None]))
        idx = err.argmin(axis=1).astype(np.int8)
        bad = (w[s:e] <= 0) | (h[s:e] <= 0)
        idx[bad] = fallback
        out[s:e] = idx
    return out


# ─────────────────────────── PFS LMDB 后端 ───────────────────────────

_LMDB_ENVS: dict[str, object] = {}
_LMDB_LOCK = threading.Lock()


def _lmdb_env(path: str):
    env = _LMDB_ENVS.get(path)
    if env is None:
        import lmdb
        with _LMDB_LOCK:   # 双检: lmdb 不允许同进程重复 open 同一 env, 需串行化 open
            env = _LMDB_ENVS.get(path)
            if env is None:
                env = lmdb.open(path, readonly=True, lock=False, subdir=False, max_readers=4096)
                _LMDB_ENVS[path] = env
    return env


def load_pfs_rows(index_dir: str):
    """复用训练 mmap 列式索引缓存。全部列保持 mmap 惰性, 只存索引不展开字符串。"""
    meta = json.load(open(os.path.join(index_dir, "meta.json")))

    def mmap_npy(name):
        return np.load(os.path.join(index_dir, name), mmap_mode="r")

    n = int(meta["n_image"])   # 行序 image 在前 edit 在后, 只取 image 行
    return {
        "kind": "pfs",
        "sid": mmap_npy("sid.npy")[:n],
        "w": mmap_npy("w.npy")[:n],
        "h": mmap_npy("h.npy")[:n],
        "src": mmap_npy("src.npy")[:n],        # uint16 → meta["sources"] 下标
        "stem": mmap_npy("stem.npy")[:n],      # uint32 → meta["stems"] 下标
        "sources": list(meta["sources"]),
        "source_dirs": list(meta["source_dirs"]),
        "stems": list(meta["stems"]),
        "n": n,
        "meta": meta,
    }


def pfs_row_at(rows, i):
    return (rows["source_dirs"][int(rows["src"][i])],
            rows["stems"][int(rows["stem"][i])],
            int(rows["sid"][i]))


def pfs_read_one(row):
    dataset_dir, stem, sid = row
    path = os.path.join(dataset_dir, "mdb", f"{stem}.mdb")
    if not os.path.exists(path):
        path = os.path.join(dataset_dir, f"{stem}.mdb")
    env = _lmdb_env(path)
    with env.begin() as txn:
        raw = txn.get(struct.pack("<Q", sid))
    if raw is None:
        raise KeyError(f"lmdb 记录不存在: {path} sid={sid}")
    import pickle
    rec = pickle.loads(raw)
    for field in _LMDB_FIELD_SINGLE:
        if field in rec:
            return rec[field]
    raise KeyError(f"记录缺图字段: fields={sorted(rec)}")


# ─────────────────────────── NFS TAR 后端 ───────────────────────────

_TAR_FDS: dict[str, object] = {}
_TAR_LOCK = threading.Lock()


def _tar_fd(path: str):
    fd = _TAR_FDS.get(path)
    if fd is None:
        fd = open(path, "rb")
        with _TAR_LOCK:
            _TAR_FDS[path] = fd
    return fd


def scan_nfs_rows(sources, tar_root=NFS_TAR_ROOT):
    """扫 tar 侧 parquet (w/h/images[{shard,off,size}]) → 采样表 (source 用整数 id)。"""
    import pyarrow.parquet as pq
    shard_paths: list[str] = []
    shard_ids: dict[str, int] = {}
    source_ids: dict[str, int] = {}
    sources_list: list[str] = []
    out = {k: [] for k in ("shard", "off", "size", "w", "h", "src_id")}
    t0 = time.time()
    for source in sources:
        files = sorted(glob.glob(os.path.join(tar_root, source, "parquet", "*.parquet")))
        if not files:
            print(f"[nfs-scan] ⚠️ {source}: 无 parquet, 跳过", flush=True)
            continue
        if source not in source_ids:
            source_ids[source] = len(sources_list)
            sources_list.append(source)
        sid_src = source_ids[source]
        n_src = 0
        for fp in files:
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["w", "h", "images"])
                w = tbl["w"].to_numpy()
                h = tbl["h"].to_numpy()
                for i, lst in enumerate(tbl["images"].to_pylist()):
                    if not lst:
                        continue
                    it = lst[0]   # 单图 t2i 源只有一张; 多图编辑源取第一张
                    sp = os.path.join(tar_root, source, it["shard"])
                    if sp not in shard_ids:
                        shard_ids[sp] = len(shard_paths)
                        shard_paths.append(sp)
                    out["shard"].append(shard_ids[sp])
                    out["off"].append(int(it["off"]))
                    out["size"].append(int(it["size"]))
                    out["w"].append(int(w[i]) if w[i] else 0)
                    out["h"].append(int(h[i]) if h[i] else 0)
                    out["src_id"].append(sid_src)
                    n_src += 1
        print(f"[nfs-scan] {source}: {n_src} 行 ({time.time()-t0:.0f}s)", flush=True)
    return {
        "shard_paths": np.array(shard_paths, dtype=object),
        "shard": np.asarray(out["shard"], np.int32),
        "off": np.asarray(out["off"], np.int64),
        "size": np.asarray(out["size"], np.int64),
        "w": np.asarray(out["w"], np.int32),
        "h": np.asarray(out["h"], np.int32),
        "src_id": np.asarray(out["src_id"], np.int16),
        "sources": sources_list,
    }


def nfs_row_at(rows, i):
    return (str(rows["shard_paths"][int(rows["shard"][i])]),
            int(rows["off"][i]), int(rows["size"][i]))


def nfs_read_one(row):
    shard_path, off, size = row
    fd = _tar_fd(shard_path)
    fd.seek(off)
    return fd.read(size)


# ─────────────────────────── 压测核心 ───────────────────────────

def run_level(rows, bucket_rows, bucket_ids, weights, buckets, bw_bh, concurrency,
              seconds, warmup, row_at, source_of, read_fn, decode):
    """单档并发压测: 线程池按 bucket 权重随机采样; warmup 后清零计数, 统计窗口固定时长。"""
    stop = threading.Event()
    stats = {"ok": 0, "err": 0, "bytes": 0, "read_lat": [], "total_lat": [],
             "buckets": Counter(), "sources": Counter(), "lock": threading.Lock(),
             "err_sample": ""}

    present = [b for b in buckets if len(bucket_rows.get(b, [])) > 0]
    probs = np.array([weights.get(b, 0) for b in present], np.float64)
    probs = probs / probs.sum()
    sizes = {b: bw_bh[b] for b in present}

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    def worker(seed):
        rng = np.random.default_rng(seed)
        while not stop.is_set():
            b = rng.choice(present, p=probs)
            ids = bucket_rows[b]
            i = int(ids[rng.integers(len(ids))])
            t0 = time.perf_counter()
            try:
                data = read_fn(row_at(i))
                t_read = time.perf_counter()
                if decode:
                    bw, bhh = sizes[b]
                    img = Image.open(io.BytesIO(data))
                    img.convert("RGB").resize((bw, bhh), Image.BILINEAR).load()
                t1 = time.perf_counter()
                with stats["lock"]:
                    stats["ok"] += 1
                    stats["bytes"] += len(data)
                    stats["buckets"][b] += 1
                    stats["sources"][source_of(i)] += 1
                    stats["read_lat"].append(t_read - t0)
                    stats["total_lat"].append(t1 - t0)
            except Exception as e:
                with stats["lock"]:
                    stats["err"] += 1
                    if not stats["err_sample"]:
                        stats["err_sample"] = f"{type(e).__name__}: {e}"[:150]

    pool = ThreadPoolExecutor(max_workers=concurrency)
    for k in range(concurrency):
        pool.submit(worker, (os.getpid() * 7919 + k * 104729) % (2**31))
    time.sleep(warmup)
    with stats["lock"]:
        stats.update(ok=0, err=0, bytes=0, read_lat=[], total_lat=[],
                     buckets=Counter(), sources=Counter())
    t_win = time.perf_counter()
    time.sleep(seconds)
    stop.set()
    pool.shutdown(wait=True)
    elapsed = time.perf_counter() - t_win
    with stats["lock"]:
        n_ok, n_err, nbytes = stats["ok"], stats["err"], stats["bytes"]
        read_lat = np.asarray(stats["read_lat"])
        total_lat = np.asarray(stats["total_lat"])
        buckets_c, sources_c, err_sample = (stats["buckets"].copy(), stats["sources"].copy(),
                                            stats["err_sample"])

    def pct(a, q):
        return float(np.percentile(a, q)) * 1000 if len(a) else float("nan")
    return {
        "concurrency": concurrency, "elapsed_s": round(elapsed, 1),
        "ok": n_ok, "err": n_err,
        "samples_per_s": round(n_ok / elapsed, 1),
        "mb_per_s": round(nbytes / elapsed / 1e6, 1),
        "read_p50_ms": round(pct(read_lat, 50), 1), "read_p90_ms": round(pct(read_lat, 90), 1),
        "read_p99_ms": round(pct(read_lat, 99), 1),
        "total_p50_ms": round(pct(total_lat, 50), 1), "total_p90_ms": round(pct(total_lat, 90), 1),
        "total_p99_ms": round(pct(total_lat, 99), 1),
        "buckets": dict(buckets_c), "sources": dict(sources_c.most_common(10)),
        "err_sample": err_sample,
    }


def run_seq_level(rows, bucket_of, buckets, bw_bh, concurrency, seconds, warmup,
                  source_of, decode, src_mask=None):
    """tar 顺序模式: 每个 worker 独占若干分片, 按 idx 顺序连续读 (tar 的最优访问姿势)。
    分片随机洗牌后轮转均摊给 worker → 多源混合; 记录的 resize 目标 = 该记录最近桶。"""
    stop = threading.Event()
    stats = {"ok": 0, "err": 0, "bytes": 0, "read_lat": [], "total_lat": [],
             "buckets": Counter(), "sources": Counter(), "lock": threading.Lock(),
             "err_sample": ""}

    shard_ids = np.asarray(rows["shard"])
    order = np.argsort(shard_ids, kind="stable")          # 一次排序, O(n log n) 替代逐分片全表扫描
    shard_sorted = shard_ids[order]
    uniq, starts = np.unique(shard_sorted, return_index=True)
    tasks = []
    for k, u in enumerate(uniq):
        s, e = int(starts[k]), int(starts[k + 1]) if k + 1 < len(starts) else len(order)
        ids = order[s:e]
        if src_mask is not None:
            ids = ids[src_mask[ids]]
            if not len(ids):
                continue
        ids = ids[np.argsort(rows["off"][ids], kind="stable")]   # idx 顺序即 off 升序, 再稳排一次
        tasks.append((str(rows["shard_paths"][int(u)]), ids))
    rng = np.random.default_rng(42)
    rng.shuffle(tasks)   # 洗牌保证多源混合
    assignments = [tasks[k::concurrency] for k in range(concurrency)]
    n_workers = sum(1 for a in assignments if a)
    print(f"[seq] 分片总数={len(tasks)} → {n_workers} worker 轮转均摊", flush=True)

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    def worker(k):
        fd = None
        for path, ids in assignments[k]:
            if stop.is_set():
                return
            fd = _tar_fd(path)
            for i in ids:
                if stop.is_set():
                    return
                bw, bh = bw_bh[buckets[int(bucket_of[i])]]
                t0 = time.perf_counter()
                try:
                    fd.seek(int(rows["off"][i]))
                    data = fd.read(int(rows["size"][i]))
                    t_read = time.perf_counter()
                    if decode:
                        img = Image.open(io.BytesIO(data))
                        img.convert("RGB").resize((bw, bh), Image.BILINEAR).load()
                    t1 = time.perf_counter()
                    with stats["lock"]:
                        stats["ok"] += 1
                        stats["bytes"] += len(data)
                        stats["buckets"][buckets[int(bucket_of[i])]] += 1
                        stats["sources"][source_of(i)] += 1
                        stats["read_lat"].append(t_read - t0)
                        stats["total_lat"].append(t1 - t0)
                except Exception as e:
                    with stats["lock"]:
                        stats["err"] += 1
                        if not stats["err_sample"]:
                            stats["err_sample"] = f"{type(e).__name__}: {e}"[:150]
            fd = None

    pool = ThreadPoolExecutor(max_workers=n_workers)
    for k in range(n_workers):
        pool.submit(worker, k)
    time.sleep(warmup)
    with stats["lock"]:
        stats.update(ok=0, err=0, bytes=0, read_lat=[], total_lat=[],
                     buckets=Counter(), sources=Counter())
    t_win = time.perf_counter()
    time.sleep(seconds)
    stop.set()
    pool.shutdown(wait=True)
    elapsed = time.perf_counter() - t_win
    with stats["lock"]:
        n_ok, n_err, nbytes = stats["ok"], stats["err"], stats["bytes"]
        read_lat = np.asarray(stats["read_lat"])
        total_lat = np.asarray(stats["total_lat"])
        buckets_c, sources_c, err_sample = (stats["buckets"].copy(), stats["sources"].copy(),
                                            stats["err_sample"])

    def pct(a, q):
        return float(np.percentile(a, q)) * 1000 if len(a) else float("nan")
    return {
        "concurrency": concurrency, "elapsed_s": round(elapsed, 1),
        "ok": n_ok, "err": n_err,
        "samples_per_s": round(n_ok / elapsed, 1),
        "mb_per_s": round(nbytes / elapsed / 1e6, 1),
        "read_p50_ms": round(pct(read_lat, 50), 1), "read_p90_ms": round(pct(read_lat, 90), 1),
        "read_p99_ms": round(pct(read_lat, 99), 1),
        "total_p50_ms": round(pct(total_lat, 50), 1), "total_p90_ms": round(pct(total_lat, 90), 1),
        "total_p99_ms": round(pct(total_lat, 99), 1),
        "buckets": dict(buckets_c), "sources": dict(sources_c.most_common(10)),
        "err_sample": err_sample,
    }


def pick_index_dir(index_cache_dir):
    best, best_t = None, -1
    for d in glob.glob(os.path.join(index_cache_dir, "index_*/")):
        mp = os.path.join(d, "meta.json")
        if os.path.isfile(mp):
            m = json.load(open(mp))
            if m.get("created", 0) > best_t:
                best, best_t = d, m["created"]
    if best is None:
        raise RuntimeError(f"{index_cache_dir} 下无列式索引缓存 (index_*/meta.json)")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["pfs", "nfs", "both"], default="both")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--seconds", type=float, default=20, help="每档并发压测时长 (s)")
    ap.add_argument("--warmup", type=float, default=3, help="每档预热时长 (s), 不计入统计")
    ap.add_argument("--sources", nargs="+", default=None, help="只压测指定源 (默认全部)")
    ap.add_argument("--weights", nargs="+", default=None, help="bucket 权重 256_square:0.14 ...")
    ap.add_argument("--buckets", nargs="+", default=DEFAULT_BUCKETS)
    ap.add_argument("--no-decode", action="store_true", help="只测 IO 不解码 (纯存储压力)")
    ap.add_argument("--nfs-mode", choices=["seek", "seq"], default="seek",
                    help="seek=随机 seek+read (训练现状); seq=worker 独占分片顺序读 (tar 最优姿势)")
    ap.add_argument("--index-dir", default=None, help="PFS 列式索引缓存目录 (默认取最新)")
    ap.add_argument("--rescan", action="store_true", help="强制重建 NFS 采样表缓存")
    ap.add_argument("--report", default="/tmp/bench_read_stress_report.json")
    args = ap.parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights = {k: float(v) for k, v in (w.split(":") for w in args.weights)}
    bw_bh = {b: parse_bucket(b) for b in args.buckets}
    backends = ["pfs", "nfs"] if args.backend == "both" else [args.backend]

    results = []
    for backend in backends:
        print(f"\n═══ backend={backend} ═══", flush=True)
        t0 = time.time()
        if backend == "pfs":
            index_dir = args.index_dir or pick_index_dir(INDEX_CACHE_DIR)
            print(f"[pfs] 索引缓存: {index_dir}", flush=True)
            rows = load_pfs_rows(index_dir)
            row_at = lambda i: pfs_row_at(rows, i)
            read_fn = pfs_read_one
            source_of = lambda i: rows["sources"][int(rows["src"][i])]
            src_filter = np.asarray(rows["src"])
        else:
            if os.path.isfile(NFS_SCAN_CACHE) and not args.rescan:
                print(f"[nfs] 复用采样表缓存 {NFS_SCAN_CACHE} (--rescan 重建)", flush=True)
                z = np.load(NFS_SCAN_CACHE, allow_pickle=True)
                rows = {k: z[k] for k in ("shard", "off", "size", "w", "h", "src_id")}
                rows["shard_paths"] = z["shard_paths"]
                rows["sources"] = [str(s) for s in z["sources"]]
            else:
                sources = args.sources or sorted(d for d in os.listdir(NFS_TAR_ROOT)
                                                 if os.path.isdir(os.path.join(NFS_TAR_ROOT, d, "parquet")))
                print(f"[nfs] 扫描 {len(sources)} 个源 parquet ...", flush=True)
                scanned = scan_nfs_rows(sources)
                rows = {k: scanned[k] for k in ("shard", "off", "size", "w", "h", "src_id")}
                rows["shard_paths"] = scanned["shard_paths"]
                rows["sources"] = scanned["sources"]
                np.savez(NFS_SCAN_CACHE, shard_paths=scanned["shard_paths"], sources=scanned["sources"],
                         shard=rows["shard"], off=rows["off"], size=rows["size"],
                         w=rows["w"], h=rows["h"], src_id=rows["src_id"])
                print(f"[nfs] 采样表已缓存 → {NFS_SCAN_CACHE}", flush=True)
            row_at = lambda i: nfs_row_at(rows, i)
            read_fn = nfs_read_one
            source_of = lambda i: rows["sources"][int(rows["src_id"][i])]
            src_filter = np.asarray(rows["src_id"])

        bucket_of = assign_buckets(rows["w"], rows["h"], args.buckets)
        bucket_rows = {}
        for b in args.buckets:
            ids = np.flatnonzero(bucket_of == args.buckets.index(b))
            if args.sources:
                ids = ids[np.isin(src_filter[ids], [rows["sources"].index(s) for s in args.sources if s in rows["sources"]])]
            bucket_rows[b] = ids
        dist = {b: len(v) for b, v in bucket_rows.items()}
        print(f"[{backend}] 行数={len(bucket_of)} 桶分布={dist} 建表耗时 {time.time()-t0:.0f}s", flush=True)
        if sum(dist.values()) == 0:
            print(f"[{backend}] ⚠️ 无可用行, 跳过", flush=True)
            continue

        for c in args.concurrency:
            if backend == "nfs" and args.nfs_mode == "seq":
                src_mask = None
                if args.sources:
                    keep_ids = [rows["sources"].index(s) for s in args.sources if s in rows["sources"]]
                    src_mask = np.isin(np.asarray(rows["src_id"]), keep_ids)
                r = run_seq_level(rows, bucket_of, args.buckets, bw_bh, c,
                                  args.seconds, args.warmup, source_of,
                                  decode=not args.no_decode, src_mask=src_mask)
            else:
                r = run_level(rows, bucket_rows, None, weights, args.buckets, bw_bh,
                              c, args.seconds, args.warmup, row_at, source_of, read_fn,
                              decode=not args.no_decode)
            r["backend"] = backend + (f"-{args.nfs_mode}" if backend == "nfs" else "")
            print(f"[{backend}] C={c:>3}  {r['samples_per_s']:>8.1f} samples/s  "
                  f"{r['mb_per_s']:>7.1f} MB/s  err={r['err']}  "
                  f"read p50/p90/p99 = {r['read_p50_ms']}/{r['read_p90_ms']}/{r['read_p99_ms']} ms  "
                  f"total p50/p90/p99 = {r['total_p50_ms']}/{r['total_p90_ms']}/{r['total_p99_ms']} ms"
                  + (f"  ⚠️ {r['err_sample']}" if r["err_sample"] else ""), flush=True)
            results.append(r)
        if backend == "pfs":
            _LMDB_ENVS.clear()
        else:
            for fd in _TAR_FDS.values():
                fd.close()
            _TAR_FDS.clear()

    print("\n═══ 汇总 ═══", flush=True)
    print(f"{'backend':<6} {'C':>4} {'samples/s':>10} {'MB/s':>8} {'read_p50':>9} {'read_p99':>9} {'tot_p50':>8} {'tot_p99':>8} {'err':>5}")
    for r in results:
        print(f"{r['backend']:<6} {r['concurrency']:>4} {r['samples_per_s']:>10.1f} {r['mb_per_s']:>8.1f} "
              f"{r['read_p50_ms']:>8.1f} {r['read_p99_ms']:>9.1f} {r['total_p50_ms']:>8.1f} "
              f"{r['total_p99_ms']:>8.1f} {r['err']:>5}")
    with open(args.report, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已写入 {args.report}", flush=True)


if __name__ == "__main__":
    main()
