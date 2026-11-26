#!/usr/bin/env python3
"""breakpoint-grounding-55m parquet -> tar_parquet (t2i 格式), 按美学分桶 + 每 N 行切 part.

输出布局对齐训练 loader (data/datasets/tar_database.py + unified_images_tar 约定):
  <out_root>/<part_name>/shard-NNNNN.tar              512 对齐 ustar, 成员 = 原始 JPEG bytes
  <out_root>/<part_name>/shard-NNNNN.idx              JSONL {id, off, size} (off = 数据起始偏移)
  <out_root>/<part_name>/parquet/shard-NNNNN.parquet  检索层: id, source, caption, w, h,
                                                      images[{id, shard, off, size}] (+附加列)

分桶: aesthetic_score ∈ [lo, hi) (hi=None 表示 +inf), decode_ok 必须为 true。
切 part: 桶内按 (shard 升序, 行号升序) 全局计数, 每 part_rows 行一个 part,
余数进最后一个 part。part 命名: aes{区间规范名}_part{p:03d}。

任务按 shard 粒度并行: 每个 shard 的 data parquet 只读一次 (仅含目标行的 row
group), 行路由到其所属 (桶, part) 输出文件; 一个 shard 最多产出 4 个输出文件
(2 桶 x 边界切分)。断点续传: 输出三件齐且无 .tmp 即跳过。

用法:
  python3 build_grounding55m_tar.py \
      --buckets 5.25-5.75 5.75+ \
      --out-root /home/jovyan/pfs/data/public/image_data \
      --workers 12
"""
import argparse
import glob
import json
import os
import tarfile
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

SECTOR = 512
RECORD = 20 * SECTOR
COUNTS_CACHE = "/home/jovyan/pfs/users/zhanqian.wu/exp/bpg55m_bucket_counts.json"


def parse_bucket(spec):
    """'5.25-5.75' -> (5.25, 5.75); '5.75+' -> (5.75, None); '<4.75' -> (0.0, 4.75)."""
    if spec.startswith("<"):
        return (0.0, float(spec[1:]))
    if spec.endswith("+"):
        lo = float(spec[:-1])
        return (lo, None)
    lo, hi = spec.split("-")
    return (float(lo), float(hi))


def bucket_dirname(lo, hi):
    if lo == 0.0 and hi is not None:
        return f"aeslt{hi:g}"
    if hi is None:
        return f"aes{lo:g}".replace(".", "")
    return f"aes{lo:g}_{hi:g}".replace(".", "")


def write_member(fobj, member_name, data, cur):
    assert cur % SECTOR == 0, cur
    ti = tarfile.TarInfo(name=member_name)
    ti.size = len(data)
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o644
    fobj.write(ti.tobuf(format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict"))
    off = cur + SECTOR
    fobj.write(data)
    pad = (-len(data)) % SECTOR
    if pad:
        fobj.write(b"\0" * pad)
    return off, cur + SECTOR + len(data) + pad


def finalize_tar(fobj, cur):
    end = cur + 2 * SECTOR
    fobj.write(b"\0" * (2 * SECTOR + (-end) % RECORD))


class TarPartWriter:
    """一个 (桶, part) 输出文件组: tar + idx 暂存 + meta 行."""

    def __init__(self, out_dir, k):
        self.out_dir = out_dir
        self.k = k
        self.stem = f"shard-{k:05d}"
        self.tar = open(os.path.join(out_dir, self.stem + ".tar.tmp"), "wb")
        self.cur = 0
        self.idx_lines = []
        self.meta = []
        self.n = 0

    def add(self, member_id, blob, meta_row):
        off, self.cur = write_member(self.tar, member_id, blob, self.cur)
        self.idx_lines.append(json.dumps({"id": member_id, "off": off, "size": len(blob)},
                                         ensure_ascii=False))
        meta_row["images"] = [{"id": member_id, "shard": self.stem + ".tar",
                               "off": off, "size": len(blob)}]
        self.meta.append(meta_row)
        self.n += 1

    def close(self):
        finalize_tar(self.tar, self.cur)
        self.tar.close()
        stem = self.stem
        with open(os.path.join(self.out_dir, stem + ".idx.tmp"), "w") as f:
            f.write("\n".join(self.idx_lines) + ("\n" if self.idx_lines else ""))
        schema = pa.schema([
            ("id", pa.string()),
            ("source", pa.string()),
            ("caption", pa.string()),
            ("w", pa.int64()),
            ("h", pa.int64()),
            ("aesthetic_score", pa.float64()),
            ("object_captions", pa.list_(pa.string())),
            ("normalized_boxes", pa.list_(pa.list_(pa.float32()))),
            ("images", pa.list_(pa.struct([
                ("id", pa.string()), ("shard", pa.string()),
                ("off", pa.int64()), ("size", pa.int64())]))),
        ])
        cols = {k: [r.get(k) for r in self.meta] for k in
                ("id", "source", "caption", "w", "h", "aesthetic_score",
                 "object_captions", "normalized_boxes", "images")}
        tbl = pa.table(cols, schema=schema)
        pq.write_table(tbl, os.path.join(self.out_dir, "parquet", stem + ".parquet.tmp"))
        for suf in (".tar", ".idx"):
            os.replace(os.path.join(self.out_dir, stem + suf + ".tmp"),
                       os.path.join(self.out_dir, stem + suf))
        os.replace(os.path.join(self.out_dir, "parquet", stem + ".parquet.tmp"),
                   os.path.join(self.out_dir, "parquet", stem + ".parquet"))


def scan_counts(src, qf, buckets, total, workers=48):
    """每 shard 各桶行数 (decode_ok 且 aes ∈ [lo,hi)), 带缓存."""
    if os.path.exists(COUNTS_CACHE):
        with open(COUNTS_CACHE) as f:
            raw = json.load(f)
        if raw.get("buckets") == [list(b) for b in buckets] and len(raw.get("counts", {})) == total:
            return {int(k): {int(bi): n for bi, n in v.items()}
                    for k, v in raw["counts"].items()}

    def scan_one(s):
        fp = os.path.join(qf, f"train-{s:06d}-of-{total:06d}.parquet")
        t = pq.read_table(fp, columns=["decode_ok", "aesthetic_score"])
        dec = t.column("decode_ok").to_pylist()
        aes = t.column("aesthetic_score").to_pylist()
        out = {}
        for bi, (lo, hi) in enumerate(buckets):
            n = 0
            for d, a in zip(dec, aes):
                if d and a is not None and a >= lo and (hi is None or a < hi):
                    n += 1
            out[bi] = n
        return s, out

    from concurrent.futures import ThreadPoolExecutor
    counts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (s, out) in enumerate(pool.map(scan_one, range(total))):
            counts[s] = out
            if (i + 1) % 500 == 0:
                print(f"[counts] {i + 1}/{total}", flush=True)
    os.makedirs(os.path.dirname(COUNTS_CACHE), exist_ok=True)
    with open(COUNTS_CACHE, "w") as f:
        json.dump({"buckets": [list(b) for b in buckets], "counts": {str(k): v for k, v in counts.items()}}, f)
    return counts


def build_plan(counts, buckets, part_rows, out_root, total):
    """返回 tasks: [(shard_no, outputs)], outputs=[{bucket_i, part, out_dir, k, lo, hi}].
    lo/hi 为该 shard 桶内行列表的局部切片 [lo, hi)."""
    plan = {}
    for bi, (lo, hi) in enumerate(buckets):
        name = bucket_dirname(lo, hi)
        prefix = []
        c = 0
        for s in range(total):
            prefix.append(c)
            c += counts[s][bi]
        n_parts = (c + part_rows - 1) // part_rows
        print(f"[plan] bucket {lo}-{hi if hi is not None else '+'}: rows={c:,} parts={n_parts}", flush=True)
        for s in range(total):
            start = prefix[s]
            n = counts[s][bi]
            if n == 0:
                continue
            end = start + n
            p0, p1 = start // part_rows, (end - 1) // part_rows
            for p in range(p0, p1 + 1):
                glo = max(start, p * part_rows)
                ghi = min(end, (p + 1) * part_rows)
                if ghi <= glo:
                    continue
                plan.setdefault((bi, p), {})[s] = (glo - start, ghi - start)
    # part 内 shard 顺序编号 k
    tasks = {}
    for (bi, p), shard_slices in plan.items():
        lo_b, hi_b = BUCKETS[bi]
        out_dir = os.path.join(out_root, f"{bucket_dirname(lo_b, hi_b)}_part{p:03d}")
        for k, s in enumerate(sorted(shard_slices)):
            slo, shi = shard_slices[s]
            tasks.setdefault(s, []).append({"bucket_i": bi, "part": p, "k": k,
                                            "out_dir": out_dir, "lo": slo, "hi": shi})
    return tasks


def convert_shard(task):
    shard_no, outputs, src, qf, total = task
    data_path = os.path.join(src, f"train-{shard_no:06d}-of-{total:06d}.parquet")
    if not os.path.exists(data_path):
        return (shard_no, [], "missing")
    qf_path = os.path.join(qf, os.path.basename(data_path))

    t0 = time.time()
    # 完整性检查 (三件齐无 tmp)
    todo = []
    for o in outputs:
        stem = f"shard-{o['k']:05d}"
        paths = [os.path.join(o["out_dir"], stem + s) for s in (".tar", ".idx")] + \
                [os.path.join(o["out_dir"], "parquet", stem + ".parquet")]
        if all(os.path.exists(p) for p in paths) and not any(
                os.path.exists(p + ".tmp") for p in paths):
            continue
        todo.append(o)
    done_rows = 0
    if not todo:
        return (shard_no, [(o["out_dir"], o["k"], -1) for o in outputs], "skip")

    qt = pq.read_table(qf_path, columns=["decode_ok", "aesthetic_score"])
    dec = qt.column("decode_ok").to_pylist()
    aes = qt.column("aesthetic_score").to_pylist()
    bucket_rows = {}  # bucket_i -> [row_idx]
    for o in todo:
        bi = o["bucket_i"]
        if bi in bucket_rows:
            continue
        lo, hi = BUCKETS[bi]
        bucket_rows[bi] = [i for i, (d, a) in enumerate(zip(dec, aes))
                           if d and a is not None and a >= lo and (hi is None or a < hi)]

    pf = pq.ParquetFile(data_path)
    names = set(pf.schema_arrow.names)
    cols = [c for c in ("image_caption", "img_size_wh", "image",
                        "object_captions", "normalized_boxes") if c in names]
    # 需要 row group: 行号 -> rg (基于 metadata 累计)
    wanted = set()
    for o in todo:
        rows = bucket_rows[o["bucket_i"]][o["lo"]:o["hi"]]
        wanted.update(rows)
    rg_starts = []
    c = 0
    for r in range(pf.metadata.num_row_groups):
        rg_starts.append(c)
        c += pf.metadata.row_group(r).num_rows
    need_rgs = set()
    for row in wanted:
        # 二分 row group
        lo_, hi_ = 0, len(rg_starts) - 1
        while lo_ < hi_:
            mid = (lo_ + hi_ + 1) // 2
            if rg_starts[mid] <= row:
                lo_ = mid
            else:
                hi_ = mid - 1
        need_rgs.add(lo_)

    # writers
    writers = {}
    for o in todo:
        bi = o["bucket_i"]
        rows_slice = bucket_rows[bi][o["lo"]:o["hi"]]
        o["_rows"] = rows_slice
        o["_pos"] = 0
        writers[(o["bucket_i"], o["part"])] = TarPartWriter(o["out_dir"], o["k"])

    n_written = 0
    for rg in sorted(need_rgs):
        tbl = pf.read_row_group(rg, columns=cols)
        base = rg_starts[rg]
        caps = tbl.column("image_caption").to_pylist()
        sizes = tbl.column("img_size_wh").to_pylist()
        imgs = tbl.column("image").to_pylist()
        objs = (tbl.column("object_captions").to_pylist()
                if "object_captions" in names else [None] * tbl.num_rows)
        boxes = (tbl.column("normalized_boxes").to_pylist()
                 if "normalized_boxes" in names else [None] * tbl.num_rows)
        for j in range(tbl.num_rows):
            row_idx = base + j
            for o in todo:
                pos = o["_pos"]
                rows = o["_rows"]
                if pos >= len(rows):
                    continue
                if rows[pos] != row_idx:
                    continue
                o["_pos"] = pos + 1
                blob = imgs[j].get("bytes") if isinstance(imgs[j], dict) else None
                if not blob:
                    continue
                cap = caps[j]
                if not isinstance(cap, str) or not cap.strip():
                    continue
                lo_, hi_ = BUCKETS[o["bucket_i"]]
                sid = (f"bpg55m_{bucket_dirname(lo_, hi_)}_{o['part']:03d}"
                       f"_{shard_no:05d}_{row_idx:08d}")
                wh = sizes[j] if sizes[j] else (None, None)
                meta = {
                    "id": sid,
                    "source": "BreakpointGrounding55M",
                    "caption": cap,
                    "w": int(wh[0]) if wh[0] else None,
                    "h": int(wh[1]) if wh[1] else None,
                    "aesthetic_score": aes[row_idx],
                    "object_captions": objs[j],
                    "normalized_boxes": boxes[j],
                }
                writers[(o["bucket_i"], o["part"])].add(sid + "__img_bytes.jpg", blob, meta)
                n_written += 1

    results = []
    for w in writers.values():
        w.close()
        results.append((w.out_dir, w.k, w.n))
    return (shard_no, results, f"ok {time.time() - t0:.0f}s")


BUCKETS = []


def main():
    global BUCKETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/nfs/data/zhanqian.wu/unified_canvas_images/breakpoint-grounding-55m/data")
    ap.add_argument("--qf", default="/mnt/nfs/data/zhanqian.wu/unified_canvas_images/breakpoint-grounding-55m/quality_filter")
    ap.add_argument("--out-root", default="/home/jovyan/pfs/data/public/image_data")
    ap.add_argument("--buckets", nargs="+", default=["5.25-5.75", "5.75+"])
    ap.add_argument("--part-rows", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only-parts", default=None, help="如 aes575:0-2 或 aes525_575:3,逗号分隔多组")
    ap.add_argument("--max-shards", type=int, default=None, help="试点: 只处理前 N 个 shard")
    args = ap.parse_args()

    BUCKETS = [parse_bucket(b) for b in args.buckets]
    total = 2746

    counts = scan_counts(args.src, args.qf, BUCKETS, total)
    tasks_map = build_plan(counts, BUCKETS, args.part_rows, args.out_root, total)

    if args.only_parts:
        want = set()
        for grp in args.only_parts.split(","):
            name, ps = grp.split(":")
            if "-" in ps:
                a, b = ps.split("-")
                parts = range(int(a), int(b) + 1)
            else:
                parts = [int(x) for x in ps.split("+")]
            for bi, (lo, hi) in enumerate(BUCKETS):
                if bucket_dirname(lo, hi) == name:
                    want.update((bi, p) for p in parts)
        tasks_map = {s: [o for o in outs if (o["bucket_i"], o["part"]) in want]
                     for s, outs in tasks_map.items()
                     if any((o["bucket_i"], o["part"]) in want for o in outs)}

    for (bi, p) in sorted({(o["bucket_i"], o["part"]) for outs in tasks_map.values() for o in outs}):
        lo, hi = BUCKETS[bi]
        d = os.path.join(args.out_root, f"{bucket_dirname(lo, hi)}_part{p:03d}")
        os.makedirs(os.path.join(d, "parquet"), exist_ok=True)

    tasks = [(s, outs, args.src, args.qf, total) for s, outs in sorted(tasks_map.items())]
    if args.max_shards:
        tasks = tasks[:args.max_shards]
    print(f"[run] tasks={len(tasks)} workers={args.workers} "
          f"parts={sorted({(o['bucket_i'], o['part']) for outs in tasks_map.values() for o in outs})}",
          flush=True)

    n_rows = 0
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (s, results, status) in enumerate(pool.imap_unordered(convert_shard, tasks)):
            for out_dir, k, n in results:
                if n > 0:
                    n_rows += n
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                el = time.time() - t0
                print(f"[{i + 1}/{len(tasks)}] rows={n_rows:,} {el:.0f}s "
                      f"({n_rows / max(el, 1) / 1e3:.1f}k rows/s)", flush=True)
    print(f"[done] rows={n_rows:,} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
