#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_lmdb.py — manifest + captions → unified_images_lmdb/sft_dt_bi_pixabay (JourneyDB 格式)。

输出 (OUT=/mnt/nfs/data/zhanqian.wu/unified_images_lmdb/sft_dt_bi_pixabay):
  mdb/sft_dt_bi_pixabay_{NNNN}.mdb      key=struct.pack("<Q", sid)
                                       value=pickle({id, source, platform, query, caption,
                                                                     w, h, img_bytes})
  parquet/meta_{NNNN}.parquet           与 mdb 分片按 sorted 顺序一一对应
  parquet/meta.parquet                  汇总
  grid/grid_{shard_start:08d}.png       每分片一张 (前 20 图 4x5 缩略)

mdb 先写 pfs stage (NFS mmap 随机写仅 ~3.7张/s), 再顺序拷贝到 NFS (journeydb v2 方案)。
sid 全局唯一, 按分片预分配区间 [i*per, (i+1)*per)。caption 缺失/解码失败/空 caption 的图剔除。

用法: python3 build_lmdb.py [--per-shard 20000]
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import pickle
import shutil
import struct
import time
from pathlib import Path

import lmdb
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

DATASET = "sft_dt_bi_pixabay"
SOURCE = "sft_dt_bi_pixabay"
MANIFEST_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/exp/sft_dt_bi_pixabay/manifest.parquet"
CAPTIONS_DIR_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/exp/sft_dt_bi_pixabay/captions"
OUT_DEFAULT = f"/mnt/nfs/data/zhanqian.wu/unified_images_lmdb/{DATASET}"
STAGE_DEFAULT = f"/home/jovyan/pfs/users/zhanqian.wu/exp/sft_dt_bi_pixabay/stage_mdb"

GRID_COLS, GRID_ROWS, CELL_W, CELL_H = 4, 5, 256, 256

PARQUET_SCHEMA = pa.schema([
    ("sample_id", pa.int64()),
    ("platform", pa.string()),
    ("query", pa.string()),
    ("file", pa.string()),
    ("caption", pa.string()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("file_size", pa.int64()),
])


def say(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_captions(cdir: str) -> dict[str, str]:
    cap = {}
    for p in sorted(glob.glob(os.path.join(cdir, "chunk_*.parquet"))):
        for r in pq.read_table(p).to_pylist():
            if r.get("caption"):
                cap[r["sha256"]] = r["caption"]
    return cap


def make_grid(thumbs: list) -> Image.Image:
    grid = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H), "white")
    for i, im in enumerate(thumbs[:GRID_COLS * GRID_ROWS]):
        t = im.copy()
        t.thumbnail((CELL_W, CELL_H))
        x, y = (i % GRID_COLS) * CELL_W, (i // GRID_COLS) * CELL_H
        grid.paste(t, (x + (CELL_W - t.width) // 2, y + (CELL_H - t.height) // 2))
    return grid


def publish_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)


_G_ROWS: list = []
_G_CAP: dict = {}
_G_ARGS = None


def build_shard(si: int) -> dict:
    """构建单个分片 (fork worker 调用, 数据经 _G_* 共享)。"""
    rows, cap_map, args = _G_ROWS, _G_CAP, _G_ARGS
    shard_rows = rows[si * args.per_shard:(si + 1) * args.per_shard]
    stem = f"{DATASET}_{si:04d}"
    sid_base = si * args.per_shard
    out = Path(args.out)
    stage = Path(args.stage)

    env_path = stage / f"{stem}.mdb"
    if env_path.exists():
        env_path.unlink()
    env = lmdb.open(str(env_path), subdir=False, map_size=args.map_size, writemap=True)
    pq_rows, thumbs = [], []
    st = dict(kept=0, nocap=0, decode_fail=0)
    t0 = time.time()
    with env.begin(write=True) as txn:
        for j, r in enumerate(shard_rows):
            sid = sid_base + j
            caption = cap_map.get(r["sha256"])
            if not caption:
                st["nocap"] += 1
                continue
            try:
                img_bytes = Path(r["file"]).read_bytes()
                with Image.open(io.BytesIO(img_bytes)) as im:
                    w, h = im.size
                    if len(thumbs) < GRID_COLS * GRID_ROWS:
                        thumbs.append(im.convert("RGB"))
            except Exception:
                st["decode_fail"] += 1
                continue
            val = pickle.dumps({
                "id": f"{stem}_{sid:08d}", "source": SOURCE,
                "platform": r["platform"], "query": r["query"],
                "caption": caption, "w": w, "h": h, "img_bytes": img_bytes,
            }, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(struct.pack("<Q", sid), val)
            pq_rows.append({"sample_id": sid, "platform": r["platform"],
                            "query": r["query"], "file": r["file"],
                            "caption": caption, "width": w, "height": h,
                            "file_size": len(img_bytes)})
            st["kept"] += 1
    env.sync()
    env.close()

    # pfs stage → NFS 发布
    publish_file(env_path, out / "mdb" / f"{stem}.mdb")
    env_path.unlink()
    tmp_pq = out / "parquet" / f".meta_{si:04d}.tmp"
    pq.write_table(pa.Table.from_pylist(pq_rows, schema=PARQUET_SCHEMA), tmp_pq)
    os.replace(tmp_pq, out / "parquet" / f"meta_{si:04d}.parquet")
    if thumbs:
        make_grid(thumbs).save(out / "grid" / f"grid_{sid_base:08d}.png")
    say(f"[{stem}] 发布 {st['kept']}/{len(shard_rows)} 图 ({time.time()-t0:.0f}s)")
    return st


def _run(si: int):
    return build_shard(si)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--captions-dir", default=CAPTIONS_DIR_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--stage", default=STAGE_DEFAULT)
    ap.add_argument("--per-shard", type=int, default=20000)
    ap.add_argument("--map-size", type=int, default=64 << 30)
    ap.add_argument("--shards", default="", help="逗号分隔分片序号, 只重建指定分片 (如 0,1)")
    ap.add_argument("--workers", type=int, default=4, help="分片级并行进程数 (fork)")
    args = ap.parse_args()

    t0 = time.time()
    rows = pq.read_table(args.manifest).to_pylist()
    cap_map = load_captions(args.captions_dir)
    say(f"manifest {len(rows):,} 图, caption {len(cap_map):,} 条")

    only_shards = {int(x) for x in args.shards.split(",") if x.strip()} if args.shards else None
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    (out / "parquet").mkdir(parents=True, exist_ok=True)
    (out / "grid").mkdir(parents=True, exist_ok=True)

    n_shards = (len(rows) + args.per_shard - 1) // args.per_shard
    todo = []
    stats = dict(kept=0, nocap=0, decode_fail=0)
    for si in range(n_shards):
        if only_shards is not None and si not in only_shards:
            continue
        if (out / "mdb" / f"{DATASET}_{si:04d}.mdb").exists() \
                and (out / "parquet" / f"meta_{si:04d}.parquet").exists():
            stats["kept"] += pq.ParquetFile(out / "parquet" / f"meta_{si:04d}.parquet").metadata.num_rows
            continue
        todo.append(si)
    say(f"待构建分片 {len(todo)}/{n_shards} (已发布跳过 {n_shards - len(todo)})")

    global _G_ROWS, _G_CAP, _G_ARGS
    _G_ROWS, _G_CAP, _G_ARGS = rows, cap_map, args

    if len(todo) > 1 and args.workers > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(min(args.workers, len(todo))) as pool:
            for st in pool.imap_unordered(_run, todo):
                for k, v in st.items():
                    if k == "kept":
                        stats[k] += v
                    else:
                        stats.setdefault(k, 0)
                        stats[k] += v
    else:
        for si in todo:
            for k, v in build_shard(si).items():
                if k == "kept":
                    stats[k] += v
                else:
                    stats.setdefault(k, 0)
                    stats[k] += v

    # 汇总 meta.parquet
    shards_pq = sorted((out / "parquet").glob("meta_*.parquet"))
    tables = [pq.read_table(p) for p in shards_pq]
    pq.write_table(pa.concat_tables(tables), out / "parquet" / "meta.parquet")

    n_final = sum(t.num_rows for t in tables)
    say(f"✅ DONE: 保留 {stats['kept']:,} (最终 parquet {n_final:,}) "
        f"/ nocap={stats['nocap']:,} / decode_fail={stats['decode_fail']:,} "
        f"分片 {len(shards_pq)} 个, 耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
