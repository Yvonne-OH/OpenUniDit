#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_dataset.py — 校验 sft_dt_bi_pixabay 的 lmdb 与 tar 两种格式。

lmdb 侧: mdb/parquet 分片数量一致, 每分片抽样 pickle 记录含 img_bytes/caption/w/h,
         parquet 行数 == mdb entries。
tar 侧:  parquet 行数 == meta.jsonl 行数 == idx 行数, 抽样 seek+read 校验图片字节
         可被 PIL 解码且宽高与 parquet 一致。
"""
from __future__ import annotations

import glob
import io
import json
import os
import pickle
import random
import struct
import sys
from pathlib import Path

import lmdb
import pyarrow.parquet as pq
from PIL import Image

LMDB = Path("/mnt/nfs/data/zhanqian.wu/unified_images_lmdb/sft_dt_bi_pixabay")
TAR = Path("/mnt/nfs/data/zhanqian.wu/unified_images_tar/sft_dt_bi_pixabay")
if "--lmdb" in sys.argv:
    LMDB = Path(sys.argv[sys.argv.index("--lmdb") + 1])
if "--tar" in sys.argv:
    TAR = Path(sys.argv[sys.argv.index("--tar") + 1])
N_SAMPLE = 32
random.seed(42)


def ok(msg):
    print(f"  ✅ {msg}", flush=True)


def fail(msg):
    print(f"  ❌ {msg}", flush=True)
    sys.exit(1)


def verify_lmdb():
    print("== LMDB ==")
    mdbs = sorted(glob.glob(str(LMDB / "mdb" / "*.mdb")))
    pqs = sorted(glob.glob(str(LMDB / "parquet" / "meta_*.parquet")))
    if len(mdbs) != len(pqs):
        fail(f"分片数不一致 mdb={len(mdbs)} parquet={len(pqs)}")
    ok(f"{len(mdbs)} 分片")
    total = 0
    for mf, pf in zip(mdbs, pqs):
        env = lmdb.open(mf, readonly=True, lock=False, subdir=False, max_readers=1024)
        n = env.stat()["entries"]
        nrow = pq.ParquetFile(pf).metadata.num_rows
        if n != nrow:
            fail(f"{mf}: mdb entries={n} != parquet rows={nrow}")
        # 抽样一条
        t = pq.read_table(pf)
        row = t.slice(0, 1).to_pylist()[0]
        with env.begin() as txn:
            raw = txn.get(struct.pack("<Q", row["sample_id"]))
            if raw is None:
                fail(f"{mf}: sample_id={row['sample_id']} 不存在")
            rec = pickle.loads(raw)
            for k in ("img_bytes", "caption", "w", "h", "source"):
                if k not in rec:
                    fail(f"{mf}: 记录缺字段 {k}")
            with Image.open(io.BytesIO(rec["img_bytes"])) as im:
                if im.size != (row["width"], row["height"]):
                    fail(f"{mf}: 宽高不一致 {im.size} vs ({row['width']},{row['height']})")
            if not rec["caption"].strip():
                fail(f"{mf}: 空 caption")
        env.close()
        total += n
    meta = pq.ParquetFile(LMDB / "parquet" / "meta.parquet").metadata.num_rows
    if meta != total:
        fail(f"meta.parquet {meta} != 分片合计 {total}")
    ok(f"总行数 {total:,} (meta.parquet 一致); 抽样记录/img/caption/宽高 全部通过")


def verify_tar():
    print("== TAR ==")
    tars = sorted(glob.glob(str(TAR / "shard-*.tar")))
    if not tars:
        fail("无 shard-*.tar")
    total = 0
    for tp in tars:
        stem = Path(tp).stem
        idx = TAR / f"{stem}.idx"
        meta = TAR / f"{stem}.meta.jsonl"
        pqp = TAR / "parquet" / f"{stem}.parquet"
        for p in (idx, meta, pqp):
            if not p.exists():
                fail(f"{stem}: 缺 {p.name}")
        n_idx = sum(1 for _ in open(idx))
        n_meta = sum(1 for _ in open(meta))
        n_pq = pq.ParquetFile(pqp).metadata.num_rows
        if not (n_idx == n_meta == n_pq):
            fail(f"{stem}: idx={n_idx} meta={n_meta} parquet={n_pq} 不一致")
        total += n_pq
    ok(f"{len(tars)} 分片, 总行数 {total:,}, idx/meta/parquet 行数一致")

    # 抽样 seek+read 校验
    rows = []
    for tp in tars:
        stem = Path(tp).stem
        t = pq.read_table(TAR / "parquet" / f"{stem}.parquet", columns=["id", "caption", "w", "h", "images"])
        for r in t.to_pylist():
            rows.append((tp, r))
    if not rows:
        fail("tar parquet 无行")
    sample = random.sample(rows, min(N_SAMPLE, len(rows)))
    for tp, r in sample:
        blob = r["images"][0]
        with open(tp, "rb") as f:
            f.seek(blob["off"])
            data = f.read(blob["size"])
        with Image.open(io.BytesIO(data)) as im:
            if im.size != (r["w"], r["h"]):
                fail(f"{r['id']}: 宽高 {im.size} != ({r['w']},{r['h']})")
        if not r["caption"].strip():
            fail(f"{r['id']}: 空 caption")
    ok(f"随机抽 {len(sample)} 行 seek+read 解码/宽高/caption 全部通过")


if __name__ == "__main__":
    if LMDB.is_dir():
        verify_lmdb()
    else:
        print(f"⚠️ {LMDB} 不存在, 跳过 lmdb 校验")
    if TAR.is_dir():
        verify_tar()
    else:
        print(f"⚠️ {TAR} 不存在, 跳过 tar 校验")
    print("\n🎉 校验全部通过")
