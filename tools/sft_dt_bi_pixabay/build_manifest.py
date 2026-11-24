#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_manifest.py — 合并 4 个爬虫源 metadata.jsonl → 去重后的图像 manifest.parquet。

读取 /mnt/nfs/data/zhanqian.wu/unified_images/{duitang,pixabay,bilibili,tuchong}/metadata.jsonl,
取 status=="saved" 且文件存在的条目, 按 sha256 全局去重 (同一图多 query/多平台只保留首个),
排序 (platform, query, sha256) 保证确定性, 输出 manifest.parquet:
    sha256 | platform | query | file

用法: python3 build_manifest.py [--out manifest.parquet]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC_ROOT = Path("/mnt/nfs/data/zhanqian.wu/unified_images")
PLATFORMS = ("duitang", "pixabay", "bilibili", "tuchong")
OUT_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/exp/sft_dt_bi_pixabay/manifest.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", default=str(SRC_ROOT))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--skip-exists-check", action="store_true",
                    help="跳过文件存在性检查 (NFS stat 慢时可关)")
    args = ap.parse_args()

    t0 = time.time()
    rows = []          # (platform, query, sha256, file)
    seen = set()
    stats = {}
    for plat in PLATFORMS:
        meta = Path(args.src_root) / plat / "metadata.jsonl"
        n_line = n_saved = n_missing = n_dup = 0
        with open(meta) as f:
            for line in f:
                if not line.strip():
                    continue
                n_line += 1
                o = json.loads(line)
                if o.get("status") != "saved":
                    continue
                fp = o.get("file") or ""
                sha = o.get("sha256") or ""
                if not fp or not sha:
                    continue
                n_saved += 1
                if sha in seen:
                    n_dup += 1
                    continue
                if not args.skip_exists_check and not os.path.isfile(fp):
                    n_missing += 1
                    continue
                seen.add(sha)
                rows.append((plat, o.get("query", ""), sha, fp))
        stats[plat] = dict(lines=n_line, saved=n_saved, dup=n_dup, missing=n_missing)
        print(f"[{plat}] lines={n_line:,} saved={n_saved:,} "
              f"dup={n_dup:,} missing={n_missing:,}", flush=True)

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "platform": pa.array([r[0] for r in rows], pa.string()),
        "query": pa.array([r[1] for r in rows], pa.string()),
        "sha256": pa.array([r[2] for r in rows], pa.string()),
        "file": pa.array([r[3] for r in rows], pa.string()),
    })
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, out)
    print(f"\n✅ manifest: {len(rows):,} 张唯一图像 -> {out}")
    print(f"   耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
