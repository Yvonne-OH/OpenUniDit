#!/usr/bin/env python
"""按 enchanced_prompt/ 外部文件统一匹配回写 parquet / mdb。

读取 {lmdb_root}/{source}/enchanced_prompt/{分片stem}.jsonl (generate_prompts.py 的输出),
仅取 status=ok 的记录, 回写:
- parquet: 新增/覆盖 other_format_prompt 列 (JSON 字符串), caption 保留不动;
  原子替换 (写 .tmp 后 rename)。
- mdb: 原地改写记录, 新增 other_format_prompt 字段 (dict), caption 保留不动;
  分批 commit。⚠️ 执行前确认训练进程未在读取该 mdb。

默认 dry-run 只统计匹配情况; 加 --apply 才真正写盘。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from lmdb_index import probe_key_scheme, shard_mdb_map  # noqa: E402


def load_jsonl(path: Path) -> dict[int, dict]:
    """读取单分片 jsonl, 返回 {sample_id: {prompt, metrics}} (仅 status=ok)。"""
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok" and isinstance(rec.get("enhanced_prompt"), dict):
                out[int(rec["sample_id"])] = {
                    "prompt": rec["enhanced_prompt"],
                    "metrics": rec.get("metrics"),
                }
    return out


def compact_json(prompt: dict) -> str:
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))


def _set_or_append(table: pa.Table, col_name: str, values: list) -> pa.Table:
    if col_name in table.schema.names:
        return table.set_column(table.schema.get_field_index(col_name),
                                col_name, pa.array(values, type=pa.string()))
    return table.append_column(pa.field(col_name, pa.string()),
                               pa.array(values, type=pa.string()))


def apply_parquet(shard_path: Path, prompts: dict[int, dict], dry_run: bool) -> tuple[int, int]:
    """新增/覆盖 other_format_prompt 与 metrics 两列, 返回 (命中, 未命中)。"""
    table = pq.read_table(shard_path)
    ids = table.column("sample_id").to_pylist()
    hits = [(sid in prompts) for sid in ids]
    matched = sum(hits)
    if dry_run or not matched:
        return matched, len(ids) - matched

    prompt_col = [compact_json(prompts[sid]["prompt"]) if sid in prompts else None for sid in ids]
    metrics_col = [compact_json(prompts[sid]["metrics"]) if sid in prompts
                   and prompts[sid].get("metrics") is not None else None for sid in ids]
    table = _set_or_append(table, "other_format_prompt", prompt_col)
    table = _set_or_append(table, "metrics", metrics_col)
    fd, tmp = tempfile.mkstemp(dir=str(shard_path.parent), suffix=".parquet.tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, shard_path)  # 原子替换
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return matched, len(ids) - matched


def apply_mdb(mdb_path: Path, prompts: dict[int, dict],
              dry_run: bool) -> tuple[int, int, int]:
    """原地改写 mdb 记录 (单分片单事务), 返回 (命中, 未找到, 改写)。"""
    import lmdb

    matched = missing = written = 0
    # 写模式: map_size 预留 20% 余量 (value 变长会扩页); 只读模式 map_size 无关紧要
    map_size = 0 if dry_run else int(mdb_path.stat().st_size * 1.2) + (1 << 30)
    env = lmdb.open(str(mdb_path), readonly=dry_run, subdir=False, map_size=map_size)
    try:
        with env.begin(write=not dry_run, buffers=True) as txn:
            for sid in prompts:
                item = prompts[sid]
                key = probe_key_scheme(txn, sid)
                if key is None:
                    missing += 1
                    continue
                matched += 1
                if dry_run:
                    continue
                rec = pickle.loads(txn.get(key))
                rec["other_format_prompt"] = item["prompt"]
                if item.get("metrics") is not None:
                    rec["metrics"] = item["metrics"]
                txn.put(key, pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
                written += 1
    finally:
        env.close()
    return matched, missing, written


def main():
    ap = argparse.ArgumentParser(description="按外部文件回写 other_format_prompt (默认 dry-run)")
    ap.add_argument("--lmdb_root", default="/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_images_lmdb")
    ap.add_argument("--source", action="append", default=None, help="限定 source, 默认扫描全部")
    ap.add_argument("--mdb", action="store_true", help="同时原地改写 mdb")
    ap.add_argument("--apply", action="store_true", help="真正写盘 (否则仅统计)")
    args = ap.parse_args()

    root = Path(args.lmdb_root)
    source_dirs = sorted(d for d in root.iterdir() if (d / "enchanced_prompt").is_dir())
    if args.source:
        allow = set(args.source)
        source_dirs = [d for d in source_dirs if d.name in allow]
    if not source_dirs:
        sys.exit("❌ 未找到任何 enchanced_prompt/ 产出 (先跑 generate_prompts.py)")

    mode = "APPLY" if args.apply else "DRY-RUN"
    grand = {"parquet_matched": 0, "parquet_missed": 0, "mdb_matched": 0,
             "mdb_missing": 0, "mdb_written": 0, "failed_jsonl": 0}
    t0 = time.time()
    for source_dir in source_dirs:
        jsonl_files = sorted((source_dir / "enchanced_prompt").glob("*.jsonl"))
        if not jsonl_files:
            continue
        print(f"📁 {source_dir.name}: {len(jsonl_files)} 个分片产出", flush=True)
        mdb_map = shard_mdb_map(source_dir) if args.mdb else {}
        for jf in jsonl_files:
            prompts = load_jsonl(jf)
            shard_path = source_dir / "parquet" / f"{jf.stem}.parquet"
            if not shard_path.exists():
                print(f"  ⚠️ {jf.stem}: parquet 分片不存在, 跳过", flush=True)
                continue
            m, miss = apply_parquet(shard_path, prompts, dry_run=not args.apply)
            grand["parquet_matched"] += m
            grand["parquet_missed"] += miss
            line = f"  {jf.stem}: prompt={len(prompts):,} parquet命中={m:,} 未命中={miss:,}"
            if args.mdb:
                mdb_path = mdb_map.get(jf.stem)
                if mdb_path is None:
                    line += " | mdb: 无对应分片, 跳过"
                else:
                    mm, mmiss, mw = apply_mdb(mdb_path, prompts,
                                              dry_run=not args.apply)
                    grand["mdb_matched"] += mm
                    grand["mdb_missing"] += mmiss
                    grand["mdb_written"] += mw
                    line += f" | mdb命中={mm:,} 未找到={mmiss:,} 改写={mw:,}"
            print(line, flush=True)

    print(f"\n{'✅' if args.apply else '🔍'} [{mode}] 汇总 "
          f"({time.time() - t0:.0f}s): {json.dumps(grand, ensure_ascii=False)}", flush=True)
    if not args.apply:
        print("确认无误后加 --apply 真正写盘; 加 --mdb 连同 mdb 原地改写。", flush=True)


if __name__ == "__main__":
    main()
