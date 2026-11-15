#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tar_parquet_t2i.py — LMDB edit 库 parquet_t2i 的 t2i_caption 合并进 tar edit 库

为每个 tar edit 源生成 parquet_t2i/<tar分片stem>.parquet = 原 tar parquet 行 + t2i_caption 列,
使 TarParquetDataset 能以 t2i 模式读 edit 库 (before 原图 + Qwen3-VL 描述 caption 作普通 image 样本)。

join key (与 tools/lmdb_to_tar.py 的 id 编码对应):
    tar parquet 行 id = "{env_stem}_{key_int}"  (key_int = int.from_bytes(lmdb_key, "big"))
    LMDB 读取侧 key   = struct.pack("<Q", sample_id)  (小端)
    ⇒ sid = int.from_bytes(key_int.to_bytes(8, "big"), "little") ↔ parquet_t2i/meta_{env_stem}.parquet 的 sample_id

合并前对每个源抽样比对 tar 行与 LMDB parquet_t2i 行的 edit_prompt/w_before, 不一致即中止
(防 key scheme 变化导致错配)。

用法:
    python3 tools/build_tar_parquet_t2i.py                    # 默认 7 个单图对源
    python3 tools/build_tar_parquet_t2i.py --datasets HQ-Edit # 指定源
    python3 tools/build_tar_parquet_t2i.py --skip-verify      # 跳过抽样比对
    python3 tools/build_tar_parquet_t2i.py --force            # 已存在的 parquet_t2i 分片重写
"""

import argparse
import glob
import os
import random
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_edit_images_lmdb"
DST_DEFAULT = "/mnt/nfs/data/zhanqian.wu/unified_edit_images_tar"

# 单图对源 (与 stage_1_xl.yaml t2i_edit_sources_t2i_mode 一致; 多参考图源 unipic_*/Macro 不支持)
DEFAULT_DATASETS = [
    "HQ-Edit", "GPT-UltraEdit", "GPT-HQEdit", "pico-banana-400k",
    "GPT-OmniEdit", "CrispEdit", "SEED-Data-Edit",
]

VERIFY_SAMPLES_PER_DS = 10      # 抽样 shard 数
VERIFY_ROWS_PER_SHARD = 20      # 每 shard 抽样行数


def say(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tar_id_to_sid(rid: str) -> tuple[str, int]:
    """tar parquet 行 id "{env_stem}_{key_int}" → (env_stem, lmdb sample_id)。"""
    env_stem, key_str = str(rid).rsplit("_", 1)
    key_int = int(key_str)
    sid = int.from_bytes(key_int.to_bytes(8, "big"), "little")
    return env_stem, sid


class CaptionStore:
    """env_stem → {sample_id: t2i_caption} 惰性缓存 (tar 分片按 env 顺序转换, 命中率高)。"""

    def __init__(self, t2i_dir: Path, max_envs: int = 8):
        self.t2i_dir = Path(t2i_dir)
        self.max_envs = max_envs
        self._cache: OrderedDict[str, dict[int, str]] = OrderedDict()
        self._lock = threading.Lock()

    def _load(self, env_stem: str) -> dict[int, str]:
        path = self.t2i_dir / f"meta_{env_stem}.parquet"
        if not path.exists():
            return {}
        table = pq.read_table(path, columns=["sample_id", "t2i_caption"])
        out = {}
        for sid, cap in zip(table.column("sample_id").to_pylist(),
                            table.column("t2i_caption").to_pylist()):
            if isinstance(cap, str) and cap.strip():
                out[int(sid)] = cap
        return out

    def get(self, env_stem: str, sids: set[int]) -> dict[int, str]:
        with self._lock:
            if env_stem not in self._cache:
                self._cache[env_stem] = self._load(env_stem)
                while len(self._cache) > self.max_envs:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(env_stem)
            data = self._cache[env_stem]
        return {sid: data[sid] for sid in sids if sid in data}


def process_shard(pq_path: Path, t2i_out_dir: Path, store: CaptionStore,
                  force: bool) -> tuple[str, int, int, int]:
    """单个 tar parquet 分片 → parquet_t2i/<同名>.parquet。返回 (状态, 总行, 命中, 校验不一致)。"""
    out_path = t2i_out_dir / pq_path.name
    if out_path.exists() and not force:
        return ("skip", 0, 0, 0)
    table = pq.read_table(pq_path)
    names = set(table.schema.names)
    if "id" not in names:
        return ("no_id", 0, 0, 0)
    rows = table.column("id").to_pylist()
    parsed = []
    env_sids: dict[str, set[int]] = defaultdict(set)
    for rid in rows:
        try:
            env_stem, sid = tar_id_to_sid(rid)
        except (ValueError, OverflowError):
            parsed.append(None)
            continue
        parsed.append((env_stem, sid))
        env_sids[env_stem].add(sid)

    caps: list[str | None] = [None] * len(rows)
    hit = 0
    for env_stem, sids in env_sids.items():
        found = store.get(env_stem, sids)
        for i, item in enumerate(parsed):
            if item and item[0] == env_stem and item[1] in found:
                caps[i] = found[item[1]]
                hit += 1

    if out_path.exists() and force:
        out_path.unlink()
    t2i_out_dir.mkdir(parents=True, exist_ok=True)
    table = table.append_column("t2i_caption", pa.array(caps, type=pa.string()))
    tmp = out_path.with_name(out_path.name + ".part")
    pq.write_table(table, tmp)
    os.replace(tmp, out_path)
    return ("ok", len(rows), hit, 0)


def _prompt_column(schema_names) -> str:
    """edit prompt 列名自适应: 多数源为 edit_prompt, SEED-Data-Edit 等为 instruction。"""
    for col in ("edit_prompt", "instruction"):
        if col in schema_names:
            return col
    raise KeyError(f"parquet 无 edit_prompt/instruction 列: {sorted(schema_names)}")


def verify_dataset(src_ds: Path, dst_ds: Path, sample_shards: int) -> int:
    """独立抽样校验: tar 行 join 到 LMDB parquet_t2i 后 prompt/w_before 必须一致。"""
    pq_files = sorted(glob.glob(str(dst_ds / "parquet" / "*.parquet")))
    if not pq_files:
        say(f"  ⚠️ {dst_ds.name}: parquet/ 为空, 跳过校验")
        return 0
    random.seed(0)
    bad = checked = 0
    for pq_path in random.sample(pq_files, min(sample_shards, len(pq_files))):
        prompt_col = _prompt_column(pq.ParquetFile(pq_path).schema_arrow.names)
        table = pq.read_table(pq_path, columns=["id", prompt_col, "w_before"])
        rows = table.to_pylist()
        picks = random.sample(rows, min(VERIFY_ROWS_PER_SHARD, len(rows)))
        # 按 env 分组, 每个 env 的 LMDB parquet_t2i 只读一次
        by_env: dict[str, dict[int, tuple]] = defaultdict(dict)
        for row in picks:
            try:
                env_stem, sid = tar_id_to_sid(row["id"])
            except (ValueError, OverflowError):
                continue
            by_env[env_stem][sid] = (row[prompt_col], row["w_before"])
        for env_stem, want in by_env.items():
            ref_path = src_ds / "parquet_t2i" / f"meta_{env_stem}.parquet"
            if not ref_path.exists():
                bad += len(want)
                say(f"  ❌ 缺 meta_{env_stem}.parquet ({len(want)} 行未校验)")
                continue
            ref_prompt = _prompt_column(pq.ParquetFile(ref_path).schema_arrow.names)
            ref = pq.read_table(ref_path, columns=["sample_id", ref_prompt, "w_before"])
            ref_map = {
                int(s): (p, w) for s, p, w in zip(
                    ref.column("sample_id").to_pylist(),
                    ref.column(ref_prompt).to_pylist(),
                    ref.column("w_before").to_pylist())
            }
            for sid, (tar_p, tar_w) in want.items():
                if sid not in ref_map:
                    bad += 1
                    say(f"  ❌ MISS {dst_ds.name}/{env_stem} sid={sid}")
                    continue
                checked += 1
                if ref_map[sid] != (tar_p, tar_w):
                    bad += 1
                    say(f"  ❌ MISMATCH {dst_ds.name}/{env_stem} sid={sid}")
    say(f"  校验: checked={checked} bad={bad}")
    return bad


def merge_dataset(ds: str, args) -> dict:
    src_ds = Path(args.src) / ds
    dst_ds = Path(args.dst) / ds
    t0 = time.time()
    pq_files = sorted(glob.glob(str(dst_ds / "parquet" / "*.parquet")))
    if not pq_files:
        say(f"[{ds}] 跳过: tar parquet/ 为空")
        return {"done": False}
    if not args.skip_verify:
        say(f"[{ds}] 抽样校验 join key ...")
        bad = verify_dataset(src_ds, dst_ds, VERIFY_SAMPLES_PER_DS)
        if bad:
            say(f"[{ds}] ❌ 校验不一致 {bad} 条, 中止该源 (key scheme 可能不匹配)")
            return {"done": False, "bad": bad}

    t2i_out_dir = dst_ds / "parquet_t2i"
    n_files = len(glob.glob(str(t2i_out_dir / "*.parquet")))
    if n_files == len(pq_files) and not args.force:
        say(f"[{ds}] 已完成 ({n_files}/{len(pq_files)} 分片), 跳过 (--force 重写)")
        return {"done": True, "skipped": True}

    store = CaptionStore(src_ds / "parquet_t2i")
    total = hit = mismatch = done = 0
    t2i_out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.shard_workers) as pool:
        futs = {pool.submit(process_shard, Path(p), t2i_out_dir, store,
                            args.force): Path(p) for p in pq_files}
        for i, fut in enumerate(as_completed(futs), 1):
            status, n, h, m = fut.result()
            total += n
            hit += h
            mismatch += m
            if status == "ok":
                done += 1
            if i % 500 == 0 or i == len(futs):
                say(f"[{ds}] {i}/{len(futs)} 分片 行={total:,} 命中={hit:,} "
                    f"({100.0 * hit / max(total, 1):.1f}%)")
    final = len(glob.glob(str(t2i_out_dir / "*.parquet")))
    say(f"[{ds}] DONE 分片={final}/{len(pq_files)} 行={total:,} caption命中={hit:,} "
        f"缺失={total - hit:,} 用时 {(time.time() - t0) / 60:.1f}min")
    if final != len(pq_files):
        say(f"[{ds}] ❌ 分片数不齐 ({final}/{len(pq_files)})")
        return {"done": False}
    return {"done": True, "rows": total, "hit": hit}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT, help="LMDB edit 库根目录")
    ap.add_argument("--dst", default=DST_DEFAULT, help="tar edit 库根目录")
    ap.add_argument("--datasets", default="", help="逗号分隔, 默认 7 个单图对源")
    ap.add_argument("--shard-workers", type=int, default=8, help="每源分片并行线程数")
    ap.add_argument("--ds-workers", type=int, default=min(4, len(DEFAULT_DATASETS)),
                    help="并行数据集数 (进程)")
    ap.add_argument("--force", action="store_true", help="重写已存在的 parquet_t2i 分片")
    ap.add_argument("--skip-verify", action="store_true", help="跳过抽样比对")
    args = ap.parse_args()

    want = [x.strip() for x in args.datasets.split(",") if x.strip()] or DEFAULT_DATASETS
    say(f"datasets={want} dst={args.dst} ds_workers={args.ds_workers} "
        f"shard_workers={args.shard_workers}")
    if args.ds_workers <= 1:
        results = {ds: merge_dataset(ds, args) for ds in want}
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.ds_workers) as ex:
            futs = {ex.submit(merge_dataset, ds, args): ds for ds in want}
            results = {}
            for fut, ds in futs.items():
                try:
                    results[ds] = fut.result()
                except Exception as e:
                    say(f"[{ds}] 失败: {e!r}")
                    import traceback
                    traceback.print_exc()
                    results[ds] = {"done": False}
    failed = [ds for ds, r in results.items() if not r.get("done")]
    rows = sum(r.get("rows", 0) for r in results.values())
    hit = sum(r.get("hit", 0) for r in results.values())
    say(f"ALL_DONE datasets={len(want)} 失败={failed or '无'} "
        f"总行={rows:,} 命中={hit:,}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
