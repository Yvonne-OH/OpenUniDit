"""unified_images_lmdb 分片索引共享模块。

约定 (对齐 data/datasets/image_database.py 的训练读取逻辑):
- 每个 source 目录下 parquet/ 与 mdb/ 子目录;
- parquet 分片与 mdb 文件按 sorted 顺序一一对应 (zip 索引);
- mdb key: 8 字节小端 uint64 (struct.pack("<Q", sid)) 或 ascii 十进制字符串, 两种 scheme 探测;
- 训练用 glob 优先 meta_*.parquet, 无则取除 meta.parquet 外的全部分片。
"""
from __future__ import annotations

import struct
from pathlib import Path


def list_shards(lmdb_root: str, sources: list[str]) -> list[Path]:
    """枚举各 source 的 parquet 分片 (跳过汇总 meta.parquet)。"""
    root = Path(lmdb_root)
    shards: list[Path] = []
    for source in sources:
        parquet_dir = root / source / "parquet"
        if not parquet_dir.is_dir():
            print(f"⚠️ 跳过不存在的 source: {parquet_dir}", flush=True)
            continue
        files = sorted(parquet_dir.glob("meta_*.parquet"))
        if not files:  # blip3o 等自定义命名
            files = sorted(p for p in parquet_dir.glob("*.parquet") if p.name != "meta.parquet")
        if not files:
            print(f"⚠️ source 无分片: {parquet_dir}", flush=True)
        shards.extend(files)
    return shards


def shard_mdb_map(source_dir: Path) -> dict[str, Path]:
    """按 sorted 顺序 zip, 返回 {parquet 分片 stem: mdb 文件路径}。"""
    parquet_dir, mdb_dir = source_dir / "parquet", source_dir / "mdb"
    parquet_files = sorted(parquet_dir.glob("meta_*.parquet"))
    if not parquet_files:
        parquet_files = sorted(p for p in parquet_dir.glob("*.parquet") if p.name != "meta.parquet")
    mdb_files = sorted(mdb_dir.glob("*.mdb"))
    if not mdb_files:
        return {}
    if len(parquet_files) != len(mdb_files):
        raise RuntimeError(
            f"Parquet/LMDB 分片数量不一致: {source_dir} "
            f"parquet={len(parquet_files)} mdb={len(mdb_files)}"
        )
    return {pf.stem: mf for pf, mf in zip(parquet_files, mdb_files)}


def probe_key_scheme(txn, sample_id: int) -> bytes | None:
    """探测 mdb key 编码并返回命中 key, 未命中返回 None。"""
    for key in (struct.pack("<Q", int(sample_id)), str(sample_id).encode()):
        if txn.get(key) is not None:
            return key
    return None
