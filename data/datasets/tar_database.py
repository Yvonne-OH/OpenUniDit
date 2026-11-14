"""Unified tar 分片数据集 (NFS): TarParquetDataset。

数据布局 (见 .trae/skills/unified-tar-dataset/SKILL.md 与 docs/dataset_storage_design.md):
  <root>/<source>/shard-NNNNN.tar          成员 = 原始图片字节 (不转码)
  <root>/<source>/shard-NNNNN.idx          JSONL {id, off, size} (构建/校验用, 读取路径不用)
  <root>/<source>/parquet/shard-NNNNN.parquet  检索层: id, source, caption, w, h, images 列

设计要点:
- 采样/分桶/索引缓存全部复用 LmdbParquetDataset (BucketBatchSampler 兼容, 分桶+乱序语义不变);
  唯一区别是行定位: parquet `images` 列直供 {shard, off, size}, image_path 编码为
  tar://{shard_dir}/{stem}/{off}:{size}, 解码时 seek+read 一次切出 blob (不解析 tar 结构)。
- off/size 进列式索引缓存 (v4), 命中缓存秒级启动; blob 只在命中后读。
- 实测 NFS tar seek ~816 samples/s @ C=256, 训练需求 (8 卡 b=2) ~20 samples/s, 余量 >40x;
  顺序 seq 模式 (1774 samples/s) 需要流式 sampler, 与现有分桶乱序采样器冲突, 不采用。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from data.datasets.image_database import LmdbParquetDataset, collate_image_batch


class TarParquetDataset(LmdbParquetDataset):
    """unified_images_tar 分片 tar 数据集: 只支持 image 训练 (edit/t2i 走 LMDB 库)。"""

    def __init__(self, database_path: str, buckets: list[str], lmdb_root: str | None = None,
                 image_root: str | None = None, transform=None, max_upscale_ratio: float = 1.2,
                 max_samples: int | None = None, max_retries: int = 8, seed: int = 42,
                 sources: list[str] | None = None, edit_database_path: str | None = None,
                 edit_sources: list[str] | None = None, edit_split: str = "train",
                 edit_ratio: float | str = 0.0, max_edit_samples: int | None = None,
                 image_training_enabled: bool = True, edit_training_enabled: bool = False,
                 index_cache_dir: str | None = None, refresh_index_cache: bool = False,
                 bucket_sampling_weights: dict | None = None, max_decode_candidates: int = 3,
                 preprocess_style: str = "cover",
                 t2i_edit_database_path: str | None = None, t2i_edit_sources_t2i_mode: list[str] | None = None,
                 **kwargs):
        if edit_training_enabled or edit_database_path:
            raise NotImplementedError("TarParquetDataset 暂不支持 edit 训练 (edit 库仍走 lmdb_parquet 后端)")
        if t2i_edit_database_path or t2i_edit_sources_t2i_mode:
            raise NotImplementedError("TarParquetDataset 暂不支持 t2i_edit 库 (需在 yaml 中去掉 t2i_edit_* 配置)")
        self.tar_root = Path(database_path)
        if not self.tar_root.is_dir():
            raise FileNotFoundError(f"tar 数据集根目录不存在: {self.tar_root}")
        super().__init__(
            database_path=str(database_path), buckets=buckets,
            lmdb_root=str(database_path), image_root=None, transform=transform,
            max_upscale_ratio=max_upscale_ratio, max_samples=max_samples,
            max_retries=max_retries, seed=seed, sources=sources,
            edit_training_enabled=False, edit_ratio=0.0,
            index_cache_dir=index_cache_dir, refresh_index_cache=refresh_index_cache,
            bucket_sampling_weights=bucket_sampling_weights,
            max_decode_candidates=max_decode_candidates, preprocess_style=preprocess_style,
            **kwargs,
        )

    # ─── 分片签名: (parquet 分片数, tar 分片数), 数据入库中途指纹变化自动触发索引重建 ───

    def _shard_signature(self, edit_database_path: str | None = None,
                         t2i_database_path: str | None = None) -> dict[str, list[int]]:
        signature: dict[str, list[int]] = {}
        root = self.tar_root
        ttl = float(os.environ.get("GWP_SHARD_SIGNATURE_TTL", 300))
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._load_signature_snapshot("tar", root, ttl)
        for source_dir in self._discover_source_dirs(root, self.sources):
            name = f"tar:{source_dir.name}"
            if snapshot and name in snapshot:
                signature[name] = snapshot[name]
                continue
            try:
                pq_count = sum(1 for entry in os.listdir(source_dir / "parquet")
                               if entry.startswith("shard-") and entry.endswith(".parquet"))
                tar_count = sum(1 for entry in os.listdir(source_dir)
                                if entry.startswith("shard-") and entry.endswith(".tar"))
            except OSError:
                continue
            signature[name] = [pq_count, tar_count]
        self._write_signature_snapshot("tar", root, signature)
        return signature

    # ─── 元数据扫描: 只读检索列 (列裁剪), blob 定位信息来自 images 列 ───

    def _read_parquet(self, path, sources, is_edit, lmdb_root, limit=None, t2i: bool = False):
        if is_edit or t2i:
            raise NotImplementedError("TarParquetDataset 只支持 image 模式读取")
        import pyarrow.parquet as parquet

        root = Path(path)
        if (root / "parquet").is_dir():
            source_dirs = [root]
        else:
            source_dirs = [root / source for source in sorted(sources)] if sources else sorted(
                item for item in root.iterdir() if item.is_dir()
            )
        rows: list[dict[str, Any]] = []
        stats: dict[str, dict] = {}
        excluded_sources: list[str] = []
        if root.is_dir() and sources:
            excluded_sources = sorted(
                item.name for item in root.iterdir() if item.is_dir() and item.name not in sources
            )
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        source_counts: dict[str, int] = {}
        t0 = time.time()
        source_dirs = [d for d in source_dirs if d.is_dir()]
        file_list = []
        for source_dir in source_dirs:
            source = source_dir.name
            if sources and source not in sources:
                continue
            pq_files = sorted((source_dir / "parquet").glob("shard-*.parquet"))
            tar_count = sum(1 for entry in os.listdir(source_dir)
                            if entry.startswith("shard-") and entry.endswith(".tar"))
            if not pq_files:
                print(f"⚠️ {source}: parquet/ 下无 shard-*.parquet, 已跳过该源", flush=True)
                continue
            if len(pq_files) != tar_count:
                raise RuntimeError(
                    f"Parquet/tar 分片数量不一致: {source_dir} parquet={len(pq_files)} tar={tar_count}"
                )
            file_list.extend((file, source_dir) for file in pq_files)
        # 并行读分片: pyarrow 的 footer/列读取释放 GIL, NFS 上单线程逐分片开文件
        # 被 RTT 排队卡死 (~2 shard/s); 线程池并行后纯 IO 受限, 提速一个数量级以上。
        # pool.map 保序产出 → 行处理仍按 sorted 分片顺序串行执行, 索引内容与单线程逐字节一致。
        max_workers = max(1, int(os.environ.get("GWP_TAR_INDEX_THREADS", "32")))

        def load_shard(task):
            file, source_dir = task
            pf = parquet.ParquetFile(str(file))   # footer 读一次, schema 与列读共用, 不再二次开文件
            columns = [name for name in ("caption", "text", "w", "h", "width", "height", "images")
                       if name in set(pf.schema_arrow.names)]
            return file, source_dir, pf.read(columns=columns)

        def iter_tables():
            # 分块提交限制在途表格数: IO 快于行处理时避免 2.6 万分片的 table 全部积压内存
            chunk = max_workers * 4
            if max_workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for i in range(0, len(file_list), chunk):
                        yield from pool.map(load_shard, file_list[i:i + chunk])
            else:
                yield from (load_shard(task) for task in file_list)

        iterator = iter_tables()
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(file_list), desc="Loading tar parquet metadata",
                            unit="shard", dynamic_ncols=True)
        for file, source_dir, table in iterator:
            source = source_dir.name
            source_stats = stats.setdefault(source, {
                "shards": 0, "total": 0, "valid": 0, "filtered_no_image": 0,
                "limit_stopped": False,
            })
            if limit is not None and source_counts.get(source, 0) >= int(limit):
                source_stats["limit_stopped"] = True
                continue
            stem = file.stem   # shard-NNNNN, 与 tar/idx 同 stem 一一对应
            shard_dir = str(source_dir)
            source_stats["shards"] += 1
            for pos, row in enumerate(table.to_pylist()):
                if limit is not None and source_counts.get(source, 0) >= int(limit):
                    source_stats["limit_stopped"] = True
                    break
                source_stats["total"] += 1
                images = row.get("images") or []
                if not images:
                    source_stats["filtered_no_image"] += 1
                    continue
                # image-only 模式: blob 取 images[0]; edit 对 (before/after) 尚未接入 tar 后端
                blob = images[0]
                # images[].shard 形如 "shard-00000.tar", 去掉扩展名得到 stem; 缺失时退回 parquet stem
                blob_stem = str(blob.get("shard") or "").removesuffix(".tar") or stem
                rows.append({
                    "sample_id": pos,          # 分片内行号 (tar 后端 id 为字符串, 不入 int64 列式索引)
                    "source": source,
                    "is_edit": False,
                    "caption": row.get("caption") or row.get("text") or "",
                    "width": row.get("w") if row.get("w") is not None else row.get("width"),
                    "height": row.get("h") if row.get("h") is not None else row.get("height"),
                    "image_path": f"tar://{shard_dir}/{blob_stem}/{int(blob['off'])}:{int(blob['size'])}",
                    "blob_off": int(blob["off"]),
                    "blob_size": int(blob["size"]),
                })
                source_counts[source] = source_counts.get(source, 0) + 1
                source_stats["valid"] += 1
        if not rows:
            raise RuntimeError(f"tar parquet 数据没有可用样本: {path}")
        self.read_stats["image"] = stats
        if excluded_sources:
            self.read_stats["image_excluded"] = excluded_sources
        grand = {key: sum(int(info.get(key, 0)) for info in stats.values())
                 for key in ("shards", "total", "valid")}
        print(
            f"✅ tar parquet 读取完成[image]: sources={len(stats)} 分片={grand['shards']:,} "
            f"总数={grand['total']:,} 有效={grand['valid']:,} "
            f"过滤={grand['total'] - grand['valid']:,} ({time.time() - t0:.1f}s)"
            + (f" 未加载目录={excluded_sources}" if excluded_sources else ""),
            flush=True,
        )
        return rows


__all__ = ["TarParquetDataset", "collate_image_batch"]
