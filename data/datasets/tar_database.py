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

import json
import os
import time
from pathlib import Path
from typing import Any

from data.datasets.image_database import LmdbParquetDataset, collate_image_batch


class TarParquetDataset(LmdbParquetDataset):
    """unified_images_tar 分片 tar 数据集: image 训练 + edit 库 t2i_caption 模式
    (before 原图 + 描述 caption 作普通 image 样本); edit 对训练仍走 lmdb_parquet 后端。"""

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
                 freeze_index_signature: bool = False,
                 **kwargs):
        if edit_training_enabled or edit_database_path:
            raise NotImplementedError("TarParquetDataset 暂不支持 edit 对训练 (edit 库仍走 lmdb_parquet 后端)")
        if type(freeze_index_signature) is not bool:
            raise ValueError("freeze_index_signature 必须为 bool")
        self.freeze_index_signature = freeze_index_signature
        if freeze_index_signature:
            if refresh_index_cache or os.environ.get("GWP_DATASET_INDEX_CACHE_DISABLE"):
                raise ValueError("freeze_index_signature=true 不允许强制刷新或禁用索引缓存")
            if not (index_cache_dir or os.environ.get("GWP_DATASET_INDEX_CACHE")):
                raise ValueError("freeze_index_signature=true 必须显式指定已复制的 index_cache_dir")
            if not sources:
                raise ValueError("freeze_index_signature=true 必须显式指定 image_sources")
        self.tar_root = Path(database_path)
        # Frozen runs trust the copied signatures/cache, never stat a remote root.
        if not freeze_index_signature and not self.tar_root.is_dir():
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
            t2i_edit_database_path=t2i_edit_database_path,
            t2i_edit_sources_t2i_mode=t2i_edit_sources_t2i_mode,
            **kwargs,
        )

    def _load_frozen_signature(self, label: str, root: Path, sources) -> dict[str, list[int]]:
        """Read the copied snapshot without TTL checks or remote source discovery."""
        if not sources:
            raise ValueError(f"冻结 {label} 签名必须显式指定 sources")
        path = self._signature_snapshot_path(self.index_cache_dir, label, root)
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"冻结索引签名缺失或损坏，禁止扫描数据源: {path}") from exc
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"冻结索引签名必须为 JSON object: {path}")
        signature = {}
        for source in sorted(sources):
            key = f"{label}:{source}"
            value = snapshot.get(key)
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(count) is not int or count < 0 for count in value)):
                raise RuntimeError(f"冻结索引签名缺失/无效: {path}: {key} (需两个非负整数)")
            signature[key] = value
        return signature

    def _load_or_read_records(self, image_enabled: bool, edit_enabled: bool,
                              max_samples, max_edit_samples, edit_database_path):
        if not self.freeze_index_signature:
            return super()._load_or_read_records(
                image_enabled, edit_enabled, max_samples, max_edit_samples, edit_database_path)
        # Keep the existing fingerprint schema, and never enter the rebuild or
        # distributed "rank 0 might rebuild" path for an immutable cache.
        self.index_fingerprint = self._index_fingerprint(
            image_enabled, edit_enabled, max_samples, max_edit_samples, edit_database_path)
        payload = self._load_index_cache(self.index_fingerprint, allow_fresh=True)
        if payload is None:
            raise RuntimeError(
                "冻结索引缓存缺失或无效，禁止重建/扫描数据源: "
                f"{self._cache_paths(self.index_fingerprint)[0]}")
        self.read_stats = payload.get("stats") or {"image": {}, "edit": {}}
        return payload["store"]

    def _build_bucket_index(self, index_store, max_samples, max_edit_samples, edit_ratio):
        if not self.freeze_index_signature:
            return super()._build_bucket_index(index_store, max_samples, max_edit_samples, edit_ratio)
        runtime_dir = self._runtime_cache_dir(max_samples, max_edit_samples, edit_ratio)
        if not self._load_runtime_cache(runtime_dir, index_store):
            raise RuntimeError(f"冻结运行期 bucket 缓存缺失或无效，禁止重建: {runtime_dir}")

    # ─── 分片签名: (parquet 分片数, tar 分片数), 数据入库中途指纹变化自动触发索引重建 ───

    def _shard_signature(self, edit_database_path: str | None = None,
                         t2i_database_path: str | None = None) -> dict[str, list[int]]:
        signature: dict[str, list[int]] = {}
        if self.freeze_index_signature:
            signature.update(self._load_frozen_signature("tar", self.tar_root, self.sources))
            if t2i_database_path:
                signature.update(self._load_frozen_signature(
                    "t2i", Path(t2i_database_path), self.t2i_edit_sources_t2i_mode))
            return signature
        ttl = float(os.environ.get("GWP_SHARD_SIGNATURE_TTL", 300))
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        roots: list[tuple[str, Path, list[str] | None]] = [("tar", self.tar_root, self.sources)]
        if t2i_database_path:
            roots.append(("t2i", Path(t2i_database_path), self.t2i_edit_sources_t2i_mode))
        for label, root, sources in roots:
            snapshot = self._load_signature_snapshot(label, root, ttl)
            for source_dir in self._discover_source_dirs(root, sources):
                name = f"{label}:{source_dir.name}"
                if snapshot and name in snapshot:
                    signature[name] = snapshot[name]
                    continue
                try:
                    pq_subdir = "parquet_t2i" if label == "t2i" else "parquet"
                    pq_count = sum(1 for entry in os.listdir(source_dir / pq_subdir)
                                   if entry.endswith(".parquet"))
                    tar_count = sum(1 for entry in os.listdir(source_dir)
                                    if entry.endswith(".tar"))
                except OSError:
                    continue
                signature[name] = [pq_count, tar_count]
            self._write_signature_snapshot(label, root, signature)
        return signature

    # ─── 元数据扫描: 只读检索列 (列裁剪), blob 定位信息来自 images 列 ───

    def _read_parquet(self, path, sources, is_edit, lmdb_root, limit=None, t2i: bool = False):
        if self.freeze_index_signature:
            raise RuntimeError("冻结索引模式禁止扫描 parquet 数据源")
        if is_edit:
            raise NotImplementedError("TarParquetDataset 只支持 image 模式读取 (edit 对训练走 LMDB 库)")
        import pyarrow.parquet as parquet

        # t2i 模式: 读 edit tar 库各 source 的 parquet_t2i/ (原 tar parquet + t2i_caption 列,
        # 由 tools/build_tar_parquet_t2i.py 生成), 以 before 原图 + 描述 caption 生成普通 image 行。
        pq_subdir = "parquet_t2i" if t2i else "parquet"
        # tar edit 库分片命名不统一 (shard-* / {env_stem}-NNNNN), 统一按 *.parquet / *.tar 计数;
        # image 库保持 shard-* glob 以防 parquet/ 下出现其他 parquet 文件
        pq_glob = "*.parquet" if t2i else "shard-*.parquet"

        root = Path(path)
        if (root / pq_subdir).is_dir():
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
            pq_files = sorted((source_dir / pq_subdir).glob(pq_glob))
            tar_count = sum(1 for entry in os.listdir(source_dir)
                            if entry.endswith(".tar"))
            if not pq_files:
                print(f"⚠️ {source}: {pq_subdir}/ 下无 *.parquet, 已跳过该源", flush=True)
                continue
            if len(pq_files) != tar_count:
                raise RuntimeError(
                    f"Parquet/tar 分片数量不一致: {source_dir} {pq_subdir}={len(pq_files)} tar={tar_count}"
                )
            file_list.extend((file, source_dir) for file in pq_files)
        # 并行读分片: pyarrow 的 footer/列读取释放 GIL, NFS 上单线程逐分片开文件
        # 被 RTT 排队卡死 (~2 shard/s); 线程池并行后纯 IO 受限, 提速一个数量级以上。
        # pool.map 保序产出 → 行处理仍按 sorted 分片顺序串行执行, 索引内容与单线程逐字节一致。
        max_workers = max(1, int(os.environ.get("GWP_TAR_INDEX_THREADS", "32")))

        def load_shard(task):
            file, source_dir = task
            pf = parquet.ParquetFile(str(file))   # footer 读一次, schema 与列读共用, 不再二次开文件
            if t2i:
                want = ("t2i_caption", "w_before", "h_before", "images")
            else:
                want = ("caption", "text", "w", "h", "width", "height", "images")
            columns = [name for name in want if name in set(pf.schema_arrow.names)]
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
                "filtered_no_caption": 0, "limit_stopped": False,
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
                if t2i:
                    # t2i 行: before 原图 + Qwen3-VL 描述 caption (宽高取 w_before/h_before)。
                    # 多参考图源 t2i_caption 是 list → 非 str 视为无效; 空 caption 同样跳过。
                    cap = row.get("t2i_caption")
                    if not isinstance(cap, str) or not cap.strip():
                        source_stats["filtered_no_caption"] += 1
                        continue
                # image-only 模式: blob 取 images[0]; t2i 模式取 id 含 before 的成员
                # (edit 对 images[0]=img_before, images[1]=img_after; PATHS 去重条目同样带 shard/off/size)
                if t2i:
                    blob = next((b for b in images if "before" in str(b.get("id", ""))), images[0])
                else:
                    blob = images[0]
                # images[].shard 形如 "shard-00000.tar", 去掉扩展名得到 stem; 缺失时退回 parquet stem
                blob_stem = str(blob.get("shard") or "").removesuffix(".tar") or stem
                entry = {
                    "sample_id": pos,          # 分片内行号 (tar 后端 id 为字符串, 不入 int64 列式索引)
                    "source": source,
                    "is_edit": False,
                    "image_path": f"tar://{shard_dir}/{blob_stem}/{int(blob['off'])}:{int(blob['size'])}",
                    "blob_off": int(blob["off"]),
                    "blob_size": int(blob["size"]),
                }
                if t2i:
                    entry.update(caption=cap,
                                 width=row.get("w_before"), height=row.get("h_before"))
                else:
                    entry.update(
                        caption=row.get("caption") or row.get("text") or "",
                        width=row.get("w") if row.get("w") is not None else row.get("width"),
                        height=row.get("h") if row.get("h") is not None else row.get("height"),
                    )
                rows.append(entry)
                source_counts[source] = source_counts.get(source, 0) + 1
                source_stats["valid"] += 1
        if not rows:
            raise RuntimeError(f"tar parquet 数据没有可用样本: {path}")
        if t2i:
            # t2i 行并入 image 模式统计 (普通 image 读取已写过该 key, 不能覆盖)
            self.read_stats.setdefault("image", {}).update(stats)
        else:
            self.read_stats["image"] = stats
        if excluded_sources:
            self.read_stats["image_excluded"] = excluded_sources
        grand = {key: sum(int(info.get(key, 0)) for info in stats.values())
                 for key in ("shards", "total", "valid")}
        print(
            f"✅ tar parquet 读取完成[{'t2i' if t2i else 'image'}]: sources={len(stats)} 分片={grand['shards']:,} "
            f"总数={grand['total']:,} 有效={grand['valid']:,} "
            f"过滤={grand['total'] - grand['valid']:,} ({time.time() - t0:.1f}s)"
            + (f" 未加载目录={excluded_sources}" if excluded_sources else ""),
            flush=True,
        )
        return rows


__all__ = ["TarParquetDataset", "collate_image_batch"]
