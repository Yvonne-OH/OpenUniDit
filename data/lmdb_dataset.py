"""Adapter for the unified LMDB+Parquet image dataset (copied from yisu-omni).

索引后端选择 (环境变量 GIGA_INDEX_BACKEND):
  - mmap (默认): 列式索引 mmap 只读共享, 同节点多 rank/worker 共享物理页, 启动秒级。
  - legacy: 旧版整份 pickle 反序列化 (每 rank 独立加载 ~20GB, fork worker COW),
    仅供压测对比, 不建议训练使用。
两个后端的缓存文件相互独立 (pickle 单文件 vs 列式目录), 可共存于同一 cache 目录。
"""
from __future__ import annotations

import os

if os.environ.get("GIGA_INDEX_BACKEND", "mmap") == "legacy":
    from data.datasets.image_database_legacy import LmdbParquetDataset, collate_image_batch
else:
    from data.datasets.image_database import LmdbParquetDataset, collate_image_batch


class GigaLMDBDataset(LmdbParquetDataset):
    def __init__(self, cfg, seed=42, index_cache_dir=None):
        super().__init__(
            database_path=cfg["database_path"],
            buckets=cfg["buckets"],
            lmdb_root=cfg.get("lmdb_root"),
            image_root=cfg.get("image_root"),
            sources=cfg.get("image_sources"),
            image_training_enabled=True,
            edit_training_enabled=False,
            edit_ratio=0.0,
            max_upscale_ratio=cfg.get("max_upscale_ratio", 1.2),
            max_retries=cfg.get("max_retries", 16),
            seed=seed,
            refresh_index_cache=cfg.get("refresh_index_cache", False),
            index_cache_dir=index_cache_dir,
            bucket_sampling_weights=cfg.get("bucket_sampling_weights"),
            preprocess_style=cfg.get("preprocess_style", "cover"),
        )

__all__ = ["GigaLMDBDataset", "collate_image_batch"]
