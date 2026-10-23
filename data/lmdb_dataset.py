"""Adapter for the unified LMDB+Parquet image dataset (copied from yisu-omni)."""
from __future__ import annotations

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
