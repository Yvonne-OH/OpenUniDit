"""Adapter for the unified LMDB+Parquet / tar 分片 image dataset (copied from yisu-omni).

索引后端选择 (环境变量 GIGA_INDEX_BACKEND):
  - mmap (默认): 列式索引 mmap 只读共享, 同节点多 rank/worker 共享物理页, 启动秒级。
  - legacy: 旧版整份 pickle 反序列化 (每 rank 独立加载 ~20GB, fork worker COW),
    仅供压测对比, 不建议训练使用。
两个后端的缓存文件相互独立 (pickle 单文件 vs 列式目录), 可共存于同一 cache 目录。

数据后端选择 (yaml dataset.dataset_type):
  - lmdb_parquet (默认): PFS LMDB, image_path = lmdb://...
  - tar_parquet: NFS unified tar 分片 (TarParquetDataset), image_path = tar://...
    分桶/乱序采样/索引缓存与 LMDB 后端完全一致。
"""
from __future__ import annotations

import os

if os.environ.get("GIGA_INDEX_BACKEND", "mmap") == "legacy":
    from data.datasets.image_database_legacy import LmdbParquetDataset, collate_image_batch
else:
    from data.datasets.image_database import LmdbParquetDataset, collate_image_batch


class GigaLMDBDataset(LmdbParquetDataset):
    def __init__(self, cfg, seed=42, index_cache_dir=None):
        dataset_type = str(cfg.get("dataset_type", "lmdb_parquet"))
        if dataset_type == "tar_parquet":
            from data.datasets.tar_database import TarParquetDataset
            # 纯 Python 子类, 实例布局兼容, 直接切换后端类 (保持 GigaLMDBDataset 调用点不变)
            self.__class__ = TarParquetDataset
            TarParquetDataset.__init__(
                self,
                database_path=cfg["database_path"],
                buckets=cfg["buckets"],
                sources=cfg.get("image_sources"),
                max_upscale_ratio=cfg.get("max_upscale_ratio", 1.2),
                max_retries=cfg.get("max_retries", 16),
                seed=seed,
                image_training_enabled=cfg.get("image_training_enabled", True),
                edit_training_enabled=cfg.get("edit_training_enabled", False),
                edit_database_path=cfg.get("edit_database_path"),
                edit_sources=cfg.get("edit_sources"),
                max_edit_samples=cfg.get("max_edit_samples"),
                refresh_index_cache=cfg.get("refresh_index_cache", False),
                index_cache_dir=index_cache_dir,
                bucket_sampling_weights=cfg.get("bucket_sampling_weights"),
                preprocess_style=cfg.get("preprocess_style", "cover"),
                # t2i_caption 编辑库 7 源: 之前漏传导致 t2i_active=False, 指纹算出
                # 无 t2i 版本并永久命中复制来的旧缓存 (46.8M 无编辑库行);
                # 传参后指纹变化自动重建, ~741 万 t2i 行进训练
                t2i_edit_database_path=cfg.get("t2i_edit_database_path"),
                t2i_edit_sources_t2i_mode=cfg.get("t2i_edit_sources_t2i_mode"),
            )
            return
        if dataset_type != "lmdb_parquet":
            raise ValueError(f"不支持的 dataset_type: {dataset_type!r} (可选: lmdb_parquet, tar_parquet)")
        LmdbParquetDataset.__init__(
            self,
            database_path=cfg["database_path"],
            buckets=cfg["buckets"],
            lmdb_root=cfg.get("lmdb_root"),
            image_root=cfg.get("image_root"),
            sources=cfg.get("image_sources"),
            t2i_edit_database_path=cfg.get("t2i_edit_database_path"),
            t2i_edit_sources_t2i_mode=cfg.get("t2i_edit_sources_t2i_mode"),
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
