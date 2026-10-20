"""预构建 stage_1 数据集索引缓存到 output 目录.

单进程运行一次, 之后训练启动直接命中 output/dataset_index_cache 索引缓存,
不再重复扫描 NFS 上的数千个 parquet 分片. 首次构建过程带 tqdm 进度条.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from data.lmdb_dataset import GigaLMDBDataset


def main():
    parser = argparse.ArgumentParser(description="预构建 giga_exp_image 数据集索引缓存")
    parser.add_argument("--config", default="configs/stage_1.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text() if cfg_path.is_absolute() else (ROOT / args.config).read_text())

    # 与训练脚本保持一致: 缓存优先放 output 目录
    index_cache_dir = Path(cfg["output_dir"]) / "dataset_index_cache"
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    print("🗂️  index_cache_dir = %s" % index_cache_dir, flush=True)
    print("🗃️  database_path   = %s" % cfg["dataset"]["database_path"], flush=True)
    print("🪣  buckets         = %s" % cfg["dataset"]["buckets"], flush=True)

    dataset = GigaLMDBDataset(cfg["dataset"], seed=int(cfg.get("seed", 42)), index_cache_dir=str(index_cache_dir))
    print(f"\n✅ 索引构建/加载完成: records={len(dataset.records):,}", flush=True)
    for bucket, indices in sorted(dataset.bucket_indices.items()):
        print(f"   {bucket:<16s} {len(indices):>10,}", flush=True)
    print(f"💾 缓存目录: {index_cache_dir}", flush=True)


if __name__ == "__main__":
    main()
