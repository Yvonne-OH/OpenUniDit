# -*- coding: utf-8 -*-
"""从 HuggingFace 镜像下载 cyberagent/opencole 数据集并读取海报初始 prompt。

本节点直连 huggingface.co 不通, 通过 HF_ENDPOINT=https://hf-mirror.com 镜像下载 parquet,
本地缓存后用 pyarrow 读取。每个样例包含 intention/description/keywords/captions/headings,
作为海报生成器的初始 prompt。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "cyberagent/opencole"
CACHE_ROOT = Path("/home/jovyan/pfs/users/zhanqian.wu/ckpt/opencole")


def _download_split(split: str) -> list[Path]:
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.startswith(f"data/{split}-") and f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"opencole 仓库 {split}/ 下没有 parquet 文件")
    paths = [Path(hf_hub_download(REPO, f, repo_type="dataset", local_dir=CACHE_ROOT))
             for f in files]
    return paths


def load_opencole(split: str = "test", limit: int | None = None) -> list[dict]:
    """返回 opencole 样例列表, 字段: id/intention/description/keywords/captions_*/headings_*"""
    parquets = _download_split(split)
    frames = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(frames, ignore_index=True)
    if limit:
        df = df.head(limit)
    records = []
    for row in df.to_dict("records"):
        # pyarrow list 列 / numpy array 转普通 list
        for k, v in row.items():
            if isinstance(v, (list, tuple)) or (hasattr(v, "tolist") and not isinstance(v, str)):
                row[k] = [str(x) for x in v]
            elif isinstance(v, bytes):
                row[k] = v.decode("utf-8", "replace")
            elif v is not None and not isinstance(v, str):
                row[k] = str(v)
        records.append(row)
    return records


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    recs = load_opencole(args.split, args.limit)
    print(f"loaded {len(recs)} records from split={args.split}")
    for r in recs[:2]:
        print({k: (v if not isinstance(v, str) else v[:80]) for k, v in r.items()})
