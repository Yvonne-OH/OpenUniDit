#!/usr/bin/env python3
"""抽取 edit 库在 t2i 训练模式下实际使用的 (图片 + t2i_caption prompt) 对, 存盘预览。
每库抽 3 条: 图片按 parquet images[0] 的 off/size 从对应 tar 里 seek+read (与训练管道一致)。"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import io

ROOT = Path("/home/jovyan/pfs/data/public/image_data")
OUT = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/t2i_pairs_preview")
DATASETS = ["HQ-Edit", "GPT-UltraEdit", "GPT-HQEdit", "pico-banana-400k"]
N_PER = 3

OUT.mkdir(parents=True, exist_ok=True)
for name in DATASETS:
    pq = sorted((ROOT / name / "parquet_t2i").glob("*.parquet"))[0]
    df = pd.read_parquet(pq)
    n_done = 0
    for _, row in df.iterrows():
        if n_done >= N_PER:
            break
        cap = row.get("t2i_caption")
        if not isinstance(cap, str) or not cap.strip():
            continue
        imgs = row.get("images")
        if imgs is None or len(imgs) == 0:
            continue
        blob = imgs[0]
        if not isinstance(blob, dict) or "off" not in blob:
            continue
        tar_path = ROOT / name / blob["shard"]
        with open(tar_path, "rb") as f:
            f.seek(int(blob["off"]))
            data = f.read(int(blob["size"]))
        stem = f"{name}_t2i_{n_done}"
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((768, 768))
        im.save(OUT / f"{stem}.jpg", quality=85)
        (OUT / f"{stem}.prompt.txt").write_text(cap, encoding="utf-8")
        print(f"{stem}: {blob['shard']} | {im.size[0]}x{im.size[1]} | prompt {len(cap)} 字符")
        n_done += 1
print(f"完成, 存在 {OUT}")
