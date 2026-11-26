#!/usr/bin/env python3
"""从 4 个 edit 库各抽 2 对 (img_before, img_after + edit_prompt) 存为预览图。"""
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/home/jovyan/pfs/data/public/image_data")
OUT = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/edit_pairs_preview")
DATASETS = ["HQ-Edit", "GPT-UltraEdit", "GPT-HQEdit", "pico-banana-400k"]

OUT.mkdir(parents=True, exist_ok=True)
for name in DATASETS:
    d = ROOT / name
    tar_path = sorted(d.glob("*.tar"))[0]
    meta_path = tar_path.with_suffix(".meta.jsonl")
    n_target = 2
    n_done = 0
    for line in meta_path.open():
        if n_done >= n_target:
            break
        rec = json.loads(line)
        if len(rec.get("images", [])) < 2:
            continue
        before, after = rec["images"][:2]
        with open(tar_path, "rb") as f:
            f.seek(before["off"]); img_b = f.read(before["size"])
            f.seek(after["off"]); img_a = f.read(after["size"])
        stem = f"{name}_{n_done}"
        for tag, blob in (("before", img_b), ("after", img_a)):
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            im.thumbnail((768, 768))
            im.save(OUT / f"{stem}_{tag}.jpg", quality=85)
        print(f"== {stem} | edit_type={rec.get('edit_type', '-')}")
        print(f"   edit_prompt: {rec.get('edit_prompt', '')[:200]}")
        if rec.get("input_caption"):
            print(f"   input_caption: {rec['input_caption'][:150]}")
        n_done += 1
print(f"预览已存到 {OUT}")
