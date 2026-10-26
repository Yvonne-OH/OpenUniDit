#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_pipeline.py — LVIS lmdb → bbox 过滤 → 画质过滤 → Qwen caption → JSONL。

用法:
  单卡冒烟:  python run_pipeline.py --config configs/lvis_v1.yaml --limit 24
  分片并行:  CUDA_VISIBLE_DEVICES=$i python run_pipeline.py --config ... --shard $i --num-shards 8
  续跑:      同命令重跑即可 (已写入 output jsonl 的 key 自动跳过)

修复 CUDA 803 (libcuda compat 旧库遮蔽): 进程启动前重写 LD_LIBRARY_PATH 并 execv 自身。
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── CUDA 803 修复: 必须在 import torch 前生效 → 改环境后 execv 重启自身 ──
_LIB = "/usr/lib/x86_64-linux-gnu"
_lp = os.environ.get("LD_LIBRARY_PATH", "")
if _LIB not in _lp.split(os.pathsep):
    os.environ["LD_LIBRARY_PATH"] = _LIB + (os.pathsep + _lp if _lp else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, _HERE)

import lmdb
import pickle
import yaml
from tqdm import tqdm

from bbox_filter import filter_annotations
from quality_filter import quality_check

TS = time.strftime("%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{TS}] {msg}", flush=True)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def enumerate_records(lmdb_dir: str) -> list[tuple[str, int, str]]:
    """[(stem, sample_id, key), ...] 从 parquet 元数据枚举全部样本。"""
    import pyarrow.parquet as pq
    records = []
    for pf in sorted(glob.glob(f"{lmdb_dir}/parquet/*.parquet")):
        stem = os.path.basename(pf).replace("meta_", "").replace(".parquet", "")
        t = pq.read_table(pf, columns=["sample_id", "key"])
        for sid, key in zip(t["sample_id"].to_pylist(), t["key"].to_pylist()):
            records.append((stem, sid, key))
    return records


def read_record(lmdb_dir: str, stem: str, sid: int) -> dict:
    env = lmdb.open(f"{lmdb_dir}/mdb/{stem}.mdb", readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        rec = pickle.loads(txn.get(struct.pack("<Q", sid)))
    env.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "configs/lvis_v1.yaml"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条 (冒烟)")
    ap.add_argument("--no-caption", action="store_true", help="跳过 VLM caption")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    lmdb_dir = cfg["input"]["lmdb_dir"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(
        out_dir, f"{cfg['output']['jsonl_prefix']}_shard{args.shard}of{args.num_shards}.jsonl")

    all_records = enumerate_records(lmdb_dir)
    work = [r for i, r in enumerate(all_records) if i % args.num_shards == args.shard]
    if args.limit:
        work = work[: args.limit]

    # 续跑: 跳过已写入的 key
    done = set()
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass
    work = [r for r in work if r[2] not in done]
    log(f"shard {args.shard}/{args.num_shards}: 共 {len(all_records)} 样本, "
        f"本 shard {len(work)} 待处理 (已跳过 {len(done)}) → {jsonl_path}")

    cap_cfg = cfg["caption"]
    captioner = None
    if cap_cfg["enabled"] and not args.no_caption:
        from qwen_caption import QwenCaptioner, pil_from_bytes
        captioner = QwenCaptioner(
            cap_cfg["model_path"], cap_cfg["dtype"], cap_cfg["max_new_tokens"],
            cap_cfg["language"], cap_cfg["max_regions"],
        )
        log(f"🤖 Qwen3-VL 就绪: {cap_cfg['model_path']}")

    bs = cap_cfg.get("batch_size", 16)
    stats = {"total": 0, "pass_quality": 0, "fail_quality": 0, "n_conf_dropped": 0,
             "n_nms_dropped": 0, "n_ann_in": 0, "n_ann_kept": 0, "caption_ok": 0}
    fail_reasons = {}
    t0 = time.time()
    fout = open(jsonl_path, "a", buffering=1)

    pbar = tqdm(total=len(work), desc=f"shard{args.shard}", unit="img",
                bar_format="{l_bar}{bar:24}{r_bar}")
    i = 0
    while i < len(work):
        batch = work[i: i + bs]
        items = []
        for stem, sid, key in batch:
            rec = read_record(lmdb_dir, stem, sid)
            items.append((stem, sid, key, rec))
        i += len(batch)

        # ── (1) bbox 过滤 + (2) 画质过滤 ──
        prepped = []   # 通过全部过滤、待 caption 的
        for stem, sid, key, rec in items:
            stats["total"] += 1
            from qwen_caption import pil_from_bytes
            img = pil_from_bytes(rec["img_bytes"])

            bcfg = cfg["bbox_filter"]
            kept, bstat = filter_annotations(
                rec["annotations"], bcfg["conf_key"], bcfg["conf_min"], bcfg["iou_threshold"])
            max_r = bcfg.get("max_regions_per_image", 0)
            if max_r and len(kept) > max_r:
                stats["fail_dense"] = stats.get("fail_dense", 0) + 1
                stats["n_ann_in"] += bstat["n_input"]
                continue
            stats["n_conf_dropped"] += bstat["n_conf_dropped"]
            stats["n_nms_dropped"] += bstat["n_nms_dropped"]
            stats["n_ann_in"] += bstat["n_input"]
            stats["n_ann_kept"] += bstat["n_kept"]

            q = quality_check(img, cfg["quality_filter"])
            if not q["pass"]:
                stats["fail_quality"] += 1
                for r in q["fails"]:
                    fail_reasons[r.split("<")[0].split(">")[0]] = fail_reasons.get(r.split("<")[0].split(">")[0], 0) + 1
                continue
            stats["pass_quality"] += 1
            prepped.append((stem, sid, key, rec, kept, q, img))

        # ── (3) Qwen caption (batch) ──
        cap_results = [None] * len(prepped)
        if captioner is not None and prepped:
            try:
                cap_results = captioner.caption_batch(
                    [{"img": p[6], "annotations": p[4]} for p in prepped])
            except Exception as e:  # 整批失败: 记日志继续 (坐标信息不丢)
                log(f"⚠️ caption batch 失败: {type(e).__name__}: {e}")
                cap_results = [None] * len(prepped)

        for (stem, sid, key, rec, kept, q, _), cap in zip(prepped, cap_results):
            regions = []
            for j, ann in enumerate(kept):
                desc = None
                if cap is not None and cap.get("ok") and j < len(cap["regions"]):
                    desc = cap["regions"][j]["description"]
                regions.append({
                    "id": j, "ann_id": ann["id"],
                    "category": ann["category"], "synset": ann["synset"],
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"], "area": ann["area"],
                    "segmentation": ann["segmentation"],
                    "conf": ann.get("score"),
                    "description": desc,
                })
            out = {
                "key": key, "sample_id": sid, "file_name": rec["file_name"],
                "split": rec["split"], "stem": stem,
                "image": {"width": rec["w"], "height": rec["h"],
                          "ref": f"unified_canvas_images/lvis_v1/mdb/{stem}.mdb",
                          "lmdb_sample_id": sid},
                "quality": q,
                "bbox_filter": {"conf_key": cfg["bbox_filter"]["conf_key"],
                                "iou_threshold": cfg["bbox_filter"]["iou_threshold"]},
                "caption": (None if cap is None else {
                    "scene": cap.get("scene"), "model": os.path.basename(cfg["caption"]["model_path"]),
                    "language": cfg["caption"]["language"], "ok": bool(cap.get("ok")),
                    "missing_regions": cap.get("missing", []),
                }),
                "regions": regions,
                "lvis_categories_caption": rec["caption"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            if cap is not None and cap.get("ok"):
                stats["caption_ok"] += 1
        pbar.update(len(batch))
        pbar.set_postfix(quality=f"{stats['pass_quality']}/{stats['total']}",
                         cap_ok=stats["caption_ok"])

    fout.close()
    pbar.close()
    stats["fail_reasons"] = fail_reasons
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats_path = jsonl_path.replace(".jsonl", ".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"✅ 完成: {stats['total']} 图, 画质通过 {stats['pass_quality']}, "
        f"过滤 bbox conf={stats['n_conf_dropped']} nms={stats['n_nms_dropped']}, "
        f"caption ok={stats['caption_ok']} → {stats_path}")


if __name__ == "__main__":
    main()
