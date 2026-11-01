#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_pipeline_ocr.py — OCR-Data parquet 流式 → bbox/画质过滤 → LAION 美学 → Qwen-VL caption → JSONL。

流程 (对每张图):
  (1) bbox 过滤:  小框 (w/h<min_box_side 或面积<min_box_area) 丢弃;
                  重合框占比 > max_overlap_ratio 整图丢弃; 空文本框丢弃。
  (2) 画质过滤:   图片过小/过大、宽高比异常、过暗/过亮、模糊、低对比 → 整图丢弃。
  (3) 美学过滤:   LAION aesthetic head 打分 < threshold 整图丢弃。
  (4) VLM 描述:   Qwen2.5-VL 输出 {"scene", "text_style"} (批量)。
  输出 JSONL: 场景描述 + 文字风格 + regions(bbox+ocr 文本) + 美学分; 原图字节落盘 (不重编码)。

用法:
  单卡冒烟:  CUDA_VISIBLE_DEVICES=0 python run_pipeline_ocr.py --config configs/ocr_sa1b.yaml --limit 64
  分片并行:  CUDA_VISIBLE_DEVICES=$i python run_pipeline_ocr.py --config ... --shard $i --num-shards 8
  续跑:      同命令重跑即可 (已写入 jsonl 的 key 自动跳过)
  仅过滤:    --no-caption (跳过 VLM, 快速出过滤产物)
"""
import argparse
import glob
import io
import json
import os
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

import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from ocr_bbox_filter import filter_ocr_boxes
from quality_filter import quality_check

TS = time.strftime("%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{TS}] {msg}", flush=True)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stream_rows(parquet_dir: str, shard: int, num_shards: int, done: set):
    """流式产出 (key, split, row, record dict)，跳过已完成的 key。"""
    files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{parquet_dir} 下没有 parquet")
    for fp in files:
        split = os.path.basename(fp).split("-")[0]
        pf = pq.ParquetFile(fp)
        row = 0
        for batch in pf.iter_batches(batch_size=32):
            for r in batch.to_pylist():
                if row % num_shards == shard:
                    key = f"{split}_{row:06d}"
                    if key not in done:
                        yield key, split, row, r
                row += 1
        log(f"{split}: 遍历完毕")


class Pipeline:
    def __init__(self, cfg: dict, shard: int, num_shards: int, no_caption: bool):
        self.cfg = cfg
        self.shard, self.num_shards = shard, num_shards
        out_dir = cfg["output"]["dir"]
        self.img_dir = os.path.join(out_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.jsonl_path = os.path.join(
            out_dir, f"{cfg['output']['jsonl_prefix']}_shard{shard}of{num_shards}.jsonl")
        self.save_images = cfg["output"].get("save_images", True)

        bcfg, qcfg = cfg["bbox_filter"], cfg["quality_filter"]
        self.bcfg, self.qcfg = bcfg, qcfg
        acfg, ccfg = cfg["aesthetic"], cfg["caption"]
        self.aes_threshold = acfg.get("threshold", 5.0) if acfg.get("enabled", True) else None
        self.aes_bs = acfg.get("batch_size", 128)
        self.cap_bs = ccfg.get("batch_size", 16)

        self.scorer = None
        if self.aes_threshold is not None:
            from aesthetic_scorer import AestheticScorer
            self.scorer = AestheticScorer(acfg["clip_backbone"], acfg["head_path"],
                                          device="cuda:0")
            log(f"🎨 美学打分就绪: backbone={self.scorer.pretrained_tag}, "
                f"threshold={self.aes_threshold}")
        self.captioner = None
        if ccfg["enabled"] and not no_caption:
            from ocr_caption import Qwen25VLCaptioner
            self.captioner = Qwen25VLCaptioner(
                ccfg["model_path"], ccfg["dtype"], ccfg["max_new_tokens"],
                ccfg.get("max_pixels", 4194304), ccfg["language"])
            log(f"🤖 Qwen2.5-VL 就绪: {ccfg['model_path']}")

        self.stats = {"total": 0, "fail_decode": 0, "fail_quality": 0,
                      "fail_bbox": 0, "fail_aesthetic": 0, "pass_all": 0,
                      "n_box_in": 0, "n_box_kept": 0, "caption_ok": 0}
        self.fail_reasons = {}
        self.aes_buf: list[dict] = []   # 通过画质+bbox, 待美学打分
        self.cap_buf: list[dict] = []   # 通过美学, 待 caption
        self.fout = open(self.jsonl_path, "a", buffering=1)

    # ── 各级处理 ──
    def process_one(self, key, split, row, r):
        self.stats["total"] += 1
        try:
            img_bytes = r["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            self.stats["fail_decode"] += 1
            return

        # (2) 画质过滤
        q = quality_check(img, self.qcfg)
        if not q["pass"]:
            self.stats["fail_quality"] += 1
            for f in q["fails"]:
                k = f.split("<")[0].split(">")[0]
                self.fail_reasons[k] = self.fail_reasons.get(k, 0) + 1
            return

        # (1) bbox 过滤
        kept, bfails, bstat = filter_ocr_boxes(
            r["texts"], r["bboxes"], self.bcfg["min_box_side"],
            self.bcfg["min_box_area"], self.bcfg["overlap_iou"],
            self.bcfg["max_overlap_ratio"])
        self.stats["n_box_in"] += bstat["n_input"]
        self.stats["n_box_kept"] += bstat["n_kept"]
        max_r = self.bcfg.get("max_regions_per_image", 0)
        if bfails or (max_r and len(kept) > max_r):
            self.stats["fail_bbox"] += 1
            for f in bfails or ["too_many_regions"]:
                k = f.split("<")[0].split(">")[0]
                self.fail_reasons[k] = self.fail_reasons.get(k, 0) + 1
            return

        self.aes_buf.append({
            "key": key, "split": split, "row": row,
            "src": r["image"]["path"], "img": img,
            "img_bytes": img_bytes, "kept": kept, "q": q,
        })
        if len(self.aes_buf) >= self.aes_bs:
            self.flush_aesthetic()
        if len(self.cap_buf) >= self.cap_bs:
            self.flush_caption()

    def flush_aesthetic(self):
        if not self.aes_buf:
            return
        batch, self.aes_buf = self.aes_buf, []
        if self.scorer is None:  # 美学关闭: 全部通过
            for it in batch:
                it["score"] = 1.0
                self.cap_buf.append(it)
            return
        try:
            scores = self.scorer.score_batch([it["img"] for it in batch])
        except Exception as e:
            log(f"⚠️ aesthetic batch 失败: {type(e).__name__}: {e}")
            scores = [float("nan")] * len(batch)
        for it, s in zip(batch, scores):
            if s != s or s < self.aes_threshold:  # NaN 或低于阈值
                self.stats["fail_aesthetic"] += 1
                self.fail_reasons["aesthetic"] = self.fail_reasons.get("aesthetic", 0) + 1
            else:
                it["score"] = s
                self.cap_buf.append(it)

    def flush_caption(self):
        if not self.cap_buf:
            return
        batch, self.cap_buf = self.cap_buf, []
        results = self._caption_with_retry([it["img"] for it in batch])
        for it, cap in zip(batch, results):
            self.write_record(it, cap)
            self.stats["pass_all"] += 1
            if cap and cap.get("ok"):
                self.stats["caption_ok"] += 1

    def _caption_with_retry(self, imgs: list) -> list:
        """VLM 描述; OOM 等失败时二分重试, 尽量不丢 caption。"""
        try:
            return self.captioner.caption_batch(imgs)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"⚠️ caption batch 失败: {type(e).__name__}: {e}")
            return [None] * len(imgs)
        if len(imgs) == 1:
            log("⚠️ caption 单图仍失败, 置 None")
            return [None]
        mid = len(imgs) // 2
        return self._caption_with_retry(imgs[:mid]) + self._caption_with_retry(imgs[mid:])

    def write_record(self, it, cap):
        key, split = it["key"], it["split"]
        img = it["img"]
        fname = f"{key}_{os.path.basename(it['src'])}"
        img_rel = f"images/{split}/{fname}"
        if self.save_images:
            os.makedirs(os.path.join(self.img_dir, split), exist_ok=True)
            with open(os.path.join(self.img_dir, split, fname), "wb") as f:
                f.write(it["img_bytes"])  # 原始字节, 不重编码
        self.fout.write(json.dumps({
            "key": key, "split": split, "row": it["row"],
            "source_image": os.path.basename(it["src"]),
            "image": img_rel if self.save_images else None,
            "width": img.size[0], "height": img.size[1],
            "aesthetic_score": round(it["score"], 4),
            "scene": cap["scene"] if cap else None,
            "text_style": cap["text_style"] if cap else None,
            "scene_zh": cap["scene_zh"] if cap else None,
            "text_style_zh": cap["text_style_zh"] if cap else None,
            "caption_ok": bool(cap and cap.get("ok")),
            "quality": {k: it["q"][k] for k in ("blur_var", "brightness", "contrast")},
            "regions": [{"bbox": b["bbox"], "text": b["text"]} for b in it["kept"]],
        }, ensure_ascii=False) + "\n")

    def finish(self, t0):
        self.flush_aesthetic()
        self.flush_caption()
        self.fout.close()
        self.stats["fail_reasons"] = self.fail_reasons
        self.stats["elapsed_s"] = round(time.time() - t0, 1)
        stats_path = self.jsonl_path.replace(".jsonl", ".stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        log(f"✅ 完成: {self.stats['total']} 图 | 解码失败 {self.stats['fail_decode']}, "
            f"画质丢弃 {self.stats['fail_quality']}, bbox 丢弃 {self.stats['fail_bbox']}, "
            f"美学丢弃 {self.stats['fail_aesthetic']}, 通过 {self.stats['pass_all']} → {stats_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "configs/ocr_sa1b.yaml"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条 (冒烟)")
    ap.add_argument("--no-caption", action="store_true", help="跳过 VLM caption")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    jsonl_path = os.path.join(
        cfg["output"]["dir"],
        f"{cfg['output']['jsonl_prefix']}_shard{args.shard}of{args.num_shards}.jsonl")

    done = set()
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass

    pl = Pipeline(cfg, args.shard, args.num_shards, args.no_caption)
    pbar = tqdm(total=args.limit or None, desc=f"shard{args.shard}", unit="img",
                bar_format="{l_bar}{bar:24}{r_bar}")
    t0 = time.time()
    n_seen = 0
    for key, split, row, r in stream_rows(cfg["input"]["parquet_dir"],
                                          args.shard, args.num_shards, done):
        if args.limit and n_seen >= args.limit:
            break
        n_seen += 1
        pl.process_one(key, split, row, r)
        pbar.update(1)
        pbar.set_postfix(pass_all=pl.stats["pass_all"], aes_fail=pl.stats["fail_aesthetic"])
    pbar.close()
    pl.finish(t0)


if __name__ == "__main__":
    main()
