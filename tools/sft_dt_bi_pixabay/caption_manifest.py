#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""caption_manifest.py — Qwen3-VL-4B-Instruct 给 manifest 图像批量打 t2i caption。

读取 build_manifest.py 的 manifest.parquet, 按 chunk (默认 1000 图) 切分,
每 GPU 一个 worker 从队列领 chunk, 批量推理, 写 captions/chunk_{i:05d}.parquet
(列: sha256, caption)。分片级断点续跑: 重复执行自动跳过已完成 chunk。
解码失败的图 caption 记空串 (build 阶段剔除)。

用法:
  python3 caption_manifest.py --num-gpus 8          # 全量
  python3 caption_manifest.py --sample 16           # 首个 chunk 前 N 图试跑
  python3 caption_manifest.py --status              # 查看进度
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/exp/sft_dt_bi_pixabay/manifest.parquet"
CAPTIONS_DIR_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/exp/sft_dt_bi_pixabay/captions"
MODEL_PATH = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/Qwen3-VL-4B-Instruct"

PROMPT = (
    "Describe this image as a text-to-image prompt in one paragraph of about 80-100 words. "
    "Cover the subject and its attributes, composition and framing, art or photographic "
    "style, colors, lighting and atmosphere. Output only the description itself."
)


def chunk_path(cdir: Path, i: int) -> Path:
    return cdir / f"chunk_{i:05d}.parquet"


def load_manifest(path: str):
    t = pq.read_table(path)
    return t.to_pylist()


def done_chunks(cdir: Path, n: int) -> set[int]:
    done = set()
    for i in range(n):
        p = chunk_path(cdir, i)
        if p.exists():
            try:
                pq.ParquetFile(p)  # 可完整读取即视为完成 (0 行 = 全部解码失败, 重试无意义)
                done.add(i)
            except Exception:
                pass
    return done


def make_processor(max_pixels: int, min_pixels: int):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.min_pixels = min_pixels
    processor.tokenizer.padding_side = "left"  # decoder-only 批量生成必须左填充
    return processor


def worker_loop(gpu_id: int, task_queue, result_queue, args_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import gc

    import torch
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration

    cdir = Path(args_dict["captions_dir"])
    log_path = cdir / "_failed.log"

    def log_fail(msg: str):
        with open(log_path, "a") as f:
            f.write(f"[gpu{gpu_id} {time.strftime('%H:%M:%S')}] {msg}\n")

    print(f"[gpu{gpu_id}] 加载 {MODEL_PATH} ...", flush=True)
    processor = make_processor(args_dict["max_pixels"], args_dict["min_pixels"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    @torch.inference_mode()
    def generate_batch(images):
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
            tokenize=False, add_generation_prompt=True) for _ in images]
        bs = len(images)
        while True:
            try:
                inputs = processor(text=texts[:bs], images=[[im] for im in images[:bs]],
                                   padding=True, return_tensors="pt").to("cuda")
                out = model.generate(**inputs, max_new_tokens=args_dict["max_new_tokens"],
                                      do_sample=False)
                trimmed = out[:, inputs.input_ids.shape[1]:]
                result = [t.strip() for t in processor.batch_decode(trimmed, skip_special_tokens=True)]
                if bs < len(images):
                    result += generate_batch(images[bs:])
                return result
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    log_fail(f"gpu{gpu_id} 单图仍 OOM, 本批 {len(images)} 图返回空")
                    return [""] * len(images)
                bs = max(1, bs // 2)
                print(f"[gpu{gpu_id}] OOM, batch 减到 {bs}", flush=True)

    def process_chunk(chunk_rows):
        t0 = time.time()
        sha_list, caps, ok = [], [], 0
        bs = args_dict["batch_size"]
        for i in range(0, len(chunk_rows), bs):
            part = chunk_rows[i:i + bs]
            images, shas = [], []
            for r in part:
                try:
                    images.append(Image.open(r["file"]).convert("RGB"))
                    shas.append(r["sha256"])
                except Exception as e:
                    log_fail(f"{r['file']}: 解码失败 {e}")
            if images:
                for sha, cap in zip(shas, generate_batch(images)):
                    sha_list.append(sha)
                    caps.append(cap)
                    ok += 1
        table = pa.table({"sha256": pa.array(sha_list, pa.string()),
                          "caption": pa.array(caps, pa.string())})
        tmp = chunk_path(cdir, chunk_rows[0]["_chunk"]) .with_suffix(".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, chunk_path(cdir, chunk_rows[0]["_chunk"]))
        dt = time.time() - t0
        result_queue.put(("ok", gpu_id, chunk_rows[0]["_chunk"], ok, len(chunk_rows),
                          len(chunk_rows) / max(dt, 1e-6)))
        gc.collect()
        torch.cuda.empty_cache()

    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            process_chunk(task)
        except Exception as e:
            import traceback
            log_fail(f"chunk 失败 #{task[0]['_chunk']}: {e}\n{traceback.format_exc()}")
            result_queue.put(("error", gpu_id, task[0]["_chunk"], 0, 0, 0.0))


def run_all(args):
    from tqdm import tqdm

    rows = load_manifest(args.manifest)
    cdir = Path(args.captions_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    per = args.chunk_size
    chunks = []
    for i in range(0, len(rows), per):
        part = [dict(r, _chunk=i // per) for r in rows[i:i + per]]
        chunks.append(part)
    done = done_chunks(cdir, len(chunks))
    pending = [c for idx, c in enumerate(chunks) if idx not in done]
    print(f"✍️ Qwen3-VL caption: {len(rows):,} 图 / {len(chunks)} chunk, "
          f"已完成 {len(done)}, 待处理 {len(pending)}", flush=True)
    if not pending:
        print("✅ 全部完成", flush=True)
        return

    ctx = get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    for c in pending:
        task_queue.put(c)
    n_workers = min(args.num_gpus, len(pending))
    for _ in range(n_workers):
        task_queue.put(None)
    workers = [ctx.Process(target=worker_loop, daemon=True,
                           args=(g, task_queue, result_queue,
                                 dict(captions_dir=str(cdir), batch_size=args.batch_size,
                                      max_new_tokens=args.max_new_tokens,
                                      max_pixels=args.max_pixels, min_pixels=args.min_pixels)))
               for g in range(n_workers)]
    for w in workers:
        w.start()

    t0, n_ok, n_err, n_img = time.time(), 0, 0, 0
    pbar = tqdm(total=len(pending), desc="✍️ caption", unit="chunk")
    while n_ok + n_err < len(pending):
        try:
            ev = result_queue.get(timeout=5)
        except Exception:
            ev = None
            if not any(w.is_alive() for w in workers):
                print("⚠️ worker 全部退出但任务未完成, 中止 (可重跑断点续跑)", flush=True)
                break
        if ev:
            kind, gpu, cidx, ok, n, rate = ev
            if kind == "ok":
                n_ok += 1
                n_img += ok
                speed = n_img / max(time.time() - t0, 1)
                pbar.set_postfix(gpu=gpu, imgs=f"{ok}/{n}", speed=f"{speed:,.0f} img/s")
            else:
                n_err += 1
                print(f"\n❌ [gpu{gpu}] chunk {cidx} 失败, 见 captions/_failed.log", flush=True)
            pbar.update(1)
    pbar.close()
    for w in workers:
        w.join(timeout=10)
    print(f"✅ 运行结束: 成功 {n_ok} chunk / 失败 {n_err} chunk, "
          f"已打标 {n_img:,} 图, 用时 {(time.time()-t0)/60:.0f} 分钟", flush=True)


def run_sample(args):
    import torch
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration

    rows = load_manifest(args.manifest)[:args.sample]
    print(f"🔎 试跑前 {len(rows)} 图", flush=True)
    processor = make_processor(args.max_pixels, args.min_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    images = [Image.open(r["file"]).convert("RGB") for r in rows]
    texts = [processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        tokenize=False, add_generation_prompt=True) for _ in images]
    with torch.inference_mode():
        inputs = processor(text=texts, images=[[im] for im in images], padding=True,
                           return_tensors="pt").to("cuda")
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        dt = time.time() - t0
    caps = [t.strip() for t in processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                                     skip_special_tokens=True)]
    print(f"{len(images)} 图耗时 {dt:.1f}s ({len(images)/dt:.1f} 图/s/卡) "
          f"→ 8 卡估算 {len(images)/dt*8*3600:,.0f} 图/小时\n", flush=True)
    for r, cap in zip(rows, caps):
        print(f"--- {r['platform']}/{r['query']} {Path(r['file']).name} ---\n{cap}\n", flush=True)


def show_status(args):
    rows = load_manifest(args.manifest)
    cdir = Path(args.captions_dir)
    per = args.chunk_size
    n_chunks = (len(rows) + per - 1) // per
    done = done_chunks(cdir, n_chunks)
    print(f"📊 进度: {len(done)}/{n_chunks} chunk "
          f"({len(done)*per:,}/{len(rows):,} 图, chunk={per})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--captions-dir", default=CAPTIONS_DIR_DEFAULT)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=24, help="每卡每批图像数")
    ap.add_argument("--max-new-tokens", type=int, default=240)
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--min-pixels", type=int, default=256 * 256)
    ap.add_argument("--sample", type=int, default=0, help="仅试跑前 N 图")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        show_status(args)
    elif args.sample:
        run_sample(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
