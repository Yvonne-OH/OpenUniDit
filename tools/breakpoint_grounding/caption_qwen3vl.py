"""breakpoint-grounding-55m caption stage 2/2: Qwen3-VL-4B 给过滤后图像打 t2i caption。

读取 stage 1 sidecar (quality_filter/<stem>.parquet) 按阈值过滤, 只给通过质量过滤的行打标:
  sidecar: <root>/qwen_caption/<stem>.parquet (index, caption; index 为分片内行号)
  分片级断点续跑 (重复执行自动跳过已完成分片)
merge: 原分片 + 质量指标 + caption 按阈值合并 → <root>/filtered_parquet/<stem>.parquet
       (image/image_caption/object_captions/normalized_boxes/img_size_wh + 新增列), 原目录不动。

8 卡每卡一个 worker, 分片队列消费; --all-rows 跳过质量过滤给全部行打标。

用法:
  python caption_qwen3vl.py --sample 16      # 首分片前 N 行(过滤后)试跑 + html 预览
  python caption_qwen3vl.py --num-gpus 8     # 全量 8 卡
  python caption_qwen3vl.py --status
  python caption_qwen3vl.py --merge          # 合并出 filtered_parquet/
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import io
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pyarrow.compute as pc

ROOT_DEFAULT = "/mnt/nfs/data/zhanqian.wu/unified_canvas_images/breakpoint-grounding-55m"
MODEL_PATH = "/mnt/nfs/models/huggingface/Qwen/Qwen3-VL-4B-Instruct"

FILTER_DIR = "quality_filter"
CAPTION_DIR = "qwen_caption"
PROGRESS_RELPATH = f"{CAPTION_DIR}/_progress.json"
LOCK_RELPATH = f"{CAPTION_DIR}/_progress.lock"

PROMPT = (
    "Describe this image as a text-to-image prompt in one paragraph of about 80-100 words. "
    "Cover the subject and its attributes, composition and framing, art or photographic "
    "style, colors, lighting and atmosphere. Output only the description itself."
)

# 过滤阈值默认值 (与 stage 1 指标对应, 可 CLI 调整)
DEFAULT_THRESHOLDS = dict(min_side=384, blur_th=60.0, jpeg_th=1.6, aes_th=4.5)


def row_pass(m: dict | None, th: dict) -> bool:
    """单行质量判定; m 为 stage 1 sidecar 行 (dict)。"""
    if not m or not m.get("decode_ok"):
        return False
    return (m.get("min_side") or 0) >= th["min_side"] \
        and (m.get("laplacian_var") or 0.0) >= th["blur_th"] \
        and (m.get("jpeg_block_score") or 0.0) <= th["jpeg_th"] \
        and (m.get("aesthetic_score") or 0.0) >= th["aes_th"]


def load_filter_rows(root: Path, stem: str) -> list[dict] | None:
    import pyarrow.parquet as pq
    p = root / FILTER_DIR / f"{stem}.parquet"
    if not p.exists():
        return None
    return pq.read_table(p).to_pylist()


# ─────────────────────────── 分片级进度 ───────────────────────────

class Progress:
    def __init__(self, root: Path):
        self.path = root / PROGRESS_RELPATH
        self.lock_path = root / LOCK_RELPATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _update(self, fn):
        with open(self.lock_path, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {}
            data = fn(data)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
            os.replace(tmp, self.path)

    def done_shards(self) -> set[str]:
        if not self.path.exists():
            return set()
        return set(json.loads(self.path.read_text()).keys())

    def mark_done(self, stem: str, rows: int, seconds: float, worker: int):
        def fn(data):
            data[stem] = dict(rows=rows, seconds=round(seconds, 1), worker=worker, ts=time.time())
            return data
        self._update(fn)

    def snapshot(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}


def list_shards(root: Path) -> list[Path]:
    return sorted((root / "data").glob("train-*.parquet"))


def pq_rows(shard: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(shard).metadata.num_rows


# ─────────────────────────── GPU worker ───────────────────────────

def make_processor(max_pixels: int, min_pixels: int):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.min_pixels = min_pixels
    processor.tokenizer.padding_side = "left"  # decoder-only 批量生成必须左填充
    return processor


def worker_loop(gpu_id: int, task_queue, result_queue, root_str: str, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import gc

    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration

    root = Path(root_str)
    progress = Progress(root)
    log_path = root / CAPTION_DIR / "_failed.log"

    def log_fail(msg: str):
        with open(log_path, "a") as f:
            f.write(f"[gpu{gpu_id} {time.strftime('%H:%M:%S')}] {msg}\n")

    print(f"[gpu{gpu_id}] 加载 {MODEL_PATH} ...", flush=True)
    processor = make_processor(args_dict["max_pixels"], args_dict["min_pixels"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    @torch.inference_mode()
    def generate_batch(images: list) -> list[str]:
        texts = []
        for _ in images:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        bs = len(images)
        while True:
            try:
                inputs = processor(text=texts[:bs], images=[[im] for im in images[:bs]],
                                   padding=True, return_tensors="pt").to("cuda")
                out = model.generate(**inputs, max_new_tokens=args_dict["max_new_tokens"], do_sample=False)
                trimmed = out[:, inputs.input_ids.shape[1]:]
                result = [t.strip() for t in processor.batch_decode(trimmed, skip_special_tokens=True)]
                if bs < len(images):  # OOM 减半后先处理前半, 后半递归补上
                    result += generate_batch(images[bs:])
                return result
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    log_fail(f"gpu{gpu_id} 单图仍 OOM, 本批 {len(images)} 图返回空")
                    return [""] * len(images)
                bs = max(1, bs // 2)
                print(f"[gpu{gpu_id}] OOM, batch 减到 {bs}", flush=True)

    def process_task(task: dict):
        t0 = time.time()
        shard = Path(task["shard"])
        n_total = pq_rows(shard)

        # 过滤行选择: stage 1 sidecar + 阈值; sidecar 缺失时退化为全行并记日志
        if args_dict["all_rows"]:
            pass_idx = list(range(n_total))
        else:
            frows = load_filter_rows(root, shard.stem)
            if frows is None:
                log_fail(f"{shard.stem}: quality_filter sidecar 缺失, 退化为全行打标")
                pass_idx = list(range(n_total))
            else:
                pass_idx = [int(r["index"]) for r in frows if row_pass(r, args_dict["thresholds"])]

        if not pass_idx:
            out_table = pa.table({"index": pa.array([], pa.int32()),
                                  "caption": pa.array([], pa.string())})
        else:
            table = pq.read_table(shard, columns=["image"])
            raw_list = pc.struct_field(table.column("image"), "bytes").to_pylist()
            del table
            images = []
            for idx in pass_idx:
                try:
                    images.append(Image.open(io.BytesIO(raw_list[idx])).convert("RGB"))
                except Exception as e:
                    log_fail(f"{shard.stem} 行 {idx} 解码失败: {e}")
                    images.append(None)
            del raw_list

            captions: list[str] = [""] * len(images)
            bs = args_dict["batch_size"]
            for i in range(0, len(images), bs):
                part = images[i:i + bs]
                valid = [j for j, im in enumerate(part) if im is not None]
                if valid:
                    for j, cap in zip(valid, generate_batch([part[j] for j in valid])):
                        captions[i + j] = cap
            out_table = pa.table({"index": pa.array(pass_idx, pa.int32()),
                                  "caption": pa.array(captions, pa.string())})
            del images

        out_dir = root / CAPTION_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".{shard.stem}.tmp"
        import pyarrow.parquet as pq_w
        pq_w.write_table(out_table, tmp)
        os.replace(tmp, out_dir / f"{shard.stem}.parquet")

        dt = time.time() - t0
        progress.mark_done(shard.stem, len(pass_idx), dt, gpu_id)
        result_queue.put(("ok", gpu_id, shard.stem, n_total, len(pass_idx), len(pass_idx) / max(dt, 1e-6)))
        gc.collect()
        torch.cuda.empty_cache()

    while True:
        task = task_queue.get()
        if task is None:
            return
        try:
            process_task(task)
        except Exception as e:
            import traceback
            log_fail(f"分片失败 {task['shard']}: {e}\n{traceback.format_exc()}")
            result_queue.put(("error", gpu_id, Path(task["shard"]).stem, 0, 0, 0.0))


# ─────────────────────────── 主驱动 ───────────────────────────

def run_all(args):
    from tqdm import tqdm

    root = Path(args.root)
    shards = list_shards(root)
    progress = Progress(root)
    pending = [s for s in shards if s.stem not in progress.done_shards()]
    total_rows = sum(pq_rows(s) for s in shards)
    done_rows = total_rows - sum(pq_rows(s) for s in pending)
    print(f"✍️ Qwen3-VL caption: {len(shards)} 分片 / {total_rows:,} 行, "
          f"已完成 {len(shards) - len(pending)} 片 ({done_rows:,} 行), 待处理 {len(pending)} 片 "
          f"(阈值: {args.thresholds()})", flush=True)
    if not pending:
        print("✅ 全部完成", flush=True)
        return

    ctx = get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    for s in pending:
        task_queue.put(dict(shard=str(s)))
    n_workers = min(args.num_gpus, len(pending))
    for _ in range(n_workers):
        task_queue.put(None)
    workers = [ctx.Process(target=worker_loop,
                           args=(g, task_queue, result_queue, str(root),
                                 dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                                      max_pixels=args.max_pixels, min_pixels=args.min_pixels,
                                      all_rows=args.all_rows, thresholds=args.thresholds())),
                           daemon=True)
               for g in range(n_workers)]
    for w in workers:
        w.start()

    t0, n_ok, n_err = time.time(), 0, 0
    pbar = tqdm(total=len(pending), desc="✍️ qwen3vl caption", unit="shard")
    while n_ok + n_err < len(pending):
        try:
            ev = result_queue.get(timeout=5)
        except Exception:
            ev = None
            if not any(w.is_alive() for w in workers):
                print("⚠️ worker 全部退出但任务未完成, 中止 (可重新运行断点续跑)", flush=True)
                break
        if ev:
            kind, gpu, stem, n_total, n_cap, rate = ev
            if kind == "ok":
                n_ok += 1
                pbar.set_postfix(gpu=gpu, cap=f"{n_cap}/{n_total}", speed=f"{rate:.1f} img/s")
            else:
                n_err += 1
                print(f"\n❌ [gpu{gpu}] {stem} 失败, 见 qwen_caption/_failed.log", flush=True)
            pbar.update(1)
            snap = progress.snapshot()
            speed = sum(v.get("rows", 0) for v in snap.values()) / max(time.time() - t0, 1)
            pbar.set_description(f"✍️ caption {speed:,.0f} rows/s")
    pbar.close()
    for w in workers:
        w.join(timeout=10)
    print(f"✅ 运行结束: 成功 {n_ok} 片, 失败 {n_err} 片, 用时 {(time.time()-t0)/60:.0f} 分钟", flush=True)


def show_status(args):
    root = Path(args.root)
    shards = list_shards(root)
    snap = Progress(root).snapshot()
    total_rows = sum(pq_rows(s) for s in shards)
    cap_rows = sum(v.get("rows", 0) for v in snap.values())
    print(f"📊 进度: {len(snap)}/{len(shards)} 分片, 已打标 {cap_rows:,} 行 (总行 {total_rows:,})", flush=True)


# ─────────────────────────── 小样本试跑 ───────────────────────────

def run_sample(args):
    import torch
    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration

    root = Path(args.root)
    shard = list_shards(root)[0]
    th = args.thresholds()
    print(f"🔎 试跑 {shard.name}: 前 {args.sample} 个通过过滤的行 (阈值 {th})", flush=True)

    frows = load_filter_rows(root, shard.stem)
    if frows is None:
        print("❌ quality_filter sidecar 不存在, 先跑 stage 1 (--sample)", flush=True)
        return
    pass_idx = [int(r["index"]) for r in frows if row_pass(r, th)][:args.sample]
    n_pass = sum(1 for r in frows if row_pass(r, th))
    print(f"该分片全量通过过滤: {n_pass}/{len(frows)} 行 ({100*n_pass/len(frows):.1f}%)", flush=True)

    processor = make_processor(args.max_pixels, args.min_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    table = pq.read_table(shard, columns=["image"])
    raw_list = pc.struct_field(table.column("image"), "bytes").to_pylist()
    images = [Image.open(io.BytesIO(raw_list[i])).convert("RGB") for i in pass_idx]

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
    rate = len(images) / max(dt, 1e-6)
    print(f"\n{len(images)} 图耗时 {dt:.1f}s ({rate:.1f} 图/s/卡) → 8 卡估算 {rate*8*3600:,.0f} 图/小时\n", flush=True)

    out_dir = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/breakpoint_grounding/caption_sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_html = []
    for idx, (im, cap) in zip(pass_idx, zip(images, caps)):
        thumb = im.copy()
        thumb.thumbnail((384, 384))
        p = out_dir / f"s{idx}.jpg"
        thumb.save(p, quality=88)
        b64 = base64.b64encode(p.read_bytes()).decode()
        rows_html.append(f'<div style="display:flex;gap:16px;margin:12px 0;padding:12px;'
                         f'border:1px solid #ddd;border-radius:8px">'
                         f'<img src="data:image/jpeg;base64,{b64}" style="max-width:384px;border-radius:4px">'
                         f'<div style="font-size:13px">idx={idx}</div><div style="margin-top:6px">{cap}</div></div>')
        print(f"--- idx={idx} ---\n{cap}\n", flush=True)
    html = ('<!doctype html><meta charset="utf-8"><title>caption sample</title>'
            '<body style="font-family:sans-serif;max-width:900px;margin:auto">'
            f'<h2>Qwen3-VL caption 样本 ({shard.stem})</h2>' + "".join(rows_html) + "</body>")
    (out_dir / "index.html").write_text(html)
    print(f"🖼 预览页: {out_dir / 'index.html'}", flush=True)


# ─────────────────────────── merge ───────────────────────────

def run_merge(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(args.root)
    out_dir = root / "filtered_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    th = args.thresholds()
    shards = list_shards(root)
    stats = dict(total=0, kept=0, drop_small=0, drop_blur=0, drop_blocky=0, drop_aes=0,
                 drop_decode=0, drop_nocap=0)

    def drop_reason(m: dict, th: dict) -> str | None:
        if not m or not m.get("decode_ok"):
            return "drop_decode"
        if (m.get("min_side") or 0) < th["min_side"]:
            return "drop_small"
        if (m.get("laplacian_var") or 0.0) < th["blur_th"]:
            return "drop_blur"
        if (m.get("jpeg_block_score") or 0.0) > th["jpeg_th"]:
            return "drop_blocky"
        if (m.get("aesthetic_score") or 0.0) < th["aes_th"]:
            return "drop_aes"
        return None

    for shard in shards:
        cap_path = root / CAPTION_DIR / f"{shard.stem}.parquet"
        if not cap_path.exists():
            print(f"⏭ {shard.stem}: 无 caption sidecar, 跳过", flush=True)
            continue
        frows = load_filter_rows(root, shard.stem) or []
        fmap = {int(r["index"]): r for r in frows}
        cmap = {int(r["index"]): r["caption"] for r in pq.read_table(cap_path).to_pylist()}

        table = pq.read_table(shard)
        keep, aes_col, cap_col, lap_col = [], [], [], []
        for i in range(table.num_rows):
            stats["total"] += 1
            m = fmap.get(i)
            reason = drop_reason(m, th)
            if reason:
                stats[reason] += 1
                continue
            if i not in cmap or not cmap[i]:
                stats["drop_nocap"] += 1
                continue
            keep.append(i)
            aes_col.append(float(m["aesthetic_score"]))
            lap_col.append(float(m["laplacian_var"]))
            cap_col.append(cmap[i])

        kept = table.take(pa.array(keep, pa.int64()))
        kept = kept.append_column(pa.field("qwen_caption", pa.string()), pa.array(cap_col))
        kept = kept.append_column(pa.field("aesthetic_score", pa.float32()), pa.array(aes_col))
        kept = kept.append_column(pa.field("laplacian_var", pa.float32()), pa.array(lap_col))
        tmp = out_dir / f".{shard.stem}.tmp"
        pq.write_table(kept, tmp)
        os.replace(tmp, out_dir / f"{shard.stem}.parquet")
        del table, kept
        print(f"✅ {shard.stem}: 保留 {len(keep)}/{table.num_rows} 行", flush=True)

    print(f"\n════ merge 汇总 (阈值 {th}) ════")
    print(f"  总行: {stats['total']:,}")
    for k in ("drop_decode", "drop_small", "drop_blur", "drop_blocky", "drop_aes", "drop_nocap"):
        print(f"  {k}: {stats[k]:,} ({100*stats[k]/max(stats['total'],1):.1f}%)")
    print(f"  保留: {stats['kept']:,} ({100*stats['kept']/max(stats['total'],1):.1f}%) → {out_dir}", flush=True)


# ─────────────────────────── CLI ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=24, help="每卡每批图像数")
    ap.add_argument("--max-new-tokens", type=int, default=240)
    ap.add_argument("--max-pixels", type=int, default=768 * 768, help="单图最大像素 (控制 vision token 数)")
    ap.add_argument("--min-pixels", type=int, default=256 * 256)
    ap.add_argument("--all-rows", action="store_true", help="跳过质量过滤, 全部行打标")
    ap.add_argument("--min-side", type=int, default=DEFAULT_THRESHOLDS["min_side"])
    ap.add_argument("--blur-th", type=float, default=DEFAULT_THRESHOLDS["blur_th"])
    ap.add_argument("--jpeg-th", type=float, default=DEFAULT_THRESHOLDS["jpeg_th"])
    ap.add_argument("--aes-th", type=float, default=DEFAULT_THRESHOLDS["aes_th"])
    ap.add_argument("--sample", type=int, default=0, help="仅跑首分片前 N 个通过行做人工确认")
    ap.add_argument("--status", action="store_true", help="只查看进度")
    ap.add_argument("--merge", action="store_true", help="合并输出 filtered_parquet/")
    args = ap.parse_args()
    args.thresholds = lambda: dict(min_side=args.min_side, blur_th=args.blur_th,
                                   jpeg_th=args.jpeg_th, aes_th=args.aes_th)

    if args.status:
        show_status(args)
    elif args.merge:
        run_merge(args)
    elif args.sample:
        run_sample(args)
    else:
        run_all(args)


if __name__ == "__main__":
    sys.exit(main())
