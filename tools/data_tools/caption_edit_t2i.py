"""用 Qwen3-VL (HF transformers) 给图像编辑数据集的原始图像打 text-to-image 描述。

对 unified_edit_images_lmdb 各 source 逐分片 (parquet+mdb 一一对应) 处理:
- 单图对源 (img_before/img_after 字节存 mdb): 给 img_before 打标, t2i_caption 存 str;
- 多参考图源 (ref_paths 指向本地文件): 每张参考图各打一条, t2i_caption 存 list[str]。
一个分片全部打完才原子提交 (tmp + os.replace), 不影响原 parquet/mdb;
输出 sidecar: <root>/<source>/t2i_caption/<shard_stem>.parquet (sample_id, t2i_caption)。

8 进程 8 卡, 分片级断点续跑 (重复执行自动跳过已完成分片)。

用法:
  # 8 卡全量
  python caption_edit_t2i.py --num-gpus 8
  # 小样本确认 (指定 source 第一个分片前 N 行, 存 html 预览)
  python caption_edit_t2i.py --sources pico-banana-400k --sample 16
  # 查看进度 / 合并 (原 parquet + t2i_caption 列 → <source>/parquet_t2i/)
  python caption_edit_t2i.py --status
  python caption_edit_t2i.py --merge
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import pickle
import struct
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_edit_images_lmdb"
MODEL_PATH = "/mnt/nfs/models/huggingface/Qwen/Qwen3-VL-4B-Instruct"

# mdb 记录里原始图像字节字段候选 (对齐 data/datasets/image_database.py 的 _LMDB_FIELD_BEFORE)
BEFORE_FIELDS = ("img_before", "img_before_bytes", "input_img", "img_input", "before")

PROMPT = (
    "Describe this image as a text-to-image prompt in one paragraph of about 80-100 words. "
    "Cover the subject and its attributes, composition and framing, art or photographic "
    "style, colors, lighting and atmosphere. Output only the description itself."
)

PROGRESS_RELPATH = "t2i_caption/_progress.json"
LOCK_RELPATH = "t2i_caption/_progress.lock"


# ─────────────────────────── 数据枚举 ───────────────────────────

def source_parquet_shards(source_dir: Path) -> list[Path]:
    """parquet 分片, 与 mdb 分片按 sorted 顺序一一对应 (对齐 lmdb_index.py)。"""
    files = sorted(source_dir.glob("parquet/meta_*.parquet"))
    if not files:
        files = sorted(p for p in source_dir.glob("parquet/*.parquet") if p.name != "meta.parquet")
    return files


def count_rows(parquet_files: list[Path]) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)


def build_tasks(root: Path, only_sources: list[str] | None) -> tuple[list[dict], dict]:
    """按 source 数据量升序生成分片任务 (小数据集先打)。"""
    import pyarrow.parquet as pq
    sources = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")) or d.name == "t2i_caption":
            continue
        if only_sources and d.name not in only_sources:
            continue
        pqs = source_parquet_shards(d)
        mdbs = sorted((d / "mdb").glob("*.mdb"))
        if not pqs or len(pqs) != len(mdbs):
            print(f"⚠️ 跳过分片不匹配的 source: {d.name} parquet={len(pqs)} mdb={len(mdbs)}", flush=True)
            continue
        sources.append((count_rows(pqs), d, pqs, mdbs))
    sources.sort(key=lambda x: x[0])
    tasks, totals = [], {}
    for rows, d, pqs, mdbs in sources:
        totals[d.name] = rows
        for pf, mf in zip(pqs, mdbs):
            tasks.append(dict(source=d.name, source_dir=str(d), parquet=str(pf),
                              mdb=str(mf), stem=pf.stem, rows=pq.ParquetFile(pf).metadata.num_rows))
    return tasks, totals


# ─────────────────────────── 进度 (分片级, 断点续跑) ───────────────────────────

class Progress:
    """分片级进度, flock 保护并发读写; 一个分片完成后才整体更新。"""

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

    def done_shards(self, source: str) -> set[str]:
        if not self.path.exists():
            return set()
        return set(json.loads(self.path.read_text()).get(source, {}).keys())

    def mark_done(self, source: str, stem: str, rows: int, seconds: float, worker: int):
        def fn(data):
            data.setdefault(source, {})[stem] = dict(rows=rows, seconds=round(seconds, 1),
                                                     worker=worker, ts=time.time())
            return data
        self._update(fn)

    def snapshot(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}


# ─────────────────────────── 图像加载 ───────────────────────────

def load_mdb_records(mdb_path: str, sample_ids: list[int]) -> list[dict | None]:
    """批量读 mdb pickle 记录, 自动探测 key scheme (uint64 小端 / ascii 十进制)。"""
    import lmdb
    env = lmdb.open(mdb_path, readonly=True, lock=False, subdir=False, max_readers=1024)
    records: list[dict | None] = [None] * len(sample_ids)
    with env.begin() as txn:
        scheme = None
        with txn.cursor() as cur:
            if cur.first():
                scheme = "pack" if len(cur.key()) == 8 else "ascii"
        keys = ([struct.pack("<Q", int(s)) for s in sample_ids] if scheme == "pack"
                else [str(int(s)).encode() for s in sample_ids])
        for i, key in enumerate(keys):
            raw = txn.get(key)
            if raw is not None:
                records[i] = pickle.loads(raw)
    env.close()
    return records


def before_image_from_record(rec: dict) -> bytes | None:
    for f in BEFORE_FIELDS:
        v = rec.get(f)
        if isinstance(v, bytes):
            return v
    return None


# ─────────────────────────── GPU worker (HF transformers) ───────────────────────────

def make_processor(args_dict: dict | None = None):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    if args_dict:
        processor.image_processor.max_pixels = args_dict["max_pixels"]
        processor.image_processor.min_pixels = args_dict["min_pixels"]
    processor.tokenizer.padding_side = "left"  # decoder-only 批量生成必须左填充
    return processor


def worker_loop(gpu_id: int, task_queue, result_queue, root_str: str, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import io
    import gc
    import torch
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import Qwen3VLForConditionalGeneration

    root = Path(root_str)
    progress = Progress(root)
    log_path = root / "t2i_caption" / "_failed.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_fail(msg: str):
        with open(log_path, "a") as f:
            f.write(f"[gpu{gpu_id} {time.strftime('%H:%M:%S')}] {msg}\n")

    print(f"[gpu{gpu_id}] 加载模型 {MODEL_PATH} ...", flush=True)
    processor = make_processor(args_dict)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    def open_image(record: dict | None, sample_id: int, ref_idx: int | None) -> Image.Image | None:
        try:
            if ref_idx is None:  # 单图对源: mdb 字节
                raw = before_image_from_record(record)
                if raw is None:
                    raise KeyError(f"record 无 before 字段: fields={sorted(record or {})}")
                return Image.open(io.BytesIO(raw)).convert("RGB")
            paths = (record or {}).get("ref_paths") or []
            return Image.open(paths[ref_idx]).convert("RGB")
        except Exception as e:
            log_fail(f"读图失败 sample_id={sample_id} ref={ref_idx}: {e}")
            return None

    @torch.inference_mode()
    def generate_batch(images: list) -> list[str]:
        """一批图生成 caption, CUDA OOM 自动减半重试。"""
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
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
                result = [t.strip() for t in decoded]
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
        table = pq.read_table(task["parquet"], columns=["sample_id"])
        sample_ids = [int(x) for x in table.column("sample_id").to_pylist()]
        records = load_mdb_records(task["mdb"], sample_ids)
        n_missing = sum(r is None for r in records)
        if n_missing:
            log_fail(f"{task['stem']}: {n_missing}/{len(sample_ids)} 条 mdb 记录缺失")

        is_multi = records and isinstance(records[0], dict) and "ref_paths" in records[0]
        flat = []  # (行号, 参考图序号) — 单图源 ref_idx=None
        for ri, rec in enumerate(records):
            for k in (range(len(rec["ref_paths"])) if is_multi else [None]):
                flat.append((ri, k))

        captions: dict[tuple[int, int], str] = {}
        bs = args_dict["batch_size"]
        for i in range(0, len(flat), bs):
            part = flat[i:i + bs]
            images = [open_image(records[ri], sample_ids[ri], k) for ri, k in part]
            valid = [(j, im) for j, im in enumerate(images) if im is not None]
            outs = [""] * len(part)
            if valid:
                generated = generate_batch([im for _, im in valid])
                for (j, _), cap in zip(valid, generated):
                    outs[j] = cap
            for (ri, k), cap in zip(part, outs):
                captions[(ri, k)] = cap

        # 组装行级 t2i_caption
        result = []
        for ri in range(len(records)):
            if is_multi:
                n_refs = len(records[ri]["ref_paths"])
                result.append([captions.get((ri, k), "") for k in range(n_refs)])
            else:
                result.append(captions.get((ri, None), ""))

        # 原子提交 sidecar
        out_dir = Path(task["source_dir"]) / "t2i_caption"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_table = pa.table({"sample_id": sample_ids, "t2i_caption": result})
        tmp = out_dir / f".{task['stem']}.tmp"
        pq.write_table(out_table, tmp)
        os.replace(tmp, out_dir / f"{task['stem']}.parquet")

        dt = time.time() - t0
        progress.mark_done(task["source"], task["stem"], len(sample_ids), dt, gpu_id)
        result_queue.put(("shard", gpu_id, task["source"], task["stem"],
                          len(sample_ids), len(flat), len(flat) / max(dt, 1e-6)))
        del records, captions
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
            log_fail(f"分片失败 {task['stem']}: {e}\n{traceback.format_exc()}")
            result_queue.put(("error", gpu_id, task["source"], task["stem"], 0, 0, 0.0))


# ─────────────────────────── 主驱动 ───────────────────────────

def run_all(args):
    root = Path(args.root)
    tasks, totals = build_tasks(root, args.sources or None)
    total_rows = sum(totals.values())
    print(f"共 {len(totals)} 个 source, {len(tasks)} 个分片, {total_rows:,} 行 (多图源按参考图数会更多)", flush=True)

    progress = Progress(root)
    pending = [t for t in tasks if t["stem"] not in progress.done_shards(t["source"])]
    done_rows = total_rows - sum(t["rows"] for t in pending)
    print(f"断点检查: 已完成 {len(tasks) - len(pending)} 个分片 ({done_rows:,} 行), 待处理 {len(pending)} 个分片", flush=True)
    if not pending:
        print("✅ 全部完成", flush=True)
        return

    ctx = get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    args_dict = dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                     max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    for t in pending:
        task_queue.put(t)
    n_workers = min(args.num_gpus, len(pending))
    for _ in range(n_workers):
        task_queue.put(None)
    workers = [ctx.Process(target=worker_loop, args=(g, task_queue, result_queue, str(root), args_dict),
                           daemon=True) for g in range(n_workers)]
    for w in workers:
        w.start()

    t0 = time.time()
    done_shards, n_err, last_summary = 0, 0, 0.0
    while done_shards + n_err < len(pending):
        try:
            ev = result_queue.get(timeout=5)
        except Exception:
            ev = None
        if ev:
            kind, gpu, source, stem, rows, n_imgs, rate = ev
            if kind == "shard":
                done_shards += 1
                print(f"✅ [gpu{gpu}] {source}/{stem}: {rows} 行, {n_imgs} 图, {rate:.1f} 图/s", flush=True)
            else:
                n_err += 1
                print(f"❌ [gpu{gpu}] {source}/{stem} 失败, 见 _failed.log", flush=True)
            if time.time() - last_summary >= args.status_interval:
                last_summary = print_summary(progress, totals, t0, done_shards, len(pending))
        if not any(w.is_alive() for w in workers) and done_shards + n_err < len(pending):
            print("⚠️ 有 worker 退出但任务未完成, 中止 (可重新运行断点续跑)", flush=True)
            break
    for w in workers:
        w.join(timeout=10)
    print_summary(progress, totals, t0, done_shards, len(pending))
    print("✅ 本次运行结束", flush=True)


def print_summary(progress: Progress, totals: dict, t0: float, done_shards: int, total_shards: int) -> float:
    snap = progress.snapshot()
    elapsed = time.time() - t0
    print(f"\n════ 进度 {done_shards}/{total_shards} 分片 · 已运行 {elapsed/60:.0f} 分钟 ════", flush=True)
    for source, total_rows in totals.items():
        shards = snap.get(source, {})
        rows = sum(v.get("rows", 0) for v in shards.values())
        print(f"  {source}: {rows:,}/{total_rows:,} 行 ({100*rows/max(total_rows,1):.1f}%), {len(shards)} 分片", flush=True)
    return time.time()


def show_status(args):
    root = Path(args.root)
    progress = Progress(root)
    tasks, totals = build_tasks(root, args.sources or None)
    snap = progress.snapshot()
    total_done = 0
    for source, total_rows in totals.items():
        shards = snap.get(source, {})
        rows = sum(v.get("rows", 0) for v in shards.values())
        total_done += rows
        print(f"  {source}: {rows:,}/{total_rows:,} 行 ({100*rows/max(total_rows,1):.1f}%), "
              f"{len(shards)} 分片完成", flush=True)
    print(f"合计: {total_done:,}/{sum(totals.values()):,} 行", flush=True)


# ─────────────────────────── 小样本预览 ───────────────────────────

def run_sample(args):
    """指定 source 第一个分片前 N 行打标, 存 html 预览供人工确认。"""
    import io
    import torch
    from PIL import Image
    import pyarrow.parquet as pq
    from transformers import Qwen3VLForConditionalGeneration

    root = Path(args.root)
    tasks, totals = build_tasks(root, args.sources or None)
    task = tasks[0]
    print(f"样本来源: {task['source']}/{task['stem']} (该 source 共 {totals[task['source']]:,} 行)", flush=True)

    args_dict = dict(max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    processor = make_processor(args_dict)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()

    table = pq.read_table(task["parquet"], columns=["sample_id"])
    sample_ids = [int(x) for x in table.column("sample_id").to_pylist()][:args.sample]
    records = load_mdb_records(task["mdb"], sample_ids)
    is_multi = records and "ref_paths" in records[0]

    flat_imgs, flat_meta = [], []  # meta=(sample_id, ref_idx)
    for ri, sid in enumerate(sample_ids):
        rec = records[ri]
        refs = list(range(len(rec["ref_paths"]))) if is_multi else [None]
        for k in refs:
            try:
                if k is None:
                    img = Image.open(io.BytesIO(before_image_from_record(rec))).convert("RGB")
                else:
                    img = Image.open(rec["ref_paths"][k]).convert("RGB")
                flat_imgs.append(img)
                flat_meta.append((sid, k))
            except Exception as e:
                print(f"⚠️ 读图失败 sample_id={sid} ref={k}: {e}", flush=True)

    texts = []
    for _ in flat_imgs:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    with torch.inference_mode():
        inputs = processor(text=texts, images=[[im] for im in flat_imgs], padding=True,
                           return_tensors="pt").to("cuda")
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        dt = time.time() - t0
    caps = [t.strip() for t in processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                                      skip_special_tokens=True)]
    rate = len(flat_imgs) / max(dt, 1e-6)
    print(f"\n{len(flat_imgs)} 图耗时 {dt:.1f}s ({rate:.1f} 图/s/卡) → 8 卡估算 {rate*8*3600:,.0f} 图/小时\n", flush=True)

    out_dir = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/edit_t2i_caption_sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_html = []
    for (sid, k), (im, cap) in zip(flat_meta, zip(flat_imgs, caps)):
        im = im.copy()
        im.thumbnail((384, 384))
        thumb = out_dir / f"s{sid}_r{k}.jpg"
        im.convert("RGB").save(thumb, quality=88)
        b64 = base64.b64encode(thumb.read_bytes()).decode()
        label = f"sample_id={sid}" + (f" ref_{k}" if k is not None else "")
        rows_html.append(f'<div style="display:flex;gap:16px;margin:12px 0;padding:12px;border:1px solid #ddd;border-radius:8px">'
                         f'<img src="data:image/jpeg;base64,{b64}" style="max-width:384px;border-radius:4px">'
                         f'<div><div style="color:#888;font-size:12px">{task["source"]} · {label}</div>'
                         f'<div style="margin-top:6px">{cap}</div></div></div>')
        print(f"--- sample_id={sid}" + (f" ref_{k}" if k is not None else "") + " ---\n" + cap + "\n", flush=True)
    html = f'<!doctype html><meta charset="utf-8"><title>t2i_caption sample</title><body style="font-family:sans-serif;max-width:900px;margin:auto"><h2>Qwen3-VL t2i_caption 样本 ({task["source"]})</h2>' + "".join(rows_html) + "</body>"
    (out_dir / "index.html").write_text(html)
    print(f"🖼 预览页: {out_dir / 'index.html'}", flush=True)


# ─────────────────────────── 合并 ───────────────────────────

def run_merge(args):
    """原 parquet + t2i_caption 列 → <source>/parquet_t2i/, 原目录不动。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(args.root)
    tasks, totals = build_tasks(root, args.sources or None)
    for source in totals:
        src_tasks = [t for t in tasks if t["source"] == source]
        merged = 0
        for t in src_tasks:
            sidecar = Path(t["source_dir"]) / "t2i_caption" / f"{t['stem']}.parquet"
            if not sidecar.exists():
                continue
            table = pq.read_table(t["parquet"])
            caps = pq.read_table(sidecar)
            cap_map = dict(zip(caps.column("sample_id").to_pylist(), caps.column("t2i_caption").to_pylist()))
            sids = table.column("sample_id").to_pylist()
            col = pa.array([cap_map.get(int(s), None) for s in sids])
            table = table.append_column(pa.field("t2i_caption", col.type), col)
            out_dir = Path(t["source_dir"]) / "parquet_t2i"
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / f".{t['stem']}.tmp"
            pq.write_table(table, tmp)
            os.replace(tmp, out_dir / f"{t['stem']}.parquet")
            merged += 1
        print(f"✅ {source}: 合并 {merged}/{len(src_tasks)} 分片 → parquet_t2i/", flush=True)


# ─────────────────────────── CLI ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=24, help="每卡每批图像数")
    ap.add_argument("--max-new-tokens", type=int, default=240)
    ap.add_argument("--max-pixels", type=int, default=768 * 768, help="单图最大像素 (控制 vision token 数)")
    ap.add_argument("--min-pixels", type=int, default=256 * 256)
    ap.add_argument("--sources", nargs="*", help="只处理指定 source")
    ap.add_argument("--sample", type=int, default=0, help="仅跑指定 source 前 N 行做人工确认")
    ap.add_argument("--status", action="store_true", help="只查看进度")
    ap.add_argument("--merge", action="store_true", help="sidecar 合并回带 t2i_caption 列的新 parquet")
    ap.add_argument("--status-interval", type=int, default=120)
    args = ap.parse_args()

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
