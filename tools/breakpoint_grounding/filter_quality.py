"""breakpoint-grounding-55m 质量过滤 stage 1/2: CPU 清晰度/尺寸指标 + GPU LAION aesthetic 打分。

逐分片扫 data/train-*.parquet (2746 片, 每片 2 万行, 图片内嵌), 每行计算:
  CPU (cv2, 进程池, 复用 data_tools/quality_metrics.cv_metrics):
    min_side / laplacian_var / tenengrad / jpeg_block_score / 曝光 / 饱和度 / colorfulness
  GPU (open_clip ViT-L/14 + laion-aesthetic-heads sac+logos+ava 线性头, 量程 ~1-10):
    aesthetic_score
sidecar: <root>/quality_filter/<stem>.parquet (index 为分片内行号), 本阶段只存原始指标,
过滤阈值在 stage 2 (caption_qwen3vl.py) / merge 阶段才应用, 阈值可后调不用重算。
分片级断点续跑 (重复执行自动跳过已完成分片)。

架构: 每卡一个 worker 进程, worker 内先 fork CPU 进程池 (解码+指标, 不碰 CUDA) 再加载
GPU aesthetic 模型; GPU 做批量打分。CPU 解码是瓶颈, 用 --workers-per-gpu 调节。

用法:
  python filter_quality.py --sample 32       # 首分片前 32 行试跑, 打印指标 + html 预览
  python filter_quality.py --num-gpus 8      # 全量 8 卡 (每卡 24 CPU worker)
  python filter_quality.py --status          # 查看进度
"""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from PIL import Image  # spawn 子进程 re-import 时也必须可用 (_row_worker/aes_thumb 依赖)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_tools"))
from quality_metrics import cv_metrics  # noqa: E402  (CPU 指标, 与 unified 库同尺度)

ROOT_DEFAULT = "/mnt/nfs/data/zhanqian.wu/unified_canvas_images/breakpoint-grounding-55m"
OPENCLIP_DIR = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/CLIP-ViT-L-14-laion2B-s32B-b82K"
AES_HEAD = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/laion-aesthetic-heads/sac+logos+ava1-l14-linearMSE.pth"

SIDECAR_DIR = "quality_filter"
PROGRESS_RELPATH = f"{SIDECAR_DIR}/_progress.json"
LOCK_RELPATH = f"{SIDECAR_DIR}/_progress.lock"

# open_clip OpenAI CLIP 归一化参数 (thumb 在 CPU 端只做 resize+crop 存 uint8, 归一化在 GPU 做)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
AES_SIZE = 224

SIDECAR_SCHEMA = [
    ("index", "int32"), ("width", "int32"), ("height", "int32"), ("min_side", "int32"),
    ("decode_ok", "bool"),
    ("laplacian_var", "float32"), ("tenengrad", "float32"), ("jpeg_block_score", "float32"),
    ("brightness_mean", "float32"), ("overexposed_ratio", "float32"),
    ("underexposed_ratio", "float32"), ("saturation_mean", "float32"),
    ("colorfulness", "float32"), ("aesthetic_score", "float32"),
]


# ─────────────────────────── CPU 行 worker ───────────────────────────

def aes_thumb(img, size: int = AES_SIZE) -> np.ndarray | None:
    """open_clip Resize(short=224 bicubic) + CenterCrop 的等价实现, 返回 uint8 RGB HWC。"""
    w, h = img.size
    if w < 1 or h < 1:
        return None
    scale = size / min(w, h)
    tw, th = max(size, int(w * scale + 0.5)), max(size, int(h * scale + 0.5))
    im = img.resize((tw, th), Image.BICUBIC) if hasattr(Image, "BICUBIC") else img.resize((tw, th), 3)
    left, top = (tw - size) // 2, (th - size) // 2
    im = im.crop((left, top, left + size, top + size))
    return np.asarray(im, dtype=np.uint8)


def _row_worker(job):
    """(index, img_bytes) → (index, metrics|None, thumb|None); 解码失败 metrics=None。"""
    import cv2
    cv2.setNumThreads(0)
    idx, raw = job
    if not raw:
        return idx, None, None
    try:
        m = cv_metrics(raw)
        if m is None:
            return idx, None, None
        img = Image.open(io.BytesIO(raw))
        img.load()
        w, h = img.size
        thumb = aes_thumb(img.convert("RGB"))
        if thumb is None:
            return idx, None, None
        m["width"], m["height"], m["min_side"] = w, h, min(w, h)
        return idx, m, thumb
    except Exception:
        return idx, None, None


# ─────────────────────────── GPU aesthetic scorer ───────────────────────────

def build_scorer(batch_size: int = 256):
    """open_clip ViT-L/14 (fp16) + LAION aesthetic 线性头 (fp32), 返回 scores(thumbs) 闭包。"""
    import open_clip
    import torch

    device = "cuda"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=str(Path(OPENCLIP_DIR) / "open_clip_pytorch_model.bin"))
    visual = model.visual.to(device).eval().half()
    del model  # 只留 visual 塔

    head = torch.nn.Sequential(
        torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
        torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
    sd = torch.load(AES_HEAD, map_location=device, weights_only=True)
    head.load_state_dict({k.removeprefix("layers."): v for k, v in sd.items()})
    head = head.to(device).eval()

    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1).half()
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1).half()

    @torch.no_grad()
    def scores(thumbs: list[np.ndarray]) -> list[float]:
        x = torch.from_numpy(np.stack(thumbs)).to(device).permute(0, 3, 1, 2)
        x = x.half().div_(255.0).sub_(mean).div_(std)
        emb = visual(x).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return head(emb).squeeze(-1).float().cpu().tolist()

    return scores, batch_size


# ─────────────────────────── 分片级进度 (flock 断点续跑) ───────────────────────────

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


def write_sidecar(out_path: Path, rows: list[dict]):
    """rows: 每行一个 dict (字段对齐 SIDECAR_SCHEMA), 原子提交。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {name: [] for name, _ in SIDECAR_SCHEMA}
    for r in rows:
        for name, _ in SIDECAR_SCHEMA:
            data[name].append(r.get(name))
    arrays = []
    for name, typ in SIDECAR_SCHEMA:
        arrays.append(pa.array(data[name], type=pa.type_for_alias(typ)))
    table = pa.table(arrays, names=[n for n, _ in SIDECAR_SCHEMA])
    tmp = out_path.with_name(f".{out_path.stem}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, out_path)


def metrics_rows(n: int, results: list, aes_scores: list[float | None]) -> list[dict]:
    """按行号组装 sidecar 行; results 为 _row_worker 输出 (有序)。"""
    rows = []
    for idx in range(n):
        _, m, _ = results[idx]
        row = {"index": idx, "decode_ok": m is not None, "aesthetic_score": aes_scores[idx]}
        for k in ("width", "height", "min_side"):
            row[k] = int(m[k]) if m else None
        for k in ("laplacian_var", "tenengrad", "jpeg_block_score", "brightness_mean",
                  "overexposed_ratio", "underexposed_ratio", "saturation_mean", "colorfulness"):
            row[k] = float(m[k]) if m else None
        rows.append(row)
    return rows


def process_shard(shard: Path, root: Path, pool, scorer, aes_bs: int, gpu_id: int, progress: Progress):
    import pyarrow.parquet as pq

    t0 = time.time()
    table = pq.read_table(shard, columns=["image"])
    raw_list = pc.struct_field(table.column("image"), "bytes").to_pylist()
    del table
    n = len(raw_list)

    results = list(pool.imap(_row_worker, enumerate(raw_list), chunksize=64))
    del raw_list

    aes_scores: list[float | None] = [None] * n
    thumbs, thumb_idx = [], []
    for idx, (_, _, thumb) in enumerate(results):
        if thumb is not None:
            thumbs.append(thumb)
            thumb_idx.append(idx)
        if len(thumbs) >= aes_bs:
            for i, s in zip(thumb_idx, scorer(thumbs)):
                aes_scores[i] = s
            thumbs, thumb_idx = [], []
    if thumbs:
        for i, s in zip(thumb_idx, scorer(thumbs)):
            aes_scores[i] = s

    write_sidecar(root / SIDECAR_DIR / f"{shard.stem}.parquet",
                  metrics_rows(n, results, aes_scores))
    dt = time.time() - t0
    progress.mark_done(shard.stem, n, dt, gpu_id)
    return n, dt


# ─────────────────────────── GPU worker 主循环 ───────────────────────────

def worker_loop(gpu_id: int, task_queue, result_queue, root_str: str, args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    root = Path(root_str)

    # 先建 fork CPU 池 (子进程不碰 CUDA), 再加载 GPU 模型
    pool = get_context("fork").Pool(args_dict["workers_per_gpu"])
    scorer, aes_bs = build_scorer()
    print(f"[gpu{gpu_id}] aesthetic 模型就绪 (cpu_workers={args_dict['workers_per_gpu']})", flush=True)

    progress = Progress(root)
    while True:
        task = task_queue.get()
        if task is None:
            pool.close()
            return
        shard = Path(task)
        try:
            n, dt = process_shard(shard, root, pool, scorer, aes_bs, gpu_id, progress)
            result_queue.put(("ok", gpu_id, shard.stem, n, n / max(dt, 1e-6)))
        except Exception as e:
            import traceback
            err_log = root / SIDECAR_DIR / "_failed.log"
            with open(err_log, "a") as f:
                f.write(f"[gpu{gpu_id} {time.strftime('%H:%M:%S')}] {shard.stem}: {e}\n{traceback.format_exc()}\n")
            result_queue.put(("error", gpu_id, shard.stem, 0, 0.0))


# ─────────────────────────── 主驱动 ───────────────────────────

def run_all(args):
    from tqdm import tqdm

    root = Path(args.root)
    shards = list_shards(root)
    progress = Progress(root)
    done = progress.done_shards()
    pending = [s for s in shards if s.stem not in done]
    done_rows = sum(v.get("rows", 0) for v in progress.snapshot().values())
    print(f"🖼 breakpoint-grounding-55m 质量过滤: {len(shards)} 分片, "
          f"已完成 {len(shards) - len(pending)} 片 ({done_rows:,} 行), 待处理 {len(pending)} 片", flush=True)
    if not pending:
        print("✅ 全部完成", flush=True)
        return

    ctx = get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    for s in pending:
        task_queue.put(str(s))
    n_workers = min(args.num_gpus, len(pending))
    for _ in range(n_workers):
        task_queue.put(None)
    workers = [ctx.Process(target=worker_loop, args=(g, task_queue, result_queue, str(root),
                                                      dict(workers_per_gpu=args.workers_per_gpu)),
                           daemon=False)  # worker 内要 fork CPU 池, 不能是 daemon
               for g in range(n_workers)]
    for w in workers:
        w.start()

    t0, n_ok, n_err, rows_this_run = time.time(), 0, 0, 0
    pbar = tqdm(total=len(pending), desc="🔧 quality filter", unit="shard")
    while n_ok + n_err < len(pending):
        try:
            ev = result_queue.get(timeout=5)
        except Exception:
            ev = None
            if not any(w.is_alive() for w in workers):
                print("⚠️ worker 全部退出但任务未完成, 中止 (可重新运行断点续跑)", flush=True)
                break
        if ev:
            kind, gpu, stem, rows, rate = ev
            if kind == "ok":
                n_ok += 1
                rows_this_run += rows
                pbar.set_postfix(gpu=gpu, rows=f"{rows}", speed=f"{rate:.0f} rows/s")
            else:
                n_err += 1
                print(f"\n❌ [gpu{gpu}] {stem} 失败, 见 quality_filter/_failed.log", flush=True)
            pbar.update(1)
            elapsed = time.time() - t0
            speed = rows_this_run / max(elapsed, 1)
            remain_rows = (len(pending) - n_ok - n_err) * 20000  # 每片约 2 万行
            pbar.set_description(f"🔧 filter {speed:,.0f} rows/s · ETA {remain_rows/max(speed,1)/3600:.1f}h")
    pbar.close()
    for w in workers:
        w.join(timeout=10)
    print(f"✅ 运行结束: 成功 {n_ok} 片, 失败 {n_err} 片, 用时 {(time.time()-t0)/60:.0f} 分钟", flush=True)


def pq_rows(shard: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(shard).metadata.num_rows


def show_status(args):
    root = Path(args.root)
    shards = list_shards(root)
    snap = Progress(root).snapshot()
    total_rows = sum(pq_rows(s) for s in shards)
    done_rows = sum(v.get("rows", 0) for v in snap.values())
    print(f"📊 进度: {len(snap)}/{len(shards)} 分片, {done_rows:,}/{total_rows:,} 行 "
          f"({100*done_rows/max(total_rows,1):.1f}%)", flush=True)


# ─────────────────────────── 小样本试跑 ───────────────────────────

def run_sample(args):
    root = Path(args.root)
    shard = list_shards(root)[0]
    print(f"🔎 试跑 {shard.name} 前 {args.sample} 行", flush=True)
    pool = get_context("fork").Pool(8)
    import pyarrow.parquet as pq
    table = pq.read_table(shard, columns=["image"]).slice(0, args.sample)
    raw_list = pc.struct_field(table.column("image"), "bytes").to_pylist()
    results = list(pool.imap(_row_worker, enumerate(raw_list), chunksize=8))
    scorer, _ = build_scorer()
    thumbs = [t for _, _, t in results if t is not None]
    scores = scorer(thumbs) if thumbs else []
    it = iter(scores)
    aes = [next(it) if t is not None else None for _, _, t in results]

    n_fail = sum(m is None for _, m, _ in results)
    print(f"\n{'='*72}\n解码失败: {n_fail}/{len(results)}")
    hdr = f"{'idx':>4} {'WxH':>11} {'min_side':>8} {'lap_var':>9} {'jpeg_blk':>8} {'aes':>5}"
    print(hdr + "\n" + "-" * len(hdr))
    for (idx, m, _), a in zip(results, aes):
        if m is None:
            print(f"{idx:>4}  ❌ decode failed")
            continue
        print(f"{idx:>4} {str(m['width'])+'x'+str(m['height']):>11} {m['min_side']:>8} "
              f"{m['laplacian_var']:>9.1f} {m['jpeg_block_score']:>8.3f} "
              f"{(f'{a:.2f}' if a is not None else 'n/a'):>5}")

    # html 预览
    out_dir = Path("/home/jovyan/pfs/users/zhanqian.wu/exp/breakpoint_grounding/filter_sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    import base64
    rows_html = []
    for (idx, m, thumb), a in zip(results, aes):
        if thumb is None:
            continue
        b64 = base64.b64encode(cv2_jpeg(thumb)).decode()
        rows_html.append(
            f'<div style="display:inline-block;margin:8px;padding:8px;border:1px solid #ddd;border-radius:8px;width:240px;vertical-align:top">'
            f'<img src="data:image/jpeg;base64,{b64}" style="width:224px;border-radius:4px">'
            f'<div style="font-size:12px;color:#555;margin-top:4px">idx={idx} · {m["width"]}x{m["height"]} · '
            f'lap={m["laplacian_var"]:.0f} · aes=<b>{a:.2f}</b></div></div>')
    html = ('<!doctype html><meta charset="utf-8"><title>filter sample</title>'
            '<body style="font-family:sans-serif;max-width:1100px;margin:auto">'
            f'<h2>breakpoint-grounding-55m 质量指标样本 ({shard.stem} 前 {args.sample} 行)</h2>'
            + "".join(rows_html) + "</body>")
    (out_dir / "index.html").write_text(html)
    print(f"\n🖼 预览页: {out_dir / 'index.html'}", flush=True)


def cv2_jpeg(thumb: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", thumb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else b""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--workers-per-gpu", type=int, default=24, help="每卡 CPU 解码进程数 (8 卡 × 24 = 192 核)")
    ap.add_argument("--sample", type=int, default=0, help="仅跑首分片前 N 行验证")
    ap.add_argument("--status", action="store_true", help="只查看进度")
    args = ap.parse_args()

    if args.status:
        show_status(args)
    elif args.sample:
        run_sample(args)
    else:
        run_all(args)


if __name__ == "__main__":
    sys.exit(main())
