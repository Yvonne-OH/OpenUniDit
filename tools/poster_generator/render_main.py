# -*- coding: utf-8 -*-
"""MOSS-Image 渲染 worker 主进程 (独立进程, CUDA_VISIBLE_DEVICES 由启动方设置)。

轮询 tasks 目录: req_<id>_layouts.json -> 渲染 -> out_dir/req_<id>_p<j>.png
通过排他创建 .lock 文件领取任务。
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_worker import load_moss_pipeline, render_one


def shard_of(req_id: str, shard_size: int) -> Path | None:
    """从 req_<run>_<i:05d> 解析序号, 计算分片目录名 (无数字结尾则不分片)。"""
    m = re.search(r"_(\d+)$", req_id)
    if not m or shard_size <= 0:
        return None
    return f"shard_{int(m.group(1)) // shard_size:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--shard-size", type=int, default=200,
                    help="每个分片子文件夹放多少个请求的产出, 0=不分子文件夹")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    args = ap.parse_args()

    tasks_dir, out_dir = args.tasks_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import zlib

    pipe = load_moss_pipeline()
    print(f"[render gpu{args.gpu_id}] MOSS-Image ready", flush=True)

    stop_flag = tasks_dir / "ALL_STOP"

    def work_dir(req_id: str) -> Path:
        shard = shard_of(req_id, args.shard_size)
        d = out_dir / shard if shard else out_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def find_work():
        work = []
        for layouts_file in sorted(p for p in tasks_dir.glob("req_*_layouts.json")
                                   if not p.name.removesuffix("_layouts.json").endswith("_layouts")):
            req_id = layouts_file.stem.removesuffix("_layouts")
            try:
                layouts = json.loads(layouts_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(layouts, list):
                continue  # 错误占位 {"error": ...}
            d = work_dir(req_id)
            for j, layout in enumerate(layouts):
                png = d / f"{req_id}_p{j}.png"
                lock = d / f"{req_id}_p{j}.lock"
                if png.exists() or lock.exists() or (d / f"{req_id}_p{j}.bad").exists():
                    continue
                work.append((req_id, j, layout, lock, png, d))
        return work

    while True:
        if stop_flag.exists():
            break  # 当前任务已完成, 到循环顶部时再退出
        work = find_work()
        if work:
            req_id, j, layout, lock, png, d = work[0]
            try:
                lock.touch(exist_ok=False)  # 原子排他领取
            except FileExistsError:
                continue
            seed = (zlib.crc32(f"{req_id}_p{j}".encode()) % (2 ** 31)
                    + args.gpu_id * 100003)
            layouts_file = tasks_dir / f"{req_id}_layouts.json"
            (d / f"{req_id}_p{j}.json").write_text(
                json.dumps(layout, ensure_ascii=False, indent=2),
                encoding="utf-8")
            meta_path = d / "results.jsonl"
            t0 = time.time()
            try:
                render_one(pipe, layout, png, height=args.height,
                           width=args.width, steps=args.steps,
                           guidance_scale=args.guidance_scale, seed=seed)
            except Exception as e:
                print(f"[render gpu{args.gpu_id}] req {req_id} p{j} FAILED: {e}",
                      flush=True)
                meta_path.open("a", encoding="utf-8").write(json.dumps(
                    {"req_id": req_id, "poster_idx": j, "error": str(e)},
                    ensure_ascii=False) + "\n")
                continue
            meta = {"req_id": req_id, "poster_idx": j, "image": str(png),
                    "seed": seed, "seconds": round(time.time() - t0, 1)}
            meta_path.open("a", encoding="utf-8").write(
                json.dumps(meta, ensure_ascii=False) + "\n")
            print(f"[render gpu{args.gpu_id}] req {req_id} p{j} done "
                  f"({time.time() - t0:.1f}s) -> {png}", flush=True)
            continue
        time.sleep(5)
    print(f"[render gpu{args.gpu_id}] exit", flush=True)


if __name__ == "__main__":
    main()
