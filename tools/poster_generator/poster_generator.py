# -*- coding: utf-8 -*-
"""海报广告自动生成器主入口。

流程:
  1. fetch_opencole 从 cyberagent/opencole 拉取初始 prompt (经 hf-mirror);
  2. GPU 7 上的 Qwen3-VL-4B 对每个请求做中文润色, 概括成通用 prompt,
     并生成 2 个海报 layout JSON (文字用 "" 框住);
  3. GPU 0-6 共 7 张卡各跑一个 MOSS-Image-SFT worker 并行出图;
  4. GPU 7 上的 Qwen3-VL captioner 对渲染出的图片反打自然语言 prompt;
  5. 结果按分片子文件夹保存到 pfs_data/poster_ads/<run_name>/shard_XXX/。

支持断点继续: 任务文件/产出均有确定性命名, 重启后自动跳过已完成部分
(重新运行同样命令加 --resume 即可)。
支持 --max-hours 到时后写入 ALL_STOP, worker 完成手头任务后自动退出。

输出结构:
  <run_name>/
    tasks/req_<id>.json / req_<id>_layouts.json / ALL_STOP
    shard_000/req_<id>_p<j>.png|.json|_caption.txt|results.jsonl
    refiner.log render_gpu*.log captioner.log

用法示例:
  python poster_generator.py --num-requests 1200 --max-hours 12
  python poster_generator.py --num-requests 1200 --max-hours 12 --resume
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

OUT_ROOT = Path("/home/jovyan/pfs/users/zhanqian.wu/pfs_data/poster_ads")
PYTHON = sys.executable


def parse_args():
    ap = argparse.ArgumentParser(description="海报广告自动生成器")
    ap.add_argument("--num-requests", type=int, default=1200,
                    help="从 opencole 取多少个请求 (每个请求生成 2 张海报)")
    ap.add_argument("--num-posters", type=int, default=2,
                    help="每个请求生成的海报数")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--run-name", default=None, help="输出子目录名, 默认按时间戳")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--num-gpus", type=int, default=7, help="渲染用卡数 (默认前 7 张)")
    ap.add_argument("--vlm-gpu", type=int, default=7, help="Qwen3-VL (润色+caption) 用的卡号")
    ap.add_argument("--shard-size", type=int, default=200,
                    help="每个分片子文件夹放多少个请求的产出, 0=不分子文件夹")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="最长运行小时数, 到时自动停止 (写入 ALL_STOP)")
    ap.add_argument("--resume", action="store_true",
                    help="断点继续: 复用已有 run 目录, 跳过已完成任务")
    return ap.parse_args()


def count_pngs(out_dir: Path) -> int:
    n = 0
    for d in out_dir.glob("shard_*"):
        n += len([p for p in d.glob("*.png") if p.stem.rsplit("_p", 1)[-1].isdigit()])
    if not any(out_dir.glob("shard_*")):  # 未分片的旧结构
        n += len([p for p in out_dir.glob("*.png") if p.stem.rsplit("_p", 1)[-1].isdigit()])
    return n


def count_bad(out_dir: Path) -> int:
    return sum(len(list(d.glob("*.bad"))) for d in out_dir.glob("shard_*"))


def count_captions(out_dir: Path) -> int:
    return sum(len(list(d.glob("*_caption.txt"))) for d in out_dir.glob("shard_*"))


def main():
    args = parse_args()
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / run_name
    tasks_dir = out_dir / "tasks"
    resuming = args.resume and any(tasks_dir.glob("req_*.json"))
    tasks_dir.mkdir(parents=True, exist_ok=True)
    # 清除上次运行留下的停止标记 (断点继续时 worker 才会重新领任务)
    (tasks_dir / "ALL_STOP").unlink(missing_ok=True)
    print(f"[main] run={run_name} out={out_dir} resume={resuming} "
          f"max_hours={args.max_hours}", flush=True)

    from fetch_opencole import load_opencole

    records = load_opencole(args.split, args.num_requests)
    print(f"[main] loaded {len(records)} opencole records (split={args.split})", flush=True)

    # 写任务文件 (确定性命名; --resume 时已存在的不动, 实现断点继续)
    for i, rec in enumerate(records):
        req_id = f"{run_name}_{i:05d}"
        f = tasks_dir / f"req_{req_id}.json"
        if not f.exists():
            f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    total_requests = len([p for p in tasks_dir.glob("req_*.json")
                          if not p.name.endswith("_layouts.json")])
    expected = total_requests * args.num_posters
    print(f"[main] tasks total={total_requests} (new {len(records)}), "
          f"expected posters={expected}, already done={count_pngs(out_dir)}", flush=True)

    env_base = dict(os.environ)

    procs = []

    def launch(name, cmd, gpu, log_name):
        env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
        log = open(out_dir / log_name, "a")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             cwd=str(SCRIPT_DIR))
        procs.append((name, p, log))
        print(f"[main] launched {name} on gpu{gpu} pid={p.pid}", flush=True)

    # 渲染 workers: 前 num_gpus 张卡 (错开启动, 降低并发 CUDA init 竞态)
    for gpu in range(args.num_gpus):
        launch(f"render_gpu{gpu}",
               [PYTHON, str(SCRIPT_DIR / "render_main.py"),
                "--tasks-dir", str(tasks_dir), "--out-dir", str(out_dir),
                "--gpu-id", str(gpu), "--shard-size", str(args.shard_size),
                "--height", str(args.height), "--width", str(args.width),
                "--steps", str(args.steps),
                "--guidance-scale", str(args.guidance_scale)],
               gpu, f"render_gpu{gpu}.log")
        time.sleep(4)

    # VLM refiner (润色) + captioner (反打 prompt): 共用 vlm_gpu
    launch("refiner",
           [PYTHON, str(SCRIPT_DIR / "refiner_main.py"),
            "--tasks-dir", str(tasks_dir), "--num-posters", str(args.num_posters)],
           args.vlm_gpu, "refiner.log")
    launch("captioner",
           [PYTHON, str(SCRIPT_DIR / "captioner_main.py"),
            "--out-dir", str(out_dir)],
           args.vlm_gpu, "captioner.log")

    t0 = time.time()
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None
    stopped = False
    try:
        while True:
            n_png = count_pngs(out_dir)
            n_bad = count_bad(out_dir)
            n_cap = count_captions(out_dir)
            n_layouts = len([p for p in tasks_dir.glob("req_*_layouts.json")
                             if not p.name.removesuffix("_layouts.json").endswith("_layouts")])
            elapsed_h = (time.time() - t0) / 3600
            print(f"[main] {time.strftime('%m-%d %H:%M:%S')} "
                  f"layouts {n_layouts}/{total_requests} "
                  f"posters {n_png}/{expected} bad {n_bad} captions {n_cap}/{n_png} "
                  f"elapsed {elapsed_h:.2f}h", flush=True)
            if n_png + n_bad >= expected:
                print("[main] all posters done", flush=True)
                break
            if deadline and time.time() >= deadline:
                print(f"[main] max-hours {args.max_hours} reached, stopping", flush=True)
                break
            ref_alive = all(p.poll() is None for n, p, _ in procs
                            if n in ("refiner", "captioner"))
            if not ref_alive and n_png < expected:
                # VLM 进程挂了且没有更多产出 -> 停
                if all(p.poll() is not None for _, p, _ in procs):
                    print("[main] all workers exited early", flush=True)
                    break
                if any(p.poll() is not None and p.returncode != 0
                       for n, p, _ in procs if n == "refiner"):
                    print("[main] refiner crashed, stopping", flush=True)
                    break
            time.sleep(60)
    except KeyboardInterrupt:
        print("[main] interrupted", flush=True)
    finally:
        stopped = True
        (tasks_dir / "ALL_STOP").write_text("stop", encoding="utf-8")
        for name, p, log in procs:
            try:
                p.wait(timeout=3600)
                print(f"[main] {name} exited rc={p.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[main] {name} timeout, kill", flush=True)
                p.kill()
            log.close()

    n_png = count_pngs(out_dir)
    n_cap = count_captions(out_dir)
    print(f"[main] ALL DONE in {(time.time() - t0) / 3600:.2f}h: "
          f"{n_png}/{expected} posters, {n_cap} captions at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
