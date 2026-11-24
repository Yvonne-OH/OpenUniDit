#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
url_dataset_image_filter.py — URL 数据集批量下载 + CPU 质量过滤 + 分片上传中心 BOS。

支持源:
  wukong  : Mxode/Noah-Wukong-100M, data/wukong_100m_*.csv (url, caption)
  danqing : DeepGlint-AI/DanQing100M, data/train-*.parquet (url, alt_text, recaption)

过滤 (全部 CPU, 下载后内存中判定, 不合格不落盘):
  1) 短边 > min-short (默认 512)
  2) 模糊   : 灰度降采样 (max side 1024, 对齐 data_tools/quality_metrics.py 尺度) 后
              Laplacian 方差 < sharpness-min 判模糊
  3) 灰度   : HSV 饱和度均值 < min-saturation
  4) 暗黑   : 灰度均值 (0-1) < min-brightness
  5) 过曝   : 灰度均值 > max-brightness 或 高亮像素 (>=251) 占比 > max-overexposed
  6) 欠曝   : 低亮像素 (<6) 占比 > max-underexposed
  7) 低对比 : 灰度标准差 < min-std
  8) 损坏图 : cv2 解码失败; 体积 < min-bytes

输出: 图片按原始字节保存 (不重编码), 文件名 sha256, 分 part 目录:
  {staging}/{dataset}/part_{i:05d}/<sha256>.{jpg|png|webp}
每个 part 满后由后台线程调 bos_center.py 上传到 BOS, 成功后删除本地目录。
上传目标: {bos-prefix}/{dataset}/part_XXXXX/  (默认
  users/zhanqian.wu/unified_images_ori/{dataset}/part_XXXXX/)

断点续跑: {staging}/{dataset}/_state.json 记录 (file_idx, next_row, part_idx, part_count),
  周期落盘 + 收尾时落盘; BOS 侧 done-file 防重传。重跑同命令自动续。
尽量多扫: 不设目标张数, 按 URL 顺序处理到结束 / --max-hours 到时 / 被 SIGTERM 杀,
  三种情况均先收尾 (flush checkpoint + 上传未满 part) 再退出。
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback

import cv2
import numpy as np
import requests

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BOS_TOOL = os.path.join(TOOLS_DIR, "highres_crawler", "bos_center.py")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

METRICS_MAX_SIDE = 1024

EXT_BY_FMT = {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# URL 流
# --------------------------------------------------------------------------

def iter_wukong_rows(src_dir: str, num_shards: int = 1, shard_idx: int = 0):
    import pandas as pd
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".csv"))
    files = files[shard_idx::num_shards] if num_shards > 1 else files
    log(f"[wukong] shard {shard_idx}/{num_shards}: {len(files)} csv files")
    for fi, fname in enumerate(files):
        path = os.path.join(src_dir, fname)
        try:
            df = pd.read_csv(path, usecols=["url"])
        except Exception as e:
            log(f"[wukong] skip {fname}: {e}")
            continue
        for row_i, url in enumerate(df["url"].tolist()):
            yield fi, row_i, fname, url


def iter_danqing_rows(src_dir: str, num_shards: int = 1, shard_idx: int = 0):
    import pyarrow.parquet as pq
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".parquet"))
    files = files[shard_idx::num_shards] if num_shards > 1 else files
    log(f"[danqing] shard {shard_idx}/{num_shards}: {len(files)} parquet files")
    for fi, fname in enumerate(files):
        path = os.path.join(src_dir, fname)
        pf = pq.ParquetFile(path)
        for row_i, batch in enumerate(pf.iter_batches(batch_size=8192, columns=["url"])):
            yield fi, row_i, fname, batch.column(0).to_pylist()


# ------------------------------------------------------------------
# 质量过滤
# ------------------------------------------------------------------

def quality_check(raw: bytes, p) -> tuple[dict | None, str]:
    """通过返回 (metrics, ""), 拒绝返回 (None, reason)。"""
    if len(raw) < p.min_bytes:
        return None, "too_small_bytes"
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "decode_failed"
    h, w = img.shape[:2]
    short = min(w, h)
    if short <= p.min_short:
        return None, f"short_edge_le({w}x{h})"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = METRICS_MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        small_gray = gray

    laplacian_var = float(cv2.Laplacian(small_gray, cv2.CV_32F).var())
    if laplacian_var < p.sharpness_min:
        return None, f"blurry({laplacian_var:.1f})"

    mean_0_1 = float(small_gray.mean()) / 255.0
    if mean_0_1 < p.min_brightness:
        return None, f"too_dark({mean_0_1:.3f})"
    if mean_0_1 > p.max_brightness:
        return None, f"too_bright({mean_0_1:.3f})"

    std = float(small_gray.std())
    if std < p.min_std:
        return None, f"low_contrast({std:.1f})"

    hist = np.bincount(small_gray.ravel(), minlength=256).astype(np.float64)
    n = hist.sum()
    over_ratio = float(hist[251:].sum() / n)
    under_ratio = float(hist[:6].sum() / n)
    if over_ratio > p.max_overexposed:
        return None, f"overexposed({over_ratio:.2f})"
    if under_ratio > p.max_underexposed:
        return None, f"underexposed({under_ratio:.2f})"

    sat_mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 255.0
    if sat_mean < p.min_saturation:
        return None, f"grayscale({sat_mean:.3f})"

    return {"width": w, "height": h, "sharpness": round(laplacian_var, 1),
            "brightness": round(mean_0_1, 3), "contrast": round(std, 1),
            "saturation": round(sat_mean, 3)}, ""


# ------------------------------------------------------------------
# Part 管理 + 上传
# ------------------------------------------------------------------

class PartManager:
    """图片落盘到当前 part 目录, 满后关闭并交给上传线程。"""

    def __init__(self, args, out_root: str):
        self.args = args
        self.out_root = out_root            # {staging}/{dataset} (/dev/shm)
        self.lock = threading.Condition()   # 兼职缓存水位控制
        self.bytes_written = 0              # 本地缓存已占用字节
        self.part_idx = args.state["part_idx"]
        self.count = args.state["part_count"]
        self.dir = None
        if self.count > 0:
            self.dir = self._path(self.part_idx)
            os.makedirs(self.dir, exist_ok=True)
        self.closed_parts: list[str] = []
        self._up_stop = threading.Event()
        self.up_thread = threading.Thread(target=self._upload_loop, daemon=True)

    def _path(self, idx: int) -> str:
        return os.path.join(self.out_root, f"part_{idx:05d}")

    def start(self):
        self.up_thread.start()

    def save(self, raw: bytes, fmt: str) -> bool:
        ext = EXT_BY_FMT.get(fmt, ".jpg")
        name = hashlib.sha256(raw).hexdigest() + ext
        with self.lock:
            # 本地缓存水位: 超过 shm-cap 暂停写入, 等上传线程释放
            while self.bytes_written > self.args.shm_cap:
                self.lock.wait(timeout=30)
            if self.dir is None:
                self.dir = self._path(self.part_idx)
                os.makedirs(self.dir, exist_ok=True)
            dest = os.path.join(self.dir, name)
            if os.path.exists(dest):
                return True                      # 跨 part 重复, 视为已保存
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(raw)
            os.replace(tmp, dest)
            self.bytes_written += len(raw)
            self.count += 1
            if self.count >= self.args.part_size:
                self._close_locked()
            return True

    def _close_locked(self):
        final = self._path(self.part_idx)
        self.closed_parts.append(final)
        self.part_idx += 1
        self.count = 0
        self.dir = None

    def flush_state(self):
        with self.lock:
            self.args.state["part_idx"] = self.part_idx
            self.args.state["part_count"] = self.count
        save_state(self.args.state_path, self.args.state)

    def current_dir(self) -> str | None:
        with self.lock:
            return self.dir

    def stop(self):
        """收尾: 关闭未满 part, 等上传队列清空。"""
        with self.lock:
            if self.count > 0 and self.dir is not None:
                self._close_locked()
        self._up_stop.wait(timeout=3600)
        self.up_thread.join(timeout=7200)

    def _upload_loop(self):
        while True:
            path = None
            with self.lock:
                if self.closed_parts:
                    path = self.closed_parts.pop(0)
            if path is None:
                if self._up_stop.is_set():
                    return
                self._up_stop.wait(2.0)
                continue
            try:
                self._upload_one(path)
            except Exception:
                traceback.print_exc()
                with self.lock:
                    self.closed_parts.insert(0, path)   # 留待重试
                self._up_stop.wait(30)

    def _upload_one(self, part_dir: str):
        """打包成单个 tar 上传 (BOS 对象 key = <prefix>/<dataset>/part_XXXXX.tar),
        stat 校验大小一致后清理本地缓存并释放水位。"""
        part_name = os.path.basename(part_dir)
        tar_path = part_dir + ".tar"
        t0 = time.time()
        n_files, part_bytes = 0, 0
        with tarfile.open(tar_path, "w") as tf:
            for fn in sorted(os.listdir(part_dir)):
                p = os.path.join(part_dir, fn)
                if os.path.isfile(p):
                    tf.add(p, arcname=f"{part_name}/{fn}")
                    n_files += 1
                    part_bytes += os.path.getsize(p)
        log(f"[upload] {part_name}.tar packed {n_files} files "
            f"{part_bytes/1e6:.0f}MB in {time.time()-t0:.0f}s")
        # 分片隔离: 每个分片一个 BOS 子前缀, 避免不同实例同名 part 相互覆盖
        prefix = f"{self.args.bos_prefix}/{self.args.dataset}/s{self.args.shard_idx}/"
        cmd = ["python3", BOS_TOOL, "--cred-file", self.args.cred_file, "up",
               tar_path, prefix, "--done-file", os.path.join(self.out_root, "bos_done.json")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bos upload rc={r.returncode}: {r.stdout[-500:]} {r.stderr[-500:]}")
        # 校验真实对象大小 (不信进度条)
        key = f"{prefix}{part_name}.tar"
        chk = subprocess.run(["python3", BOS_TOOL, "--cred-file", self.args.cred_file,
                              "stat", key], capture_output=True, text=True)
        m = re.search(r"content_length:\s*(\d+)", chk.stdout)
        if not m or int(m.group(1)) != os.path.getsize(tar_path):
            raise RuntimeError(f"upload verify failed for {key}: {chk.stdout[-300:]} {chk.stderr[-300:]}")
        log(f"[upload] {part_name}.tar verified on BOS in {time.time()-t0:.0f}s")
        # 成功后清理本地: tar + part 目录, 释放缓存水位
        subprocess.run(["rm", "-rf", part_dir, tar_path], check=False)
        with self.lock:
            self.args.state["uploaded_parts"] = self.args.state.get("uploaded_parts", []) + [part_name]
            if len(self.args.state["uploaded_parts"]) > 200:
                self.args.state["uploaded_parts"] = self.args.state["uploaded_parts"][-200:]
            self.bytes_written = max(0, self.bytes_written - part_bytes)
            self.lock.notify_all()


# ------------------------------------------------------------------
# 状态
# ------------------------------------------------------------------

def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"file_idx": 0, "next_row": 0, "part_idx": 0, "part_count": 0}


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


# ------------------------------------------------------------------
# 下载 worker
# ------------------------------------------------------------------

def download(url: str, timeout: int, retries: int) -> bytes | None:
    headers = {"User-Agent": UA, "Accept": "image/*, */*"}
    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
        time.sleep(0.5)
    return None


def handle_url(url: str, p, pm: PartManager, stats_lock, stats) -> None:
    if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        with stats_lock:
            stats["bad_url"] += 1
        return
    raw = download(url, p.timeout, p.retries)
    with stats_lock:
        stats["downloaded"] += 1
    if raw is None:
        with stats_lock:
            stats["failed"] += 1
        return
    try:
        from PIL import Image
        fmt = Image.open(io.BytesIO(raw)).format or "JPEG"
    except Exception:
        with stats_lock:
            stats["rejected:decode_failed"] += 1
        return
    metrics, reason = quality_check(raw, p)
    if metrics is None:
        with stats_lock:
            stats[f"rejected:{reason.split('(')[0]}"] += 1
        return
    pm.save(raw, fmt)
    with stats_lock:
        stats["saved"] += 1


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["wukong", "danqing"])
    ap.add_argument("--src-dir", required=True, help="csv/parquet 所在目录")
    ap.add_argument("--staging", required=True,
                    help="本地 part 暂存目录 (建议 /dev/shm, 上传成功即删; 丢失最多损失一个未传 part)")
    ap.add_argument("--state-path", default="",
                    help="断点 state json 路径 (落 PFS 以支持跨重启续跑; 空则放 staging 内)")
    ap.add_argument("--bos-prefix", default="users/zhanqian.wu/unified_images_ori")
    ap.add_argument("--cred-file", default=os.path.expanduser("~/.bos_center_cred"))
    ap.add_argument("--threads", type=int, default=256, help="下载+过滤并发")
    ap.add_argument("--num-shards", type=int, default=1, help="按源文件取模分片总数")
    ap.add_argument("--shard-idx", type=int, default=0, help="本实例处理的分片号 fi %% num-shards")
    ap.add_argument("--part-size", type=int, default=1000, help="每 part 打包成一个 tar 的文件数")
    ap.add_argument("--shm-cap", type=float, default=5e9,
                    help="本地缓存水位上限字节, 超过则暂停下载等上传释放")
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--min-bytes", type=int, default=4096)
    ap.add_argument("--ckpt-every", type=int, default=50000, help="每处理多少行落一次 state")
    ap.add_argument("--max-hours", type=float, default=0, help="0=不限时")
    ap.add_argument("--max-rows", type=int, default=0, help="0=不限行数 (调试用)")
    # 过滤阈值
    ap.add_argument("--min-short", type=int, default=512, help="短边必须严格大于该值")
    ap.add_argument("--sharpness-min", type=float, default=60.0)
    ap.add_argument("--min-saturation", type=float, default=0.05)
    ap.add_argument("--min-brightness", type=float, default=0.05)
    ap.add_argument("--max-brightness", type=float, default=0.92)
    ap.add_argument("--min-std", type=float, default=12.0)
    ap.add_argument("--max-overexposed", type=float, default=0.5)
    ap.add_argument("--max-underexposed", type=float, default=0.5)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.staging, exist_ok=True)
    args.state_path = args.state_path or os.path.join(args.staging, f"{args.dataset}", "_state.json")
    os.makedirs(os.path.dirname(args.state_path), exist_ok=True)
    args.state = load_state(args.state_path)
    log(f"state: {args.state}  state_path: {args.state_path}")

    out_root = os.path.join(args.staging, args.dataset)
    os.makedirs(out_root, exist_ok=True)
    pm = PartManager(args, out_root)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: (log("SIGTERM, graceful stop..."), stop.set()))
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    stats_lock = threading.Lock()
    stats: collections.Counter = collections.Counter()
    t_start = time.time()
    next_log = 10000

    if args.dataset == "wukong":
        row_iter = iter_wukong_rows(args.src_dir, args.num_shards, args.shard_idx)
    else:
        row_iter = iter_danqing_rows(args.src_dir, args.num_shards, args.shard_idx)

    pm.start()
    log(f"threads={args.threads} part_size={args.part_size} staging={out_root} "
        f"bos={args.bos_prefix}/{args.dataset}/")

    pool = cf.ThreadPoolExecutor(max_workers=args.threads)
    inflight = set()
    processed = 0
    skipped_state = 0

    def drain(completed_block: bool):
        nonlocal inflight
        if not inflight:
            return
        if completed_block:
            done, inflight = cf.wait(inflight, return_when=cf.ALL_COMPLETED)
        else:
            done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
        for fu in done:
            try:
                fu.result()
            except Exception:
                traceback.print_exc()
                with stats_lock:
                    stats["worker_error"] = stats.get("worker_error", 0) + 1

    try:
        for fi, row_i, fname, payload in row_iter:
            if stop.is_set():
                break
            # 断点续跑: 跳过已完成源文件
            if fi < args.state["file_idx"]:
                continue
            urls = payload if isinstance(payload, list) else [payload]
            for u in urls:
                if stop.is_set():
                    break
                if fi == args.state["file_idx"] and row_i < args.state["next_row"]:
                    skipped_state += 1
                    continue
                while len(inflight) >= args.threads * 2:
                    drain(False)
                    if stop.is_set():
                        break
                if stop.is_set():
                    break
                fut = pool.submit(handle_url, u, args, pm, stats_lock, stats)
                inflight.add(fut)
                processed += 1

            # 断点: 只在已处理过数据时落盘, 且 next_row 单调不回退
            # (processed=0 时 0%N==0 会立即触发, 用 0 覆盖已加载的断点)
            if processed and processed % args.ckpt_every < len(urls) and not stop.is_set():
                drain(False)
                # 续跑基准 = 下一个未处理行 (只支持按 file 粒度精确, row 粒度近似)
                args.state["file_idx"] = fi
                args.state["next_row"] = max(row_i, args.state.get("next_row", 0))
                pm.flush_state()

            if args.max_rows and processed >= args.max_rows:
                log(f"max-rows {args.max_rows} reached")
                break
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                log(f"max-hours {args.max_hours} reached")
                break

            # 日志: 按跨阈值触发 (兼容 danqing 8192 步长, % 10000 永不命中)
            while processed >= next_log:
                with stats_lock:
                    s = dict(stats)
                rate = processed / max(time.time() - t_start, 1)
                log(f"rows={processed} rate={rate:.1f}/s saved={s.get('saved', 0)} "
                    f"dl_fail={s.get('failed', 0)} "
                    + " ".join(f"{k}={v}" for k, v in s.items() if str(k).startswith("rejected")))
                next_log += 10000
    finally:
        drain(True)
        pool.shutdown(wait=False, cancel_futures=True)
        args.state["file_idx"] = args.state.get("file_idx", 0)
        args.state["next_row"] = args.state.get("next_row", 0)
        pm.flush_state()
        pm.stop()

    with stats_lock:
        s = dict(stats)
    log(f"DONE elapsed={(time.time()-t_start)/60:.1f}min processed={processed} "
        f"saved={s.get('saved', 0)} failed={s.get('failed', 0)}")
    for k, v in sorted(s.items()):
        if str(k).startswith("rejected") or k in ("bad_url", "worker_error"):
            log(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
