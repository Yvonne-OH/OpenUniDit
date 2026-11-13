#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_image_collector.py — 三平台小样本统一采集：图虫 / 堆糖 / B站

来源：
  tuchong    图虫标签页 API（摄影图；匿名直链上限长边 1200，走例外短边阈值 tuchong-min-short）
  duitang    堆糖关键词搜索 API（自动尝试把缩略 URL 升级为原图）
  bilibili   B站热门(popular)/排行榜(ranking)视频封面（免登录公开接口）

统一质量过滤（全部 CPU、下载后在内存中执行，不合格不落盘）：
  1) 尺寸        : 短边 >= min-short（默认 1080，即 1080p），长边 >= min-long（0=不限）
  2) 清晰度      : 灰度图短边归一到 1080 基准后算 Laplacian 方差，
                   < sharpness-min（默认 60）判模糊拒绝
  3) 低信息量    : 灰度标准差 < min-std（纯色/渐变/大面积留白）拒绝
  4) 亮度异常    : 均值 < min-mean（过暗）或 > max-mean（过曝）拒绝
  5) 极端宽高比  : 超出 [1/max-aspect, max-aspect]（长条图/拼接图）拒绝
  6) 损坏图/动图 : cv2 解码失败（含 GIF）拒绝
  7) 去重        : sha256 精确去重 + dHash 感知去重（汉明距离 <= hamming），
                   索引持久化到 _meta/，跨运行、跨平台、跨 query 生效

输出结构（图片按 query 分目录、每 per-dir 张再分 part_XXX 分片，避免单目录过大）：
  {out}/tuchong/{query}/part_000/*.jpg  ...  + 平台根目录聚合 metadata.jsonl
  {out}/duitang/{query}/part_000/*.jpg
  {out}/bilibili/feed_popular/part_000/*.jpg
  {out}/_meta/  sha256_index.txt  dhash_index.txt  all_metadata.jsonl
  （metadata.jsonl 每行含 file 绝对路径与 subdir 相对路径）

metadata.jsonl 每行一条 JSON，通过/被拒/失败都记录（被拒带 reject_reason），
便于事后统计阈值分布、调参。

用法示例：
  python3 unified_image_collector.py                                    # 全平台，queries=猫,夜景
  python3 unified_image_collector.py --platform all --queries 猫 --per-query 20
  python3 unified_image_collector.py --platform tuchong --queries 夜景 --per-query 30
  python3 unified_image_collector.py --platform bilibili --per-query 30
  python3 unified_image_collector.py --platform duitang --queries 猫 --dry-run

使用边界：仅供个人学习 / 内部研究使用，图片版权归原作者与各平台所有，
勿商用、勿二次分发；默认限速低并发，请勿用于大规模抓取。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

PRINT_LOCK = threading.Lock()


def say(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# 基础设施：限速 / 任务对象
# --------------------------------------------------------------------------

class RateLimiter:
    """全局请求限速：任意两次请求至少间隔 delay 秒，附加随机抖动。"""

    def __init__(self, delay: float) -> None:
        self.delay = max(0.0, delay)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.delay <= 0:
            return
        with self._lock:
            gap = self.delay - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic() + random.uniform(0, self.delay * 0.3)


@dataclass
class Task:
    urls: list          # 候选下载地址，从前往后尝试（第一个成功且合格的即用）
    referer: str
    name: str           # 文件名主干（不含扩展名）
    platform: str
    query: str
    meta: dict = field(default_factory=dict)
    # 平台级短边阈值覆盖（0 = 用全局 min-short）；
    # 图虫匿名上限为长边 1200，走 tuchong-min-short 例外档
    min_short_override: int = 0


# --------------------------------------------------------------------------
# CPU 质量闸门
# --------------------------------------------------------------------------

def dhash_hex(gray: "np.ndarray", size: int = 8) -> str:
    """dHash 64bit：比较相邻列亮度，返回 16 位 hex。"""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    v = 0
    for b in bits.flatten():
        v = (v << 1) | int(b)
    return f"{v:016x}"


@dataclass
class QualityGate:
    min_short: int
    min_long: int
    max_aspect: float
    sharpness_min: float
    min_std: float
    min_mean: int
    max_mean: int

    def check(self, raw: bytes, min_short: int | None = None):
        """返回 (metrics, "") 或 (None, 拒绝原因)。min_short 可按任务覆盖。"""
        short_req = self.min_short if min_short is None else min_short
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "decode_failed_or_gif"
        h, w = img.shape[:2]
        short, long = min(w, h), max(w, h)
        if short < short_req:
            return None, f"too_small({w}x{h})"
        if self.min_long and long < self.min_long:
            return None, f"long_side_lt_{self.min_long}({w}x{h})"
        aspect = w / h
        if aspect > self.max_aspect or aspect < 1.0 / self.max_aspect:
            return None, f"extreme_aspect({aspect:.2f})"

        # 清晰度：短边归一到 1080 基准，使不同分辨率间可比
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = 1080.0 / short
        if abs(scale - 1.0) > 0.05:
            gray = cv2.resize(gray, (int(round(gray.shape[1] * scale)),
                                     int(round(gray.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean = float(gray.mean())
        std = float(gray.std())
        if sharpness < self.sharpness_min:
            return None, f"blurry(sharpness={sharpness:.1f})"
        if std < self.min_std:
            return None, f"low_info(std={std:.1f})"
        if mean < self.min_mean or mean > self.max_mean:
            return None, f"brightness_out_of_range(mean={mean:.1f})"

        metrics = {"width": w, "height": h, "aspect": round(aspect, 4),
                   "sharpness": round(sharpness, 2), "brightness": round(mean, 2),
                   "contrast": round(std, 2), "dhash": dhash_hex(gray)}
        return metrics, ""


# --------------------------------------------------------------------------
# 下载（requests 主路 + curl 兜底）
# --------------------------------------------------------------------------

def download(task: Task, limiter: RateLimiter, timeout: int, retries: int, min_bytes: int):
    """依次尝试 task.urls；requests 失败回退 curl。返回 (raw, url, via) 或 (None, None, None)。"""
    headers = {"User-Agent": UA, "Referer": task.referer, "Accept": "image/*, */*"}
    for url in task.urls:
        for attempt in range(retries):
            limiter.wait()
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code == 200 and len(r.content) >= min_bytes:
                    return r.content, url, "requests"
            except Exception:
                pass
            # curl 兜底（部分 CDN 拦 requests 指纹）
            limiter.wait()
            fd, tmp = tempfile.mkstemp(suffix=".part")
            os.close(fd)
            try:
                code = subprocess.run(
                    ["curl", "-sL", "-o", tmp, "-w", "%{http_code}", "-m", str(timeout),
                     "-H", f"User-Agent: {UA}", "-H", f"Referer: {task.referer}", url],
                    capture_output=True, text=True).stdout
                if code == "200" and os.path.getsize(tmp) >= min_bytes:
                    with open(tmp, "rb") as f:
                        raw = f.read()
                    return raw, url, "curl"
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            time.sleep(1.2 * (attempt + 1))
    return None, None, None


# --------------------------------------------------------------------------
# 平台枚举器（生成器：边翻页边产出候选 Task）
# --------------------------------------------------------------------------

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*"})
    return s


TUCHONG_MAX_LONG = 1200   # 图虫匿名直链 f/ 的输出上限：长边压缩到 1200


def enum_tuchong(tag: str, args, limiter: RateLimiter):
    """图虫标签页：/rest/tags/{tag}/posts，直链 photo.tuchong.com/{uid}/f/{img_id}.jpg
    注意：匿名拿到的 f/ 图是长边 <=1200 的档位，故图虫走 tuchong-min-short 例外阈值，
    并用 API 尺寸模拟 f/ 输出尺寸做预筛。"""
    s = _new_session()
    for page in range(1, args.pages + 1):
        url = (f"https://tuchong.com/rest/tags/{urllib.parse.quote(tag)}"
               f"/posts?page={page}&count=20")
        try:
            limiter.wait()
            r = s.get(url, headers={"Referer": "https://tuchong.com/"}, timeout=20)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            say(f"[tuchong/{tag}] page {page} 请求失败: {e}")
            break
        posts = d.get("postList") or []
        if not posts:
            break
        for p in posts:
            for im in (p.get("images") or []):
                w, h = im.get("width") or 0, im.get("height") or 0
                if not w or not h:
                    continue
                # 预筛：模拟 f/ 输出（长边压到 1200）后的短边是否达标
                scale = min(1.0, TUCHONG_MAX_LONG / max(w, h))
                est_short = round(min(w, h) * scale)
                if est_short < args.tuchong_min_short:
                    continue
                img_id = im.get("img_id")
                if not img_id:
                    continue
                img_url = f"https://photo.tuchong.com/{im['user_id']}/f/{img_id}.jpg"
                yield Task([img_url], "https://tuchong.com/", f"tuchong_{img_id}",
                           "tuchong", tag,
                           {"origin_url": img_url,
                            "post_url": f"https://tuchong.com/{p['site_id']}/",
                            "post_id": p.get("post_id"),
                            "img_id": img_id,
                            "author_id": im.get("user_id"),
                            "title": (p.get("title") or "")[:120],
                            "api_width": w, "api_height": h,
                            "views": p.get("views"), "favorites": p.get("favorites"),
                            "comments": p.get("comments"),
                            "published_at": p.get("published_at")},
                           min_short_override=args.tuchong_min_short)
    say(f"[tuchong/{tag}] 枚举完成（{args.pages} 页上限）")


duitang_thumb_re = re.compile(r"\.thumb\.\d+(?:_\d+)?(?=\.|_)", re.I)
duitang_size_re = re.compile(r"_(?:650|600|blurthumb_\d+)(?=\.(?:jpe?g|png|webp))", re.I)


def duitang_candidates(path: str):
    """堆糖 path 通常是缩略图 URL，尝试去掉 .thumb.xxx / _650 等片段拿原图，原图候选优先。"""
    up = duitang_thumb_re.sub("", path)
    up = duitang_size_re.sub("", up)
    cands = [up, path] if up != path else [path]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def enum_duitang(kw: str, args, limiter: RateLimiter):
    """堆糖关键词搜索：/napi/blog/list/by_search/，start 递增翻页。"""
    s = _new_session()
    start = 0
    for _ in range(args.pages):
        url = (f"https://www.duitang.com/napi/blog/list/by_search/"
               f"?kw={urllib.parse.quote(kw)}&start={start}&limit=24")
        try:
            limiter.wait()
            r = s.get(url, headers={"Referer": "https://www.duitang.com/"}, timeout=20)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            say(f"[duitang/{kw}] start={start} 请求失败: {e}")
            break
        objs = (d.get("data") or {}).get("object_list") or []
        if not objs:
            break
        for o in objs:
            ph = o.get("photo") or {}
            path = ph.get("path")
            if not path:
                continue
            w, h = ph.get("width") or 0, ph.get("height") or 0
            if w and min(w, h) < args.min_short:
                continue
            if w and args.min_long and max(w, h) < args.min_long:
                continue
            bid = o.get("id")
            yield Task(duitang_candidates(path), "https://www.duitang.com/",
                       f"duitang_{bid}", "duitang", kw,
                       {"origin_url": path,
                        "blog_url": f"https://www.duitang.com/blog/?id={bid}",
                        "blog_id": bid,
                        "title": (ph.get("msg") or "")[:120],
                        "api_width": w, "api_height": h,
                        "favorite_count": o.get("favorite_count")})
        start += 24
    say(f"[duitang/{kw}] 枚举完成（{args.pages} 轮上限）")


def _bili_task(pic: str, bvid: str, it: dict, source: str) -> Task:
    pic = pic.replace("http://", "https://")
    pic = re.sub(r"(@\d+[wh].*)$", "", pic)   # 去掉 @480w_270h 之类裁剪后缀，取原图
    title = re.sub(r"<[^>]+>", "", it.get("title") or "")
    return Task([pic], "https://www.bilibili.com/", f"bilibili_{bvid}",
                "bilibili", f"feed_{source}",
                {"origin_url": pic,
                 "video_url": f"https://www.bilibili.com/video/{bvid}/",
                 "bvid": bvid,
                 "title": title[:120],
                 "owner": (it.get("owner") or {}).get("name", ""),
                 "view": (it.get("stat") or {}).get("view"),
                 "like": (it.get("stat") or {}).get("like"),
                 "pubdate": it.get("pubdate")})


def enum_bilibili(args, limiter: RateLimiter):
    """B站封面：优先 popular 分页；风控/失败自动降级 ranking（一次全量）。"""
    s = _new_session()

    def bili_api(url: str) -> dict:
        limiter.wait()
        r = s.get(url, headers={"Referer": "https://www.bilibili.com/"}, timeout=20)
        r.raise_for_status()
        d = r.json()
        if d.get("code") != 0:
            raise RuntimeError(f"code={d.get('code')} {d.get('message', '')}")
        return d.get("data") or {}

    items, source = [], args.bili_source
    if source == "popular":
        for pn in range(1, args.pages + 1):
            try:
                d = bili_api(f"https://api.bilibili.com/x/web-interface/popular?ps=20&pn={pn}")
            except RuntimeError as e:
                say(f"[bilibili] popular page {pn} 失败: {e}，降级 ranking")
                source = "ranking"
                break
            batch = d.get("list") or []
            if not batch:
                break
            items.extend(batch)
    if source == "ranking" or not items:
        try:
            d = bili_api("https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all")
            items.extend(d.get("list") or [])
        except Exception as e:
            say(f"[bilibili] ranking 失败: {e}")

    say(f"[bilibili] source={source}，候选 {len(items)} 条")
    seen = set()
    for it in items:
        bvid, pic = it.get("bvid"), it.get("pic")
        if not pic or not bvid or bvid in seen:
            continue
        seen.add(bvid)
        yield _bili_task(pic, bvid, it, source)


# --------------------------------------------------------------------------
# Runner：调度、过滤、落盘、去重、metadata
# --------------------------------------------------------------------------

class Runner:
    def __init__(self, args):
        self.args = args
        self.limiter = RateLimiter(args.delay)
        self.gate = QualityGate(args.min_short, args.min_long, args.max_aspect,
                                args.sharpness_min, args.min_std,
                                args.min_mean, args.max_mean)
        self.out_root = os.path.abspath(args.out)
        self.meta_dir = os.path.join(self.out_root, "_meta")
        os.makedirs(self.meta_dir, exist_ok=True)
        self.run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        self._idx_lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self.sha_seen = set()
        self.dhash_seen = []
        self._load_indexes()

        # 分片目录计数器：{(platform, query_slug): [part_idx, 已放张数]}
        # 仅在持有 _idx_lock 时读写（_save 由 _work 在锁内调用）
        self.dir_counters = {}

        self.stats = {}            # platform -> Counter
        self.sharp_saved = {}      # platform -> [sharpness of saved]

    # ---- 索引 ----
    def _load_indexes(self):
        sha_fp = os.path.join(self.meta_dir, "sha256_index.txt")
        dh_fp = os.path.join(self.meta_dir, "dhash_index.txt")
        if os.path.exists(sha_fp):
            with open(sha_fp) as f:
                for line in f:
                    s = line.strip()
                    if s:
                        self.sha_seen.add(s.split("\t")[0])
        if os.path.exists(dh_fp):
            with open(dh_fp) as f:
                for line in f:
                    s = line.strip()
                    if s:
                        try:
                            self.dhash_seen.append(int(s.split("\t")[0], 16))
                        except ValueError:
                            pass
        say(f"[索引] 已加载 sha256 {len(self.sha_seen)} 条 / dhash {len(self.dhash_seen)} 条")

    def _append_indexes(self, sha: str, dhash: str, dest: str) -> None:
        with self._idx_lock:
            with open(os.path.join(self.meta_dir, "sha256_index.txt"), "a") as f:
                f.write(f"{sha}\t{dest}\n")
            with open(os.path.join(self.meta_dir, "dhash_index.txt"), "a") as f:
                f.write(f"{dhash}\t{dest}\n")

    # ---- 落盘 ----
    @staticmethod
    def _query_slug(query: str) -> str:
        """query 转子目录名：替换不安全字符与空白。"""
        cleaned = re.sub(r'[\\/:*?"<>|\s\x00-\x1f]+', "_", (query or "").strip()).strip(" ._")
        return cleaned or "default"

    def _alloc_subdir(self, task: Task) -> str:
        """分配该图所属子目录：{query_slug}/part_{i:03d}。
        注意：依赖调用者已持有 _idx_lock。"""
        qslug = self._query_slug(task.query)
        per_dir = self.args.per_dir
        if per_dir <= 0:
            return qslug
        key = (task.platform, qslug)
        part, cnt = self.dir_counters.get(key, (0, 0))
        if cnt >= per_dir:
            part += 1
            cnt = 0
        self.dir_counters[key] = (part, cnt + 1)
        return f"{qslug}/part_{part:03d}"

    def _save(self, task: Task, raw: bytes, url_used: str):
        m = re.search(r"\.(jpe?g|png|webp)(?=$|\?|@)", url_used, re.I)
        ext = "." + (m.group(1).lower().replace("jpeg", "jpg") if m else "jpg")
        subdir = self._alloc_subdir(task)
        pdir = os.path.join(self.out_root, task.platform)
        os.makedirs(os.path.join(pdir, subdir), exist_ok=True)
        dest = os.path.join(pdir, subdir, task.name + ext)
        if os.path.exists(dest):
            dest = os.path.join(pdir, subdir, f"{task.name}_{self.run_id}{ext}")
            if os.path.exists(dest):
                return None, subdir
        tmp = dest + f".{threading.get_ident()}.part"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, dest)
        return dest, subdir

    # ---- metadata ----
    def _write_meta(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._meta_lock:
            with open(os.path.join(self.out_root, rec["platform"], "metadata.jsonl"),
                      "a", encoding="utf-8") as f:
                f.write(line + "\n")
            with open(os.path.join(self.meta_dir, "all_metadata.jsonl"),
                      "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ---- 单任务 worker ----
    def _work(self, task: Task) -> dict:
        rec = {"platform": task.platform, "query": task.query, "run_id": self.run_id,
               "status": ""}
        rec.update(task.meta)
        raw, url_used, via = download(task, self.limiter, self.args.timeout,
                                      self.args.retries, self.args.min_bytes)
        if raw is None:
            rec["status"] = "failed"
            rec["reject_reason"] = "download_failed"
            if rec.get("origin_url") is None:
                rec["origin_url"] = task.urls[0] if task.urls else ""
            return rec
        rec["origin_url"] = url_used
        rec["via"] = via
        rec["bytes"] = len(raw)
        sha = hashlib.sha256(raw).hexdigest()
        rec["sha256"] = sha

        metrics, reason = self.gate.check(raw, min_short=task.min_short_override or None)
        if metrics is None:
            rec["status"] = "rejected"
            rec["reject_reason"] = reason
            return rec
        rec.update(metrics)

        dh = int(metrics["dhash"], 16)
        with self._idx_lock:
            if sha in self.sha_seen:
                rec["status"], rec["reject_reason"] = "duplicate", "sha256"
                return rec
            if any((h ^ dh).bit_count() <= self.args.hamming for h in self.dhash_seen):
                rec["status"], rec["reject_reason"] = "duplicate", "dhash"
                return rec
            dest, subdir = self._save(task, raw, url_used)
            if dest is None:
                rec["status"], rec["reject_reason"] = "failed", "save_failed"
                return rec
            self.sha_seen.add(sha)
            self.dhash_seen.append(dh)
        self._append_indexes(sha, metrics["dhash"], dest)
        rec["status"] = "saved"
        rec["file"] = dest
        rec["subdir"] = f"{task.platform}/{subdir}"
        return rec

    # ---- 平台批量执行 ----
    def run_platform(self, platform: str, tasks, need: int) -> int:
        os.makedirs(os.path.join(self.out_root, platform), exist_ok=True)
        stats = Counter()
        sharp_list = []
        saved = 0
        pool = cf.ThreadPoolExecutor(max_workers=max(1, self.args.workers))
        # 在途任务窗口：既保证 workers 打满，又限制超采（达标后取消未开始的）
        window = min(max(2, self.args.workers * 2), need + 4)

        def handle(rec):
            stats[rec["status"]] += 1
            if rec["status"] == "rejected":
                stats[f"reject:{str(rec.get('reject_reason', '?')).split('(')[0]}"] += 1
            if rec["status"] == "saved":
                sharp_list.append(rec["sharpness"])
                say(f"  [{platform}] + {os.path.basename(rec['file']):34s} "
                    f"{rec['width']}x{rec['height']}  sharp={rec['sharpness']:7.1f}  "
                    f"{rec['bytes'] // 1024:5d}KB  {rec['via']}")
            self._write_meta(rec)

        inflight = set()
        task_iter = iter(tasks)
        interrupted = False
        try:
            while saved < need:
                while len(inflight) < window:
                    try:
                        t = next(task_iter)
                    except StopIteration:
                        break
                    inflight.add(pool.submit(self._work, t))
                if not inflight:
                    break
                done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                for fu in done:
                    rec = fu.result()
                    handle(rec)
                    if rec["status"] == "saved":
                        saved += 1
        except KeyboardInterrupt:
            say(f"[{platform}] 已中断，收尾…")
            interrupted = True
        finally:
            for fu in inflight:
                fu.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=True)
        self.stats[platform] = (stats, sharp_list)
        if interrupted:
            raise KeyboardInterrupt
        return saved


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="图虫/堆糖/B站 三平台统一小样本采集（≥1080p + CPU 质量过滤 + 去重）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--platform", default="all",
                    help="all / tuchong / duitang / bilibili，可逗号组合")
    ap.add_argument("--queries", default="猫,夜景", help="图虫标签 & 堆糖关键词，逗号分隔")
    ap.add_argument("--queries-file", type=Path, default=None,
                    help="从文件读词表（每行一个词或 类别<TAB>词，# 开头为注释），优先于 --queries")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="词表随机打乱种子（指定同一 seed 可复现/续跑顺序；不指定则每次随机）")
    ap.add_argument("--per-query", type=int, default=10,
                    help="每个 query（B站为一个 feed）目标保存张数（通过过滤后）")
    ap.add_argument("--pages", type=int, default=6, help="每平台/query 最多枚举页数")
    ap.add_argument("--bili-source", choices=["popular", "ranking"], default="popular")
    ap.add_argument("--out", default="/mnt/nfs/data/zhanqian.wu/unified_images")
    ap.add_argument("--workers", type=int, default=8, help="下载+过滤并发数")
    ap.add_argument("--per-dir", type=int, default=1000,
                    help="每个 part_XXX 子目录最多放多少张（图片按 {平台}/{query}/part_XXX 分片存放，0=不分片只按 query 分目录）")
    ap.add_argument("--delay", type=float, default=0.4, help="全局请求最小间隔秒")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--min-bytes", type=int, default=10000, help="小于此字节数视为无效")
    # 质量过滤阈值
    ap.add_argument("--min-short", type=int, default=1080, help="短边最小像素（1080p）")
    ap.add_argument("--tuchong-min-short", type=int, default=800,
                    help="图虫例外档短边最小值（匿名上限为长边1200，如 800x1200）")
    ap.add_argument("--min-long", type=int, default=0, help="长边最小像素，0=不限")
    ap.add_argument("--max-aspect", type=float, default=3.5, help="宽高比上限（同时取倒数下限）")
    ap.add_argument("--sharpness-min", type=float, default=60.0,
                    help="清晰度阈值（短边归一 1080 后 Laplacian 方差）")
    ap.add_argument("--min-std", type=float, default=12.0, help="灰度标准差下限")
    ap.add_argument("--min-mean", type=int, default=10, help="灰度均值下限（过暗）")
    ap.add_argument("--max-mean", type=int, default=245, help="灰度均值上限（过曝）")
    ap.add_argument("--hamming", type=int, default=8, help="dHash 汉明距离去重阈值")
    ap.add_argument("--dry-run", action="store_true", help="只枚举候选不下载")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    runner = Runner(args)
    platforms = [p.strip() for p in args.platform.split(",") if p.strip()]
    if platforms == ["all"]:
        platforms = ["tuchong", "duitang", "bilibili"]

    # ---- 词表加载（文件优先），随机打乱避免类别/平台扎堆 ----
    if args.queries_file:
        queries = []
        for line in args.queries_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            w = line.split("\t")[-1].strip()      # 兼容 类别<TAB>词
            if w:
                queries.append(w)
    else:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()] if args.queries else []
    seen_q = set()
    queries = [q for q in queries if not (q in seen_q or seen_q.add(q))]
    if args.shuffle_seed is not None:
        random.seed(args.shuffle_seed)
    random.shuffle(queries)
    say(f"词表: {len(queries)} 个 query，已随机打乱"
        + (f"（seed={args.shuffle_seed}）" if args.shuffle_seed is not None else "（本次随机顺序）"))

    # 已达标 query 跳过（断点续跑）：从平台 metadata.jsonl 统计 saved 数
    skip_info = {}
    if args.queries_file:
        for p in platforms:
            path = os.path.join(args.out, p, "metadata.jsonl")
            cnt = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r.get("status") == "saved" and r.get("query"):
                            cnt[r["query"]] = cnt.get(r["query"], 0) + 1
            skip_info[p] = cnt

    # plans: (platform, label, 工厂函数, need)。tuchong/duitang 的 query 计划按 round-robin 交错，
    # 使两个平台数据同步增长；bilibili 单 feed 计划放最前。
    query_plans = []          # (platform, query)
    for p in platforms:
        if p in ("tuchong", "duitang"):
            for q in queries:
                if args.queries_file and skip_info.get(p, {}).get(q, 0) >= args.per_query:
                    continue
                query_plans.append((p, q))
    # round-robin 交错
    by_p = {}
    for item in query_plans:
        by_p.setdefault(item[0], []).append(item)
    interleaved = []
    iters = {p: iter(v) for p, v in by_p.items()}
    while iters:
        for p in list(iters):
            try:
                interleaved.append(next(iters[p]))
            except StopIteration:
                del iters[p]

    plans = []
    if "bilibili" in platforms:
        plans.append(("bilibili", "bilibili[feed]",
                      lambda: enum_bilibili(args, runner.limiter), args.per_query))
    for p, q in interleaved:
        if p == "tuchong":
            plans.append(("tuchong", f"tuchong[{q}]",
                          lambda q=q: enum_tuchong(q, args, runner.limiter), args.per_query))
        else:
            plans.append(("duitang", f"duitang[{q}]",
                          lambda q=q: enum_duitang(q, args, runner.limiter), args.per_query))
    skipped = sum(1 for p in ("tuchong", "duitang")
                  if p in platforms for q in queries
                  if skip_info.get(p, {}).get(q, 0) >= args.per_query)
    if skipped:
        say(f"跳过已达标 query: {skipped} 个（上次运行已保存 ≥ {args.per_query} 张）")
    if not plans:
        say("没有可执行的采集计划")
        return 2

    if args.dry_run:
        for platform, label, factory, need in plans:
            say(f"\n===== dry-run {label}（候选上限 {need * 3}）=====")
            for i, t in enumerate(factory()):
                if i >= need * 3:
                    break
                say(f"  {t.platform:9s} {t.name:30s} {t.urls[0][:100]}")
        return 0

    say(f"运行 {runner.run_id} | 输出 {runner.out_root} | "
        f"过滤: 短边>={args.min_short} 清晰度>={args.sharpness_min} "
        f"std>={args.min_std} 宽高比<={args.max_aspect} 汉明<={args.hamming}")
    t0 = time.time()
    total_saved = 0
    try:
        for platform, label, factory, need in plans:
            say(f"\n===== {label}（目标 {need} 张通过过滤）=====")
            saved = runner.run_platform(platform, factory(), need)
            total_saved += saved
            say(f"[{label}] 通过并保存 {saved}/{need}")
    except KeyboardInterrupt:
        say("\n已被用户中断，以下为部分结果汇总")

    say("\n===== 汇总 =====")
    for platform, (stats, sharp_list) in runner.stats.items():
        sharp_summary = ""
        if sharp_list:
            qs = np.percentile(np.array(sharp_list), [25, 50, 75]).round(1).tolist()
            sharp_summary = f" | 清晰度 p25/50/75={qs}"
        say(f"  {platform:9s} saved={stats['saved']:4d} dup={stats['duplicate']:3d} "
            f"rejected={stats['rejected']:4d} failed={stats['failed']:3d}{sharp_summary}")
        reasons = {k: v for k, v in sorted(stats.items()) if k.startswith("reject:")}
        if reasons:
            say(f"      拒绝明细: " + ", ".join(f"{k[7:]}={v}" for k, v in reasons.items()))
    say(f"总保存 {total_saved} 张，耗时 {time.time() - t0:.1f}s")
    say(f"metadata: 各平台目录 metadata.jsonl + {runner.meta_dir}/all_metadata.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
