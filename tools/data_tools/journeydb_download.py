#!/usr/bin/env python3
"""JourneyDB data/train 并行下载器（断点续传 + sha256 校验 + 进度心跳）。

- 文件清单来自 ModelScope API（含每个 tgz 的 sha256）；
- 每个文件用 HTTP Range 追加到 .part，完成后校验并改名；
- 多 worker 并行（默认 16），单流约 18MB/s，16 流可到 ~300MB/s；
- tqdm 总进度条 + 周期性状态日志（速度/剩余/心跳）。

用法:
  python journeydb_download.py --local-dir /mnt/nfs/data/zhanqian.wu/unified_images/JourneyDB \
      --workers 16 [--include-anno] [--only-anno]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://www.modelscope.cn/datasets/AI-ModelScope/JourneyDB/resolve/master/"
API = "https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/JourneyDB/repo/tree"

GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"

_bytes_lock = threading.Lock()
_done_bytes = 0


def list_files(root: str) -> list[dict]:
    """分页拉取远端文件清单 (path/size/sha256)。"""
    from modelscope.hub.api import HubApi
    api = HubApi()
    out: list[dict] = []
    page = 1
    while True:
        res = api.get_dataset_files(repo_id="AI-ModelScope/JourneyDB", revision="master",
                                    root_path=root, page_number=page, page_size=100)
        if not res:
            break
        out.extend(f for f in res if f.get("Type") != "tree")
        if len(res) < 100:
            break
        page += 1
    if not out:
        raise RuntimeError("文件清单为空")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(entry: dict, local_dir: Path, pbar: tqdm, skip_sha: bool = False,
                 retries: int = 8) -> bool:
    rel = entry["Path"]
    url = BASE + requests.utils.requote_uri(rel)
    dst = local_dir / rel
    part = dst.with_suffix(dst.suffix + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == entry["Size"]:
        pbar.update(entry["Size"])
        return True
    for attempt in range(1, retries + 1):
        pos = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 416:  # 已下完
                    pass
                elif r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                else:
                    if r.status_code == 200 and pos:  # 服务器不支持 Range，重头来
                        pos = 0
                        mode = "wb"
                    else:
                        mode = "ab" if r.status_code == 206 else "wb"
                        pos = 0 if r.status_code == 200 else pos
                    with open(part, mode) as f:
                        f.seek(pos)
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            with _bytes_lock:
                                global _done_bytes
                                _done_bytes += len(chunk)
                            pbar.update(len(chunk))
            if part.stat().st_size != entry["Size"]:
                raise RuntimeError(f"size mismatch {part.stat().st_size} != {entry['Size']}")
            if entry.get("Sha256") and not skip_sha:
                if sha256_file(part) != entry["Sha256"]:
                    part.unlink(missing_ok=True)
                    raise RuntimeError("sha256 mismatch, restart")
            part.rename(dst)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"{YELLOW}⚠️ {rel} attempt {attempt}/{retries} 失败: {e}{RESET}", flush=True)
            time.sleep(min(30, 2 * attempt))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only-anno", action="store_true", help="只下 train_anno*.tgz")
    ap.add_argument("--exclude-anno", action="store_true")
    ap.add_argument("--skip-sha", action="store_true", help="跳过 sha256 校验 (NFS strict 缓存下回读代价极高)")
    args = ap.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    files = list_files("data/train")
    anno = [f for f in files if "anno" in f["Name"]]
    imgs = [f for f in files if f not in anno]
    if args.only_anno:
        files = anno
    elif args.exclude_anno:
        files = imgs
    files.sort(key=lambda f: f["Path"])

    total = sum(f["Size"] for f in files)
    done = sum(f["Size"] for f in files if (local_dir / f["Path"]).exists())
    print(f"{BLUE}📦 JourneyDB 下载: {len(files)} 个文件, 共 {total/1e9:.1f} GB, 已存在 {done/1e9:.1f} GB{RESET}", flush=True)

    pbar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                initial=done, desc="🚚 JourneyDB", smoothing=0.1)

    stop = threading.Event()

    def heartbeat() -> None:
        last = done
        t0 = time.time()
        while not stop.wait(60):
            cur = _done_bytes + done
            rate = (cur - last) / 60
            remain = (total - cur) / rate / 3600 if rate > 0 else -1
            print(f"{GREEN}💓 [{time.strftime('%m-%d %H:%M:%S')}] "
                  f"进度 {cur/1e9:.1f}/{total/1e9:.1f} GB ({cur/total*100:.2f}%) "
                  f"速度 {rate/1e6:.1f} MB/s 预计剩余 {remain:.1f} h{RESET}", flush=True)
            last = cur

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, f, local_dir, pbar, args.skip_sha): f for f in files}
        for fut, f in futs.items():
            if not fut.result():
                failed.append(f["Path"])

    stop.set()
    pbar.close()
    elapsed = time.time() - pbar.start_t
    print(f"{BLUE}✅ 下载结束: 用时 {elapsed/3600:.2f} h, 平均 {total/elapsed/1e6:.1f} MB/s{RESET}", flush=True)
    if failed:
        print(f"{YELLOW}⚠️ 失败 {len(failed)} 个: {failed[:10]}{RESET}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
