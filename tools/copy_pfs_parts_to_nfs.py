#!/usr/bin/env python3
"""PFS -> NFS 多线程拷贝 (rsync 不可用), 文件粒度断点续传."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SRC_PARTS = [
    "/home/jovyan/pfs/data/public/image_data/aes525_575_part%03d" % i for i in range(10)
] + [
    "/home/jovyan/pfs/data/public/image_data/aes575_part%03d" % i for i in range(6)
]
DST_ROOT = "/mnt/nfs/data/zhanqian.wu/unified_images_tar"
WORKERS = 12


def collect():
    jobs = []
    for src in SRC_PARTS:
        dst = os.path.join(DST_ROOT, os.path.basename(src))
        for dirpath, _, files in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            for fn in files:
                if fn.endswith(".tmp"):
                    continue
                s = os.path.join(dirpath, fn)
                d = os.path.join(dst, rel, fn) if rel != "." else os.path.join(dst, fn)
                jobs.append((s, d))
    return jobs


def copy_one(job):
    s, d = job
    try:
        ssz = os.path.getsize(s)
        if os.path.exists(d) and os.path.getsize(d) == ssz:
            return ("skip", ssz)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        tmp = d + ".copytmp"
        with open(s, "rb") as fi, open(tmp, "wb") as fo:
            while True:
                buf = fi.read(4 << 20)
                if not buf:
                    break
                fo.write(buf)
        os.replace(tmp, d)
        return ("copy", ssz)
    except Exception as e:
        return ("err", f"{s}: {e}")


def main():
    t0 = time.time()
    jobs = collect()
    total = sum(os.path.getsize(s) for s, _ in jobs)
    print(f"[plan] {len(jobs)} files, {total/1e12:.2f} TB, workers={WORKERS}", flush=True)
    done = 0
    errs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (st, v) in enumerate(pool.map(copy_one, jobs)):
            done += 0 if st == "err" else (v if st == "copy" else 0)
            if st == "err":
                errs.append(v)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(jobs)}] copied={done/1e12:.2f}TB {done/1e9/el:.0f}MB/s err={len(errs)}", flush=True)
    print(f"[done] copied={done/1e12:.2f}TB in {(time.time()-t0)/3600:.1f}h errors={len(errs)}", flush=True)
    for e in errs[:20]:
        print("ERR", e, flush=True)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
