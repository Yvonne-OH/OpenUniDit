#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pfs_to_nfs_copy.py — PFS → NFS 整库拷贝（多进程 + 断点续传 + 进度）

用法:
    python3 pfs_to_nfs_copy.py --src <src库> --dst <dst库> [--workers 12]
    python3 pfs_to_nfs_copy.py --src ... --dst ... --dry-run

实现要点:
  - 枚举：单次 find 全树（输出落临时文件，避免管道缓冲）。PFS 上 os.walk 遍历
    数百万小文件极慢，find 一次拿到全量 + 大小。
  - 拷贝：多进程（ProcessPool，每进程一个线程池），文件级并发。
  - 断点：目标已存在且大小一致则跳过（幂等，可随时重跑）。
  - 原子写：先写 .part 再 os.replace，避免下游读到半成品。
  - 符号链接按实体内容复制（目标端是真实文件）。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

LOCK = __import__("threading").Lock()


def say(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_one(src_root: str, rel_dir: str):
    """对单个子目录做 find，返回 [(rel, size)]。"""
    out = []
    with tempfile.NamedTemporaryFile("w+", suffix=".lst", delete=False) as tf:
        lst = tf.name
    try:
        with open(lst, "w") as f:
            subprocess.run(["find", "-H", os.path.join(src_root, rel_dir),
                            "(", "-type", "f", "-o", "-type", "l", ")",
                            "-printf", "%s\\t%P\\n"],
                           stdout=f, stderr=subprocess.DEVNULL, check=False)
        with open(lst) as f:
            for line in f:
                try:
                    size, rel = line.rstrip("\n").split("\t", 1)
                except ValueError:
                    continue
                if size.isdigit():
                    out.append((f"{rel_dir}/{rel}", int(size)))
    finally:
        os.unlink(lst)
    return out


def enumerate_files(src_root: str, skip_dirs=(), workers=8):
    """按 (数据集/子目录) 并行 find，汇总 (rel_paths, total_bytes)。
    skip_dirs 里的子目录整体跳过（文件数极多，用整目录 cp 更划算）。"""
    tasks = []
    for ds in sorted(os.listdir(src_root)):
        dp = os.path.join(src_root, ds)
        if not os.path.isdir(dp):
            continue
        for sub in sorted(os.listdir(dp)):
            sp = os.path.join(dp, sub)
            rel = f"{ds}/{sub}"
            if os.path.isdir(sp) and rel not in skip_dirs:
                tasks.append(rel)
    rels, total = [], 0
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
        futs = {ex.submit(_find_one, src_root, t): t for t in tasks}
        for fu in as_completed(futs):
            part = fu.result()
            rels.extend(part)
            total += sum(s for _r, s in part)
    return rels, total


def copy_worker(src_root: str, dst_root: str, force: bool):
    """子进程入口：从 stdin 读 "size\trel" 行，逐个拷贝。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    state = {"new": 0, "skip": 0, "err": 0, "bytes": 0}

    def one(rel, size):
        dst = os.path.join(dst_root, rel)
        if not force:
            try:
                if os.path.getsize(dst) == size:
                    with lock:
                        state["skip"] += 1
                        state["bytes"] += size
                    return
            except OSError:
                pass
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        try:
            with open(os.path.join(src_root, rel), "rb") as fi, \
                    open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo, length=16 << 20)
                fo.flush()
                os.fsync(fo.fileno())
            os.replace(tmp, dst)
            with lock:
                state["new"] += 1
                state["bytes"] += size
        except Exception as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            with lock:
                state["err"] += 1
            say(f"  ERROR {rel}: {e!r}")

    buf = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = []
        for line in sys.stdin:
            try:
                size, rel = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue
            buf.append((rel, int(size)))
            if len(buf) >= 200:
                futs = [pool.submit(one, r, s) for r, s in buf]
                for f in futs:
                    f.result()
                buf = []
        if buf:
            for f in [pool.submit(one, r, s) for r, s in buf]:
                f.result()
    return state


def _cp_subdir(args):
    """拷贝一个子目录（模块级函数才能被 ProcessPool pickle）。"""
    s, d = args
    if os.path.isdir(d) and os.listdir(d):
        return 0, 0                            # 已拷过（幂等：非空则跳过）
    r = subprocess.run(["cp", "-rL", os.path.join(s, "."), d],
                       stderr=subprocess.DEVNULL)
    return (1, 0) if r.returncode == 0 else (0, 1)


def cp_dir_parallel(sdir: str, ddir: str, jobs: int = 24):
    """并行拷贝一个目录树。cp -rL 单线程处理几十万小文件太慢，
    这里按顶层子目录并行 cp -rL。
    返回 (成功数, 失败数)。"""
    os.makedirs(ddir, exist_ok=True)
    subs = [d for d in sorted(os.listdir(sdir))
            if os.path.isdir(os.path.join(sdir, d))]
    top_files = [f for f in sorted(os.listdir(sdir))
                 if os.path.isfile(os.path.join(sdir, f))]
    for f in top_files:                       # 顶层散文件直接拷
        s = os.path.join(sdir, f)
        d = os.path.join(ddir, f)
        if not os.path.exists(d):
            try:
                with open(s, "rb") as fi, open(d + ".part", "wb") as fo:
                    shutil.copyfileobj(fi, fo, length=16 << 20)
                os.replace(d + ".part", d)
            except OSError:
                pass

    ok = err = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_cp_subdir, (os.path.join(sdir, s),
                                        os.path.join(ddir, s))) for s in subs]
        for fu in as_completed(futs):
            a, b = fu.result()
            ok += a
            err += b
    return ok, err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=12, help="并行进程数")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="只拷指定 dataset 前缀（逗号分隔）")
    ap.add_argument("--skip-dirs", default="",
                    help="跳过这些 数据集/子目录（用 --cp-dirs 整目录拷）")
    ap.add_argument("--cp-dirs", default="",
                    help="用整目录 cp 拷的 数据集/子目录（逗号分隔），-L 跟随符号链接")
    args = ap.parse_args()

    skip_dirs = {x.strip() for x in args.skip_dirs.split(",") if x.strip()}
    cp_dirs = [x.strip() for x in args.cp_dirs.split(",") if x.strip()]

    say(f"src={args.src}")
    say(f"dst={args.dst}")
    t0 = time.time()
    rels, total = enumerate_files(args.src, skip_dirs, args.workers)
    if args.only:
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        rels = [(r, s) for r, s in rels if any(r.startswith(w + "/") for w in want)]
        total = sum(s for _r, s in rels)
    say(f"文件 {len(rels)} 个, {total/1e9:.1f}GB, 枚举耗时 {time.time()-t0:.0f}s")

    # ---- 整目录 cp（大量小文件时比逐文件枚举快得多）----
    for d in cp_dirs:
        sdir = os.path.join(args.src, d)
        ddir = os.path.join(args.dst, d)
        if not os.path.isdir(sdir):
            say(f"skip {d}: 源不存在")
            continue
        say(f"整目录并行拷贝 {d} ...")
        tc = time.time()
        ok, err = cp_dir_parallel(sdir, ddir, args.workers * 2)
        say(f"  {d} 子目录={ok} err={err} 用时 {time.time()-tc:.0f}s")

    if args.dry_run:
        for r, s in rels[:20]:
            print(f"  {s/1e6:.1f}MB {r}")
        return

    os.makedirs(args.dst, exist_ok=True)
    # 按大小降序分片，把大文件（mdb）均分到各 worker，避免长尾
    rels.sort(key=lambda t: -t[1])
    nw = max(args.workers, 1)
    chunks = [rels[i::nw] for i in range(nw)]

    procs = []
    for ch in chunks:
        if not ch:
            continue
        p = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--_worker",
             "--src", args.src, "--dst", args.dst]
            + (["--force"] if args.force else []),
            stdin=subprocess.PIPE, text=True)
        for rel, size in ch:
            p.stdin.write(f"{size}\t{rel}\n")
        p.stdin.close()
        procs.append(p)

    done_files = done_bytes = 0
    last = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(20)
        now = time.time()
        try:
            cur_files = sum(1 for p in procs if p.poll() is not None)
        except Exception:
            cur_files = 0
        # 用目标端实际大小估算进度（粗略但无额外 IO 压力）
        el = now - t0
        say(f"进度: 活跃 worker {sum(1 for p in procs if p.poll() is None)}/{len(procs)}, "
            f"已结束 {cur_files}, 用时 {el/60:.1f}min")
    for p in procs:
        p.wait()
    say(f"ALL_DONE 用时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    if "--_worker" in sys.argv:
        i = sys.argv.index("--_worker")
        src = sys.argv[i + 2]
        dst = sys.argv[i + 4]
        st = copy_worker(src, dst, "--force" in sys.argv)
        say(f"worker done: new={st['new']} skip={st['skip']} err={st['err']} "
            f"{st['bytes']/1e9:.1f}GB")
        sys.exit(1 if st["err"] else 0)
    main()
