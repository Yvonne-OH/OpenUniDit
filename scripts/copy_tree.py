#!/usr/bin/env python3
"""流式多线程断点续传复制: copy_tree.py SRC DST PROGRESS_FILE [WORKERS]
- 流式: 边遍历边复制, 无预扫描, 启动即传
- 续传: worker 并行检查(目标已存在且大小相同 -> 跳过)
- 原子写: .copytmp -> rename, 中断不留半文件
- PROGRESS_FILE 每 2s 重写为干净快照(供心跳展示)
"""
import os, sys, time, threading, queue

CHUNK = 4 * 1024 * 1024
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 32

def fmt_bytes(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}" if u != "B" else f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def fmt_hms(s):
    s = int(s)
    return f"{s // 3600:02d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"

def main():
    src, dst, plog = sys.argv[1], sys.argv[2], sys.argv[3]
    q = queue.Queue(maxsize=WORKERS * 8)
    lock = threading.Lock()
    st = {"bytes": 0, "sbytes": 0, "done": 0, "skip": 0, "err": 0,
          "tbytes": 0, "tfiles": 0, "walk": False, "cur": ""}
    errs = []
    win = []          # (时间, 字节) 滑动窗口, 用于瞬时速度
    t0 = time.time()

    def walk_dir(dst_root, src_root):
        n = tb = 0
        for root, dnames, files in os.walk(src_root):
            rel = os.path.relpath(root, src_root)
            d = dst_root if rel == "." else os.path.join(dst_root, rel)
            os.makedirs(d, exist_ok=True)
            for f in files:
                s = os.path.join(root, f)
                q.put((s, os.path.join(d, f)))
                try:
                    tb += os.path.getsize(s)
                except OSError:
                    pass
                n += 1
        return n, tb

    def worker():
        while True:
            job = q.get()
            if job is None:
                q.task_done()
                return
            s, t = job
            try:
                ssz = os.path.getsize(s)
                if os.path.exists(t) and os.path.getsize(t) == ssz:
                    with lock:
                        st["skip"] += 1
                        st["sbytes"] += ssz
                    q.task_done()
                    continue
                tmp = t + ".copytmp"
                with open(s, "rb") as fi, open(tmp, "wb") as fo:
                    while True:
                        b = fi.read(CHUNK)
                        if not b:
                            break
                        fo.write(b)
                        with lock:
                            st["bytes"] += len(b)
                os.replace(tmp, t)
                try:
                    fst = os.stat(s)
                    os.utime(t, (fst.st_atime, fst.st_mtime))
                except OSError:
                    pass
                with lock:
                    st["done"] += 1
                    st["cur"] = os.path.basename(t)
            except Exception as e:
                with lock:
                    st["err"] += 1
                    errs.append(f"{s}: {e}")
            q.task_done()

    def snapshot(pf, final=False):
        C_G, C_Y, C_R, C_C, C_D, C_0 = ("\033[32m", "\033[33m", "\033[31m",
                                        "\033[36m", "\033[2m", "\033[0m")
        with lock:
            b, dn, sk, err = st["bytes"], st["done"], st["skip"], st["err"]
            tb, tf, walked, cur = st["tbytes"], st["tfiles"], st["walk"], st["cur"]
            sk_bytes = st["sbytes"]
        now = time.time()
        el = max(now - t0, 1e-9)
        sp_avg = b / el
        # 瞬时速度: 最近 10s 滑动窗口
        win.append((now, b))
        while len(win) > 2 and win[-1][0] - win[0][0] > 10:
            win.pop(0)
        dt = win[-1][0] - win[0][0]
        sp = (win[-1][1] - win[0][1]) / dt if dt > 0.5 else sp_avg
        if final:
            sp = sp_avg
        prog_bytes = b + sk_bytes
        def c_speed(s):
            mbs = s / 1048576
            c = C_G if mbs >= 100 else (C_Y if mbs >= 1 else C_R)
            return f"{c}{fmt_bytes(s)}/s{C_0}"
        def c_pct(p):
            c = C_G if p >= 99.9 else (C_Y if p < 10 else C_C)
            return f"{c}{p:5.1f}%{C_0}"
        if walked and tb:
            pct = 100.0 * prog_bytes / tb
            eta = fmt_hms((tb - prog_bytes) / sp_avg) if sp_avg > 1 else "?"
            head = (f"{c_pct(pct)}  {C_C}{fmt_bytes(b)}{C_D}/{fmt_bytes(tb)}{C_0}  "
                    f"{c_speed(sp)}  {C_D}ETA={eta}  已运行={fmt_hms(el)}{C_0}")
        else:
            head = (f"{C_Y}统计中{C_0}  已传 {C_C}{fmt_bytes(b)}{C_0}  "
                    f"{c_speed(sp)}  {C_D}已运行={fmt_hms(el)}{C_0}")
        pf.seek(0)
        pf.write(f"{C_C}STAT{C_0}  {head}\n")
        pf.write(f"文件  {C_G}完成 {dn}{C_0}  {C_D}跳过 {sk}{C_0}  "
                 f"总数 {tf if walked else '?'}  "
                 f"{C_R + f'错误 {err}' if err else f'错误 0'}{C_0}\n")
        pf.write(f"当前  {C_D}{cur or '-'}{C_0}\n")
        if errs:
            pf.write(f"{C_R}错误明细(最近3条):{C_0}\n")
            for e in errs[-3:]:
                pf.write(f"  {C_R}{e}{C_0}\n")
        if final:
            pf.write(f"{C_G if err == 0 else C_R}DONE  完成 {dn} 跳过 {sk} 错误 {err} "
                     f"{fmt_bytes(b)} 平均 {fmt_bytes(sp)}/s 用时 {fmt_hms(el)}{C_0}\n")
        pf.truncate()
        pf.flush()

    os.makedirs(dst, exist_ok=True)
    ws = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for w in ws:
        w.start()

    stop = threading.Event()
    with open(plog, "w") as pf:
        def reporter():
            while not stop.wait(2):
                snapshot(pf)
        rt = threading.Thread(target=reporter, daemon=True)
        rt.start()
        tf, tb = walk_dir(dst, src)
        with lock:
            st["tfiles"], st["tbytes"], st["walk"] = tf, tb, True
        for _ in ws:
            q.put(None)
        q.join()
        stop.set()
        rt.join(timeout=1)
        snapshot(pf, final=True)
    el = time.time() - t0
    with lock:
        dn, sk, err, b = st["done"], st["skip"], st["err"], st["bytes"]
    print(f"{'OK' if err == 0 else 'PARTIAL'} {dn} copied, {sk} skipped, {err} errors, "
          f"{fmt_bytes(b)} in {fmt_hms(el)}")
    sys.exit(0 if err == 0 else 2)

if __name__ == "__main__":
    main()
