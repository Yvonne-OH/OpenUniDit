#!/usr/bin/env python3
"""JourneyDB tgz → unified_images_lmdb 并行流式转换器 v2（格式对齐 4KLSDB）。

瓶颈背景: NFS 上 mmap 随机写 LMDB 仅 ~3.7张/s。v2 改为:
  Lustre 暂存写分片 → 顺序拷贝到 NFS 输出 → 删除暂存。
  每个源 tgz 一个分片 (~2.18万条), N 个 worker 进程并行。

输出 (OUT=/mnt/nfs/data/zhanqian.wu/unified_images_lmdb/JourneyDB):
  mdb/JourneyDB_0000.mdb ...     key=struct.pack("<Q", sid) value=pickle({id,source,caption,w,h,img_bytes})
  parquet/meta_0000.parquet ...  与 mdb sorted zip 对应; 全部完成后生成 meta.parquet
  grid/grid_{sid_start:08d}.png  每分片一张 (前 20 图 4x5 缩略, 2800x3600)

sample_id: 按 batch 序号预分配区间 (区间宽度=anno 中该目录图片数), 全局单调、可断点续跑。
caption=Task2.Caption, caption_full=prompt。

用法: python journeydb_to_lmdb.py [--src DIR] [--out DIR] [--stage DIR] [--workers 5]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
import shutil
import struct
import sys
import time
from pathlib import Path

import lmdb
import pandas as pd
import pyarrow as pa
from PIL import Image

GREEN, BLUE, YELLOW, RED, RESET = "\033[32m", "\033[34m", "\033[33m", "\033[31m", "\033[0m"

SOURCE = "JourneyDB"
GRID_COLS, GRID_ROWS, CELL_W, CELL_H = 4, 5, 700, 720

PARQUET_SCHEMA = pa.schema([
    ("sample_id", pa.int64()), ("source_index", pa.int64()),
    ("batch", pa.string()), ("file", pa.string()), ("caption", pa.string()),
    ("width", pa.int32()), ("height", pa.int32()), ("file_size", pa.int64()),
    ("caption_full", pa.string()), ("url", pa.string()),
])

_TABLE: dict = {}  # fork 继承


def log(msg: str, color: str = GREEN) -> None:
    print(f"{color}[{time.strftime('%m-%d %H:%M:%S')}] {msg}{RESET}", flush=True)


def load_caption_table(anno_tgz: Path, cache: Path) -> dict:
    if cache.exists():
        log(f"📖 加载 caption 缓存 {cache.name} ...", BLUE)
        with open(cache, "rb") as f:
            return pickle.load(f)
    import re
    import tarfile
    log(f"📖 解析标注 {anno_tgz.name} ...", BLUE)
    table: dict[str, tuple[str, str, str]] = {}
    with tarfile.open(anno_tgz, "r|gz") as tf:
        f = tf.extractfile(tf.next())
        buf = b""
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            lines = buf.split(b"\n")
            buf = lines.pop()
            for l in lines:
                if not l.strip():
                    continue
                o = json.loads(l)
                key = re.sub(r"^\./(\./)?", "", o["img_path"])
                cap = (o.get("Task2") or {}).get("Caption") or ""
                table[key] = (cap, o.get("prompt") or "", o.get("url") or "")
    with open(cache, "wb") as fh:
        pickle.dump(table, fh, protocol=4)
    return table


def process_batch(idx: int) -> dict:
    """worker: tgz idx → staging 分片 → 拷贝 NFS → 清理。返回统计。"""
    cfg = _CFG
    out_root, stage = Path(cfg["out"]), Path(cfg["stage"])
    stem = f"{idx:03d}"
    tgz = Path(cfg["src"]) / "data" / "train" / "imgs" / f"{stem}.tgz"
    sid_start = cfg["sid_start"][idx]
    import tarfile
    t0 = time.time()
    mdb_stage = stage / f"{SOURCE}_{idx:04d}.mdb"
    env = lmdb.open(str(mdb_stage), subdir=False, map_size=30 * 2**30,
                    meminit=False, map_async=True)
    rows, thumbs = [], []
    n_ok = n_miss = n_bad = 0
    with env.begin(write=True) as txn, tarfile.open(tgz, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            name = m.name.split("/")[-1]
            ann = _TABLE.get(f"{stem}/{name}")
            if ann is None:
                n_miss += 1
                continue
            caption, prompt, url = ann
            try:
                raw = tf.extractfile(m).read()
                with Image.open(io.BytesIO(raw)) as im:
                    w, h = im.width, im.height
            except Exception:  # noqa: BLE001
                n_bad += 1
                continue
            sid = sid_start + n_ok
            rec = {"id": sid, "source": SOURCE, "caption": caption, "w": w, "h": h,
                   "img_bytes": raw}
            txn.put(struct.pack("<Q", sid), pickle.dumps(rec, protocol=4))
            rows.append({"sample_id": sid, "source_index": sid, "batch": stem,
                         "file": name, "caption": caption, "width": w, "height": h,
                         "file_size": len(raw), "caption_full": prompt, "url": url})
            if len(thumbs) < GRID_COLS * GRID_ROWS:
                thumbs.append(raw)
            n_ok += 1
            if n_ok % 2000 == 0:
                with open(stage / "progress" / f"{stem}.txt", "w") as pf:
                    pf.write(str(n_ok))
    env.sync()
    env.close()
    df = pd.DataFrame(rows, columns=[c.name for c in PARQUET_SCHEMA])
    pq_stage = stage / f"meta_{idx:04d}.parquet"
    df.to_parquet(pq_stage, schema=PARQUET_SCHEMA, index=False)
    grid_stage = stage / f"grid_{sid_start:08d}.png"
    grid = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H), (16, 16, 16))
    for i, b in enumerate(thumbs):
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        im.thumbnail((CELL_W, CELL_H))
        grid.paste(im, ((i % GRID_COLS) * CELL_W + (CELL_W - im.width) // 2,
                        (i // GRID_COLS) * CELL_H + (CELL_H - im.height) // 2))
    grid.save(grid_stage)
    # 顺序拷贝到 NFS (原子: .tmp → rename)
    dst_mdb = out_root / "mdb" / f"{SOURCE}_{idx:04d}.mdb"
    dst_pq = out_root / "parquet" / f"meta_{idx:04d}.parquet"
    dst_grid = out_root / "grid" / f"grid_{sid_start:08d}.png"
    for src_p, dst_p in ((mdb_stage, dst_mdb), (pq_stage, dst_pq), (grid_stage, dst_grid)):
        tmp = dst_p.with_suffix(dst_p.suffix + ".copytmp")
        shutil.copyfile(src_p, tmp)
        os.replace(tmp, dst_p)
        src_p.unlink()
    (stage / "progress" / f"{stem}.txt").unlink(missing_ok=True)
    res = {"idx": idx, "count": n_ok, "miss": n_miss, "bad": n_bad,
           "sid_start": sid_start, "sec": round(time.time() - t0)}
    with open(stage / "done" / f"{stem}.json", "w") as f:
        json.dump(res, f)
    return res


def main() -> None:
    global _CFG, _TABLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/nfs/data/zhanqian.wu/unified_images/JourneyDB")
    ap.add_argument("--out", default="/mnt/nfs/data/zhanqian.wu/unified_images_lmdb/JourneyDB")
    ap.add_argument("--stage", default="/home/jovyan/pfs/users/zhanqian.wu/pfs_data/journeydb_staging")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    src, out_root, stage = Path(args.src), Path(args.out), Path(args.stage)
    imgs_dir = src / "data" / "train" / "imgs"
    for d in (out_root / "mdb", out_root / "parquet", out_root / "grid",
              stage / "done", stage / "progress"):
        d.mkdir(parents=True, exist_ok=True)
    anno_tgz = src / "data" / "train" / "train_anno_realease_repath.jsonl.tgz"
    if not anno_tgz.exists():
        print(f"{RED}❌ 标注未就绪: {anno_tgz}{RESET}")
        sys.exit(1)

    _TABLE = load_caption_table(anno_tgz, out_root / "caption_table.pkl")
    log(f"📚 caption 表: {len(_TABLE)} 条", GREEN)

    # 期望 batch 清单 + anno 计数 → sid 区间
    expected = sorted({p[:-4] for p in os.listdir(imgs_dir)
                       if p.endswith(".tgz") or p.endswith(".tgz.part")})
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from journeydb_download import list_files
        remote = sorted(os.path.basename(f["Path"])[:-4] for f in list_files("data/train/imgs"))
        if len(remote) > len(expected):
            expected = remote
    except Exception as e:  # noqa: BLE001
        log(f"⚠️ 远端清单失败, 用本地发现: {e}", YELLOW)
    counts = {b: 0 for b in expected}
    for key in _TABLE:
        b = key.split("/", 1)[0]
        if b in counts:
            counts[b] += 1
    sid_start, acc = {}, 0
    for b in expected:
        sid_start[int(b)] = acc
        acc += counts[b]
    log(f"🎯 {len(expected)} 个 batch, 预计总量 {acc} 张", BLUE)

    _CFG = {"src": str(src), "out": str(out_root), "stage": str(stage), "sid_start": sid_start}

    done = {}
    for p in glob.glob(str(stage / "done" / "*.json")):
        r = json.load(open(p))
        done[r["idx"]] = r
    # 校验 NFS 最终文件存在 (done 标记在拷贝之后写, 理论上一致)
    todo = [int(b) for b in expected if int(b) not in done]
    total_written = sum(r["count"] for r in done.values())
    if done:
        log(f"🔄 断点续跑: 已完成 {len(done)} 个 batch, 共 {total_written} 张", BLUE)

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers)
    pending = set()
    t_start = time.time()
    i = 0
    while i < len(expected) or pending:
        # 派发已就绪的 batch
        while i < len(expected):
            stem = expected[i]
            tgz = imgs_dir / f"{stem}.tgz"
            if not tgz.exists():
                break
            part = imgs_dir / f"{stem}.tgz.part"
            if part.exists():  # 下载器 rename 前不应存在 .tgz; 双保险
                break
            if int(stem) not in done:
                pending.add(pool.apply_async(process_batch, (int(stem),)))
            i += 1
        # 收割完成的任务
        for r in list(pending):
            if r.ready():
                pending.discard(r)
                res = r.get()
                done[res["idx"]] = res
                total_written += res["count"]
                log(f"✅ batch {res['idx']:03d}: {res['count']} 张 "
                    f"(anno缺失{res['miss']}/坏图{res['bad']}), 耗时 {res['sec']}s, "
                    f"累计 {total_written}, 队列 {len(pending)}/{args.workers}", GREEN)
        if i < len(expected) and not pending:
            time.sleep(20)  # 等下载
            continue
        time.sleep(5)
    pool.close()
    pool.join()

    # 汇总 meta.parquet
    files = sorted((out_root / "parquet").glob("meta_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tmp = out_root / "parquet" / "meta.parquet.copytmp"
    df.to_parquet(tmp, schema=PARQUET_SCHEMA, index=False)
    os.replace(tmp, out_root / "parquet" / "meta.parquet")
    log(f"🏁 完成: {len(df)} 条 / {len(files)} 分片, 用时 {(time.time()-t_start)/3600:.2f} h", BLUE)


if __name__ == "__main__":
    main()
