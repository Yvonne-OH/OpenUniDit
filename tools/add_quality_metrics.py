#!/usr/bin/env python3
"""给 unified_*_images_lmdb 所有数据集的 parquet meta 增加 quality 字典列。

列名 quality, 类型 map<string,float32>, 每行一个 dict:
  lap_var      Laplacian 方差, <80 判糊
  blockiness   8px 块边界梯度均值 / 内部梯度均值, >1.6 判强压缩
  colorfulness Hasler-Süsstrunk 指标, <15 判近灰/单色
  over_expo    (gray>250) 占比, >0.2 判过曝
  under_expo   (gray<5) 占比, >0.2 判欠曝
  black_border 上下左右黑边占比 (四边平均), >0.15 判截图/视频抽帧
  saturation   HSV S 通道均值 (0-255), 极低 = 灰图
编辑库 (img_before/img_after) 额外输出 before_* 前缀的同一组指标 (基于 before 图)。

用法:
  python add_quality_metrics.py --limit-shards 5 --datasets anime_captions   # 试跑测速
  python add_quality_metrics.py                        # 全量, 断点续跑
  python add_quality_metrics.py --workers 64

分片粒度断点: 每处理完一个 mdb+parquet 对, 原子更新 checkpoint 并原子替换 parquet。
blip3o-long-caption 这种"单合并 meta.parquet"形态: 先落 sidecar 缓存, 全部完成后再合并写回。
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import traceback
import time
from pathlib import Path

import numpy as np

ROOTS = [
    "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_images_lmdb",
    "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_edit_images_lmdb",
]
CKPT_PATH = Path(__file__).resolve().parent / ".quality_metrics_ckpt.json"
CACHE_DIR = Path(__file__).resolve().parent / "quality_cache"

SINGLE_FIELDS = ("img_bytes", "img", "img_single")
BEFORE_FIELD = "img_before"
AFTER_FIELD = "img_after"

METRICS = ("lap_var", "blockiness", "colorfulness", "over_expo",
           "under_expo", "black_border", "saturation")


def quality_struct_type(is_edit):
    import pyarrow as pa
    fields = list(METRICS)
    if is_edit:
        fields += [f"before_{m}" for m in METRICS]
    return pa.struct([(f, pa.float32()) for f in fields])

BORDER_TH = 16        # 行/列均值 <16 视为黑边像素
BLOCK = 8             # JPEG 8x8 块


def compute_metrics(img) -> dict:
    """img: BGR uint8 ndarray → 7 项指标 (float, round 4 位)。"""
    import cv2
    cv2.setNumThreads(0)  # 限制 cv2 内部线程, 避免 128 worker × ncore 线程超订

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # blockiness: 梯度图 = 相邻像素差; 8px 倍数行列视为块边界
    grad = np.zeros((h, w), dtype=np.float32)
    grad[:, 1:] += cv2.absdiff(gray[:, 1:], gray[:, :-1])
    grad[1:, :] += cv2.absdiff(gray[1:, :], gray[:-1, :])
    yy, xx = np.mgrid[0:h, 0:w]
    boundary = ((xx % BLOCK == 0) | (yy % BLOCK == 0))
    inner = grad[~boundary]
    bmean = float(grad[boundary].mean()) if boundary.any() else 0.0
    imean = float(inner.mean()) if inner.size else 0.0
    blockiness = bmean / (imean + 1e-6)

    # colorfulness (Hasler-Süsstrunk)
    r = img[..., 2].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 0].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    over_expo = float((gray > 250).mean())
    under_expo = float((gray < 5).mean())

    # 黑边: 每条边连续的黑行/列数, 四边占比取平均
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    nz_r = np.nonzero(row_mean >= BORDER_TH)[0]
    nz_c = np.nonzero(col_mean >= BORDER_TH)[0]
    top = int(nz_r[0]) if nz_r.size else h
    bottom = int(h - 1 - nz_r[-1]) if nz_r.size else h
    left = int(nz_c[0]) if nz_c.size else w
    right = int(w - 1 - nz_c[-1]) if nz_c.size else w
    black_border = ((top + bottom) / h + (left + right) / w) / 2.0

    # saturation = HSV S 均值, 用 max-min 等价实现省一次颜色空间转换
    mx = img.max(axis=2).astype(np.float32)
    mn = img.min(axis=2).astype(np.float32)
    s = np.divide(mx - mn, mx, out=np.zeros((h, w), dtype=np.float32), where=mx > 0) * 255.0
    saturation = float(s.mean())

    return {
        "lap_var": round(lap_var, 4),
        "blockiness": round(float(blockiness), 4),
        "colorfulness": round(colorfulness, 4),
        "over_expo": round(over_expo, 4),
        "under_expo": round(under_expo, 4),
        "black_border": round(float(black_border), 4),
        "saturation": round(saturation, 4),
    }


def _image_bytes(rec, field):
    raw = rec.get(field)
    if isinstance(raw, (bytes, bytearray)) and raw:
        return bytes(raw)
    return None


def process_shard(task):
    """处理一个 mdb+parquet 对: 读图 → 算指标 → 原子写回 parquet。

    mode='sharded': meta_{stem}.parquet 直接加列替换。
    mode='merged' : 指标落 sidecar 缓存 (由 finalize 阶段合并写回 meta.parquet)。
    返回 (key, rows, n_with_image, elapsed, one_sample)。
    """
    import cv2
    cv2.setNumThreads(0)
    import pyarrow as pa
    import pyarrow.parquet as pq
    import lmdb

    root, dataset, stem, pq_path, mdb_path, mode, is_edit = task
    key = f"{dataset}/{stem}"
    t0 = time.time()

    # 读 parquet 时排除旧 quality 列 (重跑/历史 map 类型残留), 之后统一追加 struct 列
    md = pq.ParquetFile(pq_path).metadata
    top_cols = []
    for i in range(md.num_columns):
        col = md.schema.column(i)
        name = (col.path.split(".")[0] if getattr(col, "path", None) else col.name)
        if name != "quality" and name not in top_cols:
            top_cols.append(name)
    table = pq.read_table(pq_path, columns=top_cols)
    sids = table["sample_id"].to_pylist() if "sample_id" in table.column_names else []

    env = lmdb.open(str(mdb_path), readonly=True, lock=False, subdir=False, max_readers=1024)
    metrics_by_sid = {}
    n_img = 0
    one_sample = None
    with env.begin() as txn:
        for k, v in txn.cursor():
            sid = int.from_bytes(k, "little") if len(k) == 8 else int(k.decode())
            rec = pickle.loads(v)
            m = {}
            # after / single 为主图
            main = _image_bytes(rec, AFTER_FIELD)
            if main is None:
                for f in SINGLE_FIELDS:
                    main = _image_bytes(rec, f)
                    if main is not None:
                        break
            if main is not None:
                img = cv2.imdecode(np.frombuffer(main, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    try:
                        m.update(compute_metrics(img))
                        n_img += 1
                        if one_sample is None:
                            one_sample = (sid, m)
                    except Exception as e:
                        print(f"[warn] {key} sid={sid} 主图指标计算失败, 跳过该图: {e}",
                              file=sys.stderr, flush=True)
            before = _image_bytes(rec, BEFORE_FIELD)
            if is_edit and before is not None:
                img = cv2.imdecode(np.frombuffer(before, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    try:
                        m.update({f"before_{k2}": v2 for k2, v2 in compute_metrics(img).items()})
                    except Exception as e:
                        print(f"[warn] {key} sid={sid} before图指标计算失败, 跳过该图: {e}",
                              file=sys.stderr, flush=True)
            if m:
                metrics_by_sid[sid] = m
    env.close()

    quality = []
    for sid in sids:
        m = metrics_by_sid.get(sid)
        quality.append(m if m else None)
    arr = pa.array(quality, type=quality_struct_type(is_edit))
    elapsed = time.time() - t0

    if mode == "sharded":
        table = table.append_column("quality", arr)
        tmp = f"{pq_path}.tmp_quality"
        pq.write_table(table, tmp)
        os.replace(tmp, pq_path)
    else:  # merged: 落 sidecar, 之后统一合并
        out = CACHE_DIR / dataset / f"{stem}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        side = pa.table({"sample_id": pa.array(sids, pa.int64()),
                         "quality": arr})
        tmp = str(out) + ".tmp"
        pq.write_table(side, tmp)
        os.replace(tmp, str(out))

    return key, len(sids), n_img, elapsed, one_sample


def enumerate_tasks(roots, datasets=None, limit_shards=0):
    tasks = []
    skipped_meta_only = []
    for root in roots:
        root = Path(root)
        for ds_dir in sorted(root.iterdir()):
            if not ds_dir.is_dir() or (datasets and ds_dir.name not in datasets):
                continue
            pq_dir, mdb_dir = ds_dir / "parquet", ds_dir / "mdb"
            if not mdb_dir.is_dir() or not pq_dir.is_dir():
                continue
            is_edit = "unified_edit" in root.name
            n_before = len(tasks)
            mdb_files = sorted(mdb_dir.glob("*.mdb"))
            pq_files = sorted(pq_dir.glob("meta_*.parquet"))
            if pq_files:
                # 常规形态: sorted(meta_*.parquet) 与 sorted(*.mdb) 按序 1:1 配对
                # (与训练加载器一致; 分片编号与 stem 名不必相同)
                if len(pq_files) != len(mdb_files):
                    skipped_meta_only.append(
                        f"{ds_dir.name}: parquet={len(pq_files)} mdb={len(mdb_files)} 数量不一致")
                for pq_file, mdb_file in zip(pq_files, mdb_files):
                    tasks.append((str(root), ds_dir.name, mdb_file.stem,
                                  str(pq_file), str(mdb_file), "sharded", is_edit))
            elif (pq_dir / "meta.parquet").exists():
                # 合并形态: 单 meta.parquet + 多 mdb 分片 → sidecar + finalize
                for mdb_file in mdb_files:
                    tasks.append((str(root), ds_dir.name, mdb_file.stem,
                                  str(pq_dir / "meta.parquet"), str(mdb_file),
                                  "merged", is_edit))
            if limit_shards and len(tasks) - n_before > limit_shards:
                tasks = tasks[:n_before + limit_shards]
    return tasks, skipped_meta_only


def load_ckpt():
    if CKPT_PATH.exists():
        return json.loads(CKPT_PATH.read_text())
    return {"done": {}}


def save_ckpt(ckpt):
    tmp = f"{CKPT_PATH}.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(ckpt, ensure_ascii=False))
    os.replace(tmp, CKPT_PATH)


def finalize_merged(roots):
    """合并形态: 把 sidecar quality 按 sample_id 合入 meta.parquet。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for root in roots:
        root = Path(root)
        for ds_dir in sorted(root.iterdir()):
            pq_dir, mdb_dir = ds_dir / "parquet", ds_dir / "mdb"
            if not pq_dir.is_dir() or not mdb_dir.is_dir():
                continue
            meta_stems = {f[5:-8] for f in os.listdir(pq_dir)
                          if f.startswith("meta_") and f.endswith(".parquet")}
            merged = pq_dir / "meta.parquet"
            if meta_stems or not merged.exists():
                continue
            sidecars = sorted((CACHE_DIR / ds_dir.name).glob("*.parquet"))
            if not sidecars:
                continue
            mdb_stems = {f[:-4] for f in os.listdir(mdb_dir) if f.endswith(".mdb")}
            done_stems = {s.stem for s in sidecars}
            if not mdb_stems <= done_stems:
                print(f"⏳ {ds_dir.name}: 合并形态分片未全部完成, 跳过 finalize", flush=True)
                continue
            t0 = time.time()
            lookup = {}
            for sc in sidecars:
                t = pq.read_table(sc)
                for sid, q in zip(t["sample_id"].to_pylist(), t["quality"].to_pylist()):
                    if q is not None:
                        lookup[int(sid)] = q
            md = pq.ParquetFile(merged).metadata
            lookup_cols = []
            for i in range(md.num_columns):
                col = md.schema.column(i)
                name = (col.path.split(".")[0] if getattr(col, "path", None) else col.name)
                if name != "quality" and name not in lookup_cols:
                    lookup_cols.append(name)
            table = pq.read_table(merged, columns=lookup_cols)
            quality = [lookup.get(int(sid)) for sid in table["sample_id"].to_pylist()]
            arr = pa.array(quality, type=quality_struct_type("unified_edit" in root.name))
            table = table.append_column("quality", arr)
            tmp = str(merged) + ".tmp_quality"
            pq.write_table(table, tmp)
            os.replace(tmp, str(merged))
            print(f"✅ finalize {ds_dir.name}: {table.num_rows} 行, {time.time()-t0:.1f}s", flush=True)


def total_rows_estimate(tasks):
    """读 parquet footer 估算总行数 (不加载列数据)。合并形态只按 meta.parquet 计一次。"""
    import pyarrow.parquet as pq

    total = 0
    counted_merged = set()
    for t in tasks:
        try:
            if t[5] == "merged":
                if t[1] in counted_merged:
                    continue
                counted_merged.add(t[1])
            total += pq.ParquetFile(t[3]).metadata.num_rows
        except Exception:
            pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=ROOTS)
    ap.add_argument("--datasets", nargs="*", default=None, help="只处理指定数据集")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-shards", type=int, default=0, help="只跑前 N 个分片 (测速)")
    ap.add_argument("--no-write", action="store_true", help="只测速, 不改 parquet/不记 checkpoint")
    ap.add_argument("--finalize-only", action="store_true", help="只做合并形态的最终写回")
    args = ap.parse_args()

    if args.finalize_only:
        finalize_merged(args.roots)
        return

    tasks, skipped = enumerate_tasks(args.roots, args.datasets, args.limit_shards)
    ckpt = load_ckpt()
    pending = [t for t in tasks if args.no_write or f"{t[1]}/{t[2]}" not in ckpt["done"]]
    total_rows_all = total_rows_estimate(tasks)
    total_rows_done = sum(ckpt["done"].get(f"{t[1]}/{t[2]}", {}).get("rows", 0) for t in tasks)
    print(f"任务分片 {len(tasks)}, 待处理 {len(pending)}, "
          f"总行数(含已完成) {total_rows_all}, 已完成行数 {total_rows_done}", flush=True)
    for s in skipped[:20]:
        print(f"  ⚠️ meta 无对应 mdb, 跳过: {s}", flush=True)
    if not pending:
        finalize_merged(args.roots)
        print("全部完成。", flush=True)
        return

    n_done, rows_done, img_done = 0, 0, 0
    t_start = time.time()
    if args.workers <= 1:
        results = (process_shard(t) for t in pending)
    else:
        pool = mp.Pool(args.workers)
        async_results = [(t, pool.apply_async(process_shard, (t,))) for t in pending]

    if args.workers > 1:
        iterator = async_results  # 逐个 r.get(), 每完成一个就写断点/日志
    else:
        iterator = [(t, r) for t, r in zip(pending, results)]
    n_fail = 0
    for t, r in iterator:
        try:
            key, rows, n_img, elapsed, sample = r if args.workers <= 1 else r.get()
        except Exception as e:
            print(f"[fail] 分片 {t[1]}/{t[2]} 处理异常, 跳过(下次重启会重试): {e}",
                  flush=True)
            traceback.print_exc()
            n_fail += 1
            continue
        n_done += 1
        rows_done += rows
        img_done += n_img
        if not args.no_write:
            ckpt["done"][key] = {"rows": rows, "sec": round(elapsed, 2),
                                 "ts": time.strftime("%F %T")}
            save_ckpt(ckpt)
        sp = f" sample={sample[0]}:{sample[1]}" if sample else ""
        print(f"[{n_done}/{len(pending)}] {key}: {rows} 行 ({n_img} 有图), "
              f"{elapsed:.1f}s, {rows/max(elapsed,1e-6):.0f} 行/s{sp}", flush=True)
        # 速度与 ETA
        wall = time.time() - t_start
        rate = rows_done / max(wall, 1e-6)
        remain_rows = total_rows_all - total_rows_done - rows_done
        print(f"    累计 {rows_done} 行, 平均 {rate:.0f} 行/s, "
              f"剩余 {remain_rows} 行 → 预计 {remain_rows/max(rate,1e-6)/3600:.2f} h "
              f"(workers={args.workers})", flush=True)

    if args.workers > 1:
        pool.close()
        pool.join()

    finalize_merged(args.roots)
    wall = time.time() - t_start
    print(f"完成 {n_done} 分片 / {rows_done} 行, 用时 {wall/3600:.2f} h, "
          f"平均 {rows_done/max(wall,1e-6):.0f} 行/s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
