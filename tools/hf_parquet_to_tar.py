#!/usr/bin/env python3
"""HF 内嵌图片 parquet → unified_images_tar 子数据库转换工具。

将 HF 风格 parquet（image: struct<bytes,path> 内嵌字节）转换为
unified_images_tar 的分片 tar 四件套格式：
    shard-XXXXX-YYY.tar          blob 容器（原始字节，不转码）
    shard-XXXXX-YYY.idx          JSONL {"id","off","size"} 定位索引
    shard-XXXXX-YYY.meta.jsonl   记录级元数据
    parquet/shard-XXXXX-YYY.parquet 检索层（含 images 引用）

针对 breakpoint-grounding-55m：
    id      = breakpoint_{file:04d}_{row:07d}
    caption = image_caption
    w/h     = img_size_wh
    local_caption      = JSON(object_captions)
    normalized_boxes   = 原样保留（grounding 框，额外列）
    caption_zh/local_caption_zh = ""（源无中文）

断点续传：以源 parquet 文件为最小恢复单元，done 标记在 <dst>/.state/。
恢复时删除该文件已写的半成品分片后整文件重做，保证不重不漏。
"""
import argparse
import glob
import io
import json
import os
import shutil
import sys
import tarfile
import time
import traceback
from multiprocessing import get_context

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------- 常量
TAR_ROOT = "/mnt/nfs/data/zhanqian.wu/unified_images_tar"
SRC_DIR = "/mnt/nfs/data/zhanqian.wu/unified_canvas_images/breakpoint-grounding-55m/data"
SOURCE = "breakpoint_grounding_55m"
ID_PREFIX = "breakpoint"

MAX_TAR_BYTES = 1 << 30          # 单 tar ~1GB
MAX_ENTRIES = 5000               # 单 tar 成员上限
GREEN, BLUE, YELLOW, RED, END = "\033[32m", "\033[34m", "\033[33m", "\033[31m", "\033[0m"


def log(tag, msg, color=GREEN):
    print(f"{color}[{tag}]{END} {msg}", flush=True)


# ---------------------------------------------------------------- 扩展名
def ext_of(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tiff"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".bin"


# ---------------------------------------------------------------- 分片写入器（对齐 lmdb_to_tar.ShardWriter）
class ShardWriter:
    """一个 shard 的 tar + idx + meta + parquet 四件套写入器。off 指向成员数据起始。"""

    def __init__(self, ds_dir: str, stem: str):
        self.stem = stem
        self.tar_path = os.path.join(ds_dir, stem + ".tar.part")
        self.idx_path = os.path.join(ds_dir, stem + ".idx.part")
        self.meta_path = os.path.join(ds_dir, stem + ".meta.jsonl.part")
        self.pq_path = os.path.join(ds_dir, "parquet", stem + ".parquet.part")
        self.tf = tarfile.open(self.tar_path, "w", format=tarfile.GNU_FORMAT)
        self.idx_f = open(self.idx_path, "w")
        self.meta_f = open(self.meta_path, "w")
        self.rows = []
        self.n_entries = 0
        self.n_bytes = 0

    def add_blob(self, name: str, data: bytes) -> dict:
        header_off = self.tf.offset
        data_off = header_off + 512
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        ti.mtime = 0
        self.tf.addfile(ti, io.BytesIO(data))
        expect = header_off + 512 + ((len(data) + 511) // 512) * 512
        if self.tf.offset != expect:  # 长名触发 GNU 扩展块时宁可报错
            raise RuntimeError(f"tar offset mismatch for {name!r}: {self.tf.offset} != {expect}")
        self.idx_f.write(json.dumps({"id": name, "off": data_off, "size": len(data)}) + "\n")
        self.n_entries += 1
        self.n_bytes += len(data)
        return {"id": name, "shard": self.stem + ".tar", "off": data_off, "size": len(data)}

    def add_meta(self, obj):
        self.rows.append(obj)
        self.meta_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self):
        self.tf.close()
        self.idx_f.flush()
        os.fsync(self.idx_f.fileno())
        self.idx_f.close()
        self.meta_f.flush()
        os.fsync(self.meta_f.fileno())
        self.meta_f.close()
        schema = pa.schema([
            ("id", pa.string()), ("source", pa.string()), ("caption", pa.string()),
            ("w", pa.int64()), ("h", pa.int64()),
            ("caption_zh", pa.string()), ("local_caption", pa.string()),
            ("local_caption_zh", pa.string()), ("url", pa.string()),
            ("shard", pa.string()), ("row", pa.int64()),
            ("normalized_boxes", pa.list_(pa.list_(pa.float32()))),
            ("images", pa.list_(pa.struct([
                ("id", pa.string()), ("shard", pa.string()),
                ("off", pa.int64()), ("size", pa.int64())]))),
        ])
        pq.write_table(pa.Table.from_pylist(self.rows, schema=schema), self.pq_path)
        # 原子发布：.part fsync 后 rename，见正式名即完整
        for suffix in (".tar", ".idx", ".meta.jsonl"):
            src = os.path.join(os.path.dirname(self.tar_path), self.stem + suffix + ".part")
            with open(src, "rb") as f:
                os.fsync(f.fileno())
            os.replace(src, src[:-5])
        with open(self.pq_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(self.pq_path, self.pq_path[:-5])


# ---------------------------------------------------------------- 单文件处理
def process_file(args):
    (src_path, file_idx, ds_dir, state_dir, n_images_ref, n_bytes_ref) = args
    fname = os.path.basename(src_path)
    done_marker = os.path.join(state_dir, f"file-{file_idx:05d}.done")
    if os.path.exists(done_marker):
        return file_idx, 0, 0, True  # skipped

    # 断点恢复：上次中途崩溃的半成品分片整文件重做（先清理）
    for f in glob.glob(os.path.join(ds_dir, f"shard-{file_idx:05d}-*.part")) + \
             glob.glob(os.path.join(ds_dir, "parquet", f"shard-{file_idx:05d}-*.parquet.part")):
        os.remove(f)
    for f in glob.glob(os.path.join(ds_dir, f"shard-{file_idx:05d}-*.tar")) + \
             glob.glob(os.path.join(ds_dir, f"shard-{file_idx:05d}-*.idx")) + \
             glob.glob(os.path.join(ds_dir, f"shard-{file_idx:05d}-*.meta.jsonl")) + \
             glob.glob(os.path.join(ds_dir, "parquet", f"shard-{file_idx:05d}-*.parquet")):
        # 完整分片存在但 done 标记缺失 → 同样重做，避免行级重复
        os.remove(f)

    pf = pq.ParquetFile(src_path)
    n_rows_total = pf.metadata.num_rows
    t0 = time.time()
    row_base = 0
    n_img, n_byte = 0, 0
    k = 0
    w = None

    def new_writer():
        return ShardWriter(ds_dir, f"shard-{file_idx:05d}-{k:03d}")

    w = new_writer()
    for gi in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(gi, columns=[
            "image", "image_caption", "object_captions",
            "normalized_boxes", "img_size_wh"])
        imgs = tbl.column("image").to_pylist()
        caps = tbl.column("image_caption").to_pylist()
        objs = tbl.column("object_captions").to_pylist()
        boxes = tbl.column("normalized_boxes").to_pylist()
        whs = tbl.column("img_size_wh").to_pylist()
        for j in range(len(imgs)):
            row = row_base + j
            data = imgs[j].get("bytes")
            if not data:
                continue
            rec_id = f"{ID_PREFIX}_{file_idx:04d}_{row:07d}"
            ext = ext_of(data[:16])
            member = w.add_blob(f"{rec_id}__img_bytes{ext}", data)
            w.add_meta({
                "id": rec_id, "source": SOURCE,
                "caption": caps[j] or "",
                "w": int(whs[j][0]), "h": int(whs[j][1]),
                "caption_zh": "",
                "local_caption": json.dumps(objs[j], ensure_ascii=False) if objs[j] else "[]",
                "local_caption_zh": "",
                "url": imgs[j].get("path") or "",
                "shard": fname, "row": row,
                "normalized_boxes": boxes[j],
                "images": [member],
            })
            n_img += 1
            n_byte += len(data)
            if w.n_entries >= MAX_ENTRIES or w.tf.offset >= MAX_TAR_BYTES:
                w.close()
                k += 1
                w = new_writer()
        row_base += len(imgs)
    w.close()

    with open(done_marker, "w") as f:
        f.write(json.dumps({"file": fname, "rows": n_rows_total,
                            "images": n_img, "bytes": n_byte,
                            "shards": k + 1, "sec": round(time.time() - t0, 1)}) + "\n")
    return file_idx, n_img, n_byte, False


# ---------------------------------------------------------------- 心跳
def heartbeat(stop_flag, done_files_ref, n_images_ref, n_bytes_ref, n_total_files):
    t0 = time.time()
    last_img = 0
    while True:
        time.sleep(60)
        if stop_flag.value:
            return
        el = time.time() - t0
        img = n_images_ref.value
        rate = (img - last_img) / 60.0
        last_img = img
        eta_h = ((n_total_files * (img / max(done_files_ref.value, 1))) - img) / max(rate, 1) / 3600
        log("心跳", f"文件 {done_files_ref.value}/{n_total_files} | 图片 {img:,} | "
                    f"{n_bytes_ref.value / 1e12:.2f} TB | {rate:.0f} img/s | ETA ~{eta_h:.1f}h "
                    f"⏳ {time.strftime('%H:%M:%S')}", BLUE)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个源文件（试跑用）")
    ap.add_argument("--dst", default=os.path.join(TAR_ROOT, SOURCE))
    ap.add_argument("--src", default=SRC_DIR)
    args = ap.parse_args()

    ds_dir = args.dst
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(os.path.join(ds_dir, "parquet"), exist_ok=True)
    state_dir = os.path.join(ds_dir, ".state")
    os.makedirs(state_dir, exist_ok=True)

    src_files = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    if not src_files:
        print(f"{RED}源目录无 parquet: {args.src}{END}")
        sys.exit(1)
    if args.limit:
        src_files = src_files[:args.limit]
    n_done_before = len(glob.glob(os.path.join(state_dir, "file-*.done")))
    total_bytes = sum(os.path.getsize(f) for f in src_files)
    log("启动", f"源文件 {len(src_files)} 个 (~{total_bytes / 1e12:.1f} TB) | 已完成 {n_done_before} | "
                f"workers={args.workers} | 目标 {ds_dir} 🚀")

    mgr = get_context("spawn").Manager()
    n_images_ref = mgr.Value("i", 0)
    n_bytes_ref = mgr.Value("l", 0)
    done_files_ref = mgr.Value("i", n_done_before)
    stop_flag = mgr.Value("b", False)

    import threading
    hb = threading.Thread(target=heartbeat, daemon=True,
                          args=(stop_flag, done_files_ref, n_images_ref, n_bytes_ref,
                                len(src_files)))
    hb.start()

    tasks = [(p, i, ds_dir, state_dir, n_images_ref, n_bytes_ref)
             for i, p in enumerate(src_files)]
    t0 = time.time()
    n_ok, n_skip = 0, 0
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        try:
            for file_idx, n_img, n_byte, skipped in pool.imap_unordered(process_file, tasks, chunksize=1):
                if skipped:
                    n_skip += 1
                else:
                    n_ok += 1
                n_images_ref.value += n_img
                n_bytes_ref.value += n_byte
                done_files_ref.value += 1
        except Exception:
            stop_flag.value = True
            traceback.print_exc()
            pool.terminate()
            log("异常", "任务中断，可安全重启（按源文件断点续传）", RED)
            sys.exit(2)

    stop_flag.value = True
    hb.join(timeout=2)
    el = time.time() - t0
    log("完成", f"本次处理 {n_ok} 个文件 (跳过已完成 {n_skip}) | "
                f"图片 {n_images_ref.value:,} 张 | {n_bytes_ref.value / 1e12:.2f} TB | "
                f"耗时 {el / 3600:.2f}h | 平均 {n_images_ref.value / max(el, 1):.0f} img/s ✅")


if __name__ == "__main__":
    main()
