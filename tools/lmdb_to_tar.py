#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmdb_to_tar.py — unified_edit_images_lmdb → tar 分片 + sidecar 偏移索引（不转码）

输出布局（每个数据集一个目录，互不干扰，可并行、可单独重跑）:
    {dst}/{dataset}/shard-00000.tar          # ~shard-size 的 tar，成员为原始图片字节
    {dst}/{dataset}/shard-00000.idx          # 一行一条: {"id": "...", "off": <数据起始偏移>, "size": N}
    {dst}/{dataset}/shard-00000.meta.jsonl   # 一行一条记录的元数据（prompt 等非字节字段）+ images 引用

读取方式: f.seek(off); f.read(size) —— 一次 seek+read 切出一个 blob。

数据形态自动识别:
    BYTES  : pickle value 含 bytes 字段（img_before/img_after 等）→ 每个字段一个 tar 成员
    PATHS  : pickle value 只有图片路径（ref_paths/output_path 等，指向磁盘上已存在的图片）
             → 按 路径去重 后把文件字节写进 tar（同一文件只存一份）
    SCAN   : mdb 为空但存在 images/ 目录 → 直接按目录树把所有文件收进 tar（无元数据）

断点续传:
    - 已发布 shard: dst 上存在 shard-XXXXX.tar（.part 临时名写完 fsync 后原子 rename）
    - 断点: staging/{dataset}/state.json 记录下一记录位置 (env_idx, key_hex) 与 shard 序号，
      写于每个 shard 构建之前（原子替换）；崩溃后丢弃当前半成品，从断点重建整个 shard
    - 每数据集完成后写 staging/{dataset}/_DONE，重跑直接跳过

进度: 每 60s 打一行 [dataset] shard-XXXXX n/N (xx.x%) speed；n 为全数据集累计（跨重启保留）；结束打印汇总。

用法:
    python3 lmdb_to_tar.py                          # 全量
    python3 lmdb_to_tar.py --datasets CrispEdit     # 指定数据集
    python3 lmdb_to_tar.py --limit-records 100 --dst /tmp/dbg --staging /tmp/dbg_stg   # 调试
    python3 lmdb_to_tar.py --verify 200             # 对已发布数据抽样 seek+read 校验
"""

import argparse
import glob
import io
import json
import os
import pickle
import re
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_edit_images_lmdb"
DST_DEFAULT = "/mnt/nfs/data/zhanqian.wu/unified_edit_images_tar"
STG_DEFAULT = "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_edit_images_tar_staging"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")

MAGIC_EXTS = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG", ".png"),
    (b"GIF8", ".gif"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
)


def ext_of(b: bytes) -> str:
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    for magic, ext in MAGIC_EXTS:
        if b.startswith(magic):
            return ext
    return ".bin"


def say(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def open_env(path: str):
    import lmdb
    return lmdb.open(path, readonly=True, lock=False, subdir=False,
                     map_size=1 << 40, max_readers=4096, readahead=False)


def env_entries(path: str) -> int:
    env = open_env(path)
    n = env.stat()["entries"]
    env.close()
    return n


def key_to_int(k: bytes) -> int:
    return int.from_bytes(k, "big")


def int_to_key(i: int, width: int = 8) -> bytes:
    return i.to_bytes(width, "big")


class ShardWriter:
    """一个 shard 的 tar + idx + meta 三件套写入器。off 指向成员数据起始（seek+read 直接可读）。"""

    def __init__(self, tar_path: str, idx_path: str, meta_path: str, pq_path: str = None):
        self.tar_path, self.idx_path, self.meta_path = tar_path, idx_path, meta_path
        self.pq_path = pq_path
        self.tf = tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT)
        self.idx_f = open(idx_path, "w")
        self.meta_f = open(meta_path, "w") if meta_path else None
        self.rows = []                       # parquet 行缓存（本 shard 内记录）
        self.size = 0
        self.n_entries = 0

    def add_blob(self, name: str, data: bytes, src: str = None) -> dict:
        header_off = self.tf.offset          # tar 头所在偏移
        data_off = header_off + 512          # 数据载荷起始偏移
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        ti.mtime = 0
        self.tf.addfile(ti, io.BytesIO(data))
        # 长文件名会触发 GNU 扩展块，偏移计算失效 —— 宁可报错也不写错索引
        expect = header_off + 512 + ((len(data) + 511) // 512) * 512
        if self.tf.offset != expect:
            raise RuntimeError(f"tar offset mismatch for {name!r}: "
                               f"{self.tf.offset} != {expect}")
        rec = {"id": name, "off": data_off, "size": len(data)}
        if src:                               # PATHS 模式记录源路径，供去重恢复与校验
            rec["src"] = src
        self.idx_f.write(json.dumps(rec) + "\n")
        self.size = self.tf.offset
        self.n_entries += 1
        return {"id": name, "shard": os.path.basename(self.tar_path),
                "off": data_off, "size": len(data)}

    def add_meta(self, obj):
        self.rows.append(obj)
        if self.meta_f:
            self.meta_f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def flush(self):
        self.idx_f.flush()

    def close(self):
        self.tf.close()
        self.idx_f.flush()
        os.fsync(self.idx_f.fileno())
        self.idx_f.close()
        if self.meta_f:
            self.meta_f.flush()
            os.fsync(self.meta_f.fileno())
            self.meta_f.close()
        if self.pq_path and self.rows:
            import pyarrow as pa
            import pyarrow.parquet as pq
            try:
                pq.write_table(pa.Table.from_pylist(self.rows), self.pq_path)
            except Exception:
                # 字段类型冲突/空结构体(如 metadata={})时降级：
                # 除 images 外全部字符串化，保证能落盘
                safe = []
                for r in self.rows:
                    safe.append({k: (v if k == "images" or isinstance(
                        v, (int, float, str, bool)) or v is None else str(v))
                        for k, v in r.items()})
                pq.write_table(pa.Table.from_pylist(safe), self.pq_path)

    def discard(self):
        """异常/崩溃时由重启方调用；本对象只在正常结束时 close。"""
        pass


def publish(ds_dir: str, stem: str):
    """staging -> dst 原子发布: .part 写完 fsync 后 rename。parquet 落 ds_dir/parquet/。"""
    for suffix, sub in ((".tar", ""), (".idx", ""), (".meta.jsonl", ""), (".parquet", "parquet/")):
        src = stem + suffix
        if not os.path.exists(src):
            continue
        dst = os.path.join(ds_dir, sub, os.path.basename(stem) + suffix)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, dst)


# --------------------------------------------------------------------------
# 记录 → (blobs, meta) 的三种模式
# --------------------------------------------------------------------------

def record_from_bytes_mode(d: dict, env_stem: str, key_int: int):
    """返回 ([(name, data, False, dedup_key)], meta_dict)"""
    base = f"{env_stem}_{key_int:08d}"
    blobs, meta = [], {}
    for k, v in d.items():
        if isinstance(v, bytes):
            name = f"{base}__{k}{ext_of(v)}"
            blobs.append((name, v, False, name))
        else:
            meta[k] = v
    meta["id"] = base
    return blobs, meta


_PATH_CACHE = {}


def _resolve_path(p: str, roots) -> str:
    """相对路径按候选根依次解析；带缓存避免重复 stat。"""
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    key = p
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    got = ""
    if not os.path.isabs(p):
        for r in roots:
            c = os.path.join(r, p)
            if os.path.isfile(c):
                got = c
                break
    _PATH_CACHE[key] = got
    return got


def collect_paths(d: dict, roots):
    """收集 dict 中指向已存在图片文件的路径，返回 [(字段名, 序号, 解析后路径)]。

    相对路径按 roots 依次解析；同名字段的 "src_<field>" 绝对路径作为回退
    （镜像拷贝缺失时直接取公共盘原件）。"""
    out = []
    for k, v in d.items():
        cands = v if isinstance(v, list) else ([v] if isinstance(v, str) else [])
        for i, p in enumerate(cands):
            if not (isinstance(p, str) and p.lower().endswith(IMG_EXTS)):
                continue
            r = _resolve_path(p, roots)
            if not r:
                fb = d.get(f"src_{k}")
                fbc = fb[i] if isinstance(fb, list) and i < len(fb) else fb
                if isinstance(fbc, str) and os.path.isfile(fbc):
                    r = fbc
            if r:
                out.append((k, i, r))
    return out


def record_from_paths_mode(d: dict, env_stem: str, key_int: int, roots):
    """PATHS 模式：成员名 {env}_{key:08d}__{field}[{i}].ext（短名，避免 tar 长名扩展）。
    dedup_key 用 realpath，同一源文件跨记录只写一次 tar。"""
    base = f"{env_stem}_{key_int:08d}"
    meta = dict(d)
    meta["id"] = base
    blobs = []
    for k, i, p in collect_paths(d, roots):
        with open(p, "rb") as f:
            head = f.read(12)
        name = f"{base}__{k}[{i}]{ext_of(head)}"
        blobs.append((name, p, True, os.path.realpath(p)))
    return blobs, meta


# --------------------------------------------------------------------------
# env 级并行（--parallel-envs）：BYTES 型数据集按 env 拆任务，worker 间互不干扰
# --------------------------------------------------------------------------

def scan_progress(dst_ds: str):
    """扫描已发布 idx（含旧全局命名 shard-XXXXX），返回:
    done_envs: 该 env 所有 key 均已转出的判断交由 _DONE 标记，这里返回 (env_stem -> max_key+1)
    next_key: dict env_stem -> 下一个待转 key int（旧 shard 中的最大 key+1）
    n_shards: dict env_stem -> 已有 {env_stem}-NNNNN.tar 数量"""
    next_key, n_shards = {}, {}
    for p in glob.glob(os.path.join(dst_ds, "*.idx")):
        base = os.path.basename(p)[:-4]
        m = re.match(r"^(shard-\d+)$", base)
        if not m:
            em = re.match(r"^(.+)-(\d{5})$", base)
            if em:
                n_shards[em.group(1)] = max(n_shards.get(em.group(1), 0),
                                            int(em.group(2)) + 1)
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                id_ = r.get("id", "")
                head = id_.split("__", 1)[0]
                if "[" in head or "_" not in head:
                    continue
                try:
                    env_stem, kstr = head.rsplit("_", 1)
                    k = int(kstr)
                except Exception:
                    continue
                if k + 1 > next_key.get(env_stem, 0):
                    next_key[env_stem] = k + 1
    return next_key, n_shards


def convert_env(ds: str, env_path: str, args, next_key: int, shard_start: int):
    """单 env 的 BYTES 转换，产出 {env_stem}-{seq:05d}.tar/.idx/.meta.jsonl/.parquet。"""
    src_ds = os.path.join(args.src, ds)
    dst_ds = os.path.join(args.dst, ds)
    stg_ds = os.path.join(args.staging, ds)
    env_stem = os.path.splitext(os.path.basename(env_path))[0]
    done_marker = os.path.join(stg_ds, "envstate", env_stem + ".done")
    os.makedirs(os.path.dirname(done_marker), exist_ok=True)
    if os.path.exists(done_marker) and not args.force:
        return
    env = open_env(env_path)
    with env.begin() as tx:
        _k = next(tx.cursor().iternext(), None)
    if _k is None:
        env.close()
        atomic_json(done_marker, {"done": True})
        return
    key_width = len(_k[0])
    total = env.stat()["entries"]
    t0 = time.time()
    seq = shard_start
    key_int = next_key - 1
    n_records = 0
    try:
        while True:
            # 先定位：没有剩余 key 就结束，避免产出空 shard
            with env.begin() as tx:
                c0 = tx.cursor()
                start = key_int + 1
                ok = (c0.set_range(int_to_key(start, key_width)) if start > 0
                      else c0.first())
            if not ok:
                break
            seq += 1
            stem = os.path.join(stg_ds, f"{env_stem}-{seq:05d}")
            w = ShardWriter(stem + ".tar", stem + ".idx", stem + ".meta.jsonl",
                            pq_path=stem + ".parquet")
            shard_bytes = 0
            with env.begin() as tx:
                cur = tx.cursor()
                if key_int + 1 > 0:
                    cur.set_range(int_to_key(key_int + 1, key_width))
                else:
                    cur.first()
                while True:
                    key_b, val = cur.item()
                    key_i = key_to_int(key_b)
                    d = pickle.loads(val)
                    blobs, meta = record_from_bytes_mode(d, env_stem, key_i)
                    if shard_bytes > 0 and w.size + sum(
                            len(b[1]) for b in blobs) > args.shard_size:
                        break
                    rec_images = [w.add_blob(name, src) for name, src, _, _ in blobs]
                    meta["images"] = rec_images
                    w.add_meta(meta)
                    shard_bytes = w.size
                    n_records += 1
                    key_int = key_i
                    if args.limit_records and n_records >= args.limit_records:
                        break
                    if not cur.next():
                        break
            w.close()
            publish(ds_dir=dst_ds, stem=stem)
            for suffix in (".tar", ".idx", ".meta.jsonl", ".parquet"):
                os.remove(stem + suffix)
            if args.limit_records and n_records >= args.limit_records:
                break
        atomic_json(done_marker, {"done": True, "records": n_records,
                                  "shards": seq - shard_start,
                                  "secs": round(time.time() - t0, 1)})
        say(f"[{ds}/{env_stem}] DONE records={n_records}/{total} "
            f"shards={seq-shard_start} {(time.time()-t0)/60:.1f}min")
    finally:
        env.close()


# --------------------------------------------------------------------------
# 单数据集转换（一个 worker 进程一个数据集）
# --------------------------------------------------------------------------

def list_published(dst_ds: str):
    """已发布 shard 的 stem 列表（排序）。"""
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(dst_ds, "shard-*.tar")))


def rebuild_seen(dst_ds: str, mode: str):
    """PATHS 模式跨 shard 去重：从已发布 idx 重建 realpath -> (id, shard, off, size)。"""
    seen = {}
    for stem in list_published(dst_ds):
        idx = os.path.join(dst_ds, stem + ".idx")
        if not os.path.exists(idx):
            continue
        with open(idx) as f:
            for line in f:
                r = json.loads(line)
                if "src" in r:
                    seen[os.path.realpath(r["src"])] = (r["id"], stem, r["off"], r["size"])
    return seen


def convert_dataset(ds: str, args):
    src_ds = os.path.join(args.src, ds)
    dst_ds = os.path.join(args.dst, ds)
    stg_ds = os.path.join(args.staging, ds)
    os.makedirs(dst_ds, exist_ok=True)
    os.makedirs(stg_ds, exist_ok=True)
    t0 = time.time()

    done_marker = os.path.join(stg_ds, "_DONE")
    if os.path.exists(done_marker) and not args.force:
        say(f"[{ds}] 已完成，跳过（--force 可重跑）")
        return

    envs = sorted(glob.glob(os.path.join(src_ds, "mdb", "*.mdb")))
    scan_dir = os.path.join(src_ds, "images")
    if not envs:
        if os.path.isdir(scan_dir):
            say(f"[{ds}] mdb 为空，走 SCAN 模式收 images/ 目录")
            return convert_scan(ds, scan_dir, dst_ds, stg_ds, args)
        say(f"[{ds}] 跳过：无 mdb 且无 images/")
        return

    # 模式探测：取第一条记录
    mode = None
    env0 = open_env(envs[0])
    with env0.begin() as tx:
        first = next(tx.cursor().iternext(), None)
    env0.close()
    if first is None:
        say(f"[{ds}] 跳过：{envs[0]} 为空")
        return
    d0 = pickle.loads(first[1])
    roots = [src_ds, args.src, os.path.join(src_ds, "mdb")]
    if any(isinstance(v, bytes) for v in d0.values()):
        mode = "BYTES"
    elif collect_paths(d0, roots):
        mode = "PATHS"
    else:
        say(f"[{ds}] 跳过：既无 bytes 也无有效图片路径")
        return

    total = sum(env_entries(e) for e in envs)
    say(f"[{ds}] mode={mode} envs={len(envs)} entries={total}")

    # 断点恢复：state.json 或从已发布 shard 反推
    state_path = os.path.join(stg_ds, "state.json")
    state = None
    if os.path.exists(state_path) and not args.force:
        with open(state_path) as f:
            state = json.load(f)
    # 累计已发布记录数：从独立 sidecar 恢复，不依赖进程内 n_records，重启不清零
    progress_path = os.path.join(stg_ds, "progress.json")
    if state is not None and "done_records" not in state and os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                state["done_records"] = int(json.load(f).get("done_records", 0))
        except Exception:
            pass
    if state is None or state.get("mode") != mode:
        pub = list_published(dst_ds)
        seq = len(pub)
        env_idx, key_int = 0, -1
        if pub:
            last_idx = os.path.join(dst_ds, pub[-1] + ".idx")
            if os.path.exists(last_idx):
                with open(last_idx, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 4096))
                    tail = f.read().decode().strip().splitlines()
                if tail:
                    name = json.loads(tail[-1])["id"]          # {env}_{key:08d}__field.ext
                    env_stem = name.split("__")[0].rsplit("_", 1)[0]
                    key_int = int(name.split("__")[0].rsplit("_", 1)[1])
                    env_idx = next(i for i, e in enumerate(envs)
                                   if os.path.splitext(os.path.basename(e))[0] == env_stem)
        state = {"mode": mode, "shard_seq": seq, "env_idx": env_idx, "key_int": key_int,
                 "done_records": 0}
        atomic_json(state_path, state)
        say(f"[{ds}] 断点恢复: shard={seq} env_idx={env_idx} key={key_int}")
    else:
        # 清掉上次崩溃的半成品
        cur = os.path.join(stg_ds, f"shard-{state['shard_seq']:05d}")
        for suffix in (".tar", ".idx", ".meta.jsonl", ".parquet"):
            if os.path.exists(cur + suffix):
                os.remove(cur + suffix)
        # dst 上可能残留 .part
        for p in glob.glob(os.path.join(dst_ds, "*.part")):
            os.remove(p)

    # 兜底：state 为重建时同样从 sidecar 恢复累计记录数
    if "done_records" not in state and os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                state["done_records"] = int(json.load(f).get("done_records", 0))
        except Exception:
            pass

    key_width = 8
    seen = rebuild_seen(dst_ds, mode) if mode == "PATHS" else {}
    # n_records 为全数据集累计（跨重启从 state/progress.json 恢复），百分比才是真实总进度
    n_records = int(state.get("done_records", 0))
    n_nopath = 0
    last_log = time.time()
    shard_seq = state["shard_seq"]
    env_idx, key_int = state["env_idx"], state["key_int"]

    while env_idx < len(envs):
        env = open_env(envs[env_idx])
        env_stem = os.path.splitext(os.path.basename(envs[env_idx]))[0]
        # key 宽度自适应（默认 8 字节大端）
        with env.begin() as tx:
            _k = next(tx.cursor().iternext(), None)
        key_width = len(_k[0]) if _k else 8

        # ---- 写断点：本 shard 从 (env_idx, key_int+1) 开始 ----
        shard_seq += 1
        state = {"mode": mode, "shard_seq": shard_seq - 1,
                 "env_idx": env_idx, "key_int": key_int,
                 "done_records": n_records}
        atomic_json(state_path, state)
        stem = os.path.join(stg_ds, f"shard-{shard_seq - 1:05d}")
        w = ShardWriter(stem + ".tar", stem + ".idx", stem + ".meta.jsonl",
                        pq_path=stem + ".parquet")

        with env.begin() as tx:
            cur = tx.cursor()
            start = key_int + 1
            if start > 0:
                if not cur.set_range(int_to_key(start, key_width)):
                    cur = None
            elif not cur.first():        # 新 cursor 必须先定位，否则 item() 为空
                cur = None
            shard_bytes = 0
            while cur is not None:
                now = time.time()
                if now - last_log >= 60:
                    last_log = now
                    pct = 100.0 * min(n_records, total) / max(total, 1)
                    say(f"[{ds}] shard-{shard_seq-1:05d} 进行中 records={n_records}/{total} "
                        f"({pct:.1f}%) tar={w.size>>20}MiB")
                kv = cur.item()
                key_b, val = kv
                key_i = key_to_int(key_b)
                d = pickle.loads(val)
                if mode == "BYTES":
                    blobs, meta = record_from_bytes_mode(d, env_stem, key_i)
                else:
                    blobs, meta = record_from_paths_mode(d, env_stem, key_i, roots)
                # 预估超限则先收尾（保证记录完整，shard 可略超）
                if shard_bytes > 0 and w.size + sum(len(b[1]) for b in blobs
                                                    if not b[2]) > args.shard_size:
                    break
                rec_images = []
                for name, src, is_path, dk in blobs:
                    if is_path:
                        if dk in seen:
                            old = seen[dk]
                            rec_images.append({"id": name, "src_id": old[0], "shard": old[1],
                                               "off": old[2], "size": old[3],
                                               "dedup": True})
                            continue
                        with open(src, "rb") as fimg:
                            data = fimg.read()
                        ref = w.add_blob(name, data, src=src)
                        seen[dk] = (name, os.path.basename(stem + ".tar"),
                                    ref["off"], ref["size"])
                        rec_images.append(ref)
                    else:
                        rec_images.append(w.add_blob(name, src))
                if rec_images:
                    meta["images"] = rec_images
                    w.add_meta(meta)
                elif mode == "PATHS":
                    n_nopath += 1
                shard_bytes = w.size
                n_records += 1
                key_int = key_i
                if args.limit_records and n_records >= args.limit_records:
                    cur = None
                    env_idx = len(envs)      # 提前收尾
                    break
                if not cur.next():
                    cur = None
        env.close()

        # ---- 发布 ----
        w.close()
        publish(ds_dir=dst_ds, stem=stem)
        for suffix in (".tar", ".idx", ".meta.jsonl", ".parquet"):
            os.remove(stem + suffix)
        # 每个分片发布后把累计记录数落盘，重启后进度百分比不再归零
        atomic_json(progress_path, {"done_records": n_records})
        pct = 100.0 * min(n_records, total) / max(total, 1)
        dt = time.time() - t0
        say(f"[{ds}] shard-{shard_seq-1:05d} 发布  records={n_records}/{total} "
            f"({pct:.1f}%) {dt/60:.1f}min")
        if args.limit_records and n_records >= args.limit_records:
            break
        # 本 env 榨干则前进（下一 shard 从 key_int+1 的 set_range 自然越界处理，
        # 但跨 env 需要显式推进）
        env_probe = open_env(envs[env_idx])
        with env_probe.begin() as tx:
            has_next = tx.cursor().set_range(int_to_key(key_int + 1, key_width))
        env_probe.close()
        if not has_next:
            env_idx += 1
            key_int = -1

    if not args.limit_records:
        atomic_json(done_marker, {"records": n_records, "shards": shard_seq,
                                  "nopath": n_nopath,
                                  "finished_at": time.strftime("%F %T")})
    say(f"[{ds}] DONE shards={shard_seq} records={n_records} nopath={n_nopath} "
        f"用时 {(time.time()-t0)/60:.1f}min")


def convert_scan(ds: str, scan_dir: str, dst_ds: str, stg_ds: str, args):
    """SCAN 模式：mdb 为空，直接收 images/ 目录下所有文件（无元数据）。"""
    done_marker = os.path.join(stg_ds, "_DONE")
    state_path = os.path.join(stg_ds, "state.json")
    files = []
    skipped = 0
    for root, _dirs, fnames in os.walk(scan_dir):
        for fn in sorted(fnames):
            p = os.path.join(root, fn)
            if os.path.isfile(p):          # 过滤悬空符号链接/非常规文件
                files.append(p)
            else:
                skipped += 1
    files.sort()
    total = len(files)
    if skipped:
        say(f"[{ds}] 警告: 跳过 {skipped} 个缺失/悬空文件")
    say(f"[{ds}] SCAN mode files={total}")
    shard_seq, start_i = 0, 0
    if os.path.exists(state_path) and not args.force:
        with open(state_path) as f:
            st = json.load(f)
        if st.get("mode") == "SCAN":
            shard_seq, start_i = st["shard_seq"], st["file_idx"]
    else:
        for p in glob.glob(os.path.join(dst_ds, "*.part")):
            os.remove(p)
    t0 = time.time()
    n = start_i
    while n < total:
        shard_seq += 1
        atomic_json(state_path, {"mode": "SCAN", "shard_seq": shard_seq - 1, "file_idx": n})
        stem = os.path.join(stg_ds, f"shard-{shard_seq-1:05d}")
        w = ShardWriter(stem + ".tar", stem + ".idx", None)
        shard_bytes = 0
        while n < total:
            p = files[n]
            rel = os.path.relpath(p, scan_dir)
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                skipped += 1
                n += 1
                continue
            if shard_bytes > 0 and w.size + len(data) > args.shard_size:
                break
            w.add_blob(rel, data)
            shard_bytes = w.size
            n += 1
            if args.limit_records and n >= args.limit_records:
                break
        w.close()
        publish(ds_dir=dst_ds, stem=stem)
        for suffix in (".tar", ".idx"):
            os.remove(stem + suffix)
        say(f"[{ds}] shard-{shard_seq-1:05d} 发布 files={n}/{total} "
            f"({100.0*n/max(total,1):.1f}%) {(time.time()-t0)/60:.1f}min")
        if args.limit_records and n >= args.limit_records:
            break
    if not args.limit_records:
        atomic_json(done_marker, {"files": total, "shards": shard_seq, "skipped": skipped})
    say(f"[{ds}] DONE files={total} shards={shard_seq} skipped={skipped}")


# --------------------------------------------------------------------------
# 校验：抽样 seek+read 回读，与源比对
# --------------------------------------------------------------------------

def verify(args):
    """抽样校验：seek+read 回读 tar，并与 LMDB/源文件字节级比对。"""
    import random
    import hashlib
    random.seed(0)
    datasets = sorted(os.listdir(args.dst))
    total_bad = total_srcmiss = 0
    for ds in datasets:
        dst_ds = os.path.join(args.dst, ds)
        src_ds = os.path.join(args.src, ds)
        tars = sorted(glob.glob(os.path.join(dst_ds, "shard-*.tar")))
        if not tars:
            continue
        bad = checked = srcmiss = 0
        for tar_path in random.sample(tars, min(args.verify_shards, len(tars))):
            with open(tar_path[:-4] + ".idx") as f:
                lines = f.readlines()
            picks = random.sample(lines, min(4, len(lines)))
            for line in picks:
                r = json.loads(line)
                with open(tar_path, "rb") as f:
                    f.seek(r["off"])
                    data = f.read(r["size"])
                checked += 1
                if len(data) != r["size"]:
                    bad += 1
                    say(f"  SIZE-BAD {ds}/{r['id']}")
                    continue
                # 与源比对
                src_bytes = None
                try:
                    if "__" in r["id"] and "[" not in r["id"]:   # BYTES 模式
                        stem_key, field = r["id"].split("__", 1)
                        env_stem, kstr = stem_key.rsplit("_", 1)
                        env = open_env(os.path.join(src_ds, "mdb", env_stem + ".mdb"))
                        with env.begin() as tx:
                            v = tx.get(int_to_key(int(kstr), len(next(tx.cursor().iternext())[0])))
                        env.close()
                        src_bytes = pickle.loads(v)[field.rsplit(".", 1)[0]]
                    elif "src" in r:                             # PATHS 模式
                        with open(r["src"], "rb") as fs:
                            src_bytes = fs.read()
                    elif "/" in r["id"]:                         # SCAN 模式
                        with open(os.path.join(src_ds, "images", r["id"]), "rb") as fs:
                            src_bytes = fs.read()
                except Exception:
                    srcmiss += 1
                    continue
                if src_bytes is None:
                    # 无源可比时至少校验是合法图片头
                    if not (data[:3] == b"\xff\xd8\xff" or data[:4] == b"\x89PNG"
                            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")):
                        bad += 1
                        say(f"  MAGIC-BAD {ds}/{r['id']} head={data[:8].hex()}")
                elif hashlib.md5(data).hexdigest() != hashlib.md5(src_bytes).hexdigest():
                    bad += 1
                    say(f"  MD5-BAD {ds}/{r['id']}")
        say(f"[verify] {ds}: checked={checked} bad={bad} src_miss={srcmiss}")
        total_bad += bad
        total_srcmiss += srcmiss
    say(f"[verify] TOTAL bad={total_bad} src_miss={total_srcmiss}")
    return total_bad


def probe_mode(src_ds: str, envs, src_root: str = None):
    """取首 env 首条记录判定 BYTES/PATHS；无记录返回 None。"""
    env0 = open_env(envs[0])
    with env0.begin() as tx:
        first = next(tx.cursor().iternext(), None)
    env0.close()
    if first is None:
        return None
    d0 = pickle.loads(first[1])
    roots = [src_ds, src_root or SRC_DEFAULT, os.path.join(src_ds, "mdb")]
    if any(isinstance(v, bytes) for v in d0.values()):
        return "BYTES"
    if collect_paths(d0, roots):
        return "PATHS"
    return "NONE"


def build_tasks(args, all_ds):
    """构建任务列表：BYTES 数据集按 env 拆分（跳过已完成的 env），其余整数据集。"""
    tasks = []          # (kind, ds, payload)
    for ds in all_ds:
        src_ds = os.path.join(args.src, ds)
        dst_ds = os.path.join(args.dst, ds)
        stg_ds = os.path.join(args.staging, ds)
        envs = sorted(glob.glob(os.path.join(src_ds, "mdb", "*.mdb")))
        if not envs:
            if os.path.isdir(os.path.join(src_ds, "images")):
                tasks.append(("scan", ds, None))
            else:
                say(f"[{ds}] 跳过：无 mdb 且无 images/")
            continue
        mode = probe_mode(src_ds, envs, args.src)
        if mode is None:
            say(f"[{ds}] 跳过：mdb 为空")
            continue
        if mode == "BYTES":
            next_key, n_shards = scan_progress(dst_ds)
            os.makedirs(os.path.join(stg_ds, "envstate"), exist_ok=True)
            n_todo = 0
            for e in envs:
                stem = os.path.splitext(os.path.basename(e))[0]
                if os.path.exists(os.path.join(stg_ds, "envstate", stem + ".done")) \
                        and not args.force:
                    continue
                nk = next_key.get(stem, 0)
                tasks.append(("env", ds, (e, nk, n_shards.get(stem, 0))))
                n_todo += 1
            say(f"[{ds}] BYTES envs={len(envs)} 待转={n_todo} "
                f"(断点恢复 {len(envs)-n_todo} 个)")
        elif mode == "PATHS":
            tasks.append(("ds", ds, None))
        else:
            say(f"[{ds}] 跳过：既无 bytes 也无有效图片路径")
    return tasks


def run_task(task, args):
    kind, ds, payload = task
    if kind == "env":
        env_path, nk, seq0 = payload
        convert_env(ds, env_path, args, nk, seq0)
    elif kind == "scan":
        convert_scan(ds, os.path.join(args.src, ds, "images"),
                     os.path.join(args.dst, ds), os.path.join(args.staging, ds), args)
    else:
        convert_dataset(ds, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--staging", default=STG_DEFAULT)
    ap.add_argument("--shard-size", type=int, default=1 << 30, help="目标分片大小，默认 1GiB")
    ap.add_argument("--datasets", default="", help="逗号分隔，默认全部")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--parallel-envs", action="store_true",
                    help="BYTES 型数据集按 env 拆任务并行（默认开启）")
    ap.add_argument("--no-parallel-envs", dest="parallel_envs", action="store_false")
    ap.set_defaults(parallel_envs=True)
    ap.add_argument("--limit-records", type=int, default=0,
                    help="每数据集/每 env 最多处理 N 条（调试）")
    ap.add_argument("--verify", type=int, default=0, help="校验模式：每数据集抽 N 个 shard")
    ap.add_argument("--verify-shards", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="忽略 _DONE/断点/envstate 重来")
    args = ap.parse_args()

    if args.verify:
        sys.exit(1 if verify(args) else 0)

    os.makedirs(args.dst, exist_ok=True)
    os.makedirs(args.staging, exist_ok=True)

    all_ds = sorted(d for d in os.listdir(args.src)
                    if os.path.isdir(os.path.join(args.src, d))
                    and not d.startswith((".", "_")))
    if args.datasets:
        want = [x.strip() for x in args.datasets.split(",") if x.strip()]
        all_ds = [d for d in all_ds if d in want]
    say(f"数据集({len(all_ds)}): {all_ds}")
    say(f"dst={args.dst} staging={args.staging} workers={args.workers} "
        f"shard={args.shard_size>>30}GiB limit={args.limit_records} "
        f"parallel_envs={args.parallel_envs}")

    tasks = build_tasks(args, all_ds) if args.parallel_envs else \
        [("ds", ds, None) for ds in all_ds]
    say(f"任务数: {len(tasks)}")
    t0 = time.time()
    if args.workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            try:
                run_task(t, args)
            except Exception as e:
                say(f"[{t[1]}] 失败: {e!r}")
                import traceback
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_task, t, args): t for t in tasks}
            n_done = 0
            for fu in as_completed(futs):
                t = futs[fu]
                n_done += 1
                try:
                    fu.result()
                except Exception as e:
                    say(f"[{t[1]}] 失败: {e!r}")
                    import traceback
                    traceback.print_exc()
                if n_done % 50 == 0:
                    say(f"进度: {n_done}/{len(tasks)} 个任务完成 "
                        f"{(time.time()-t0)/60:.1f}min")
    say(f"ALL_TASKS_DONE {len(tasks)} 个, 用时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
