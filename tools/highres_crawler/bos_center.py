#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bos_center.py — 通用「中心 BOS」命令行工具

连接百度智能云对象存储 BOS（默认 bj.bcebos.com / giga-video-pretrain-dataset），
提供连通性自检、目录列举、上传（断点续传+分片+进度）、下载、删除、统计等能力。

设计要点（均来自生产实测教训）:
  1. 分片上传用底层 multipart API，绕开 SDK put_super_object_from_file 的
     整文件 CRC 预读坑（NFS 高并发时会全部卡在预读、永不 complete、BOS 0 对象）。
  2. 默认低并发（4 文件 x 2 分片）。高并发(32x8) 会打爆共享 NFS 存储导致卡死。
  3. 文件级断点续传：done 文件 key->size，崩溃重跑只传未完成/大小不一致的。
  4. 凭证不硬编码：环境变量优先，其次 ~/.bos_center_cred (chmod 600)。

依赖: pip install baidubce tqdm
      （必须用官方 PyPI 版 baidubce；某些镜像源的 0.8.3 是 py2 版，不可用）

凭证（按优先级）:
  1) 环境变量 BOS_AK / BOS_SK / BOS_ENDPOINT / BOS_BUCKET
  2) ~/.bos_center_cred  JSON: {"ak":..,"sk":..,"endpoint":..,"bucket":..}
  3) --cred-file 指定的 JSON

用法示例:
  python3 bos_center.py test                          # 连通性+鉴权+写权限全链路自检
  python3 bos_center.py ls                            # 列 bucket
  python3 bos_center.py ls users/zhanqian.wu/         # 列目录
  python3 bos_center.py du users/zhanqian.wu/         # 统计前缀下对象数/总大小
  python3 bos_center.py up ./data users/zhanqian.wu/  # 上传目录/文件(断点续传)
  python3 bos_center.py down <key> ./local.bin        # 下载对象(断点续传)
  python3 bos_center.py rm <key>                      # 删除对象
  python3 bos_center.py stat <key>                    # 查看对象元信息
"""
import os
import sys
import json
import time
import shutil
import socket
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tqdm
except ImportError:
    tqdm = None

CRED_FILE = os.path.join(os.path.expanduser("~"), ".bos_center_cred")
DEFAULT_ENDPOINT = "bj.bcebos.com"
DEFAULT_BUCKET = "giga-video-pretrain-dataset"

# 默认低并发（生产实测：32x8=256 路并发会打爆共享 NFS）
FILE_WORKERS = 4
PART_THREADS = 2
PART_SIZE = 64 * 1024 * 1024      # 64MB 分片
SUPER_THRESHOLD = 256 * 1024 * 1024  # 超过此大小走分片上传
RETRY = 4
DONE_SAVE_INTERVAL = 10
_state_lock = threading.Lock()


# ------------------------------------------------------------------ 凭证 / 客户端
def load_cred(cred_file=None, ak=None, sk=None, endpoint=None, bucket=None):
    ak = ak or os.environ.get("BOS_AK") or os.environ.get("BOS_CENTER_AK")
    sk = sk or os.environ.get("BOS_SK") or os.environ.get("BOS_CENTER_SK")
    endpoint = endpoint or os.environ.get("BOS_ENDPOINT") or os.environ.get("BOS_CENTER_ENDPOINT")
    bucket = bucket or os.environ.get("BOS_BUCKET") or os.environ.get("BOS_CENTER_BUCKET")
    endpoint = endpoint or DEFAULT_ENDPOINT
    bucket = bucket or DEFAULT_BUCKET

    cf = cred_file or CRED_FILE
    if not (ak and sk) and os.path.exists(cf):
        try:
            with open(cf) as fp:
                d = json.load(fp)
            ak = ak or d.get("ak")
            sk = sk or d.get("sk")
            endpoint = d.get("endpoint") or endpoint
            bucket = d.get("bucket") or bucket
        except Exception as e:
            raise RuntimeError(f"读取凭证文件 {cf} 失败: {e}")
    if not (ak and sk):
        raise RuntimeError(
            f"缺少凭证：设置环境变量 BOS_AK/BOS_SK，或创建 {cf} "
            f'(JSON: {{"ak":..,"sk":..,"endpoint":..,"bucket":..}}, chmod 600)')
    return ak, sk, endpoint, bucket


def get_client(args):
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.services.bos.bos_client import BosClient
    ak, sk, endpoint, bucket = load_cred(
        getattr(args, "cred_file", None),
        getattr(args, "ak", None), getattr(args, "sk", None),
        getattr(args, "endpoint", None), getattr(args, "bucket", None))
    cfg = BceClientConfiguration(credentials=BceCredentials(ak, sk), endpoint=endpoint)
    return BosClient(cfg), bucket, endpoint


# ------------------------------------------------------------------ 通用工具函数
def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.2f}{unit}"
        n /= 1024


def _progress(total, desc):
    return (tqdm.tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                     desc=desc, dynamic_ncols=True, mininterval=5) if tqdm else None)


# ------------------------------------------------------------------ test 自检
def cmd_test(args):
    print("=== 中心 BOS 连接自检 ===")
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.services.bos.bos_client import BosClient

    ak, sk, endpoint, bucket = load_cred(
        args.cred_file, args.ak, args.sk, args.endpoint, args.bucket)
    print(f"[1/5] 凭证已加载  endpoint={endpoint}  bucket={bucket}  ak={ak[:6]}***")

    # DNS
    try:
        ip = socket.gethostbyname(endpoint)
        print(f"[2/5] DNS 解析: {endpoint} -> {ip}  OK")
    except Exception as e:
        print(f"[2/5] DNS 解析失败: {e}")
        sys.exit(1)

    # TCP 443
    try:
        s = socket.create_connection((endpoint, 443), timeout=10)
        s.close()
        print("[3/5] TCP 443 连通  OK")
    except Exception as e:
        print(f"[3/5] TCP 443 失败: {e}")
        sys.exit(1)

    cfg = BceClientConfiguration(credentials=BceCredentials(ak, sk), endpoint=endpoint)
    client = BosClient(cfg)

    # 鉴权 + ListBuckets
    try:
        resp = client.list_buckets()
        names = [b.name for b in (resp.buckets or [])]
        print(f"[4/5] 鉴权通过, 可见 {len(names)} 个 bucket: {names[:8]}")
    except Exception as e:
        print(f"[4/5] 鉴权失败: {type(e).__name__}: {str(e)[:200]}")
        sys.exit(1)

    # 写权限（探针）
    key = f"_probe/bos_center_selftest_{int(time.time())}.txt"
    try:
        client.put_object_from_string(bucket, key, b"ok")
        md = client.get_object_meta_data(bucket, key).metadata
        client.delete_object(bucket, key)
        print(f"[5/5] 写/读/删探针 OK  ({bucket}/{key}, length={md.content_length})")
    except Exception as e:
        print(f"[5/5] 写权限测试失败: {type(e).__name__}: {str(e)[:200]}")
        sys.exit(1)
    print("\n自检全部通过 ✅ 可以正常连接中心 BOS")


# ------------------------------------------------------------------ ls / stat / du
def cmd_ls(args):
    client, bucket, _ = get_client(args)
    prefix = args.prefix or ""
    try:
        resp = client.list_objects(bucket, prefix=prefix, delimiter="/",
                                   max_keys=args.max_keys)
        dirs = [p.prefix for p in (resp.common_prefixes or [])]
        objs = [(o.key, o.size, getattr(o, "last_modified", "")) for o in (resp.contents or [])]
    except Exception as e:
        print(f"列举失败: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
    print(f"bucket={bucket}  prefix='{prefix}'  truncated={resp.is_truncated}")
    if dirs:
        print(f"\n--- 目录 ({len(dirs)}) ---")
        for d in dirs:
            print("  D", d)
    if objs:
        print(f"\n--- 文件 ({len(objs)}) ---")
        for k, sz, lm in objs:
            print(f"  F {_human(sz):>10}  {lm}  {k}")
    if not dirs and not objs:
        print("(空)")


def cmd_buckets(args):
    client, _, _ = get_client(args)
    try:
        resp = client.list_buckets()
        for b in (resp.buckets or []):
            print(f"  {b.name:<45} {getattr(b, 'location', ''):<10} {getattr(b, 'creation_date', '')}")
    except Exception as e:
        print(f"列举 bucket 失败: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)


def cmd_stat(args):
    client, bucket, _ = get_client(args)
    try:
        md = client.get_object_meta_data(bucket, args.key).metadata
    except Exception as e:
        print(f"获取元信息失败: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
    print(f"key:            {args.key}")
    print(f"content_length: {md.content_length}  ({_human(int(md.content_length))})")
    for attr in ("content_type", "etag", "last_modified", "storage_class"):
        if hasattr(md, attr):
            print(f"{attr:<16}: {getattr(md, attr)}")


def cmd_du(args):
    client, bucket, _ = get_client(args)
    marker = None
    n = 0
    total = 0
    t0 = time.time()
    print(f"统计 {bucket}/{args.prefix} ...", flush=True)
    while True:
        kw = dict(prefix=args.prefix, max_keys=1000)
        if marker:
            kw["marker"] = marker
        resp = client.list_objects(bucket, **kw)
        for o in (resp.contents or []):
            n += 1
            total += (o.size or 0)
        if not resp.is_truncated:
            break
        marker = resp.next_marker
    print(f"对象数: {n}   总大小: {_human(total)} ({total} bytes)   耗时 {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ 断点状态
def _load_done(path):
    for attempt in range(6):
        for p in (path, path + ".bak"):
            if not os.path.exists(p):
                continue
            try:
                with open(p) as fp:
                    data = json.load(fp)
                if data:
                    return data
            except Exception:
                pass
        time.sleep(0.5)
    return {}


def _save_done(path, done):
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(done, fp)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)
    try:
        shutil.copy2(path, path + ".bak")
    except OSError:
        pass


# ------------------------------------------------------------------ 上传
def _multipart_upload(client, bucket, key, path, size, part_threads,
                      limit_mbps, pbar):
    """底层分片上传，绕开 SDK 整文件 CRC 预读坑。"""
    kw = {}
    if limit_mbps:
        kw["traffic_limit"] = int(limit_mbps * 1e6)
    upload_id = client.initiate_multipart_upload(bucket, key).upload_id
    nparts = (size + PART_SIZE - 1) // PART_SIZE
    parts = [None] * nparts
    counted = {"n": 0}

    def cb(consumed, total):
        if total and consumed >= total:
            counted["n"] += total
            if pbar:
                pbar.update(total)

    def do(pn):
        offset = (pn - 1) * PART_SIZE
        part_size = min(PART_SIZE, size - offset)
        resp = client.upload_part_from_file(
            bucket, key, upload_id, pn, part_size, path, offset,
            progress_callback=cb, **kw)
        return pn, resp.metadata.etag

    try:
        with ThreadPoolExecutor(max_workers=part_threads) as ex:
            for f in as_completed([ex.submit(do, p) for p in range(1, nparts + 1)]):
                pn, etag = f.result()
                parts[pn - 1] = {"partNumber": pn, "eTag": etag}
        if any(p is None for p in parts):
            raise RuntimeError("multipart 有分片未完成")
        client.complete_multipart_upload(bucket, key, upload_id, parts)
    except Exception:
        try:
            client.abort_multipart_upload(bucket, key, upload_id)
        except Exception:
            pass
        if counted["n"] and pbar:
            pbar.update(-counted["n"])
        raise


def _upload_one(client, bucket, key, path, size, limit_mbps, pbar,
                part_threads):
    kw = {}
    if limit_mbps:
        kw["traffic_limit"] = int(limit_mbps * 1e6)
    counted = {"n": 0}

    def cb(consumed, total):
        if total and consumed >= total:
            counted["n"] += total
            if pbar:
                pbar.update(total)

    err = None
    for i in range(RETRY):
        try:
            if size > SUPER_THRESHOLD:
                _multipart_upload(client, bucket, key, path, size,
                                  part_threads, limit_mbps, pbar)
            else:
                client.put_object_from_file(bucket, key, path,
                                            progress_callback=cb, **kw)
            return key, size, True, None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
            if counted["n"] and pbar:
                pbar.update(-counted["n"])
            counted["n"] = 0
            time.sleep(min(2 ** i * 2, 30))
    return key, size, False, err


def _collect_uploads(src, prefix, recursive):
    """收集待上传文件 -> [(local_path, key, size)]"""
    items = []
    if os.path.isfile(src):
        items.append((src, f"{prefix.rstrip('/')}/{os.path.basename(src)}", os.path.getsize(src)))
    elif os.path.isdir(src):
        base = src.rstrip("/")
        root_files = [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
        for fn in root_files:
            p = os.path.join(base, fn)
            items.append((p, f"{prefix.rstrip('/')}/{fn}", os.path.getsize(p)))
        if recursive:
            for dp, _, fns in os.walk(base):
                for fn in fns:
                    p = os.path.join(dp, fn)
                    rel = os.path.relpath(p, base)
                    items.append((p, f"{prefix.rstrip('/')}/{rel}", os.path.getsize(p)))
    else:
        raise RuntimeError(f"源路径不存在: {src}")
    return items


def cmd_upload(args):
    client, bucket, _ = get_client(args)
    try:
        items = _collect_uploads(args.src, args.prefix, not args.no_recursive)
    except Exception as e:
        print(f"收集待上传文件失败: {e}")
        sys.exit(1)
    if not items:
        print("没有找到可上传的文件")
        return

    done_path = os.path.join(os.path.dirname(os.path.abspath(args.done_file)),
                             os.path.basename(args.done_file)) if args.done_file else \
                os.path.join(os.getcwd(), ".bos_center_done.json")
    done = {} if args.fresh else _load_done(done_path)
    todo = [(p, k, s) for p, k, s in items if done.get(k) != s]
    total = sum(s for _, _, s in items)
    done_bytes = sum(s for _, k, s in items if done.get(k) == s)
    print(f"[upload] 共 {len(items)} 文件 {_human(total)} | 已完成 {len(items)-len(todo)} 个 "
          f"({_human(done_bytes)}) | 待传 {len(todo)} 个 "
          f"({_human(total-done_bytes)}) | 并发 {args.file_workers}x{args.part_threads}", flush=True)
    if args.dry_run:
        print("[dry-run] 待传前 20 个:")
        for _, k, s in todo[:20]:
            print(f"   {_human(s):>10}  {k}")
        return

    pbar = _progress(total, "upload")
    if pbar and done_bytes:
        pbar.update(done_bytes)
    save_state = {"last": 0.0}

    def maybe_save(force=False):
        now = time.time()
        if force or now - save_state["last"] >= DONE_SAVE_INTERVAL:
            save_state["last"] = now
            _save_done(done_path, done)

    todo.sort(key=lambda x: -x[2])

    def work(item):
        p, k, s = item
        return _upload_one(client, bucket, k, p, s, args.limit, pbar, args.part_threads)

    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        futs = [pool.submit(work, it) for it in todo]
        for f in as_completed(futs):
            k, s, ok, err = f.result()
            with _state_lock:
                if ok:
                    done[k] = s
                else:
                    print(f"\n[FAIL] {k} {err}", flush=True)
                maybe_save()
    maybe_save(force=True)
    if pbar:
        pbar.close()
    fails = [k for _, k, s in items if done.get(k) != s]
    print(f"[upload] 完成. 断点文件: {done_path} | 失败 {len(fails)} 个")
    for k in fails[:50]:
        print("  FAIL:", k)


# ------------------------------------------------------------------ 下载（断点续传）
def cmd_download(args):
    client, bucket, _ = get_client(args)
    key = args.key
    dst = args.dest
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    md = client.get_object_meta_data(bucket, key).metadata
    total = int(md.content_length)
    existing = os.path.getsize(dst) if os.path.exists(dst) else 0
    if existing == total and not args.fresh:
        print(f"已存在且大小一致，跳过: {dst} ({_human(total)})")
        return
    if existing and existing < total and not args.fresh:
        # 用 Range 断点续传
        range_start = existing
        mode = "ab"
        print(f"断点续传: 从 {_human(existing)} 继续")
    else:
        range_start = 0
        mode = "wb"
    pbar = _progress(total, "download")
    if pbar and existing and existing < total and not args.fresh:
        pbar.update(existing)
    try:
        resp = client.get_object(bucket, key, range_start=range_start) if range_start \
               else client.get_object(bucket, key)
        with open(dst, mode) as fp:
            while True:
                buf = resp.data.read(1024 * 1024)
                if not buf:
                    break
                fp.write(buf)
                if pbar:
                    pbar.update(len(buf))
    except Exception as e:
        print(f"下载失败: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
    if pbar:
        pbar.close()
    got = os.path.getsize(dst)
    print(f"下载完成: {dst}  {_human(got)}  "
          f"{'OK' if got == total else '大小不一致!'}")


# ------------------------------------------------------------------ 删除
def cmd_rm(args):
    client, bucket, _ = get_client(args)
    if args.prefix:
        marker = None
        n = 0
        while True:
            kw = dict(prefix=args.key, max_keys=1000)
            if marker:
                kw["marker"] = marker
            resp = client.list_objects(bucket, **kw)
            for o in (resp.contents or []):
                client.delete_object(bucket, o.key)
                n += 1
                print("  deleted", o.key)
            if not resp.is_truncated:
                break
            marker = resp.next_marker
        print(f"删除 {n} 个对象")
    else:
        try:
            client.delete_object(bucket, args.key)
            print("deleted", args.key)
        except Exception as e:
            print(f"删除失败: {type(e).__name__}: {str(e)[:300]}")
            sys.exit(1)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(
        description="通用中心 BOS 工具（连接测试/列举/上传/下载/删除/统计）")
    ap.add_argument("--cred-file", help="凭证 JSON 路径(默认 ~/.bos_center_cred)")
    ap.add_argument("--ak", help="Access Key(覆盖凭证文件)")
    ap.add_argument("--sk", help="Secret Key(覆盖凭证文件)")
    ap.add_argument("--endpoint", help="endpoint(默认 bj.bcebos.com)")
    ap.add_argument("--bucket", help="bucket(默认 giga-video-pretrain-dataset)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("test", help="连通性+鉴权+写权限自检").set_defaults(func=cmd_test)
    sub.add_parser("buckets", help="列出所有 bucket").set_defaults(func=cmd_buckets)

    p = sub.add_parser("ls", help="列举目录/对象")
    p.add_argument("prefix", nargs="?", default="")
    p.add_argument("--max-keys", type=int, default=1000)
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("stat", help="查看对象元信息")
    p.add_argument("key")
    p.set_defaults(func=cmd_stat)

    p = sub.add_parser("du", help="统计前缀下对象数/总大小")
    p.add_argument("prefix", nargs="?", default="")
    p.set_defaults(func=cmd_du)

    p = sub.add_parser("up", help="上传文件/目录(断点续传+分片+进度)")
    p.add_argument("src", help="本地文件或目录")
    p.add_argument("prefix", help="目标 key 前缀")
    p.add_argument("--file-workers", type=int, default=FILE_WORKERS, help=f"并发文件数(默认 {FILE_WORKERS})")
    p.add_argument("--part-threads", type=int, default=PART_THREADS, help=f"分片线程数(默认 {PART_THREADS})")
    p.add_argument("--limit", type=float, help="限速 Mbps(单请求)")
    p.add_argument("--no-recursive", action="store_true", help="目录只传第一层")
    p.add_argument("--done-file", help="断点文件路径(默认当前目录 .bos_center_done.json)")
    p.add_argument("--fresh", action="store_true", help="忽略断点，从头传")
    p.add_argument("--dry-run", action="store_true", help="只打印计划")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("down", help="下载对象(支持 Range 断点续传)")
    p.add_argument("key")
    p.add_argument("dest")
    p.add_argument("--fresh", action="store_true", help="忽略已有文件，从头下载")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("rm", help="删除对象")
    p.add_argument("key")
    p.add_argument("--prefix", action="store_true", help="按前缀批量删除")
    p.set_defaults(func=cmd_rm)

    args = ap.parse_args()
    # 全局参数要传给子解析器（argparse 子解析器的全局参数需在子命令前给出）
    args.func(args)


if __name__ == "__main__":
    main()
