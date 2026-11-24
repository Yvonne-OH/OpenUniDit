#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bos_center.py — 中心 BOS (bj.bcebos.com / giga-video-pretrain-dataset) 命令行工具。

命令:
  test                      DNS + TCP 443 + 鉴权 + 写/读/删探针自检
  buckets                   列出所有 bucket
  ls [prefix]               列目录 (D=前缀目录, F=文件)
  stat <key>                查看对象元信息
  du [prefix]               统计对象数与总大小
  up <src> <prefix>          上传文件/目录 (断点续传 + 大文件分片 + 低并发)
  down <key> <dest>          下载对象 (Range 断点续传)
  rm <key> [--prefix]       删除对象 / 按前缀批量删除

凭证优先级: --ak/--sk > 环境变量 BOS_AK/BOS_SK > ~/.bos_center_cred (JSON)

关键设计 (见 SKILL.md 踩坑记录):
  - 分片上传走底层 initiate/upload_part_from_file/complete, 不用 SDK 的
    put_super_object_from_file (它会串行预读整个文件算 CRC, NFS 上多文件并发
    全部卡死在预读, BOS 实际 0 对象)。
  - 默认低并发 4 文件 × 2 分片 (共享 NFS 安全值, 30-90 MB/s)。
  - 文件级断点续传: done-file 记 key->size, 重跑只传未完成/大小不一致的文件。
  - 每个文件上传后 head 校验远端大小, 以远端为准, 不只信进度条。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient

DEFAULT_ENDPOINT = "bj.bcebos.com"
DEFAULT_BUCKET = "giga-video-pretrain-dataset"
MB = 1024 * 1024
PART_SIZE = 64 * MB
MULTIPART_THRESHOLD = 256 * MB
RETRY = 5


# ---------------------------------------------------------------- 凭证
def load_conf(args) -> tuple:
    ak = getattr(args, "ak", None) or os.environ.get("BOS_AK")
    sk = getattr(args, "sk", None) or os.environ.get("BOS_SK")
    endpoint = getattr(args, "endpoint", None) or os.environ.get("BOS_ENDPOINT", DEFAULT_ENDPOINT)
    bucket = getattr(args, "bucket", None) or os.environ.get("BOS_BUCKET", DEFAULT_BUCKET)
    cred_file = getattr(args, "cred_file", None) or os.path.expanduser("~/.bos_center_cred")
    if (not ak or not sk) and os.path.isfile(cred_file):
        with open(cred_file) as f:
            c = json.load(f)
        ak = ak or c.get("ak")
        sk = sk or c.get("sk")
        endpoint = args.endpoint or c.get("endpoint", endpoint)
        bucket = args.bucket or c.get("bucket", bucket)
    if not ak or not sk:
        sys.exit("❌ 缺少凭证: 用 --ak/--sk, 环境变量 BOS_AK/BOS_SK, 或 ~/.bos_center_cred")
    return ak, sk, endpoint, bucket


def make_client(ak, sk, endpoint) -> BosClient:
    # 0.9.79: 类名 BceClientConfiguration (旧版叫 BceClientConfig)
    return BosClient(BceClientConfiguration(credentials=BceCredentials(ak, sk),
                                            endpoint=endpoint))


def remote_size(client, bucket, key) -> int | None:
    """head 对象拿远端大小; 两种途径, 均失败返回 None (=远端不存在/不可读)."""
    try:
        resp = client.get_object_meta_data(bucket, key)
        return int(resp.metadata.content_length)
    except Exception:
        pass
    try:
        resp = client.list_objects(bucket, prefix=key, max_keys=1)
        for o in resp.contents:
            if o.key == key:
                return int(o.size)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- 限速
class RateLimiter:
    """简单令牌桶, 所有上传线程共享."""

    def __init__(self, mbps: int | None):
        self.rate = mbps * MB / 8 if mbps else None
        self.lock = threading.Lock()
        self.tokens = self.rate or 0
        self.last = time.time()

    def wait(self, nbytes: int):
        if not self.rate:
            return
        with self.lock:
            now = time.time()
            self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
            self.last = now
            take = min(nbytes, self.tokens)
            self.tokens -= take
            deficit = nbytes - take
        if deficit > 0:
            time.sleep(deficit / self.rate)


# ---------------------------------------------------------------- 上传
def upload_small(client, bucket, key, path, size, limiter, pbar):
    for attempt in range(RETRY):
        try:
            limiter.wait(size)
            client.put_object_from_file(bucket, key, path)
            pbar.update(size)
            return
        except Exception as e:
            if attempt == RETRY - 1:
                raise
            print(f"  ↻ 重试 {key}: {e}", flush=True)
            time.sleep(2 ** attempt)


def upload_multipart(client, bucket, key, path, size, part_threads, limiter, pbar):
    upload_id = client.initiate_multipart_upload(bucket, key).upload_id
    try:
        n_parts = (size + PART_SIZE - 1) // PART_SIZE

        def up_one(i: int):
            off = i * PART_SIZE
            psz = min(PART_SIZE, size - off)
            for attempt in range(RETRY):
                try:
                    limiter.wait(psz)
                    r = client.upload_part_from_file(
                        bucket, key, upload_id, i + 1, psz, path, offset=off)
                    pbar.update(psz)
                    return i + 1, r.metadata.etag
                except Exception as e:
                    if attempt == RETRY - 1:
                        raise
                    print(f"  ↻ 重试分片 {key}#{i+1}: {e}", flush=True)
                    time.sleep(2 ** attempt)

        etags = {}
        with ThreadPoolExecutor(part_threads) as ex:
            for n, etag in ex.map(up_one, range(n_parts)):
                etags[n] = etag
        part_list = [{"partNumber": n, "eTag": etags[n]} for n in sorted(etags)]
        client.complete_multipart_upload(bucket, key, upload_id, part_list)
    except Exception:
        try:
            client.abort_multipart_upload(bucket, key, upload_id)
        except Exception:
            pass
        raise


def save_done(done_file: str, done: dict):
    tmp = done_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(done, f)
    os.replace(tmp, done_file)
    try:  # 备份
        import shutil
        shutil.copyfile(done_file, done_file + ".bak")
    except Exception:
        pass


def collect_files(src: str, recursive: bool):
    src = os.path.abspath(src)
    if os.path.isfile(src):
        return [(src, "")]
    out = []
    for root, dirs, names in os.walk(src):
        if not recursive:
            dirs[:] = []
        for n in sorted(names):
            full = os.path.join(root, n)
            rel = os.path.relpath(full, src).replace(os.sep, "/")
            out.append((full, rel))
    return out


def cmd_up(client, bucket, args):
    from tqdm import tqdm

    files = collect_files(args.src, not args.no_recursive)
    if not files:
        sys.exit(f"❌ 源不存在或为空: {args.src}")
    prefix = args.prefix.strip("/")
    tasks = []
    for full, rel in files:
        key = f"{prefix}/{rel}" if (prefix and rel) else (rel or prefix)
        tasks.append((full, key, os.path.getsize(full)))

    done_file = args.done_file or ".bos_center_done.json"
    done = {}
    if os.path.isfile(done_file) and not args.fresh:
        try:
            with open(done_file) as f:
                done = json.load(f)
        except Exception:
            done = {}

    pending, skipped = [], []
    for full, key, size in tasks:
        if not args.fresh and done.get(key) == size and remote_size(client, bucket, key) == size:
            skipped.append(key)
        else:
            pending.append((full, key, size))

    print(f"📋 上传计划: 共 {len(tasks)} 文件, "
          f"跳过(已传且大小一致) {len(skipped)}, 待传 {len(pending)} "
          f"({sum(s for _, _, s in pending) / 2**30:.2f} GiB)", flush=True)
    if args.dry_run:
        for _, key, size in pending[:50]:
            print(f"  {key}  {size/2**20:.1f}MiB")
        if len(pending) > 50:
            print(f"  ... 其余 {len(pending)-50} 个")
        return
    if not pending:
        print("✅ 无待传文件", flush=True)
        return

    limiter = RateLimiter(args.limit)
    lock = threading.Lock()
    ok = fail = 0

    def worker(item):
        nonlocal ok, fail
        full, key, size = item
        try:
            pbar = tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024,
                        desc=key.split("/")[-1][:28], leave=False)
            try:
                if size >= args.multipart_threshold:
                    upload_multipart(client, bucket, key, full, size,
                                     args.part_threads, limiter, pbar)
                else:
                    upload_small(client, bucket, key, full, size, limiter, pbar)
            finally:
                pbar.close()
            # 以远端为准复核
            actual = remote_size(client, bucket, key)
            if actual != size:
                raise RuntimeError(f"远端大小不一致: local={size} remote={actual}")
            with lock:
                done[key] = size
                save_done(done_file, done)
                ok += 1
                print(f"  ✅ [{ok}/{len(pending)}] {key} ({size/2**20:.1f}MiB)", flush=True)
        except Exception as e:
            with lock:
                fail += 1
            print(f"  ❌ {key}: {e}", flush=True)

    with ThreadPoolExecutor(args.file_workers) as ex:
        list(ex.map(worker, pending))

    print(f"\n{'✅' if fail == 0 else '⚠️'} 完成: 成功 {ok} / 失败 {fail} / 跳过 {len(skipped)}"
          f"{'' if fail == 0 else ' — 重跑同一条命令可断点续传'}", flush=True)
    if fail == 0:
        print(f"💡 复核: python3 {sys.argv[0]} du {prefix}/", flush=True)
    sys.exit(1 if fail else 0)


# ---------------------------------------------------------------- 其他命令
def cmd_test(client, bucket, args):
    import socket as s
    print(f"1) DNS {args.endpoint} ... ", end="", flush=True)
    ip = s.gethostbyname(args.endpoint)
    print(f"✅ {ip}")
    print("2) TCP 443 ... ", end="", flush=True)
    with s.create_connection((args.endpoint, 443), timeout=5):
        print("✅")
    print("3) AK/SK 鉴权 (ListBuckets) ... ", end="", flush=True)
    client.list_buckets()
    print("✅")
    print("4) 写/读/删探针 ... ", end="", flush=True)
    probe = f"users/zhanqian.wu/.bos_probe_{int(time.time())}"
    client.put_object_from_string(bucket, probe, "ok")
    data = client.get_object_as_string(bucket, probe)
    assert data == b"ok" or data == "ok", f"读回内容异常: {data!r}"
    client.delete_object(bucket, probe)
    print("✅")
    print("\n🎉 自检全部通过")


def cmd_ls(client, bucket, args):
    prefix = args.prefix or ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    resp = client.list_objects(bucket, prefix=prefix, delimiter="/", max_keys=1000)
    if getattr(resp, "common_prefixes", None):
        for p in resp.common_prefixes:
            print(f"D  {p}")
    if resp.contents:
        for o in resp.contents:
            ts = getattr(o, "last_modified", "")
            print(f"F  {o.size:>14,}  {ts}  {o.key}")
    if resp.is_truncated:
        print("⚠️ 超过 1000 条, 仅显示第一页")


def cmd_du(client, bucket, args):
    prefix = args.prefix or ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    marker, total, count = None, 0, 0
    while True:
        kw = dict(prefix=prefix, max_keys=1000)
        if marker:
            kw["marker"] = marker
        resp = client.list_objects(bucket, **kw)
        for o in resp.contents:
            total += int(o.size)
            count += 1
        if not resp.is_truncated:
            break
        marker = resp.next_marker
        if count % 10000 == 0:
            print(f"  ... 已遍历 {count:,} 对象", flush=True)
    print(f"📊 {prefix or '(bucket 顶层)'}: {count:,} 个对象, 共 {total/2**30:.2f} GiB ({total:,} B)")


def cmd_stat(client, bucket, args):
    size = remote_size(client, bucket, args.key)
    if size is None:
        print(f"❌ 对象不存在: {args.key}")
        sys.exit(1)
    try:
        meta = client.get_object_meta_data(bucket, args.key).metadata
        print(f"key: {args.key}\nsize: {size:,} B ({size/2**20:.2f} MiB)\n"
              f"etag: {getattr(meta, 'etag', '?')}\n"
              f"last_modified: {getattr(meta, 'last_modified', '?')}")
    except Exception:
        print(f"key: {args.key}\nsize: {size:,} B ({size/2**20:.2f} MiB)")


def cmd_down(client, bucket, args):
    size = remote_size(client, bucket, args.key)
    if size is None:
        sys.exit(f"❌ 对象不存在: {args.key}")
    dest = args.dest
    start = 0
    if os.path.exists(dest) and not args.fresh:
        start = os.path.getsize(dest)
        if start > size:
            start = 0
    if start:
        print(f"↩ 续传: 本地已有 {start:,} B / 远端 {size:,} B")
    from tqdm import tqdm
    pbar = tqdm(total=size, initial=start, unit="B", unit_scale=True, unit_divisor=1024,
                desc=os.path.basename(args.key))
    mode = "ab" if start else "wb"
    with open(dest, mode) as f:
        if start:
            resp = client.get_object(bucket, args.key, range=[start, size - 1])
        else:
            resp = client.get_object(bucket, args.key)
        while True:
            chunk = resp.data.read(8 * MB)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
    pbar.close()
    if os.path.getsize(dest) != size:
        sys.exit(f"❌ 下载不完整: {os.path.getsize(dest):,} != {size:,}")
    print(f"✅ 下载完成: {dest} ({size:,} B)")


def cmd_rm(client, bucket, args):
    if args.prefix:
        marker, n = None, 0
        while True:
            kw = dict(prefix=args.key, max_keys=1000)
            if marker:
                kw["marker"] = marker
            resp = client.list_objects(bucket, **kw)
            for o in resp.contents:
                client.delete_object(bucket, o.key)
                n += 1
                print(f"  🗑 {o.key}")
            if not resp.is_truncated:
                break
            marker = resp.next_marker
        print(f"✅ 已删除 {n} 个对象 (prefix={args.key})")
    else:
        client.delete_object(bucket, args.key)
        print(f"✅ 已删除 {args.key}")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ak", help="Access Key (或环境变量 BOS_AK)")
    ap.add_argument("--sk", help="Secret Key (或环境变量 BOS_SK)")
    ap.add_argument("--endpoint", default=None, help=f"默认 {DEFAULT_ENDPOINT}")
    ap.add_argument("--bucket", default=None, help=f"默认 {DEFAULT_BUCKET}")
    ap.add_argument("--cred-file", default=None, help="凭证文件, 默认 ~/.bos_center_cred")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("test", help="连通性+鉴权+写权限自检")
    sub.add_parser("buckets", help="列出所有 bucket")
    p = sub.add_parser("ls", help="列目录/文件")
    p.add_argument("prefix", nargs="?", default="")
    p = sub.add_parser("stat", help="对象元信息")
    p.add_argument("key")
    p = sub.add_parser("du", help="统计目录大小")
    p.add_argument("prefix", nargs="?", default="")

    p = sub.add_parser("up", help="上传文件/目录")
    p.add_argument("src")
    p.add_argument("prefix")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--multipart-threshold", type=int, default=MULTIPART_THRESHOLD)
    p.add_argument("--file-workers", type=int, default=4, help="并发文件数 (默认 4)")
    p.add_argument("--part-threads", type=int, default=2, help="每文件分片线程 (默认 2)")
    p.add_argument("--limit", type=int, default=None, help="限速 Mbps")
    p.add_argument("--done-file", default=None, help="断点续传记录文件")
    p.add_argument("--fresh", action="store_true", help="忽略断点从头传")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("down", help="下载对象")
    p.add_argument("key")
    p.add_argument("dest")
    p.add_argument("--fresh", action="store_true")

    p = sub.add_parser("rm", help="删除对象")
    p.add_argument("key")
    p.add_argument("--prefix", action="store_true", help="按前缀批量删除 (谨慎!)")

    args = ap.parse_args()
    ak, sk, endpoint, bucket = load_conf(args)
    if args.cmd == "test":
        args.endpoint = endpoint  # cmd_test 用它做 DNS/TCP 探测
    client = make_client(ak, sk, endpoint)

    if args.cmd == "test":
        cmd_test(client, bucket, args)
    elif args.cmd == "buckets":
        for b in client.list_buckets().buckets:
            print(b.name)
    elif args.cmd == "ls":
        cmd_ls(client, bucket, args)
    elif args.cmd == "du":
        cmd_du(client, bucket, args)
    elif args.cmd == "stat":
        cmd_stat(client, bucket, args)
    elif args.cmd == "up":
        cmd_up(client, bucket, args)
    elif args.cmd == "down":
        cmd_down(client, bucket, args)
    elif args.cmd == "rm":
        cmd_rm(client, bucket, args)


if __name__ == "__main__":
    main()
