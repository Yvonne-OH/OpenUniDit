"""Minimal SQLite-backed image dataset for Stage 1/2 schema smoke tests."""
from pathlib import Path
from bisect import bisect_right
import io
import os
import json
import pickle
import random
import sqlite3
import struct
import threading
import time
import logging
import resource
from typing import Any

import numpy as np
from PIL import Image
# 4K source 存在 100-144MP 大图，默认阈值(89MP)会刷 DecompressionBomb 警告；本地可信数据直接关闭
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_store
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

_LOG = logging.getLogger(__name__)


def _debug_index_event(event: str, **fields):
    fields.update(pid=os.getpid(), rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  rank=(dist.get_rank() if dist.is_available() and dist.is_initialized() else -1),
                  world=(dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1))
    _LOG.warning("[index-debug] %s %s", event, " ".join(f"{k}={v}" for k, v in fields.items()))

# ─── lmdb:// URI 支持: 图片字节存于 unified_*_images_lmdb 的 mdb 中 ───
# URI 格式: lmdb://{数据集绝对目录}/{分片stem}/{sample_id}[:before|:after]
_LMDB_ENV_CACHE: dict[str, Any] = {}
_LMDB_KEY_SCHEME: dict[str, str] = {}
_LMDB_FIELD_CACHE: dict[tuple[str, str], str] = {}
_LMDB_LOCK = threading.Lock()
_LMDB_FIELD_BEFORE = ("img_before", "img_before_bytes", "input_img", "img_input", "before")
_LMDB_FIELD_AFTER = ("img_after", "img_after_bytes", "output_img", "img_output", "after", "img_bytes")
_LMDB_FIELD_SINGLE = ("img_bytes", "image_bytes", "img", "jpg", "data")


def _lmdb_env(dataset_dir: str, stem: str):
    dataset_path = Path(dataset_dir)
    path = Path(dataset_dir) / "mdb" / f"{stem}.mdb"
    if not path.exists():
        path = dataset_path / f"{stem}.mdb"
    path = str(path)
    env = _LMDB_ENV_CACHE.get(path)
    if env is None:
        import lmdb  # 延迟导入: 仅使用 lmdb:// 数据集时需要
        try:
            import bos_data  # BOS 镜像: 在线读 (BOS_LMDB_ONLINE=1) 或物化后打开
            env = bos_data.open_lmdb(path, lmdb)
        except ImportError:
            env = lmdb.open(path, readonly=True, lock=False, subdir=False,
                            max_readers=1024)
        with _LMDB_LOCK:
            _LMDB_ENV_CACHE[path] = env
    return env


def _lmdb_record(dataset_dir: str, stem: str, sample_id: int) -> dict[str, Any]:
    env = _lmdb_env(dataset_dir, stem)
    scheme = _LMDB_KEY_SCHEME.get(dataset_dir)
    keys = ([struct.pack("<Q", int(sample_id)), str(sample_id).encode()]
            if scheme != "ascii" else [str(sample_id).encode()])
    with env.begin() as txn:
        for key in keys:
            raw = txn.get(key)
            if raw is not None:
                if scheme is None:
                    _LMDB_KEY_SCHEME[dataset_dir] = ("pack" if len(key) == 8
                                                     else "ascii")
                return pickle.loads(raw)
    raise KeyError(f"lmdb 记录不存在: {dataset_dir}/{stem} sample_id={sample_id}")


def _lmdb_image_bytes(uri: str) -> bytes:
    rest = uri[len("lmdb://"):]
    field_kind = None
    tail = rest.rsplit("/", 1)[-1]
    if ":" in tail:
        rest, field_kind = rest.rsplit(":", 1)
    dataset_dir, stem, sid = rest.rsplit("/", 2)
    rec = _lmdb_record(dataset_dir, stem, int(sid))
    if field_kind is None:
        kind = "single"
    else:
        kind = field_kind
    cached = _LMDB_FIELD_CACHE.get((dataset_dir, kind))
    if cached:
        return rec[cached]
    candidates = {"before": _LMDB_FIELD_BEFORE, "after": _LMDB_FIELD_AFTER,
                  "single": _LMDB_FIELD_SINGLE}[kind]
    field = next((c for c in candidates if c in rec), None)
    if field is None:
        raise KeyError(f"lmdb 记录缺 {kind} 图字段: {uri} (fields={sorted(rec)})")
    _LMDB_FIELD_CACHE[(dataset_dir, kind)] = field
    return rec[field]


# 普通图像库：image_path + caption；编辑库：input_image_path + output_image_path + instruction。
_IMAGE_COLUMNS = ("sample_id", "image_path", "path", "file_path", "uri", "caption", "source", "width", "height", "bucket")
_EDIT_COLUMNS = ("sample_id", "source", "source_id", "split", "instruction", "input_image_path", "output_image_path", "input_width", "input_height", "output_width", "output_height", "metadata_json")

# Wan VAE 空间压缩 16，经压缩后还须被 GWP patch 2 整除 ⇒ 像素宽高必须被 32 整除
BUCKET_PIXEL_MULTIPLE = 32


def parse_bucket(bucket: str) -> tuple[int, int]:
    if bucket.endswith("_square"):
        size = int(bucket.split("_", 1)[0])
        return size, size
    width, height = bucket.split("x", 1)
    return int(width), int(height)


def validate_bucket(bucket: str) -> tuple[int, int]:
    """硬校验 bucket 像素宽高，非法配置在数据集构建时直接报错。"""
    width, height = parse_bucket(str(bucket))
    if width <= 0 or height <= 0 or width % BUCKET_PIXEL_MULTIPLE or height % BUCKET_PIXEL_MULTIPLE:
        raise ValueError(
            f"非法 bucket {bucket!r}: 宽高必须为正且可被 {BUCKET_PIXEL_MULTIPLE} 整除 "
            f"(Wan VAE 空间压缩 16 × patch 2)，实际为 {width}x{height}"
        )
    return width, height


class ImageDatabaseDataset(Dataset):
    def __init__(self, database_path: str, buckets: list[str], image_root: str | None = None, transform=None, preprocess_style: str = "cosmos_style", crop_threshold: float = .15, upscale_threshold: float = 1.5, max_upscale_ratio: float = 1.2, max_samples: int | None = None, max_retries: int = 8, seed: int = 42, sources: list[str] | None = None, edit_database_path: str | None = None, edit_sources: list[str] | None = None, edit_split: str = "train", edit_ratio: float | str = 0.0, max_edit_samples: int | None = None, image_training_enabled: bool = True, edit_training_enabled: bool = False, lmdb_root: str | None = None, max_decode_candidates: int = 3):
        self.database_path = str(database_path)
        self.sources = {str(source) for source in (sources or [])}
        self.edit_database_path = str(edit_database_path) if edit_database_path else None
        self.edit_sources = {str(source) for source in (edit_sources or [])}
        self.edit_split = edit_split
        self.edit_ratio = edit_ratio
        if self.edit_ratio != "auto" and not 0 <= float(self.edit_ratio) <= 1:
            raise ValueError("edit_ratio 必须是 0 到 1 的数字或 auto")
        self.max_retries = max(1, int(max_retries))
        self.max_decode_candidates = max(1, int(max_decode_candidates))
        self.max_upscale_ratio = float(max_upscale_ratio)
        if self.max_upscale_ratio <= 0:
            raise ValueError("max_upscale_ratio 必须为正数")
        self.seed = int(seed)
        self.image_root = Path(image_root) if image_root else None
        self.buckets = list(buckets)
        if not self.buckets:
            raise ValueError("图片 buckets 不能为空")
        for bucket in self.buckets:
            validate_bucket(bucket)
        self.transform = transform
        self.preprocess_style = str(preprocess_style)
        records = self._load_records(self.database_path, self.sources, columns=_IMAGE_COLUMNS) if image_training_enabled else []
        edit_enabled = edit_training_enabled
        for record in records:
            record["is_edit"] = False
        edit_records = []
        if edit_enabled:
            if not self.edit_database_path:
                raise ValueError("edit_training_enabled=true 时必须配置 edit_database_path")
            edit_records = self._load_records(self.edit_database_path, self.edit_sources, self.edit_split, columns=_EDIT_COLUMNS, is_edit=True)
        if self.edit_ratio == "auto":
            records = records + edit_records
        else:
            ratio = float(self.edit_ratio)
            if not records and edit_records:
                records = edit_records
            elif records and edit_records:
                edit_count = round(len(records) * ratio / (1.0 - ratio)) if ratio < 1.0 else len(edit_records)
                records = records + edit_records[:min(edit_count, len(edit_records))]
            elif not records and not edit_records:
                raise RuntimeError("普通图像和编辑训练均已关闭")
        if max_samples is not None and not edit_records:
            records = records[:int(max_samples)]
        self.bucket_indices = {bucket: [] for bucket in self.buckets}
        self.mode_bucket_indices = {
            f"{mode}:{bucket}": []
            for mode in ("image", "edit")
            for bucket in self.buckets
        }
        self.records = []
        t0 = time.time()
        print(f"⏳ 正在构建 bucket 索引 ({len(records):,} 条, max_upscale_ratio={self.max_upscale_ratio:g})...", flush=True)
        filtered = 0
        for record in records:
            bucket = self._nearest_bucket(record.get("width"), record.get("height"))
            if bucket is None:
                filtered += 1
                continue
            record = dict(record)
            record["bucket"] = bucket
            index = len(self.records)
            self.bucket_indices[bucket].append(index)
            mode = "edit" if record.get("is_edit") else "image"
            self.mode_bucket_indices[f"{mode}:{bucket}"].append(index)
            self.records.append(record)
        if not self.records:
            raise RuntimeError(
                f"图片数据库没有合格 bucket 样本: {database_path} "
                f"(候选={len(records):,}, 过滤={filtered:,}, max_upscale_ratio={self.max_upscale_ratio:g})"
            )
        dist = ", ".join(f"{k}={len(v):,}" for k, v in self.bucket_indices.items())
        print(f"✅ bucket 索引完成 ({time.time()-t0:.2f}s): 保留={len(self.records):,}, 无合格桶过滤={filtered:,}, {dist}", flush=True)

    def _load_records(self, database_path: str, sources: set[str], split: str | None = None, columns: tuple[str, ...] = _IMAGE_COLUMNS, is_edit: bool = False) -> list[dict[str, Any]]:
        path = Path(database_path)
        if not path.exists():
            raise FileNotFoundError(f"图片数据库不存在: {path}")

        cache_path = path.with_suffix(".records.pkl")

        def load_cache():
            if not cache_path.exists() or cache_path.stat().st_mtime < path.stat().st_mtime:
                return None
            try:
                t0 = time.time()
                with open(cache_path, "rb") as f:
                    cached_records = pickle.load(f)
                print(f"📦 [rank进程PID={os.getpid()}] 从缓存加载 {len(cached_records):,} 条记录 ({time.time()-t0:.2f}s): {cache_path}", flush=True)
                return cached_records
            except Exception as exc:
                print(f"⚠️ 缓存加载失败，将重建: {cache_path}: {exc}", flush=True)
                return None

        records = load_cache()
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        if records is None and rank == 0:
            records = self._read_records(database_path, columns, is_edit)
            t1 = time.time()
            print(f"⏳ rank 0 正在写入缓存: {cache_path} ...", flush=True)
            tmp_cache = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
            try:
                with open(tmp_cache, "wb") as f:
                    pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
                size_gib = tmp_cache.stat().st_size / 1024**3
                tmp_cache.replace(cache_path)
                print(f"✅ rank 0 缓存写入完成: {cache_path} ({size_gib:.2f} GiB, {time.time()-t1:.2f}s)", flush=True)
            finally:
                if tmp_cache.exists():
                    tmp_cache.unlink()

        if distributed:
            sync_key = f"yisu/image-records/{cache_path}"
            if rank == 0:
                _get_default_store().set(sync_key, b"ready")
            elif records is None:
                print(f"⏳ rank {rank} 等待 rank 0 构建缓存: {cache_path}", flush=True)
                _get_default_store().wait([sync_key])
                records = load_cache()
                if records is None:
                    raise RuntimeError(f"rank 0 完成后缓存仍不可用: {cache_path}")

        if records is None:
            records = self._read_records(database_path, columns, is_edit)
        return self._filter_records(records, sources, split, is_edit)

    def _filter_records(self, records: list[dict[str, Any]], sources: set[str], split: str | None, is_edit: bool) -> list[dict[str, Any]]:
        filtered = [record for record in records if not sources or str(record.get("source") or "") in sources]
        if split is not None:
            filtered = [record for record in filtered if str(record.get("split") or "") == split]
        for record in filtered:
            record["is_edit"] = is_edit
        if not filtered:
            raise RuntimeError(f"数据库筛选后没有可用样本: sources={sorted(sources)}, split={split!r}")
        return filtered

    def _filter_sources(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.sources:
            return records
        filtered = [record for record in records if str(record.get("source") or "") in self.sources]
        print(
            f"✅ source 过滤: 允许={sorted(self.sources)}, 过滤={len(records) - len(filtered):,}, "
            f"保留={len(filtered):,}",
            flush=True,
        )
        if not filtered:
            raise RuntimeError(f"sources 白名单过滤后没有可用图片样本: {sorted(self.sources)}")
        return filtered

    def _read_records(self, database_path: str, columns: tuple[str, ...], is_edit: bool) -> list[dict[str, Any]]:
        path = Path(database_path)
        t0 = time.time()
        print(f"⏳ 正在读取数据库: {path} ...", flush=True)
        # 使用 uri 只读模式打开，提升并发读取性能
        uri = f"file:{path.resolve()}?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(uri, uri=True)
        except Exception:
            connection = sqlite3.connect(path)

        with connection:
            tables = [row[0] for row in connection.execute("select name from sqlite_master where type='table'")]
            if not tables:
                raise RuntimeError(f"图片数据库没有表: {path}")
            image_columns = {"image_path", "path", "file_path", "uri", "input_image_path", "output_image_path"}
            candidates = []
            for table in tables:
                table_columns = [row[1] for row in connection.execute(f'pragma table_info("{table}")')]
                if image_columns & set(table_columns):
                    candidates.append((table, table_columns))
            if not candidates:
                raise RuntimeError(f"图片数据库没有包含图片路径列的表: {path} (tables={tables})")

            table, all_cols = candidates[0]
            if len(candidates) > 1:
                for c in candidates:
                    if c[0] == "samples" or c[0] == "images":
                        table, all_cols = c
                        break

            # 仅 select 训练需要的列（caption 是文本条件，必须包含）
            selected_cols = [c for c in all_cols if c in columns]
            if is_edit and "input_image_path" not in selected_cols:
                raise RuntimeError(f"编辑数据库缺少 input_image_path: {path}")
            if not selected_cols:
                selected_cols = all_cols

            col_str = ", ".join(f'"{c}"' for c in selected_cols)
            cursor = connection.execute(f'select {col_str} from "{table}"')
            # 流式 fetchmany + tqdm 进度条，避免一次性 fetchall 卡住无反馈
            rows = []
            pbar = tqdm(desc=f"读取表 {table}", unit="条", ncols=100)
            while True:
                batch = cursor.fetchmany(8192)
                if not batch:
                    break
                rows.extend(batch)
                pbar.update(len(batch))
            pbar.close()

        result = []
        filtered_sources = 0
        for row in tqdm(rows, desc="整理样本", unit="条", ncols=100):
            item = dict(zip(selected_cols, row))
            if is_edit and self.edit_sources and str(item.get("source") or "") not in self.edit_sources:
                filtered_sources += 1
                continue
            if is_edit and self.edit_split is not None and str(item.get("split") or "") != self.edit_split:
                continue
            if not is_edit and self.sources and str(item.get("source") or "") not in self.sources:
                filtered_sources += 1
                continue
            image = (item.get("output_image_path") if is_edit else (item.get("image_path") or item.get("path") or item.get("file_path") or item.get("uri")))
            if image:
                item["image_path"] = image
                if is_edit:
                    item["width"] = item.get("output_width")
                    item["height"] = item.get("output_height")
                    item["caption"] = item.get("instruction", "")
                result.append(item)

        if self.sources:
            print(f"✅ source 过滤: 允许={sorted(self.sources)}, 过滤={filtered_sources:,}, 保留={len(result):,}", flush=True)

        print(f"✅ 数据库读取完成: 共 {len(result):,} 条记录 ({time.time()-t0:.2f}s)", flush=True)
        if not result:
            raise RuntimeError(f"图片数据库没有有效样本: {path}")
        return result

    def _nearest_bucket(self, width, height):
        if not width or not height or not self.buckets:
            return None
        source_width, source_height = float(width), float(height)
        candidates = []
        for name in self.buckets:
            target_width, target_height = parse_bucket(name)
            scale = max(target_width / source_width, target_height / source_height)
            if scale <= self.max_upscale_ratio:
                candidates.append((name, target_width, target_height))
        if not candidates:
            return None
        ratio = source_width / source_height
        return min(
            candidates,
            key=lambda item: (
                abs(ratio - item[1] / item[2]),
                abs(source_width * source_height - item[1] * item[2]),
            ),
        )[0]

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _check_deadline(deadline, stage, context):
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f'image sample budget exhausted stage={stage} remaining={remaining:.3f}s {context}')
        if os.environ.get('YISU_MEDIA_DEBUG', '0') == '1':
            print(f'[image] stage={stage} remaining={remaining} {context}', flush=True)

    def _open_image(self, path, field_kind="single", deadline=None, context=""):
        from data.datasets.video_decode import decode_isolated
        if str(path).startswith("lmdb://"):
            source = str(path)
            if ":" not in source.rsplit("/", 1)[-1]:
                source = f"{source}:{field_kind}"
        else:
            source = Path(path)
            if self.image_root and not source.is_absolute():
                source = self.image_root / source
            source = str(source)
        context = f'{context} uri={source}'
        self._check_deadline(deadline, 'decode_start', context)
        timeout = float(os.environ.get('YISU_IMAGE_TIMEOUT_SECONDS', '30'))
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
        pixels = decode_isolated(dict(kind='image', path=source), timeout, context)
        self._check_deadline(deadline, 'output_load', context)
        image = Image.fromarray(pixels)
        self._check_deadline(deadline, 'fromarray', context)
        return image

    def _load_item(self, index, deadline=None, context="", bucket_override=None):
        record = dict(self.records[index])
        context = f'{context} uri={record["image_path"]}'
        from data.datasets.video_decode import decode_isolated
        # bucket_override: 高分桶借图时按目标低分桶分辨率加载 (纯降采样)
        bucket = bucket_override or record.get("bucket") or self._nearest_bucket(record.get("width"), record.get("height"))
        width, height = parse_bucket(bucket)

        def source_path(path, kind):
            path = str(path)
            if path.startswith('lmdb://'):
                return path if ':' in path.rsplit('/', 1)[-1] else f'{path}:{kind}'
            return str(self.image_root / path) if self.image_root and not Path(path).is_absolute() else path

        request = dict(kind='images', path=source_path(record['image_path'],
                       'after' if record.get('is_edit') else 'single'), width=width, height=height,
                       style=getattr(self, 'preprocess_style', 'cover'))
        if record.get('is_edit'):
            request['condition_path'] = source_path(record['input_image_path'], 'before')
        self._check_deadline(deadline, 'decode_start', context)
        timeout = float(os.environ.get('YISU_IMAGE_TIMEOUT_SECONDS', '30'))
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
        pixels = decode_isolated(request, timeout, context)
        self._check_deadline(deadline, 'output_load', context)
        record['pixel_values'] = pixels[0].astype(np.float32) / 127.5 - 1.0
        if record.get('is_edit'):
            record['condition_pixel_values'] = pixels[1].astype(np.float32) / 127.5 - 1.0
        record["bucket"] = bucket
        if isinstance(record.get("metadata"), str):
            try:
                record["metadata"] = json.loads(record["metadata"])
            except json.JSONDecodeError:
                pass
        self._check_deadline(deadline, 'postprocess_done', context)
        return record

    def __getitem__(self, index):
        import math
        import socket
        from data.datasets.video_decode import (
            DecodeBudget, MediaCandidatesExhausted, candidate_indices, media_error_category,
        )

        # BucketBatchSampler 高分桶借图时传入 (index, target_bucket): 按低分桶分辨率降采样加载
        bucket_override = None
        if isinstance(index, tuple):
            index, bucket_override = index
        index = int(index)
        timeout = float(os.environ.get('YISU_IMAGE_TIMEOUT_SECONDS', '30'))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('YISU_IMAGE_TIMEOUT_SECONDS must be finite and positive')
        started = time.monotonic()
        worker = torch.utils.data.get_worker_info()
        context = (f'host={socket.gethostname()} rank={os.environ.get("RANK", "unknown")} '
                   f'worker={worker.id if worker else "main"} worker_pid={os.getpid()} '
                   f'index={index} request_timeout={timeout:g}s')
        record = self.records[index]
        bucket = bucket_override or record.get("bucket") or self._nearest_bucket(record.get("width"), record.get("height"))
        mode = "edit" if record.get("is_edit") else "image"
        # numpy 桶池 (mmap int32 数组) 不能用 `or` 真值判断, 多元素数组会抛 ValueError
        pool = self.mode_bucket_indices.get(f"{mode}:{bucket}")
        if pool is None or len(pool) == 0:
            pool = [index]
        candidates = candidate_indices(index, pool, self.max_retries, random.Random(self.seed + index))
        last_error = None
        attempted = 0
        debug = os.environ.get('YISU_MEDIA_DEBUG', '0') == '1'
        # 候选数与视频/AV 对齐 (max_decode_candidates=3)：20s 预算均分 16 份时每候选仅 1.25s，
        # 高并发下 PIL 解码尚未完成即被杀并立刻换候选重试，形成 CPU 内卷放大。
        candidate_limit = min(self.max_decode_candidates, self.max_retries, len(pool))
        budget = DecodeBudget(float(os.environ.get('YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS', str(timeout))),
                              candidate_limit, timeout)
        for candidate in candidates:
            if time.monotonic() >= budget.deadline or budget.candidates <= 0:
                break
            attempted += 1
            detail = (f'{context} attempt={attempted}/{candidate_limit} candidate={candidate} '
                      f'timeout={timeout:.3f}s')
            if debug:
                print(f'[image] stage=sample_start {detail}', flush=True)
            try:
                candidate_timeout = budget.timeout()
                if candidate_timeout < 2.0:
                    # 剩余预算连 spawn+lmdb open 都不够 (实测 0.5s 尝试 100% 超时),
                    # 别再空转子进程, 直接放弃换下一个样本。
                    break
                result = self._load_item(candidate, min(budget.deadline, time.monotonic() + candidate_timeout), detail, bucket_override=bucket_override)
                return result
            except Exception as exc:
                # capacity (解码子进程 MemoryError) 可能由单个超大图确定性触发, 属样本级错误:
                # 换候选重试而非立即终止整个训练作业; executor_bug (真正代码/环境 bug) 仍快速失败。
                if media_error_category(exc) == 'executor_bug':
                    raise
                last_error = exc
                if debug or isinstance(exc, TimeoutError):
                    print(f'[image] stage=attempt_failed {detail} elapsed={time.monotonic() - started:.3f}s error={exc}', flush=True)
        raise MediaCandidatesExhausted(
            f"样本读取失败，共尝试 {attempted} 个候选: {context} "
            f"elapsed={time.monotonic() - started:.3f}s error={last_error}"
        ) from last_error


# ─── 列式索引存储 (mmap 只读共享) ───
# 旧格式是整份 pickle 的 list[dict] + list[int]: 每个 rank 进程反序列化一份 25-35GB 的
# Python 对象 (8 rank 约 200-300GB), 且 DataLoader fork worker 因 refcount 写页持续
# 触发 COW 复制。新格式把索引拆成列式 .npy + 原始字节缓冲, 加载侧全部 mmap_mode='r'
# 只读映射: 同节点 8 rank + 32 worker 物理共享同一份文件页 (clean page 可被内核回收)。
#
# 目录布局 (每个 fingerprint 一个目录, 由 _cache_paths 给出; meta.json 最后写入作为完成标记):
#   sid.npy int64[N] / w.npy,h.npy int32[N] (0=缺失, 取样时还原 None)
#   src.npy uint16[N] / stem.npy uint32[N]   → meta.json 里的字符串表下标
#   is_edit.npy bool[N]
#   cap_off.npy int64[N+1] + captions.bin (utf8 原始缓冲) → caption 切片解码
#   meta_off.npy int64[N+1] + metadata.bin + meta_kind.npy (0=无/1=原始str/2=json还原)
# 行序: image 行 [0, n_image) 在前, edit 行 [n_image, n_image+n_edit) 在后。
# 旧记录里的 text/edit_prompt/instruction/w_after/h_after/split 仅在构建期
# (_read_parquet 的 caption 回退/split 过滤/宽高覆写) 被消费, 缓存后全仓库无读取点,
# 不再存储 (见 _read_parquet 白名单注释)。
_INDEX_FORMAT_VERSION = 2


def _madvise_random(arr: np.ndarray) -> np.ndarray:
    """随机访问的 mmap 只读数组关闭内核预读 (MADV_RANDOM)。

    默认预读在 NFS 上是每次缺页 128KB 的 ~32x 放大流量; 索引列全部按随机下标
    访问, 关闭预读后只取按需页, 多节点并发时显著降低 NFS 压力。
    仅对 np.memmap 生效 (堆内存数组的地址段可能混布其他对象, 不得改其 VMA 行为);
    地址按页对齐 (npy 头部只保证 64B 对齐), 失败静默跳过。"""
    try:
        if not isinstance(arr, np.memmap) or arr.nbytes == 0:
            return arr
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        page = 4096
        addr = arr.__array_interface__["data"][0]
        base, end = addr & ~(page - 1), addr + arr.nbytes
        if libc.madvise(ctypes.c_void_p(base), ctypes.c_size_t(end - base), 1) != 0:
            pass  # EINVAL/ENOMEM 等仅损失预读优化, 不影响语义
    except Exception:
        pass
    return arr


class _ColumnarIndex:
    """mmap 只读列式索引。加载后所有数组保持文件映射, 不得整份读取或 .copy()。"""

    def __init__(self, n_image, n_edit, arrays, sources, stems, source_dirs):
        self.n_image = int(n_image)
        self.n_edit = int(n_edit)
        (self.sid, self.w, self.h, self.src, self.stem, self.is_edit,
         self.cap_off, self.captions, self.meta_off, self.metadata, self.meta_kind) = arrays
        self.sources = list(sources)
        self.stems = list(stems)
        self.source_dirs = list(source_dirs)

    def __len__(self):
        return self.n_image + self.n_edit

    @staticmethod
    def _split_image_uri(image_path) -> tuple[str, str]:
        """lmdb://{dataset_dir}/{stem}/{sid}:{kind} → (dataset_dir, stem)。
        载入侧按列重组 image_path, 必须与构建时的 URI 逐字节一致。"""
        if not isinstance(image_path, str) or not image_path.startswith("lmdb://"):
            raise RuntimeError(f"列式索引要求 lmdb:// 形式的 image_path, 收到: {image_path!r}")
        dataset_dir, stem, _ = image_path[len("lmdb://"):].rsplit("/", 2)
        return dataset_dir, stem

    @classmethod
    def build_from_rows(cls, records: list, edits: list) -> "_ColumnarIndex":
        """rank0 构建期: list[dict] 行 → 内存列式数组。行序 image 在前 edit 在后。"""
        n = len(records) + len(edits)
        sid = np.empty(n, np.int64)
        w = np.empty(n, np.int32)
        h = np.empty(n, np.int32)
        src = np.empty(n, np.uint16)
        stem_col = np.empty(n, np.uint32)
        is_edit = np.empty(n, bool)
        cap_off = np.empty(n + 1, np.int64)
        meta_off = np.empty(n + 1, np.int64)
        meta_kind = np.empty(n, np.uint8)
        cap_buf = bytearray()
        meta_buf = bytearray()
        source_ids: dict[str, int] = {}
        sources: list[str] = []
        source_dirs: list[str] = []
        stem_ids: dict[str, int] = {}
        stems: list[str] = []
        cap_off[0] = meta_off[0] = 0
        pos = 0
        for edit_row, rows in ((False, records), (True, edits)):
            for item in rows:
                sid[pos] = int(item["sample_id"])
                width, height = item.get("width"), item.get("height")
                w[pos] = int(width) if width else 0   # 0 = None/缺失, row_dict 时还原
                h[pos] = int(height) if height else 0
                is_edit[pos] = edit_row
                source = str(item["source"])
                if source not in source_ids:
                    source_ids[source] = len(sources)
                    sources.append(source)
                    source_dirs.append(cls._split_image_uri(item["image_path"])[0])
                src[pos] = source_ids[source]
                stem = cls._split_image_uri(item["image_path"])[1]
                if stem not in stem_ids:
                    stem_ids[stem] = len(stems)
                    stems.append(stem)
                stem_col[pos] = stem_ids[stem]
                caption = item.get("caption")
                cap_buf += ("" if caption is None else
                            caption if isinstance(caption, str) else str(caption)).encode("utf-8")
                cap_off[pos + 1] = len(cap_buf)
                meta_value = item.get("metadata")
                if meta_value is None:
                    meta_kind[pos] = 0
                elif isinstance(meta_value, str):
                    meta_kind[pos] = 1
                    meta_buf += meta_value.encode("utf-8")
                else:
                    # parquet struct/list 列等非 str 元数据: json 编码落盘, row_dict 时还原原类型
                    meta_kind[pos] = 2
                    try:
                        meta_bytes = json.dumps(meta_value, ensure_ascii=False).encode("utf-8")
                    except (TypeError, ValueError):
                        meta_bytes, meta_kind[pos] = str(meta_value).encode("utf-8"), 1
                    meta_buf += meta_bytes
                meta_off[pos + 1] = len(meta_buf)
                pos += 1
        empty = np.empty(0, np.uint8)
        arrays = (
            sid, w, h, src, stem_col, is_edit, cap_off,
            np.frombuffer(cap_buf, dtype=np.uint8) if len(cap_buf) else empty,
            meta_off,
            np.frombuffer(meta_buf, dtype=np.uint8) if len(meta_buf) else empty,
            meta_kind,
        )
        return cls(len(records), len(edits), arrays, sources, stems, source_dirs)

    def save(self, index_dir: Path, fingerprint: str, created: float, stats) -> None:
        """逐文件原子写 (tmp + rename); meta.json 最后写, 作为目录构建完成的标记。"""
        columns = (
            ("sid.npy", self.sid), ("w.npy", self.w), ("h.npy", self.h),
            ("src.npy", self.src), ("stem.npy", self.stem), ("is_edit.npy", self.is_edit),
            ("cap_off.npy", self.cap_off), ("meta_off.npy", self.meta_off),
            ("meta_kind.npy", self.meta_kind),
        )
        for name, payload in columns:
            self._atomic_write_npy(index_dir / name, payload)
        # .bin 是无 npy 头的原始 utf8 字节流, 加载侧 np.memmap 整文件映射, 必须裸写
        for name, payload in (("captions.bin", self.captions), ("metadata.bin", self.metadata)):
            if payload.size == 0:
                payload = b"\x00"   # 0 字节文件无法 mmap (全空 caption/metadata 场景), 补 1 字节占位
            self._atomic_write_bytes(index_dir / name, payload)
        meta = {
            "format_version": _INDEX_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "created": created,
            "n_image": self.n_image,
            "n_edit": self.n_edit,
            "sources": self.sources,
            "stems": self.stems,
            "source_dirs": self.source_dirs,
            "stats": stats,
        }
        self._atomic_write_bytes(index_dir / "meta.json",
                                 json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    @staticmethod
    def _atomic_write_npy(path: Path, payload: np.ndarray) -> None:
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "wb") as handle:
                np.lib.format.write_array(handle, payload, allow_pickle=False)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _atomic_write_bytes(path: Path, payload) -> None:
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "wb") as handle:
                handle.write(memoryview(payload) if isinstance(payload, np.ndarray) else payload)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @classmethod
    def load(cls, index_dir: Path, meta: dict) -> "_ColumnarIndex":
        """mmap 只读加载: 全部数组保持文件映射, 不做任何整份读取/复制。"""
        def mmap_npy(name, dtype):
            arr = np.load(index_dir / name, mmap_mode="r")
            if arr.dtype != dtype:
                raise ValueError(f"{name} dtype={arr.dtype} 期望 {dtype}")
            return arr

        n = int(meta["n_image"]) + int(meta["n_edit"])
        sid = mmap_npy("sid.npy", np.int64)
        w = mmap_npy("w.npy", np.int32)
        h = mmap_npy("h.npy", np.int32)
        src = mmap_npy("src.npy", np.uint16)
        stem_col = mmap_npy("stem.npy", np.uint32)
        is_edit = mmap_npy("is_edit.npy", np.bool_)
        cap_off = mmap_npy("cap_off.npy", np.int64)
        meta_off = mmap_npy("meta_off.npy", np.int64)
        meta_kind = mmap_npy("meta_kind.npy", np.uint8)
        captions = np.memmap(index_dir / "captions.bin", dtype=np.uint8, mode="r")
        metadata = np.memmap(index_dir / "metadata.bin", dtype=np.uint8, mode="r")
        if (sid.shape[0] != n or w.shape[0] != n or h.shape[0] != n or src.shape[0] != n
                or stem_col.shape[0] != n or is_edit.shape[0] != n or meta_kind.shape[0] != n
                or cap_off.shape[0] != n + 1 or meta_off.shape[0] != n + 1
                or int(cap_off[-1]) > captions.shape[0] or int(meta_off[-1]) > metadata.shape[0]):
            raise ValueError("列式索引各列行数/长度不一致")
        return cls(meta["n_image"], meta["n_edit"],
                   tuple(_madvise_random(a) for a in
                         (sid, w, h, src, stem_col, is_edit, cap_off, captions, meta_off, metadata, meta_kind)),
                   meta["sources"], meta["stems"], meta["source_dirs"])

    def row_dict(self, i: int, bucket: str | None = None) -> dict[str, Any]:
        """按行现拼 item dict (瞬态分配, 用完即弃, 不落任何常驻 Python 对象)。"""
        sample_id = int(self.sid[i])
        source_id = int(self.src[i])
        dataset_dir = self.source_dirs[source_id]
        stem = self.stems[int(self.stem[i])]
        edit = bool(self.is_edit[i])
        width, height = int(self.w[i]), int(self.h[i])
        cap_start, cap_end = int(self.cap_off[i]), int(self.cap_off[i + 1])
        kind = int(self.meta_kind[i])
        if kind == 0:
            metadata = None
        else:
            meta_bytes = bytes(self.metadata[int(self.meta_off[i]):int(self.meta_off[i + 1])])
            metadata = meta_bytes.decode("utf-8") if kind == 1 else json.loads(meta_bytes)
        item = {
            "sample_id": sample_id,
            "source": self.sources[source_id],
            "is_edit": edit,
            "image_path": f"lmdb://{dataset_dir}/{stem}/{sample_id}:{'after' if edit else 'single'}",
            "caption": bytes(self.captions[cap_start:cap_end]).decode("utf-8"),
            "width": width or None,
            "height": height or None,
            "metadata": metadata,
        }
        if edit:
            item["input_image_path"] = f"lmdb://{dataset_dir}/{stem}/{sample_id}:before"
        if bucket is not None:
            item["bucket"] = bucket
        return item


class _KeptRecordView:
    """self.records 的轻量列式代理: 只持有 mmap 数组引用。
    __getitem__ 按行现拼 dict (瞬态), 绝不把整份索引物化成 Python 对象,
    使 fork 出的 DataLoader worker 与各 rank 共享同一份 mmap 文件页。"""

    def __init__(self, store: _ColumnarIndex, kept_rows, row_bucket, buckets):
        self._store = store
        self._kept_rows = kept_rows
        self._row_bucket = row_bucket
        self._buckets = buckets

    def __len__(self):
        return len(self._kept_rows)

    def __getitem__(self, index):
        row = int(self._kept_rows[index])
        bucket = self._buckets[int(self._row_bucket[index])]
        return self._store.row_dict(row, bucket=bucket)

    def __iter__(self):
        for index in range(len(self._kept_rows)):
            yield self[index]

    # ── 配比表向量化统计用的列访问 ──
    def bucket_ids(self):
        return self._row_bucket

    def sources_table(self):
        return self._store.sources

    def source_ids(self):
        # 花式索引产生瞬态副本 (仅配比表在 rank0 调用一次, 用完即释放)
        return self._store.src[self._kept_rows]

    def is_edit_flags(self):
        return self._store.is_edit[self._kept_rows]


class LmdbParquetDataset(ImageDatabaseDataset):
    INDEX_CACHE_VERSION = 2  # 索引缓存格式版本，记录结构变化时递增使旧缓存失效

    def __init__(self, database_path: str, buckets: list[str], lmdb_root: str | None = None,
                 image_root: str | None = None, transform=None, max_upscale_ratio: float = 1.2,
                 max_samples: int | None = None, max_retries: int = 8, seed: int = 42,
                 sources: list[str] | None = None, edit_database_path: str | None = None,
                 edit_sources: list[str] | None = None, edit_split: str = "train",
                 edit_ratio: float | str = 0.0, max_edit_samples: int | None = None,
                 image_training_enabled: bool = True, edit_training_enabled: bool = False,
                 index_cache_dir: str | None = None, refresh_index_cache: bool = False,
                 bucket_sampling_weights: dict | None = None, max_decode_candidates: int = 3,
                 preprocess_style: str = "cover", **kwargs):
        self.database_path = str(database_path)
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform
        self.lmdb_root = str(lmdb_root or (Path(database_path).parent if not any(c in str(database_path) for c in "*?[") else Path.cwd()))
        self.edit_lmdb_root = str(kwargs.get("edit_lmdb_root") or lmdb_root or self.lmdb_root)
        self.sources = {str(value) for value in (sources or [])}
        self.edit_sources = {str(value) for value in (edit_sources or [])}
        self.edit_split, self.edit_ratio = edit_split, edit_ratio
        self.max_retries, self.seed = max(1, int(max_retries)), int(seed)
        self.max_decode_candidates = max(1, int(max_decode_candidates))
        self.max_upscale_ratio = float(max_upscale_ratio)
        # 预处理风格: cover=短边贴桶+中心裁剪 (旧行为); resize_long_edge=长边贴桶+短边补黑 (保主体完整)
        self.preprocess_style = str(preprocess_style)
        if self.preprocess_style not in ("cover", "resize_long_edge"):
            raise ValueError(f"不支持的 preprocess_style: {self.preprocess_style!r} (可选: cover, resize_long_edge)")
        self.buckets = list(buckets)
        for bucket in self.buckets:
            validate_bucket(bucket)
        if not self.buckets:
            raise ValueError("图片 buckets 不能为空")
        if edit_training_enabled and not edit_database_path:
            raise ValueError("edit_training_enabled=true 时必须配置 edit_database_path")
        # 索引缓存目录：首次读取后在 output 持久化，yaml dataset 配置或底层分片
        # 发生变化（指纹不一致）/缓存不存在时才重新扫描 parquet。可用环境变量
        # GWP_DATASET_INDEX_CACHE 覆盖，GWP_DATASET_INDEX_CACHE_DISABLE=1 强制重建，
        # 或在 yaml dataset 段设置 refresh_index_cache: true 强制重建一次。
        self.index_cache_dir = Path(
            index_cache_dir
            or os.environ.get("GWP_DATASET_INDEX_CACHE")
            or "/home/jovyan/pfs/users/zhanqian.wu/output/dataset_index_cache"
        )
        self.index_cache_force_refresh = bool(refresh_index_cache)
        self.read_stats: dict[str, Any] = {"image": {}, "edit": {}}
        # 采样权重仅用于分布表展示（实际采样在 BucketBatchSampler）
        self._bucket_weight_view = {
            str(key): float(value) for key, value in (bucket_sampling_weights or {}).items()
        }
        index_store = self._load_or_read_records(
            bool(image_training_enabled), bool(edit_training_enabled),
            max_samples, max_edit_samples, edit_database_path,
        )
        # 索引加载后仍有数分钟的多轮 O(N) 处理 (bucket 索引/分片定位/校验), 必须提示
        _debug_index_event("bucket-phase-enter", records=index_store.n_image,
                           edits=index_store.n_edit, buckets=len(self.buckets))
        print(f"⏳ 图片记录就绪 (image={index_store.n_image:,}, edit={index_store.n_edit:,}), "
              "正在构建 bucket 索引与分片定位 (约数分钟)...", flush=True)
        self._build_bucket_index(index_store, max_samples, max_edit_samples, edit_ratio)
        self._print_distribution_table()
        # fork 共享收尾: 冻结 GC 追踪对象, DataLoader worker 子进程的 GC 扫描不再
        # 触碰对象头引用计数页, 消除索引常驻结构的 COW 复制 (mmap 页本身无此问题)。
        import gc
        gc.collect()
        gc.freeze()

    # ─── 列式索引的合并与 bucket 派生 (运行期, 结果落盘为 mmap 只读文件) ───

    def _runtime_cache_dir(self, max_samples, max_edit_samples, edit_ratio) -> Path:
        """bucket 派生运行期数组的缓存子目录。指纹不含 buckets 与合并参数, 用它们
        一起再散列, 换桶配置/换配比自动换目录互不覆盖。"""
        import hashlib
        spec = json.dumps({"buckets": self.buckets, "max_upscale_ratio": self.max_upscale_ratio,
                           "max_samples": max_samples, "max_edit_samples": max_edit_samples,
                           "edit_ratio": edit_ratio}, sort_keys=True)
        digest = hashlib.sha1(spec.encode()).hexdigest()[:16]
        return self.index_cache_dir / f"index_{self.index_fingerprint}" / f"runtime_{digest}"

    def _ensure_mmap_array(self, path: Path, array: np.ndarray) -> np.ndarray:
        """运行期数组落盘并转为 mmap 只读视图 (文件已存在则直接复用)。
        各 rank 计算确定性一致, 原子 rename 保证可见文件始终完整;
        落盘失败退回内存副本 (语义不变, 仅损失跨进程页共享)。"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
                try:
                    with open(tmp, "wb") as handle:
                        np.lib.format.write_array(handle, np.ascontiguousarray(array), allow_pickle=False)
                    os.replace(tmp, path)
                finally:
                    if tmp.exists():
                        tmp.unlink()
            return _madvise_random(np.load(path, mmap_mode="r"))
        except Exception as exc:
            print(f"⚠️ 运行期索引数组落盘失败, 退回内存副本: {path}: {exc}", flush=True)
            return np.ascontiguousarray(array)

    def _group_indices_mmap(self, runtime_dir: Path, name: str, ids: np.ndarray, groups: int):
        """按 id 稳定分组收集下标 → CSR (off int64 + vals int32), 落盘为 mmap 文件。
        稳定排序保证组内下标升序, 与旧 list.append 顺序一致。"""
        vals = np.argsort(ids, kind="stable").astype(np.int32)
        counts = np.bincount(ids, minlength=groups)
        off = np.zeros(groups + 1, np.int64)
        np.cumsum(counts, out=off[1:])
        off = self._ensure_mmap_array(runtime_dir / f"{name}_off.npy", off)
        vals = self._ensure_mmap_array(runtime_dir / f"{name}_vals.npy", vals)
        return off, vals

    def _load_runtime_cache(self, runtime_dir: Path, index_store: _ColumnarIndex) -> bool:
        """运行期 bucket 派生数组的命中快路径。

        主循环产物 (kept_rows/row_bucket/edit_val/CSR 桶池) 与 指纹+桶配置+配比参数
        一一对应且各 rank 确定性一致; done.json 标记存在 (所有 mmap 文件已落盘) 时
        直接映射加载, 跳过 34M 行 Python 重建循环 (每 rank 省 ~2 分钟热启动)。
        校验失败一律退回重建路径, 语义不变。"""
        if not (runtime_dir / "done.json").exists():
            return False
        names = ("kept_rows", "row_bucket", "edit_val",
                 "bucket_off", "bucket_vals", "mode_bucket_off", "mode_bucket_vals")
        try:
            arrs = {name: _madvise_random(np.load(runtime_dir / f"{name}.npy", mmap_mode="r"))
                    for name in names}
            n_buckets = len(self.buckets)
            kept_rows, row_bucket, edit_val = arrs["kept_rows"], arrs["row_bucket"], arrs["edit_val"]
            bucket_off, bucket_vals = arrs["bucket_off"], arrs["bucket_vals"]
            mode_bucket_off, mode_bucket_vals = arrs["mode_bucket_off"], arrs["mode_bucket_vals"]
            if (kept_rows.shape[0] != row_bucket.shape[0]
                    or bucket_off.shape[0] != n_buckets + 1 or mode_bucket_off.shape[0] != 2 * n_buckets + 1
                    or bucket_off[-1] != bucket_vals.shape[0] or mode_bucket_off[-1] != mode_bucket_vals.shape[0]
                    or int(row_bucket.max()) >= n_buckets):
                raise ValueError("运行期缓存形状不一致")
        except Exception as exc:
            print(f"⚠️ 运行期 bucket 缓存校验失败, 走重建路径: {runtime_dir}: {exc}", flush=True)
            return False
        self.records = _KeptRecordView(index_store, kept_rows, row_bucket, list(self.buckets))
        self.edit_val_indices = edit_val
        self.bucket_indices = {
            bucket: bucket_vals[bucket_off[bid]:bucket_off[bid + 1]]
            for bid, bucket in enumerate(self.buckets)
        }
        self.mode_bucket_indices = {
            f"{mode}:{bucket}": mode_bucket_vals[
                mode_bucket_off[mode_id * n_buckets + bid]:mode_bucket_off[mode_id * n_buckets + bid + 1]]
            for mode_id, mode in enumerate(("image", "edit"))
            for bid, bucket in enumerate(self.buckets)
        }
        self.read_stats["edit_val_pool"] = len(edit_val)
        _debug_index_event("runtime-cache-hit", kept=int(kept_rows.shape[0]))
        print(f"📦 运行期 bucket 索引缓存命中 (跳过 34M 行重建循环): {runtime_dir.name}", flush=True)
        return True

    def _build_bucket_index(self, index_store: _ColumnarIndex, max_samples, max_edit_samples, edit_ratio) -> None:
        """列式合并 + bucket 索引构建, 与旧 list[dict] 实现的合并/过滤/验证池语义逐行对齐:
        - edit 先按 max_edit_samples 截断, 再按 edit_ratio 并入前缀; max_samples 仅在无 edit 时截断;
        - 无合格 bucket 的行被过滤, 但仍计入 merged 长度 (与旧 first_edit_pos 口径一致);
        - edit_val_indices 追加因 edit_ratio 未入选的 edit, 不注册桶池, 训练采样不可见。"""
        runtime_dir = self._runtime_cache_dir(max_samples, max_edit_samples, edit_ratio)
        if self._load_runtime_cache(runtime_dir, index_store):
            return
        n_image, n_edit = index_store.n_image, index_store.n_edit
        e_keep = n_edit if max_edit_samples is None else min(int(max_edit_samples), n_edit)
        if edit_ratio == "auto" or n_image == 0:
            e_used = e_keep
        elif e_keep > 0:
            count = e_keep if float(edit_ratio) >= 1 else round(n_image * float(edit_ratio) / (1 - float(edit_ratio)))
            e_used = min(count, e_keep)
        else:
            e_used = 0
        edit_included = e_used
        i_keep = n_image if (max_samples is None or e_keep > 0) else min(int(max_samples), n_image)
        merged_len = i_keep + e_used
        bucket_ids = {bucket: bid for bid, bucket in enumerate(self.buckets)}
        kept_rows = np.empty(i_keep + e_keep, np.int32)
        row_bucket = np.empty(i_keep + e_keep, np.int32)
        kept = bucket_filtered = 0
        bucket_t0 = time.time()
        _debug_index_event("bucket-loop-start", total=merged_len)
        for pos in range(merged_len):
            row = pos if pos < i_keep else n_image + pos - i_keep
            bucket = self._nearest_bucket(index_store.w[row], index_store.h[row])
            if bucket is None:
                bucket_filtered += 1
            else:
                kept_rows[kept] = row
                row_bucket[kept] = bucket_ids[bucket]
                kept += 1
            if pos == 0 or (pos + 1) % 1_000_000 == 0:
                _debug_index_event("bucket-loop-progress", done=pos + 1, total=merged_len,
                                   seconds=round(time.time() - bucket_t0, 1), filtered=bucket_filtered)
        _debug_index_event("bucket-loop-done", done=merged_len,
                           seconds=round(time.time() - bucket_t0, 1), filtered=bucket_filtered)
        self.read_stats["bucket_filtered"] = bucket_filtered
        # ─── 验证专用 edit 池 ───
        # 训练集内的 edit + 因 edit_ratio 未入选的 edit。追加到 records 末尾但不注册
        # bucket_indices/mode_bucket_indices —— 训练采样完全不可见，训练分布不变；
        # validation 通过 edit_val_indices 取样本做 BEFORE|GEN|GT 对比。
        first_edit_pos = merged_len - edit_included
        kept_main = kept   # 主合并循环保留的行数; 之后的 leftover edit 不进任何桶池
        edit_val = [i for i in range(first_edit_pos, kept) if bool(index_store.is_edit[kept_rows[i]])]
        for row in range(n_image + e_used, n_image + e_keep):
            bucket = self._nearest_bucket(index_store.w[row], index_store.h[row])
            if bucket is None:
                bucket_filtered += 1
                continue
            edit_val.append(kept)
            kept_rows[kept] = row
            row_bucket[kept] = bucket_ids[bucket]
            kept += 1
        self.read_stats["edit_val_pool"] = len(edit_val)
        if not kept:
            raise RuntimeError(f"Parquet 数据没有合格 bucket 样本: {self.database_path}")
        # 运行期派生数组 (kept 行映射/bucket id/edit_val/桶池 CSR) 落盘为 mmap 只读文件:
        # 8 rank + 32 worker 共享同一份文件页, 避免 33M 级 int 列表常驻 + COW 复制;
        # 全部落盘后写 done.json 标记, 供后续 rank/重启走 _load_runtime_cache 快路径。
        kept_rows = self._ensure_mmap_array(runtime_dir / "kept_rows.npy", kept_rows[:kept])
        row_bucket = self._ensure_mmap_array(runtime_dir / "row_bucket.npy", row_bucket[:kept])
        self.edit_val_indices = self._ensure_mmap_array(
            runtime_dir / "edit_val.npy", np.asarray(edit_val, np.int32))
        n_buckets = len(self.buckets)
        # 桶池/mode 池只按主合并循环的行分组 (row_bucket[:kept_main]),
        # leftover edit 行仅在 records 末尾可见 (与旧 list 实现一致)
        bucket_off, bucket_vals = self._group_indices_mmap(
            runtime_dir, "bucket", row_bucket[:kept_main], n_buckets)
        # mode 池 key = bucket_id + (edit ? n_buckets : 0), 键序与旧 dict 一致: image:* 在前 edit:* 在后
        mode_flags = index_store.is_edit[kept_rows[:kept_main]]
        mode_keys = row_bucket[:kept_main] + np.where(mode_flags, np.int32(n_buckets), np.int32(0))
        mode_bucket_off, mode_bucket_vals = self._group_indices_mmap(
            runtime_dir, "mode_bucket", mode_keys, 2 * n_buckets)
        self.records = _KeptRecordView(index_store, kept_rows, row_bucket, list(self.buckets))
        self.bucket_indices = {
            bucket: bucket_vals[bucket_off[bid]:bucket_off[bid + 1]]
            for bid, bucket in enumerate(self.buckets)
        }
        self.mode_bucket_indices = {
            f"{mode}:{bucket}": mode_bucket_vals[
                mode_bucket_off[mode_id * n_buckets + bid]:mode_bucket_off[mode_id * n_buckets + bid + 1]]
            for mode_id, mode in enumerate(("image", "edit"))
            for bid, bucket in enumerate(self.buckets)
        }
        try:  # 全部文件就绪后原子写完成标记, 快路径只信任该标记
            marker = runtime_dir / "done.json"
            tmp = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
            tmp.write_text(json.dumps({"kept": int(kept), "n_buckets": n_buckets}))
            os.replace(tmp, marker)
        except OSError:
            pass

    # ─── 索引缓存：首次读取后在 output 持久化，指纹不一致时才重建 ───

    def _discover_source_dirs(self, root: Path, sources: set[str]) -> list[Path]:
        """与 _read_parquet 相同的 source 目录发现逻辑（仅枚举，不读内容）。"""
        root = Path(root)
        if not root.exists():
            return []
        if root.is_file():
            return [root.parent.parent]
        if (root / "parquet").is_dir():
            return [root]
        if sources:
            return [item for item in (root / source for source in sorted(sources)) if item.is_dir()]
        return sorted(item for item in root.iterdir() if item.is_dir())

    def _shard_signature(self, edit_database_path: str | None) -> dict[str, list[int]]:
        """每个 source 的 (parquet分片数, mdb分片数)。

        作用：数据集在后台持续构建（如 monet）时，分片数变化会使指纹失效并触发重建，
        避免训练用上过期索引。

        性能：pathlib.glob 会对每项 stat，上万个 mdb 在 NFS 上是分钟级元数据风暴
        （曾把启动卡死在这里）。改为 os.listdir + 后缀过滤（单次 readdir，零 stat），
        并把签名快照缓存到本地 json（TTL 默认 300s，GWP_SHARD_SIGNATURE_TTL=0 关闭）。"""
        signature: dict[str, list[int]] = {}
        roots = [("image", Path(self.database_path), self.sources)]
        if edit_database_path:
            roots.append(("edit", Path(edit_database_path), self.edit_sources))
        ttl = float(os.environ.get("GWP_SHARD_SIGNATURE_TTL", 300))
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        for label, root, sources in roots:
            snapshot = self._load_signature_snapshot(label, root, ttl)
            for source_dir in self._discover_source_dirs(root, sources):
                name = f"{label}:{source_dir.name}"
                if snapshot and name in snapshot:
                    signature[name] = snapshot[name]
                    continue
                try:
                    parquet_count = sum(
                        1 for entry in os.listdir(source_dir / "parquet")
                        if entry.startswith("meta_") and entry.endswith(".parquet")
                    )
                    mdb_count = sum(
                        1 for entry in os.listdir(source_dir / "mdb")
                        if entry.endswith(".mdb")
                    )
                except OSError:
                    continue
                signature[name] = [parquet_count, mdb_count]
        for label, root, _ in roots:
            self._write_signature_snapshot(label, root, signature)
        return signature

    @staticmethod
    def _signature_snapshot_path(cache_dir: Path, label: str, root: Path) -> Path:
        """稳定文件名: hash() 每进程随机化，必须用 sha1，否则快照永远 miss。"""
        import hashlib
        digest = hashlib.sha1(str(root).encode()).hexdigest()[:16]
        return cache_dir / f"signature_{label}_{digest}.json"

    def _load_signature_snapshot(self, label: str, root: Path, ttl: float) -> dict | None:
        """读取分片签名快照；TTL 内直接用本地 json，过期返回 None 由调用方重扫。"""
        if ttl <= 0:
            return None
        snapshot_path = self._signature_snapshot_path(self.index_cache_dir, label, root)
        try:
            if time.time() - snapshot_path.stat().st_mtime <= ttl:
                return json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return None

    def _write_signature_snapshot(self, label: str, root: Path, signature: dict) -> None:
        """把某个 root 的分片签名快照写到缓存目录（写失败不影响训练）。"""
        try:
            snapshot_path = self._signature_snapshot_path(self.index_cache_dir, label, root)
            payload = {k: v for k, v in signature.items() if k.startswith(f"{label}:")}
            snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass

    def _index_fingerprint(self, image_enabled: bool, edit_enabled: bool,
                           max_samples, max_edit_samples, edit_database_path) -> str:
        """yaml dataset 中影响加载行的配置 + 底层分片计数的 sha256 指纹。"""
        import hashlib
        spec = {
            "version": self.INDEX_CACHE_VERSION,
            "database_path": str(self.database_path),
            "sources": sorted(self.sources),
            "lmdb_root": str(self.lmdb_root),
            "image_enabled": bool(image_enabled),
            "max_samples": None if max_samples is None else int(max_samples),
            "edit_database_path": str(edit_database_path or ""),
            "edit_sources": sorted(self.edit_sources),
            "edit_split": str(self.edit_split),
            "edit_lmdb_root": str(self.edit_lmdb_root),
            "edit_enabled": bool(edit_enabled),
            "max_edit_samples": None if max_edit_samples is None else int(max_edit_samples),
            "shards": self._shard_signature(edit_database_path if edit_enabled else None),
        }
        blob = json.dumps(spec, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_paths(self, fingerprint: str) -> tuple[Path, Path]:
        # 列式缓存为每指纹一个目录 (meta.json + 各列 .npy/.bin); 旧版单文件
        # index_{fp}.pkl 不再读写, 遗留文件随 INDEX_CACHE_VERSION 指纹变化自然淘汰。
        return (self.index_cache_dir / f"index_{fingerprint}",
                self.index_cache_dir / f"index_{fingerprint}.stats.txt")

    def _load_index_cache(self, fingerprint: str, allow_fresh: bool = False) -> dict[str, Any] | None:
        # allow_fresh=True 时跳过强制刷新检查：rank0 刚重建并写入了新缓存，
        # 其余 rank 应读取该新缓存而不是再次触发强制刷新逻辑。
        if self.index_cache_force_refresh and not allow_fresh:
            print("🔄 refresh_index_cache=true: 忽略旧索引缓存，强制重新扫描构建", flush=True)
            return None
        if os.environ.get("GWP_DATASET_INDEX_CACHE_DISABLE"):
            return None
        index_dir, _ = self._cache_paths(fingerprint)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            # 旧版 pkl 缓存与列式格式不兼容, 一律走重建路径
            return None
        try:
            t0 = time.time()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (not isinstance(meta, dict) or meta.get("fingerprint") != fingerprint
                    or int(meta.get("format_version") or 0) != _INDEX_FORMAT_VERSION):
                print(f"⚠️ 索引缓存指纹/格式不匹配，将重建: {index_dir}", flush=True)
                return None
            size_gib = sum(entry.stat().st_size for entry in index_dir.iterdir() if entry.is_file()) / 1024**3
            _debug_index_event("load-start", cache=index_dir.name, size_gib=round(size_gib, 2), allow_fresh=allow_fresh)
            # 列式缓存只做 mmap 映射 (秒级), 无需整份反序列化; 加载前照常打印, 避免阶段静默
            print(f"⏳ [PID={os.getpid()}] 正在加载图片索引缓存(mmap): {index_dir.name} "
                  f"({size_gib:.2f} GiB) ...", flush=True)
            store = _ColumnarIndex.load(index_dir, meta)
            payload = {
                "fingerprint": fingerprint,
                "created": meta.get("created"),
                "store": store,
                "stats": meta.get("stats"),
            }
            _debug_index_event("load-done", cache=index_dir.name, seconds=round(time.time() - t0, 1),
                               records=store.n_image, edits=store.n_edit)
            print(
                f"📦 [PID={os.getpid()}] 数据集索引缓存命中(mmap 列式共享): {index_dir.name} "
                f"({size_gib:.2f} GiB, {time.time() - t0:.1f}s), "
                f"image={store.n_image:,}, edit={store.n_edit:,}",
                flush=True,
            )
            return payload
        except Exception as exc:
            _debug_index_event("load-error", cache=index_dir.name, error=repr(exc))
            print(f"⚠️ 索引缓存加载失败，将重建: {index_dir}: {exc}", flush=True)
            return None

    def _write_index_cache(self, fingerprint: str, records: list, edits: list) -> tuple[_ColumnarIndex, bool]:
        """rank0 构建完成后把 list[dict] 行转列式并落盘 (mmap 列式缓存)。
        返回 (索引对象, 是否成功落盘)。成功后调用方应立即释放传入的 rows 引用:
        构建期 rank0 的 ~30GB 瞬态只在写缓存阶段存在, 训练期一律走 mmap 共享。"""
        index_dir, _ = self._cache_paths(fingerprint)
        index_store = None
        try:
            t0 = time.time()
            index_store = _ColumnarIndex.build_from_rows(records, edits)
            index_dir.mkdir(parents=True, exist_ok=True)
            index_store.save(index_dir, fingerprint=fingerprint, created=time.time(), stats=self.read_stats)
            payload = self._load_index_cache(fingerprint, allow_fresh=True)
            if payload is None:
                raise RuntimeError("写盘后回读失败 (meta.json 缺失或校验不过)")
            size_gib = sum(entry.stat().st_size for entry in index_dir.iterdir() if entry.is_file()) / 1024**3
            print(f"✅ 数据集索引缓存写入(列式 mmap): {index_dir} "
                  f"({size_gib:.2f} GiB, {time.time() - t0:.1f}s), "
                  f"此后所有 rank/worker 只读映射共享", flush=True)
            return payload["store"], True
        except Exception as exc:
            print(f"⚠️ 索引缓存写入失败: {index_dir}: {exc}", flush=True)
            if index_store is None:
                raise
            # 列式转换已成功但落盘失败: 退回内存数组包装, 行为与旧版"无缓存继续训练"一致
            return index_store, False

    def _load_or_read_records(self, image_enabled: bool, edit_enabled: bool,
                              max_samples, max_edit_samples, edit_database_path):
        """优先读 output 索引缓存 (mmap 列式)；缓存缺失/指纹不一致时才扫描 parquet 重建。

        分布式下仅 rank 0 重建并写缓存，其余 rank 在 barrier 后直接读缓存，
        避免多 rank 同时扫描 NFS 上的数千个 parquet 分片。返回 mmap 列式索引
        (union 存储: image 行在前, edit 行在后)。"""
        fingerprint = self._index_fingerprint(image_enabled, edit_enabled,
                                              max_samples, max_edit_samples, edit_database_path)
        self.index_fingerprint = fingerprint
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        _debug_index_event("phase-enter", cache_dir=str(self.index_cache_dir), fingerprint=fingerprint[:12], distributed=distributed)
        cache_sync_key = f"yisu/image-index-cache/{fingerprint}"
        if distributed and rank != 0:
            _debug_index_event("store-wait-start", key=cache_sync_key)
            # 先等 global rank 0 完成一次缓存探测，再逐 rank 读取同一缓存。
            # 列式缓存为 mmap 只读映射, 多 rank 并发打开同一目录也只是各自建立
            # 映射, 共享同一份物理页, 不再有 21 GiB pickle 的逐 rank 反序列化。
            _get_default_store().wait([cache_sync_key])
            _debug_index_event("store-wait-done", key=cache_sync_key)
        if distributed and rank == 0:
            _debug_index_event("rank0-load-before")
            payload = self._load_index_cache(fingerprint)
            _get_default_store().set(cache_sync_key, b"hit" if payload is not None else b"miss")
            _debug_index_event("rank0-store-set", result=("hit" if payload is not None else "miss"))
        elif distributed:
            payload = None
            result = _get_default_store().get(cache_sync_key)
            _debug_index_event("store-get", result=result.decode(errors="replace"))
            if result == b"hit":
                # 同一节点上的 rank 串行建立 mmap 映射；对象仍由各 rank 保持自己的只读映射。
                import fcntl
                lock_path = self.index_cache_dir / f".index-load-{fingerprint}.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                _debug_index_event("rank-load-lock-wait", lock=str(lock_path))
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    _debug_index_event("rank-load-lock-acquired", lock=str(lock_path))
                    payload = self._load_index_cache(fingerprint, allow_fresh=True)
                    _debug_index_event("rank-load-after", loaded=(payload is not None))
        else:
            payload = self._load_index_cache(fingerprint)
        if payload is not None:
            self.read_stats = payload.get("stats") or {"image": {}, "edit": {}}
            return payload["store"]

        def _read_all():
            rows = (self._read_parquet(self.database_path, self.sources, False,
                                       self.lmdb_root, max_samples) if image_enabled else [])
            edit_rows = (self._read_parquet(edit_database_path, self.edit_sources, True,
                                            self.edit_lmdb_root, max_edit_samples) if edit_enabled else [])
            return rows, edit_rows

        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        index_store = None
        if not distributed or rank == 0:
            records, edits = _read_all()
            index_store, cache_written = self._write_index_cache(fingerprint, records, edits)
            # 释放构建期 list[dict] 的 ~30GB 瞬态 (rank0 已转为 mmap 列式索引)
            records = edits = None
        if distributed:
            sync_key = f"yisu/image-index/{fingerprint}"
            if rank == 0:
                _get_default_store().set(sync_key, b"ready" if cache_written else b"failed")
                if not cache_written:
                    raise RuntimeError(f"global rank0 写入图片索引缓存失败: {self._cache_paths(fingerprint)[0]}")
            else:
                print(f"⏳ rank {rank} 等待 rank 0 构建数据集索引: {self.index_cache_dir}", flush=True)
                store = _get_default_store()
                store.wait([sync_key])
                if store.get(sync_key) != b"ready":
                    raise RuntimeError(f"global rank0 写入图片索引缓存失败: {self._cache_paths(fingerprint)[0]}")
                # rank0 已完成强制重建并写入新缓存 → 此处允许读取新缓存
                payload = self._load_index_cache(fingerprint, allow_fresh=True)
                if payload is None:
                    raise RuntimeError(
                        f"global rank0 未生成图片索引缓存: {self._cache_paths(fingerprint)[0]}"
                    )
                self.read_stats = payload.get("stats") or {"image": {}, "edit": {}}
                index_store = payload["store"]
                print(f"📦 rank {rank} 复用 global rank0 图片索引: {self.index_cache_dir}", flush=True)
        return index_store

    def _print_distribution_table(self):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        try:
            from prettytable import PrettyTable
        except ImportError:
            PrettyTable = None

        CYAN, GREEN, YELLOW, DIM, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
        read_stats = getattr(self, "read_stats", {}) or {}
        total_final = len(self.records)

        # 最终配比: (source, mode) -> 训练样本数。列式索引走向量化统计, 结果与旧的
        # 逐行 record.get 统计完全一致, 但省去 3300 万行逐行拼 dict 的分钟级开销。
        records_view = self.records
        bucket_ids = records_view.bucket_ids()
        bucket_counts = {
            bucket: int(count)
            for bucket, count in zip(self.buckets, np.bincount(bucket_ids, minlength=len(self.buckets)))
        }
        is_edit_flags = records_view.is_edit_flags()
        mode_counts = {
            "image": int(np.count_nonzero(~is_edit_flags)),
            "edit": int(np.count_nonzero(is_edit_flags)),
        }
        source_ids = records_view.source_ids()
        sources_table = records_view.sources_table()
        source_total = np.bincount(source_ids, minlength=len(sources_table))
        source_edit = np.bincount(source_ids[is_edit_flags], minlength=len(sources_table))
        final_counts: dict[tuple[str, str], int] = {}
        for source_id, source in enumerate(sources_table):
            image_count = int(source_total[source_id]) - int(source_edit[source_id])
            if image_count:
                final_counts[(str(source), "image")] = image_count
            if int(source_edit[source_id]):
                final_counts[(str(source), "edit")] = int(source_edit[source_id])

        # 读取阶段统计: (source, mode) -> parquet 行数等
        load_info: dict[tuple[str, str], dict] = {}
        for mode in ("image", "edit"):
            for source, info in (read_stats.get(mode) or {}).items():
                load_info[(source, mode)] = info
        keys = sorted(set(final_counts) | set(load_info))

        fingerprint = getattr(self, "index_fingerprint", "") or "unknown"
        title = f"📊 数据集分布总表 (index={fingerprint[:16]})"
        lines = [title]
        if PrettyTable is not None:
            # ── Source × Mode 配比表 ──
            table = PrettyTable(["Source", "Mode", "Shards", "总数", "有效", "过滤", "占比"])
            table.align = "l"
            grand_total = grand_valid = 0
            for source, mode in keys:
                info = load_info.get((source, mode), {})
                row_total = int(info.get("total", 0))
                valid = final_counts.get((source, mode), 0)
                shards = int(info.get("shards", 0))
                grand_total += row_total
                grand_valid += valid
                ratio = f"{valid / total_final:.2%}" if total_final else "-"
                color = "" if mode == "image" else YELLOW
                table.add_row([
                    source, f"{color}{mode}{RESET}" if color else mode,
                    f"{shards:,}", f"{row_total:,}", f"{valid:,}",
                    f"{row_total - valid:,}", ratio,
                ])
            table.add_row(["TOTAL", "all", "-",
                           f"{grand_total:,}", f"{grand_valid:,}",
                           f"{grand_total - grand_valid:,}", "100.00%"])
            lines.append(table.get_string())

            # ── Bucket 分布表（样本数降序 + 采样权重标注）──
            weights = getattr(self, "_bucket_weight_view", None) or {}
            bucket_table = PrettyTable(["Bucket", "Samples", "占比", "采样权重"])
            bucket_table.align = "l"
            for bucket, count in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
                weight = weights.get(bucket)
                bucket_table.add_row([
                    bucket, f"{count:,}",
                    f"{count / total_final:.2%}" if total_final else "-",
                    "-" if weight is None else f"{weight:g}",
                ])
            lines.append("🪣 Bucket 分布")
            lines.append(bucket_table.get_string())

            # ── 模式汇总 ──
            lines.append(
                f"🖼️ image={GREEN}{mode_counts['image']:,}{RESET}  "
                f"🎨 edit={YELLOW}{mode_counts['edit']:,}{RESET}  "
                f"{DIM}| 无合格 bucket 过滤={read_stats.get('bucket_filtered', 0):,}{RESET}"
            )
        else:
            lines.append(str({
                "final": {f"{s}|{m}": c for (s, m), c in final_counts.items()},
                "read": {f"{s}|{m}": i for (s, m), i in load_info.items()},
                "buckets": bucket_counts,
            }))
        for mode in ("image", "edit"):
            excluded = read_stats.get(f"{mode}_excluded") or []
            if excluded:
                lines.append(f"⚠️ 未加载的 {mode} source 目录(不在白名单): {excluded}")

        plain = "\n".join(
            line.replace(CYAN, "").replace(GREEN, "").replace(YELLOW, "")
                .replace(DIM, "").replace(RESET, "")
            for line in lines
        )
        print("\n" + f"{CYAN}{plain}{RESET}" + "\n", flush=True)

        # 配比表持久化到 output 索引缓存目录，供随时查阅（纯文本无 ANSI 码）
        try:
            _, table_path = self._cache_paths(fingerprint)
            table_path.parent.mkdir(parents=True, exist_ok=True)
            table_path.write_text(plain + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"⚠️ 配比表写入失败: {exc}", flush=True)

    @staticmethod
    def _mdb_ranges(source_dir):
        import lmdb
        ranges = []
        for mdb_path in sorted((Path(source_dir) / "mdb").glob("*.mdb")):
            real = str(mdb_path)
            try:
                import bos_data  # BOS 镜像: 占位 mdb 按需下载
                real = bos_data.materialize(real)
            except ImportError:
                pass
            env = lmdb.open(real, readonly=True, lock=False, subdir=False, max_readers=1024)
            minimum = maximum = None
            with env.begin() as txn:
                for key, _ in txn.cursor():
                    if len(key) == 8:
                        value = struct.unpack("<Q", key)[0]
                    else:
                        try:
                            value = int(key)
                        except ValueError:
                            continue
                    minimum = value if minimum is None else min(minimum, value)
                    maximum = value if maximum is None else max(maximum, value)
            env.close()
            if minimum is not None:
                ranges.append((minimum, maximum, mdb_path.stem))
        ranges.sort()
        return ranges

    def _read_parquet(self, path, sources, is_edit, lmdb_root, limit=None):
        import pyarrow.parquet as parquet
        import pyarrow.compute as pc
        root = Path(path)
        if root.is_file():
            files = [(root, root.parent.parent)]
        else:
            # 支持 unified_*_lmdb 根目录和单个 source 目录两种配置。
            if (root / "parquet").is_dir():
                source_dirs = [root]
            else:
                source_dirs = [root / source for source in sorted(sources)] if sources else sorted(
                    item for item in root.iterdir() if item.is_dir()
                )
            files = []
            for source_dir in source_dirs:
                if not source_dir.is_dir():
                    continue
                shard_parquets = sorted((source_dir / "parquet").glob("meta_*.parquet"))
                if shard_parquets:
                    files.extend((file, source_dir) for file in shard_parquets)
                elif (source_dir / "mdb").is_dir() and \
                        (source_dir / "parquet" / "meta.parquet").exists():
                    # BOS 上传形态: 单个合并 meta.parquet + 多个 mdb 分片
                    files.append((source_dir / "parquet" / "meta.parquet", source_dir))

        rows = []
        source_counts = {}
        source_shards = {}
        t0 = time.time()
        mode = "edit" if is_edit else "image"
        stats: dict[str, dict] = {}
        excluded_sources = []
        if root.is_dir() and sources:
            excluded_sources = sorted(
                item.name for item in root.iterdir() if item.is_dir() and item.name not in sources
            )
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        file_iterator = files
        if tqdm is not None:
            file_iterator = tqdm(
                files,
                total=len(files),
                desc="Loading Parquet dataset",
                unit="shard",
                dynamic_ncols=True,
            )
        def _item_from_row(row, sid, stem, source, is_edit, dataset_dir):
            # 白名单复制: 只保留训练读取路径实际访问的列。parquet 里 url/
            # aesthetic/水印等几十个分析列不进索引缓存（曾致缓存 29GB、
            # 每 rank 反序列化 10 分钟）。
            item = {
                key: row[key]
                for key in ("caption", "text", "edit_prompt", "instruction", "width", "height",
                            "w_after", "h_after", "split", "metadata")
                if key in row
            }
            item.update(sample_id=sid, source=source, is_edit=is_edit)
            if is_edit:
                item["image_path"] = f"lmdb://{dataset_dir}/{stem}/{sid}:after"
                item["input_image_path"] = f"lmdb://{dataset_dir}/{stem}/{sid}:before"
                item["caption"] = row.get("edit_prompt") or row.get("instruction") or row.get("edit") or ""
                item["width"] = row.get("w_after")
                item["height"] = row.get("h_after")
            else:
                item["image_path"] = f"lmdb://{dataset_dir}/{stem}/{sid}:single"
                item["caption"] = row.get("caption") or row.get("text") or ""
                item["width"] = row.get("width")
                item["height"] = row.get("height")
            return item

        for file, source_dir in file_iterator:
            if source_dir not in source_shards:
                parquet_files = sorted((source_dir / "parquet").glob("meta_*.parquet"))
                mdb_files = sorted((source_dir / "mdb").glob("*.mdb"))
                if not mdb_files:
                    continue
                if not parquet_files:
                    merged = source_dir / "parquet" / "meta.parquet"
                    if not merged.exists():
                        print(f"⚠️ {source_dir.name}: 缺 parquet 元数据 (meta_*.parquet/"
                              f"meta.parquet 都没有), 已跳过该源", flush=True)
                        continue
                    # BOS 上传形态: 单个合并 meta.parquet + 多个 mdb 分片,
                    # 按 mdb 分片键范围把行对齐到分片
                    source_shards[source_dir] = {
                        "merged": merged, "ranges": self._mdb_ranges(source_dir)}
                else:
                    if len(parquet_files) != len(mdb_files):
                        raise RuntimeError(
                            f"Parquet/LMDB 分片数量不一致: {source_dir} "
                            f"parquet={len(parquet_files)} mdb={len(mdb_files)}"
                        )
                    source_shards[source_dir] = {
                        parquet_file: mdb_file.stem
                        for parquet_file, mdb_file in zip(parquet_files, mdb_files)
                    }
            shard_info = source_shards[source_dir]
            source = source_dir.name
            if sources and source not in sources:
                continue
            if limit is not None and source_counts.get(source, 0) >= int(limit):
                continue
            source_stats = stats.setdefault(source, {
                "shards": 0, "total": 0, "valid": 0,
                "filtered_split": 0, "filtered_no_id": 0, "limit_stopped": False,
            })
            dataset_dir = str((Path(lmdb_root) / source).resolve())

            if isinstance(shard_info, dict) and "merged" in shard_info:
                # 合并元数据模式: 只读需要的列, 按 sid 区间切到各分片
                source_stats["shards"] += 1
                table = parquet.read_table(str(file))
                sid_col = "sample_id" if "sample_id" in table.column_names else "id"
                sids = table.column(sid_col)
                for mn, mx, stem in shard_info["ranges"]:
                    mask = pc.and_(pc.greater_equal(sids, mn), pc.less_equal(sids, mx))
                    for row in table.filter(mask).to_pylist():
                        if limit is not None and source_counts.get(source, 0) >= int(limit):
                            source_stats["limit_stopped"] = True
                            break
                        source_stats["total"] += 1
                        sid = int(row.get(sid_col))
                        rows.append(_item_from_row(row, sid, stem, source, is_edit, dataset_dir))
                        source_counts[source] = source_counts.get(source, 0) + 1
                        source_stats["valid"] += 1
                continue

            stem = shard_info.get(file)
            if stem is None:
                continue
            if file.name == "meta.parquet" or not file.exists():
                continue
            if limit is not None and source_counts.get(source, 0) >= int(limit):
                continue
            source_stats["shards"] += 1
            table = parquet.read_table(file)
            for row in table.to_pylist():
                if limit is not None and source_counts.get(source, 0) >= int(limit):
                    source_stats["limit_stopped"] = True
                    break
                source_stats["total"] += 1
                split = row.get("split")
                if is_edit and self.edit_split is not None and split is not None and str(split) != self.edit_split:
                    source_stats["filtered_split"] += 1
                    continue
                sid = row.get("sample_id", row.get("id"))
                if sid is None:
                    source_stats["filtered_no_id"] += 1
                    continue
                sid = int(sid)
                # URI 必须指向实际包含 mdb/ 的 source 目录；lmdb_root 用于
                # database_path 是 parquet 路径、但载荷位于另一根目录的情况。
                rows.append(_item_from_row(row, sid, stem, source, is_edit, dataset_dir))
                source_counts[source] = source_counts.get(source, 0) + 1
                source_stats["valid"] += 1
        if not rows:
            raise RuntimeError(f"Parquet 数据没有可用样本: {path}")
        self.read_stats[mode] = stats
        if excluded_sources:
            self.read_stats[f"{mode}_excluded"] = excluded_sources
        grand = {key: sum(int(info.get(key, 0)) for info in stats.values())
                 for key in ("shards", "total", "valid")}
        print(
            f"✅ parquet 读取完成[{mode}]: sources={len(stats)} 分片={grand['shards']:,} "
            f"总数={grand['total']:,} 有效={grand['valid']:,} "
            f"过滤={grand['total'] - grand['valid']:,} ({time.time() - t0:.1f}s)"
            + (f" 未加载目录={excluded_sources}" if excluded_sources else ""),
            flush=True,
        )
        return rows


def collate_image_batch(samples):
    """Numpy 传输版 collate：worker 间只走标准管道 pickle，不触发 /dev/shm 分配。"""
    np = __import__("numpy")
    result = {
        "pixel_values": np.stack([s["pixel_values"] for s in samples]),
        "captions": [str(s.get("caption", "")) for s in samples],
        "image_paths": [s.get("image_path") for s in samples],
        "is_edit": [bool(s.get("is_edit", False)) for s in samples],
        "sample_metadata": [
            {
                "source": s.get("source", "unknown"),
                "sample_id": s.get("sample_id", "unknown"),
                "width": s.get("width"),
                "height": s.get("height"),
                "bucket": s.get("bucket", "unknown"),
                "is_edit": bool(s.get("is_edit", False)),
                "image_path": s.get("image_path"),
                "input_image_path": s.get("input_image_path"),
            }
            for s in samples
        ],
    }
    if all("condition_pixel_values" in s for s in samples):
        result["condition_pixel_values"] = np.stack([s["condition_pixel_values"] for s in samples])
    return result
