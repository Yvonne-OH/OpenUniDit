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
from typing import Any

from PIL import Image
# 4K source 存在 100-144MP 大图，默认阈值(89MP)会刷 DecompressionBomb 警告；本地可信数据直接关闭
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_store
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

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
        import numpy as np
        # bucket_override: 高分桶借图时按目标低分桶分辨率加载 (纯降采样)
        bucket = bucket_override or record.get("bucket") or self._nearest_bucket(record.get("width"), record.get("height"))
        width, height = parse_bucket(bucket)

        def source_path(path, kind):
            path = str(path)
            if path.startswith('lmdb://'):
                return path if ':' in path.rsplit('/', 1)[-1] else f'{path}:{kind}'
            return str(self.image_root / path) if self.image_root and not Path(path).is_absolute() else path

        request = dict(kind='images', path=source_path(record['image_path'],
                       'after' if record.get('is_edit') else 'single'), width=width, height=height)
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
        pool = self.mode_bucket_indices.get(f"{mode}:{bucket}") or [index]
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


class LmdbParquetDataset(ImageDatabaseDataset):
    INDEX_CACHE_VERSION = 1  # 索引缓存格式版本，记录结构变化时递增使旧缓存失效

    def __init__(self, database_path: str, buckets: list[str], lmdb_root: str | None = None,
                 image_root: str | None = None, transform=None, max_upscale_ratio: float = 1.2,
                 max_samples: int | None = None, max_retries: int = 8, seed: int = 42,
                 sources: list[str] | None = None, edit_database_path: str | None = None,
                 edit_sources: list[str] | None = None, edit_split: str = "train",
                 edit_ratio: float | str = 0.0, max_edit_samples: int | None = None,
                 image_training_enabled: bool = True, edit_training_enabled: bool = False,
                 index_cache_dir: str | None = None, refresh_index_cache: bool = False,
                 bucket_sampling_weights: dict | None = None, max_decode_candidates: int = 3, **kwargs):
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
        records, edits = self._load_or_read_records(
            bool(image_training_enabled), bool(edit_training_enabled),
            max_samples, max_edit_samples, edit_database_path,
        )
        # 索引加载后仍有数分钟的多轮 O(N) 处理 (bucket 索引/分片定位/校验), 必须提示
        print(f"⏳ 图片记录就绪 (image={len(records):,}, edit={len(edits):,}), "
              "正在构建 bucket 索引与分片定位 (约数分钟)...", flush=True)
        if max_edit_samples is not None:
            edits = edits[:int(max_edit_samples)]
        edit_included = 0
        if edit_ratio == "auto":
            records += edits
            edit_included = len(edits)
        elif records and edits:
            count = len(edits) if float(edit_ratio) >= 1 else round(len(records) * float(edit_ratio) / (1 - float(edit_ratio)))
            records += edits[:count]
            edit_included = min(count, len(edits))
        elif edits:
            records = edits
            edit_included = len(edits)
        if max_samples is not None and not edits:
            records = records[:int(max_samples)]
        self.records, self.bucket_indices = [], {bucket: [] for bucket in self.buckets}
        self.mode_bucket_indices = {f"{mode}:{bucket}": [] for mode in ("image", "edit") for bucket in self.buckets}
        bucket_filtered = 0
        for record in records:
            bucket = record.get("bucket")
            if bucket not in self.buckets:
                bucket = self._nearest_bucket(record.get("width"), record.get("height"))
            if bucket is None:
                bucket_filtered += 1
                continue
            record["bucket"] = bucket
            index = len(self.records)
            self.records.append(record)
            self.bucket_indices[bucket].append(index)
            mode = "edit" if record.get("is_edit") else "image"
            self.mode_bucket_indices[f"{mode}:{bucket}"].append(index)
        self.read_stats["bucket_filtered"] = bucket_filtered
        # ─── 验证专用 edit 池 ───
        # 训练集内的 edit + 因 edit_ratio 未入选的 edit。追加到 records 末尾但不注册
        # bucket_indices/mode_bucket_indices —— 训练采样完全不可见，训练分布不变；
        # validation 通过 edit_val_indices 取样本做 BEFORE|GEN|GT 对比。
        first_edit_pos = len(records) - edit_included
        self.edit_val_indices = [i for i in range(first_edit_pos, len(self.records)) if self.records[i].get("is_edit")]
        for record in edits[edit_included:]:
            bucket = record.get("bucket")
            if bucket not in self.buckets:
                bucket = self._nearest_bucket(record.get("width"), record.get("height"))
            if bucket is None:
                bucket_filtered += 1
                continue
            record = dict(record)
            record["bucket"] = bucket
            self.edit_val_indices.append(len(self.records))
            self.records.append(record)
        self.read_stats["edit_val_pool"] = len(self.edit_val_indices)
        if not self.records:
            raise RuntimeError(f"Parquet 数据没有合格 bucket 样本: {database_path}")
        self._print_distribution_table()

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
        return (self.index_cache_dir / f"index_{fingerprint}.pkl",
                self.index_cache_dir / f"index_{fingerprint}.stats.txt")

    def _load_index_cache(self, fingerprint: str, allow_fresh: bool = False) -> dict[str, Any] | None:
        # allow_fresh=True 时跳过强制刷新检查：rank0 刚重建并写入了新缓存，
        # 其余 rank 应读取该新缓存而不是再次触发强制刷新逻辑。
        if self.index_cache_force_refresh and not allow_fresh:
            print("🔄 refresh_index_cache=true: 忽略旧索引缓存，强制重新扫描构建", flush=True)
            return None
        if os.environ.get("GWP_DATASET_INDEX_CACHE_DISABLE"):
            return None
        pkl_path, _ = self._cache_paths(fingerprint)
        if not pkl_path.exists():
            return None
        try:
            t0 = time.time()
            size_gib = pkl_path.stat().st_size / 1024**3
            # 大索引 (24G 级) unpickle 需约 1 分钟; 加载前必须打印, 否则该阶段终端静默
            print(f"⏳ [PID={os.getpid()}] 正在加载图片索引缓存: {pkl_path.name} "
                  f"({size_gib:.2f} GiB) ...", flush=True)
            with open(pkl_path, "rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
                print(f"⚠️ 索引缓存指纹不匹配，将重建: {pkl_path}", flush=True)
                return None
            print(
                f"📦 [PID={os.getpid()}] 数据集索引缓存命中: {pkl_path.name} "
                f"({size_gib:.2f} GiB, {time.time() - t0:.1f}s), "
                f"image={len(payload['image_records']):,}, edit={len(payload['edit_records']):,}",
                flush=True,
            )
            return payload
        except Exception as exc:
            print(f"⚠️ 索引缓存加载失败，将重建: {pkl_path}: {exc}", flush=True)
            return None

    def _write_index_cache(self, fingerprint: str, records: list, edits: list) -> bool:
        pkl_path, _ = self._cache_paths(fingerprint)
        payload = {
            "fingerprint": fingerprint,
            "created": time.time(),
            "image_records": records,
            "edit_records": edits,
            "stats": self.read_stats,
        }
        try:
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = pkl_path.with_name(f".{pkl_path.name}.tmp-{os.getpid()}")
            t0 = time.time()
            with open(tmp_path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(pkl_path)
            size_gib = pkl_path.stat().st_size / 1024**3
            print(f"✅ 数据集索引缓存写入: {pkl_path} ({size_gib:.2f} GiB, {time.time() - t0:.1f}s)", flush=True)
            return True
        except Exception as exc:
            print(f"⚠️ 索引缓存写入失败: {pkl_path}: {exc}", flush=True)
            return False

    def _load_or_read_records(self, image_enabled: bool, edit_enabled: bool,
                              max_samples, max_edit_samples, edit_database_path):
        """优先读 output 索引缓存；缓存缺失/指纹不一致时才扫描 parquet 重建。

        分布式下仅 rank 0 重建并写缓存，其余 rank 在 barrier 后直接读缓存，
        避免多 rank 同时扫描 NFS 上的数千个 parquet 分片。"""
        fingerprint = self._index_fingerprint(image_enabled, edit_enabled,
                                              max_samples, max_edit_samples, edit_database_path)
        self.index_fingerprint = fingerprint
        payload = self._load_index_cache(fingerprint)
        if payload is not None:
            self.read_stats = payload.get("stats") or {"image": {}, "edit": {}}
            return payload["image_records"], payload["edit_records"]

        def _read_all():
            rows = (self._read_parquet(self.database_path, self.sources, False,
                                       self.lmdb_root, max_samples) if image_enabled else [])
            edit_rows = (self._read_parquet(edit_database_path, self.edit_sources, True,
                                            self.edit_lmdb_root, max_edit_samples) if edit_enabled else [])
            return rows, edit_rows

        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        records, edits = [], []
        if not distributed or rank == 0:
            records, edits = _read_all()
            cache_written = self._write_index_cache(fingerprint, records, edits)
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
                records, edits = payload["image_records"], payload["edit_records"]
                print(f"📦 rank {rank} 复用 global rank0 图片索引: {self.index_cache_dir}", flush=True)
        return records, edits

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

        # 最终配比: (source, mode) -> 训练样本数
        final_counts: dict[tuple[str, str], int] = {}
        bucket_counts = {bucket: 0 for bucket in self.buckets}
        mode_counts = {"image": 0, "edit": 0}
        for record in self.records:
            source = str(record.get("source", "unknown"))
            mode = "edit" if record.get("is_edit") else "image"
            final_counts[(source, mode)] = final_counts.get((source, mode), 0) + 1
            if record.get("bucket") in bucket_counts:
                bucket_counts[record["bucket"]] += 1
            mode_counts[mode] += 1

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
            env = lmdb.open(str(mdb_path), readonly=True, lock=False, subdir=False, max_readers=1024)
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
                files.extend((file, source_dir) for file in sorted(
                    (source_dir / "parquet").glob("meta_*.parquet")))

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
        for file, source_dir in file_iterator:
            if source_dir not in source_shards:
                parquet_files = sorted((source_dir / "parquet").glob("meta_*.parquet"))
                mdb_files = sorted((source_dir / "mdb").glob("*.mdb"))
                if not mdb_files:
                    continue
                if len(parquet_files) != len(mdb_files):
                    raise RuntimeError(
                        f"Parquet/LMDB 分片数量不一致: {source_dir} "
                        f"parquet={len(parquet_files)} mdb={len(mdb_files)}"
                    )
                source_shards[source_dir] = {
                    parquet_file: mdb_file.stem
                    for parquet_file, mdb_file in zip(parquet_files, mdb_files)
                }
            stem = source_shards[source_dir].get(file)
            if stem is None:
                continue
            if file.name == "meta.parquet" or not file.exists():
                continue
            source = source_dir.name
            if sources and source not in sources:
                continue
            if limit is not None and source_counts.get(source, 0) >= int(limit):
                continue
            dataset_dir = (Path(lmdb_root) / source).resolve()
            source_stats = stats.setdefault(source, {
                "shards": 0, "total": 0, "valid": 0,
                "filtered_split": 0, "filtered_no_id": 0, "limit_stopped": False,
            })
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
                dataset_dir = str(dataset_dir)
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
                rows.append(item)
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
