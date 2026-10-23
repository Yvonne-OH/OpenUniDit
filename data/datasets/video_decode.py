"""Isolated video decoding; also executable directly without importing data/torch."""
import json
import os
import math
import random
import hashlib
from pathlib import Path
import subprocess
import select
from collections import OrderedDict
import sys
import tempfile
import time
import shutil
import pickle
import struct
import stat
import fcntl
from contextlib import contextmanager

import numpy as np


# Local IPC only: no parent-side stat/open/cleanup on the source filesystem.
_STRANDED = []
_MAX_STRANDED = 2
_STAGE_CONTEXT = None
_STAGE_STARTED = 0.0
_REQUEST_DEADLINE = float('inf')


class NodeMediaCache:
    """Cross-process node cache for immutable media files."""
    def __init__(self, root=None):
        self.root = root or os.environ.get('YISU_MEDIA_CACHE_DIR', '/tmp/yisu-media-cache')
        self.max_bytes = int(os.environ.get('YISU_MEDIA_CACHE_MAX_BYTES', str(64 * 1024**3)))
        self.file_limit = int(os.environ.get('YISU_MEDIA_MAX_FILE_BYTES', str(512 * 1024**2)))
        if self.max_bytes <= 0 or self.file_limit <= 0:
            raise ValueError('media cache limits must be positive')
        self.objects = os.path.join(self.root, 'objects')
        self.locks = os.path.join(self.root, 'locks')
        self.pfs_slots = os.path.join(self.root, 'pfs_slots')
        self.pfs_concurrency = max(1, int(os.environ.get('YISU_MEDIA_PFS_CONCURRENCY', '2')))
        os.makedirs(self.objects, mode=0o700, exist_ok=True)
        os.makedirs(self.pfs_slots, mode=0o700, exist_ok=True)
        for slot in range(self.pfs_concurrency):
            Path(self.pfs_slots, f'slot-{slot}').touch(mode=0o600, exist_ok=True)
        os.makedirs(self.locks, mode=0o700, exist_ok=True)

    def _key(self, path):
        source = os.path.realpath(path)
        info = os.stat(source)
        if info.st_size > self.file_limit:
            return None, source, info
        key = hashlib.sha256(f'{source}|{info.st_size}|{info.st_mtime_ns}'.encode()).hexdigest()
        return key, source, info

    def acquire(self, path):
        key, source, info = self._key(path)
        if key is None:
            return source
        target = os.path.join(self.objects, key)
        ready = target + '.ready'
        lock_path = os.path.join(self.locks, key + '.lock')
        with open(lock_path, 'a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (os.path.isfile(target) and os.path.isfile(ready)):
                slot = None
                try:
                    while slot is None:
                        for candidate in sorted(Path(self.pfs_slots).glob('slot-*')):
                            handle = open(candidate, 'a+b')
                            try:
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                slot = handle
                                break
                            except BlockingIOError:
                                handle.close()
                        if slot is None:
                            time.sleep(0.01)
                    used = sum(p.stat().st_size for p in Path(self.objects).glob('*') if p.is_file() and not p.name.endswith('.ready'))
                    if used + info.st_size > self.max_bytes:
                        raise MediaCapacityError(f'media cache capacity exceeded: {used + info.st_size} > {self.max_bytes}')
                    temporary = f'{target}.tmp.{os.getpid()}'
                    try:
                        with open(source, 'rb') as src, open(temporary, 'wb') as dst:
                            shutil.copyfileobj(src, dst, 16 * 1024**2)
                            dst.flush()
                            os.fsync(dst.fileno())
                        if os.path.getsize(temporary) != info.st_size:
                            raise OSError('media cache source changed during copy')
                        os.replace(temporary, target)
                        Path(ready).touch()
                    finally:
                        try:
                            os.unlink(temporary)
                        except FileNotFoundError:
                            pass
                finally:
                    if slot is not None:
                        fcntl.flock(slot, fcntl.LOCK_UN)
                        slot.close()
        return target


class PersistentDecoder:
    """One long-lived decoder child with an LMDB environment LRU."""
    def __init__(self, max_envs=8):
        self.envs = OrderedDict()
        self.max_envs = max(1, int(max_envs))

    def record(self, path, sample_id, encoding):
        path = NodeMediaCache(
            os.environ.get('YISU_MEDIA_LMDB_CACHE_DIR', '/tmp/yisu-lmdb-cache')).acquire(path)
        import lmdb
        env = self.envs.get(path)
        if env is None:
            page_size = os.sysconf('SC_PAGE_SIZE')
            map_size = max(2 * page_size, os.stat(path).st_size)
            env = lmdb.open(path, readonly=True, lock=False, subdir=False,
                            readahead=False, max_readers=1024, map_size=map_size)
            self.envs[path] = env
        self.envs.move_to_end(path)
        while len(self.envs) > self.max_envs:
            _, old = self.envs.popitem(last=False)
            old.close()
        with env.begin(buffers=True) as txn:
            raw = txn.get(struct.pack('<Q', int(sample_id)))
            if raw is None:
                raw = txn.get(str(sample_id).encode())
            if raw is None:
                raise KeyError(f'missing sample_id={sample_id}')
            payload = bytes(raw)
        return json.loads(payload) if encoding == 'json' else pickle.loads(payload)


def _stage(name, event='start'):
    # Always flush to the child's local stderr file; only replay successes when
    # debugging. No stderr PIPE that could fill while the parent waits.
    if _STAGE_CONTEXT is not None:
        print(f'[media-stage] {_STAGE_CONTEXT} stage={name} event={event} '
              f'elapsed={time.monotonic() - _STAGE_STARTED:.3f}s', file=sys.stderr, flush=True)


def _stage_call(name, function, *args, **kwargs):
    _stage(name)
    result = function(*args, **kwargs)
    _stage(name, 'done')
    return result


def _reap_stranded():
    for proc, directory in list(_STRANDED):
        if proc.poll() is not None:
            shutil.rmtree(directory, ignore_errors=True)
            _STRANDED.remove((proc, directory))


class MediaError(RuntimeError):
    """Base class for media failures with a stable machine-readable category."""
    category = 'executor_bug'

    def __init__(self, message, error_type=None, error_errno=None):
        super().__init__(message)
        self.error_type = error_type
        self.error_errno = error_errno


class MediaDecodeError(MediaError):
    """Compatibility base for failures caused by media content or input."""


class MediaCorruptError(MediaDecodeError):
    category = 'corrupt'


class MediaInvalidInputError(MediaDecodeError):
    category = 'invalid_input'


class MediaResourceError(MediaError):
    """Compatibility base for failures that must not be permanently blacklisted."""


class MediaTransientIOError(MediaResourceError):
    category = 'transient_io'


class MediaDeadlineError(TimeoutError, MediaError):
    category = 'deadline'

    def __init__(self, message, error_type=None, error_errno=None):
        MediaError.__init__(self, message, error_type, error_errno)


class MediaCapacityError(MediaResourceError):
    category = 'capacity'


class MediaExecutorBugError(MediaResourceError):
    category = 'executor_bug'


class MediaCacheExhausted(MediaCapacityError):
    """media cache 槽位耗尽: 与视频文件本身无关的瞬时容量错误, 不应拉黑文件。"""


class MediaCandidatesExhausted(MediaDecodeError):
    """All bounded replacement candidates failed within one sample deadline."""
    category = 'invalid_input'


def media_error_category(exc):
    category = getattr(exc, 'category', None)
    if category:
        return category
    if isinstance(exc, TimeoutError):
        return 'deadline'
    if isinstance(exc, MemoryError):
        return 'capacity'
    if isinstance(exc, FileNotFoundError):
        return 'invalid_input'
    if isinstance(exc, OSError):
        return 'transient_io'
    if isinstance(exc, (ValueError, KeyError, IndexError)):
        return 'invalid_input'
    return 'executor_bug'


def is_resource_error(exc):
    return media_error_category(exc) in ('capacity', 'executor_bug')


def blacklist_action(exc):
    """Return permanent, ttl, or none for a path-level failure."""
    category = media_error_category(exc)
    if category in ('corrupt', 'invalid_input'):
        return 'permanent'
    if category in ('transient_io', 'deadline'):
        return 'ttl'
    return 'none'


def _child_error_category(exc):
    if isinstance(exc, MediaError):
        return exc.category
    module = type(exc).__module__.split('.', 1)[0]
    if module in ('av', 'decord', 'PIL'):
        return 'corrupt'
    if isinstance(exc, MemoryError) or type(exc).__name__ in ('MapFullError', 'ReadersFullError'):
        return 'capacity'
    if isinstance(exc, FileNotFoundError):
        return 'invalid_input'
    if isinstance(exc, OSError):
        return 'transient_io'
    if isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
        return 'invalid_input'
    return 'executor_bug'


_ERROR_CLASSES = {
    'corrupt': MediaCorruptError,
    'invalid_input': MediaInvalidInputError,
    'transient_io': MediaTransientIOError,
    'deadline': MediaDeadlineError,
    'capacity': MediaCapacityError,
    'executor_bug': MediaExecutorBugError,
}


def _lmdb_cache_path(path):
    """Stage immutable LMDB bytes once per node, shared by all decode children."""
    if os.environ.get('YISU_MEDIA_LMDB_CACHE', '1') == '0':
        return path
    source = os.path.realpath(path)
    info = os.stat(source)
    root = os.environ.get('YISU_MEDIA_LMDB_CACHE_DIR', '/tmp/yisu-lmdb-cache')
    max_bytes = int(os.environ.get('YISU_MEDIA_LMDB_CACHE_MAX_BYTES', str(64 * 1024**3)))
    if max_bytes <= 0:
        raise ValueError('YISU_MEDIA_LMDB_CACHE_MAX_BYTES must be positive')
    if info.st_size > max_bytes:
        return path
    file_limit = int(os.environ.get('YISU_MEDIA_MAX_FILE_BYTES', str(512 * 1024**2)))
    if info.st_size > file_limit:
        return path
    os.makedirs(root, mode=0o700, exist_ok=True)
    key = hashlib.sha256(f'{source}|{info.st_size}|{info.st_mtime_ns}'.encode()).hexdigest()
    cached = os.path.join(root, f'{key}.mdb')
    lock_path = cached + '.lock'
    ready = cached + '.ready'
    lock = open(lock_path, 'a+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.isfile(cached) or not os.path.isfile(ready):
            temporary = f'{cached}.tmp.{os.getpid()}'
            try:
                with open(source, 'rb') as source_file, open(temporary, 'wb') as target:
                    shutil.copyfileobj(source_file, target, 16 * 1024**2)
                    target.flush()
                    os.fsync(target.fileno())
                if os.path.getsize(temporary) != info.st_size:
                    raise OSError('LMDB staging size changed during copy')
                os.replace(temporary, cached)
                Path(ready).touch()
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return cached
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


@contextmanager
def staged_file(path):
    """Bounded streaming copy, entirely inside the killable I/O child."""
    root = os.environ.get('YISU_MEDIA_CACHE_DIR', '/tmp/yisu-media-cache')
    limit = int(os.environ.get('YISU_MEDIA_MAX_FILE_BYTES', str(512 * 1024**2)))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    # deepspeed 启动路径不设 LOCAL_WORLD_SIZE, 按 1 算只给 8 槽; 节点上 8 rank x 2 worker
    # = 16 并发解码子进程, 8 槽必耗尽。默认 32 留足余量, 多实验各用独立 cache root 互不挤占。
    slots = int(os.environ.get('YISU_MEDIA_CACHE_SLOTS', str(max(32, 4 * local_world_size))))
    wait_seconds = float(os.environ.get('YISU_MEDIA_CACHE_WAIT_SECONDS', '2'))
    if limit <= 0 or slots <= 0 or not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError('media cache limits must be non-negative')
    os.makedirs(root, mode=0o700, exist_ok=True)
    lease = None
    directory = None
    deadline = time.monotonic() + wait_seconds
    if math.isfinite(_REQUEST_DEADLINE):
        deadline = min(deadline, _REQUEST_DEADLINE - 0.1)
        if deadline <= time.monotonic():
            raise MediaDeadlineError('media cache wait has no request deadline remaining')
    try:
        # Nonblocking cross-rank reservations; a D-state child keeps its lease.
        # 全部槽位被占时轮询等待 (D-state 持锁者释放或父进程超时杀掉本子进程),
        # 等待可被 kill 打断, 不会越过父进程的解码预算。
        while True:
            for slot in range(slots):
                candidate = open(os.path.join(root, f'slot-{slot}.lock'), 'a+b')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                lease = candidate
                directory = os.path.join(root, f'slot-{slot}')
                # Only the exclusive slot owner may remove stale data in this slot.
                shutil.rmtree(directory, ignore_errors=True)
                os.mkdir(directory, mode=0o700)
                break
            if lease is not None:
                break
            if time.monotonic() >= deadline:
                raise MediaCacheExhausted(
                    f'media cache capacity exhausted after {wait_seconds}s wait: '
                    f'root={root} slots={slots} (all slots held)')
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        local = os.path.join(directory, 'source')
        with open(path, 'rb') as source, open(local, 'wb') as target:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('media source is not a regular file')
            if info.st_size > limit:
                raise ValueError(f'media file exceeds {limit} bytes')
            copied = 0
            while True:
                chunk = source.read(min(1024**2, limit - copied + 1))
                if not chunk:
                    break
                copied += len(chunk)
                if copied > limit:
                    raise ValueError(f'media file exceeds {limit} bytes')
                target.write(chunk)
        yield local
    finally:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        if lease is not None:
            lease.close()


_PERSISTENT_DECODER = None


def read_record(path, sample_id, encoding='pickle'):
    global _PERSISTENT_DECODER
    import resource

    # Runs only in the timed I/O child. Explicit map_size avoids adopting a
    # persisted writer reservation, which can be much larger than the file.
    if _PERSISTENT_DECODER is not None:
        return _PERSISTENT_DECODER.record(str(path), sample_id, encoding)
    path = _stage_call('lmdb_stage', _lmdb_cache_path, path)
    page_size = os.sysconf('SC_PAGE_SIZE')
    map_size = max(2 * page_size, _stage_call('lmdb_stat', os.stat, path).st_size)
    map_size = (map_size + page_size - 1) // page_size * page_size
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft != resource.RLIM_INFINITY:
        allowance = soft + map_size
        resource.setrlimit(resource.RLIMIT_AS,
                           (min(allowance, hard) if hard != resource.RLIM_INFINITY else allowance, hard))
    if _PERSISTENT_DECODER is not None:
        try:
            payload_obj = _PERSISTENT_DECODER.record(str(path), sample_id, encoding)
            payload = pickle.dumps(payload_obj) if encoding != 'json' else json.dumps(payload_obj).encode()
            if len(payload) > 64 * 1024**2:
                raise ValueError('LMDB record exceeds 64 MiB')
            _stage('lmdb_open', 'start')
            _stage('lmdb_open', 'done')
            _stage('lmdb_get', 'start')
            _stage('lmdb_get', 'done')
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    else:
        import lmdb
        try:
            with _stage_call('lmdb_open', lmdb.open, str(path), readonly=True, lock=False, subdir=False,
                             readahead=False, max_readers=1024, map_size=map_size) as env:
                with _stage_call('lmdb_begin', env.begin, buffers=True) as txn:
                    raw = _stage_call('lmdb_get', txn.get, struct.pack('<Q', int(sample_id)))
                    if raw is None:
                        raw = _stage_call('lmdb_get_ascii', txn.get, str(sample_id).encode())
                    if raw is None:
                        raise KeyError(f'missing sample_id={sample_id}')
                    if len(raw) > 64 * 1024**2:
                        raise ValueError('LMDB record exceeds 64 MiB')
                    payload = _stage_call('lmdb_bytes', bytes, raw)
            _stage('lmdb_close', 'done')
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    # Close the mapping before deserializing/decoding under the original budget.
    # Lowering RLIMIT_AS below existing VMS succeeds on Linux but leaves no room
    # for subsequent allocations; fail explicitly rather than retaining slack.
    if soft != resource.RLIM_INFINITY:
        with open('/proc/self/statm', encoding='ascii') as status:
            current_vms = int(status.read().split()[0]) * page_size
        if current_vms > soft:
            raise MemoryError('LMDB record leaves virtual memory above child budget')
    return _stage_call('json_loads' if encoding == 'json' else 'pickle_loads',
                       json.loads if encoding == 'json' else pickle.loads, payload)


def read_video_record(root, lmdb_path, sample_id, published):
    import pyarrow.parquet as pq
    locator = read_record(os.path.join(root, lmdb_path), sample_id)
    relative = locator['parquet']
    if relative not in published or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('invalid unpublished video locator')
    path = os.path.realpath(os.path.join(root, relative))
    if os.path.commonpath([os.path.realpath(root), path]) != os.path.realpath(root):
        raise ValueError('video locator escapes dataset')
    if not Path(path).with_suffix('.ready.json').is_file():
        raise FileNotFoundError('video shard is not ready')
    # Iterate row groups, not the entire library/shard in worker memory.
    with pq.ParquetFile(path) as parquet:
        row = int(locator['row'])
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            if 0 <= row < size:
                result = parquet.read_row_group(group, columns=['video_path', 'caption']).slice(row, 1).to_pylist()[0]
                return dict(path=result['video_path'] or '', caption=(result['caption'] or '').strip())
            row -= size
    raise IndexError('video locator row out of range')


def decode_video_record(root, lmdb_path, sample_id, published, **decode_args):
    record = read_video_record(root, lmdb_path, sample_id, published)
    path = record['path']
    if not path:
        raise FileNotFoundError(f'sample_id={sample_id} missing video_path')
    with staged_file(path) as local:
        pixels = decode(local, **decode_args)
    return {
        'pixels': pixels,
        'path': np.asarray(path),
        'caption': np.asarray(record['caption']),
    }


def decode_av_record(root, lmdb_path, sample_id, **decode_args):
    record = read_record(os.path.join(root, lmdb_path), sample_id, encoding='json')
    if not record.get('has_audio') or record.get('audio') != record.get('video'):
        raise ValueError('当前只支持同容器内嵌音轨')
    path = Path(record['video'])
    if not path.is_absolute():
        path = Path(root) / path
    path = os.path.normpath(str(path))
    with staged_file(path) as local:
        pair = decode_pair(local, **decode_args)
    fields = record.get('fields') or json.loads(record.get('fields_json') or '{}')
    pair['record_fields'] = np.asarray(json.dumps(fields, ensure_ascii=False))
    pair['video_caption'] = np.asarray(str(record.get('video_caption') or '').strip())
    pair['media_path'] = np.asarray(path)
    return pair


def _bound_image(image, target_width, target_height):
    """超大图在完整解码/转换前先降采样, 避免 4K/亿级像素图把解码子进程内存打爆。

    draft() 对 JPEG 在解码阶段直接缩小 (解码峰值内存同步下降), 其余格式 no-op;
    thumbnail() 兜底把面积压到目标 4 倍 (长宽各约 2x) 以内, 保持纵横比。
    输出最终都会 resize_frames 到 bucket 分辨率, 提前降采样不改变训练语义。
    """
    if not target_width or not target_height:
        return image
    try:
        image.draft('RGB', (int(target_width), int(target_height)))
    except Exception:
        pass
    limit = 4.0 * target_width * target_height
    if image.width * image.height > limit:
        scale = math.sqrt(limit / (image.width * image.height))
        image.thumbnail((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return image


def read_images(path, width, height, condition_path=None, style="cover", records=None):
    cache = {} if records is None else records
    result = []
    for source in (path, condition_path) if condition_path else (path,):
        pixels = read_image(source, cache, target_width=width, target_height=height)
        h, w = pixels.shape[:2]
        result.append(resize_frames([pixels], width, height, w, h, style=style))
    return np.stack(result)


def read_image(path, records=None, target_width=None, target_height=None):
    import io
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 150_000_000
    if path.startswith('lmdb://'):
        rest = path[len('lmdb://'):]
        rest, kind = rest.rsplit(':', 1)
        root, stem, sid = rest.rsplit('/', 2)
        source = os.path.join(root, 'mdb', stem + '.mdb')
        if not _stage_call('lmdb_path_exists', os.path.exists, source):
            source = os.path.join(root, stem + '.mdb')
        key = (source, int(sid))
        record = records.get(key) if records is not None else None
        if record is None:
            record = read_record(source, int(sid))
            if records is not None:
                records[key] = record
        fields = {'single': ('img_bytes', 'image_bytes', 'img', 'jpg', 'data'),
                  'before': ('img_before', 'img_before_bytes', 'input_img', 'img_input', 'before'),
                  'after': ('img_after', 'img_after_bytes', 'output_img', 'img_output', 'after', 'img_bytes')}
        field = next(key for key in fields[kind] if key in record)
        with _stage_call('pil_open', Image.open, io.BytesIO(record[field])) as image:
            if image.width * image.height > 150_000_000:
                raise ValueError('image exceeds pixel limit')
            _stage_call('image_bound', _bound_image, image, target_width, target_height)
            rgb = _stage_call('pil_decode_rgb', image.convert, 'RGB')
            return _stage_call('image_array', np.asarray, rgb)
    _stage('image_stage_file')
    with staged_file(path) as local, _stage_call('pil_open', Image.open, local) as image:
        if image.width * image.height > 150_000_000:
            raise ValueError('image exceeds pixel limit')
        _stage_call('image_bound', _bound_image, image, target_width, target_height)
        rgb = _stage_call('pil_decode_rgb', image.convert, 'RGB')
        return _stage_call('image_array', np.asarray, rgb)


class DecodeBudget:
    """Share one sample's wall-clock budget fairly across bounded candidates."""
    def __init__(self, seconds, candidates, file_timeout):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('sample_timeout_seconds 必须为有限正数')
        self.deadline = time.monotonic() + seconds
        self.candidates = candidates
        self.file_timeout = file_timeout

    def timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.candidates <= 0:
            raise MediaDeadlineError('sample decode budget exhausted')
        timeout = min(self.file_timeout, remaining / self.candidates)
        self.candidates -= 1
        return timeout


def candidate_indices(index, pool, limit, rng=random):
    yield index
    # Sample positions rather than copying a potentially million-row bucket.
    emitted = 1
    for position in rng.sample(range(len(pool)), min(len(pool), limit)):
        if emitted >= limit:
            break
        candidate = pool[position]
        if candidate != index:
            emitted += 1
            yield candidate


_MISSING = object()


class BadPathBook:
    """进程内坏文件黑名单: 硬错误永久拉黑, 超时只拉黑 TTL 秒后自动过期。

    NFS 抖动会让健康文件解码超时; 若超时也永久拉黑, 好文件会被无声吞掉
    (AV 样本占比崩塌)。因此超时仅 TTL 拉黑, 到期自动重试;
    损坏/缺音轨等硬错误不会自愈, 保持永久。
    """

    def __init__(self, timeout_ttl_seconds: float = 600.0):
        self.timeout_ttl_seconds = float(timeout_ttl_seconds)
        self._entries = {}  # path -> None(永久) 或 monotonic 过期时间戳

    def add_permanent(self, path) -> None:
        self._entries[str(path)] = None

    def add_timeout(self, path) -> None:
        if self._entries.get(str(path), _MISSING) is None:
            return  # 已永久拉黑的文件不降级为 TTL
        self._entries[str(path)] = time.monotonic() + self.timeout_ttl_seconds

    def _expired(self, expiry) -> bool:
        return expiry is not None and expiry <= time.monotonic()

    def __contains__(self, path) -> bool:
        expiry = self._entries.get(str(path), _MISSING)
        if expiry is _MISSING:
            return False
        if self._expired(expiry):
            del self._entries[str(path)]
            return False
        return True

    def __len__(self) -> int:
        for path, expiry in list(self._entries.items()):
            if self._expired(expiry):
                del self._entries[path]
        return len(self._entries)


def decode_stagger_sleep():
    """Optional legacy jitter, disabled by default; staging leases bound concurrency."""
    limit = float(os.environ.get('YISU_DECODE_JITTER_SECONDS', '0') or 0)
    if limit <= 0:
        return
    time.sleep(random.uniform(0, limit))


def resize_frames(frames, out_w, out_h, src_w, src_h, style="cover"):
    from PIL import Image

    if style == "resize_long_edge":
        # 长边贴桶、保持纵横比不裁剪, 短边居中补黑: 避免中心裁剪切掉 caption 描述的主体
        scale = min(out_w / src_w, out_h / src_h)
        scaled_w = min(out_w, max(1, round(src_w * scale)))
        scaled_h = min(out_h, max(1, round(src_h * scale)))
        left, top = (out_w - scaled_w) // 2, (out_h - scaled_h) // 2
        out = np.zeros((3, len(frames), out_h, out_w), dtype=np.uint8)
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            if img.size != (scaled_w, scaled_h):
                img = img.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
            out[:, i, top:top + scaled_h, left:left + scaled_w] = np.asarray(img).transpose(2, 0, 1)
        return out

    # 默认 cover: 短边贴桶 + 中心裁剪 (与旧行为完全一致)
    scale = max(out_w / src_w, out_h / src_h)
    scaled_w, scaled_h = round(src_w * scale), round(src_h * scale)
    left, top = (scaled_w - out_w) // 2, (scaled_h - out_h) // 2
    # 预分配最终数组逐帧填充, 避免 planes 列表 + np.stack + ascontiguousarray
    # 三重复制: f121_1280x768 桶单份输出 ~340 MiB, 三份叠加会顶破子进程 RLIMIT_AS
    out = np.empty((3, len(frames), out_h, out_w), dtype=np.uint8)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        if img.size != (scaled_w, scaled_h):
            img = img.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
        crop = np.asarray(img.crop((left, top, left + out_w, top + out_h)), dtype=np.uint8)
        out[:, i] = crop.transpose(2, 0, 1)
    return out


def decode_pyav(path, indices):
    """按目标帧号取帧; 优先关键帧 seek, PTS 异常时回退全量扫描 (与旧行为一致)."""
    wanted = [int(i) for i in indices]
    frames = _pyav_collect_frames(path, wanted, seek=True)
    if frames is None:
        frames = _pyav_collect_frames(path, wanted, seek=False)
    return frames


def _pyav_collect_frames(path, wanted, seek):
    import av

    wanted_set = set(wanted)
    by_index = {}
    last_frame = None
    last_seen = -1
    with av.open(path, mode="r") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if seek:
            rate = stream.average_rate
            fps = float(rate) if rate else 0.0
            time_base = float(stream.time_base) if stream.time_base else 0.0
            if fps <= 0 or time_base <= 0 or wanted[0] <= 0:
                return None  # 无法换算 seek 时间, 交给全量扫描
            try:
                container.seek(
                    int(round(wanted[0] / fps / time_base)),
                    stream=stream, backward=True, any_frame=False,
                )
            except Exception:
                return None  # seek 不支持/失败 → 全量扫描兜底
        for frame in container.decode(stream):
            if seek:
                # seek 后 enumerate 不再等于绝对帧号, 用 PTS 换算;
                # 缺 time 的流视为异常, 回退全量扫描
                if frame.time is None:
                    return None
                frame_index = int(round(float(frame.time) * (float(stream.average_rate) if stream.average_rate else 0.0)))
            else:
                frame_index = last_seen + 1
            last_seen = frame_index
            # swscale threads are independent of codec_context.thread_count.
            last_frame = frame.to_ndarray(format="rgb24", threads=1)
            if frame_index in wanted_set:
                by_index[frame_index] = last_frame
            if frame_index >= wanted[-1]:
                if len(by_index) == len(wanted_set):
                    break
                if seek:
                    return None  # 已越过末个目标帧仍有缺口 → PTS 异常, 全量扫描兜底
    if last_frame is None:
        if seek:
            return None  # seek 路径未解出帧 → 全量扫描兜底
        raise ValueError(f"PyAV 未解出任何视频帧: {path}")
    return [by_index.get(int(index), last_frame) for index in wanted]


def decode(path, start, count, out_w, out_h, src_w, src_h, source_fps, target_fps, codec=None):
    import av

    name = str(codec or "").strip().lower()
    tag = ""
    if not name:
        # 索引期已带 video_codec 时直接路由, 免去仅为探测编码而打开容器
        with av.open(path, mode="r") as container:
            stream = container.streams.video[0]
            name = (stream.codec_context.name or "").lower()
            tag = (getattr(stream, "codec_tag", "") or "").lower()
        print(f"{name}:{tag}", flush=True)
    indices = np.rint((start + np.arange(count, dtype=np.float64)) * source_fps / target_fps).astype(np.int64)
    if name in {"av1", "av01", "libdav1d"} or tag in {"av1", "av01"}:
        frames = decode_pyav(path, indices)
    else:
        try:
            import decord
            reader = decord.VideoReader(path, num_threads=2)
            total = len(reader)
        except Exception:
            frames = decode_pyav(path, indices)
        else:
            target_duration = max(1, int(round(max(1, total) * target_fps / source_fps)))
            start = max(0, min(start, target_duration - count))
            indices = np.rint((start + np.arange(count, dtype=np.float64)) / target_fps * source_fps).astype(np.int64)
            indices = np.clip(indices, 0, total - 1)
            try:
                frames = reader.get_batch(indices.tolist()).asnumpy()
            except Exception:
                frames = decode_pyav(path, indices)
    return resize_frames(frames, out_w, out_h, src_w, src_h)


def decode_pair(path, frames, fps, height, width, random_crop=True, sample_rate=16000):
    """PTS-clock crop: F frames cover [start, start + F/fps), never retime audio."""
    import av

    video, audio = [], []
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError('样本必须同时有音轨和视频轨')
        vs, aus = container.streams.video[0], container.streams.audio[0]
        vs.codec_context.thread_count = 2
        aus.codec_context.thread_count = 2
        print(f'{vs.codec_context.name}+{aus.codec_context.name}', flush=True)
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=sample_rate)
        for packet in container.demux(vs, aus):
            for frame in packet.decode():
                if frame.pts is None:
                    raise ValueError('无 PTS，无法保证同步')
                if isinstance(frame, av.VideoFrame):
                    video.append(float(frame.pts * frame.time_base))
                else:
                    for chunk in resampler.resample(frame):
                        audio.append((float(chunk.pts * chunk.time_base), chunk.samples))
        for chunk in resampler.resample(None):
            audio.append((float(chunk.pts * chunk.time_base), chunk.samples))
    if len(video) < 2 or not audio:
        raise ValueError('解码后音视频为空')
    video.sort()
    audio.sort(key=lambda p: p[0])
    vt = np.array(video)
    frame_period = float(np.median(np.diff(vt)))
    duration = frames / fps
    samples = round(duration * sample_rate)
    begin = max(vt[0], audio[0][0])
    audio_end = max(t + size / sample_rate for t, size in audio)
    end = min(vt[-1] + frame_period, audio_end)
    # Align to the AUDIO origin, not absolute zero: nonzero PTS need not be an
    # integral target-rate sample. Integer bounds avoid ceil/floor escaping begin.
    origin = audio[0][0]
    first = math.ceil((begin - origin) * sample_rate - 1e-6)
    last = math.floor((end - origin) * sample_rate - max(samples, duration * sample_rate) + 1e-6)
    if last < first:
        # F samples span (F-1)/fps; the final display interval is still part
        # of the F/fps audio window. Permit at most one target interval of
        # tail padding, never stretch audio to fill that interval.
        if end - begin < (frames - 1) / fps - 1 / sample_rate:
            raise ValueError(f'共同有效时长 {end - begin:.6f}s 小于 {(frames - 1) / fps:.6f}s')
        last = first
    start = origin + (random.randint(first, last) if random_crop else first) / sample_rate
    times = start + np.arange(frames) / fps
    right = np.searchsorted(vt, times).clip(0, len(vt) - 1)
    left = (right - 1).clip(0, len(vt) - 1)
    indices = np.where(np.abs(vt[left] - times) <= np.abs(vt[right] - times), left, right)
    if np.max(np.abs(vt[indices] - times)) > max(frame_period * 1.5, 1 / fps):
        raise ValueError('视频 PTS 有过大间隙')
    # First pass retains only clocks, never decoded media. Replay sequentially to
    # preserve resampler phase and nonzero origins; retain only the chosen window.
    pixels = np.empty((3, frames, height, width), dtype=np.uint8)
    waveform = np.zeros(samples, dtype=np.float32)
    covered = np.zeros(samples, dtype=bool)
    wanted = {}
    for position, timestamp in enumerate(vt[indices]):
        wanted.setdefault(float(timestamp), []).append(position)
    remaining = set(wanted)

    def collect_audio(chunk):
        t = float(chunk.pts * chunk.time_base)
        offset = round((t - start) * sample_rate)
        lo, hi = max(0, offset), min(samples, offset + chunk.samples)
        if hi > lo:
            values = chunk.to_ndarray().reshape(-1)
            waveform[lo:hi] = values[lo-offset:hi-offset]
            covered[lo:hi] = True
        return t + chunk.samples / sample_rate >= start + duration

    with av.open(str(path)) as container:
        vs, aus = container.streams.video[0], container.streams.audio[0]
        vs.codec_context.thread_count = aus.codec_context.thread_count = 2
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=sample_rate)
        audio_done = False
        for packet in container.demux(vs, aus):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    t = float(frame.pts * frame.time_base)
                    if t in remaining:
                        rgb = frame.to_ndarray(format='rgb24', threads=1)
                        h, w = rgb.shape[:2]
                        resized = resize_frames([rgb], width, height, w, h)[:, 0]
                        for position in wanted[t]:
                            pixels[:, position] = resized
                        remaining.remove(t)
                else:
                    for chunk in resampler.resample(frame):
                        audio_done = collect_audio(chunk) or audio_done
            if not remaining and audio_done:
                break
        else:
            for chunk in resampler.resample(None):
                collect_audio(chunk)
    if remaining:
        raise ValueError('视频 PTS 窗口重放缺帧')
    valid_end = min(samples, max(0, round((audio_end - start) * sample_rate)))
    if (not covered[:valid_end].all() or samples - valid_end > math.ceil(sample_rate / fps)
            or not np.isfinite(waveform).all()):
        raise ValueError('音频 PTS 有内部间隙、过长尾部或非有限值')
    return dict(video=pixels, audio=waveform[None], start_seconds=start,
                duration_seconds=duration, video_timestamps=vt[indices],
                sample_rate=sample_rate)


class _PersistentDecoderClient:
    def __init__(self):
        self.proc = None

    def _start(self):
        self.proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), '--persistent'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
            env=dict(os.environ, YISU_MEDIA_LMDB_CACHE='0'),
        )

    def close(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream is not None:
                    stream.close()
            self.proc = None

    def request(self, request, timeout):
        if self.proc is None or self.proc.poll() is not None:
            self.close()
            self._start()
        try:
            self.proc.stdin.write(json.dumps(request) + '\n')
            self.proc.stdin.flush()
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
            if not ready:
                self.close()
                raise MediaDeadlineError(f'persistent decoder timeout {timeout}s', 'TimeoutExpired')
            line = self.proc.stdout.readline()
            if not line:
                self.close()
                raise MediaExecutorBugError('persistent decoder exited without response')
            response = json.loads(line)
            if not response.get('ok'):
                category = response.get('category', 'executor_bug')
                error_class = _ERROR_CLASSES.get(category, MediaExecutorBugError)
                raise error_class(response.get('error', 'persistent decoder failed'), response.get('type'))
            return response['result']
        except (BrokenPipeError, EOFError, json.JSONDecodeError) as exc:
            self.close()
            raise MediaExecutorBugError(str(exc)) from exc


_PERSISTENT_CLIENT = None


def _persistent_decode_request(request, timeout):
    global _PERSISTENT_CLIENT
    if _PERSISTENT_CLIENT is None:
        _PERSISTENT_CLIENT = _PersistentDecoderClient()
    return _PERSISTENT_CLIENT.request(request, timeout)


def decode_isolated(request, timeout, context):
    """Use a resident LMDB decoder for metadata; isolate native media decoding."""
    persistent_images = (os.environ.get('YISU_IMAGE_PERSISTENT_DECODER', '0') == '1'
                         and request.get('kind') in ('image', 'images'))
    if persistent_images:
        # 常驻子进程解码图像: 省去每样本 fork + import + lmdb env open;
        # 大数组经 /dev/shm npy 回传; 任何故障回退到一次性隔离路径, 保持候选重试语义。
        request = dict(request)
        request.pop('_trace_context', None)
        output = tempfile.mkstemp(prefix='yisu-img-', suffix='.npy', dir='/dev/shm')[1]
        try:
            _persistent_decode_request(dict(request, _output_path=output), timeout)
            pixels = np.load(output, allow_pickle=False)
            return pixels
        except Exception:
            # 常驻通道故障或样本级解码失败 → 一次性路径给出权威的 categorized error
            pass
        finally:
            try:
                os.unlink(output)
            except OSError:
                pass
    elif (os.environ.get('YISU_MEDIA_PERSISTENT_DECODER', '0') == '1'
            and request.get('kind') in ('record', 'video_record')):
        request = dict(request)
        request.pop('_trace_context', None)
        return _persistent_decode_request(request, timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('isolated I/O timeout must be finite and positive')
    import socket
    started = time.monotonic()
    debug = os.environ.get('YISU_MEDIA_DEBUG', '0') == '1'
    context = (f'{context} host={socket.gethostname()} rank={os.environ.get("RANK", "unknown")} '
               f'worker_pid={os.getpid()} timeout={timeout:.3f}s')
    request = dict(request)
    request['_trace_context'] = context
    request['_deadline'] = started + timeout
    _reap_stranded()
    if len(_STRANDED) >= _MAX_STRANDED:
        raise MediaResourceError('isolated I/O stranded process limit reached')
    # Do not remove files under a live D-state child, and never wait without a deadline.
    directory = tempfile.mkdtemp(prefix="yisu-video-", dir="/tmp")
    proc = None
    try:
        output = str(Path(directory) / "pixels.npy")
        with tempfile.TemporaryFile(dir=directory) as errors:
            errors.write(f'[media-stage] {context} uri={request["path"]} stage=child_launch event=start elapsed=0s\n'.encode())
            errors.flush()
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), output],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, env=env,
            )
            timed_out = False
            try:
                try:
                    stdout, _ = proc.communicate(json.dumps(request).encode(), timeout=max(0, timeout - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc.kill()
                    stdout = b''
            finally:
                if proc.poll() is None:
                    proc.kill()
                try:
                    proc.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
                for pipe in (proc.stdin, proc.stdout):
                    if pipe is not None:
                        pipe.close()
            elapsed = time.monotonic() - started
            codec = stdout.decode(errors="replace").strip()[:128] or "unknown"
            detail = f"{context} child_pid={proc.pid} path={request['path']} codec={codec} elapsed={elapsed:.3f}s"
            if timed_out or proc.returncode:
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 16384))
                error = errors.read().decode(errors="replace").strip()
                message = f"[video] {detail} error={'decode timeout ' + str(timeout) + 's' if timed_out else 'exit ' + str(proc.returncode)} {error}"
                print(message, flush=True)
                if timed_out:
                    raise MediaDeadlineError(message, 'TimeoutExpired')
                try:
                    with open(output + '.error', encoding='utf-8') as handle:
                        failure = json.load(handle)
                except (OSError, ValueError):
                    failure = {}
                category = failure.get('category', 'executor_bug')
                error_type = failure.get('type')
                error_errno = failure.get('errno')
                if category == 'capacity' and error_type == 'MediaCacheExhausted':
                    raise MediaCacheExhausted(message, error_type, error_errno)
                error_class = _ERROR_CLASSES.get(category, MediaExecutorBugError)
                raise error_class(message, error_type, error_errno)
            if debug:
                errors.seek(0)
                print(errors.read().decode(errors='replace').strip(), flush=True)
                print(f'[media] stage=parent_output_load {detail}', flush=True)
            if elapsed > 5 and (request.get('kind') != 'image' or debug):
                print(f"[video] slow decode {detail}", flush=True)
            if os.path.getsize(output) > int(os.environ.get('YISU_MEDIA_MAX_OUTPUT_BYTES', str(512 * 1024**2))):
                raise ValueError('isolated output exceeds size limit')
            if request.get('kind') in ('record', 'video_record'):
                with open(output, encoding='utf-8') as result:
                    return json.load(result)
            if request.get('kind') in ('av', 'video_record_decode', 'av_record_decode'):
                with np.load(output, allow_pickle=False) as result:
                    return {key: result[key] for key in result.files}
            return np.load(output, allow_pickle=False)
    finally:
        if proc is not None and proc.poll() is None:
            _STRANDED.append((proc, directory))
        else:
            shutil.rmtree(directory, ignore_errors=True)


def persistent_main():
    import resource
    # 与一次性 child 同源的资源护栏, 但 RLIMIT_AS 放宽: LMDB env 按 mdb 全尺寸 mmap,
    # 虚拟地址空间不等于物理占用 (touched pages 才计数), 2GB 会立刻爆掉。
    memory_limit = int(os.environ.get('YISU_MEDIA_PERSISTENT_MEMORY_BYTES', str(64 * 1024**3)))
    if memory_limit > 0:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = min(memory_limit, hard) if hard != resource.RLIM_INFINITY else memory_limit
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    decoder = PersistentDecoder(os.environ.get('YISU_MEDIA_LMDB_MAX_ENVS', '8'))
    global _PERSISTENT_DECODER
    _PERSISTENT_DECODER = decoder

    class _LruRecords(dict):
        """read_image 直接以 records[key]=record 写入; 插入即驱逐, 防止常驻子进程内存无界。"""
        def __init__(self, max_entries):
            super().__init__()
            self._max = max(0, int(max_entries))
            self._order = OrderedDict()

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self._order[key] = None
            self._order.move_to_end(key)
            while self._max and len(self._order) > self._max:
                oldest, _ = self._order.popitem(last=False)
                super().__delitem__(oldest)

    # 图像 record LRU: 过采样桶的重复样本命中缓存跳过重复 LMDB 读; 含 img_bytes, 有界
    record_cache = _LruRecords(os.environ.get('YISU_IMAGE_RECORD_CACHE', '64'))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request.pop('kind', 'record')
            if kind == 'record':
                result = read_record(**request)
            elif kind == 'video_record':
                request['root'] = request.pop('path')
                result = read_video_record(**request)
            elif kind in ('image', 'images'):
                output_path = request.pop('_output_path')
                request['records'] = record_cache
                pixels = read_image(**request) if kind == 'image' else read_images(**request)
                np.save(output_path, pixels, allow_pickle=False)
                print(json.dumps({'ok': True, 'result': {'output': output_path}}), flush=True)
                continue
            else:
                raise ValueError(f'unsupported persistent kind: {kind}')
            print(json.dumps({'ok': True, 'result': result}), flush=True)
        except Exception as exc:
            print(json.dumps({'ok': False, 'category': _child_error_category(exc),
                              'type': type(exc).__name__, 'error': str(exc)}), flush=True)


def child_main():
    import resource

    memory_limit = int(os.environ.get('YISU_MEDIA_CHILD_MEMORY_BYTES', str(2 * 1024**3)))
    output_limit = int(os.environ.get('YISU_MEDIA_MAX_OUTPUT_BYTES', str(512 * 1024**2)))
    file_limit = int(os.environ.get('YISU_MEDIA_MAX_FILE_BYTES', str(512 * 1024**2)))
    if min(memory_limit, output_limit, file_limit) <= 0:
        raise ValueError('media resource limits must be positive')
    for resource_id, limit in ((resource.RLIMIT_AS, memory_limit),
                               (resource.RLIMIT_FSIZE, max(file_limit, output_limit))):
        soft, hard = resource.getrlimit(resource_id)
        resource.setrlimit(resource_id, (min(limit, hard) if hard != resource.RLIM_INFINITY else limit, hard))
    global _STAGE_CONTEXT, _STAGE_STARTED, _REQUEST_DEADLINE
    request = json.load(sys.stdin)
    _REQUEST_DEADLINE = request.pop('_deadline', float('inf'))
    _STAGE_STARTED = time.monotonic()
    _STAGE_CONTEXT = (f'{request.pop("_trace_context", "")} child_pid={os.getpid()} '
                      f'uri={request.get("path")}')
    _stage('child_request')
    kind = request.pop('kind', 'video')
    if kind == 'record':
        result = read_record(**request)
    elif kind == 'video_record':
        request['root'] = request.pop('path')
        result = read_video_record(**request)
    elif kind == 'image':
        result = read_image(**request)
    elif kind == 'images':
        result = read_images(**request)
    elif kind == 'video_record_decode':
        request['root'] = request.pop('path')
        result = decode_video_record(**request)
    elif kind == 'av_record_decode':
        request['root'] = request.pop('path')
        result = decode_av_record(**request)
    elif kind in ('video', 'av'):
        with staged_file(request['path']) as local:
            request['path'] = local
            result = decode_pair(**request) if kind == 'av' else decode(**request)
    else:
        raise ValueError(f'unknown isolated I/O kind: {kind}')
    _stage('output_write')
    if kind in ('record', 'video_record'):
        payload = json.dumps(result).encode()
        if len(payload) > output_limit:
            raise ValueError('record exceeds output limit')
        with open(sys.argv[1], 'wb') as output:
            output.write(payload)
    else:
        arrays = result.values() if isinstance(result, dict) else (result,)
        if sum(np.asarray(value).nbytes for value in arrays) > output_limit - 65536:
            raise ValueError('decoded media exceeds output limit')
        with open(sys.argv[1], 'wb') as output:
            if kind in ('av', 'video_record_decode', 'av_record_decode'):
                np.savez(output, **result)
            else:
                _stage_call('output_npy', np.save, output, result, allow_pickle=False)
    _stage('output_write', 'done')


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--persistent':
            persistent_main()
        else:
            child_main()
    except Exception as exc:
        category = _child_error_category(exc)
        try:
            with open(sys.argv[1] + '.error', 'w', encoding='utf-8') as handle:
                json.dump(dict(category=category, type=type(exc).__name__,
                               errno=getattr(exc, 'errno', None)), handle)
        except Exception:
            pass
        raise
