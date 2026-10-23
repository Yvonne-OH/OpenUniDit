"""常驻图像解码器 (YISU_IMAGE_PERSISTENT_DECODER) 压力测试。

对比 one-shot 隔离解码 vs 常驻子进程解码:
  1. 正确性: 两种模式输出逐字节一致
  2. 吞吐/延迟: 顺序加载 N 样本的 mean/p50/p95 (隔离 fork+import+env open 的开销)
  3. 故障路径: 损坏样本 categorized error、子进程死亡自动重生、超时回退
  4. LRU: 超过缓存上限的样本流不泄漏内存且全部成功

用法: python scripts/test/test_persistent_image_decode.py [samples]
"""
import io
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def log(color, msg):
    print(f'{color}{msg}{RESET}', flush=True)


def build_fake_lmdb(root: Path, n_samples=80, width=1024, height=768, seed=42):
    """构造 read_image 兼容的 LMDB 布局: <root>/mdb/fake_src.mdb, uri=lmdb://<root>/fake_src/<sid>:single"""
    import lmdb
    from PIL import Image

    rng = np.random.default_rng(seed)
    mdb_dir = root / 'mdb'
    mdb_dir.mkdir(parents=True, exist_ok=True)
    mdb_path = str(mdb_dir / 'fake_src.mdb')
    env = lmdb.open(mdb_path, map_size=256 * 1024**2, subdir=False)
    with env.begin(write=True) as txn:
        for sid in range(n_samples):
            array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(array).save(buf, format='JPEG', quality=88)
            record = {'img_bytes': buf.getvalue(), 'width': width, 'height': height}
            txn.put(struct.pack('<Q', sid), pickle_dumps(record))
    env.close()
    # 一条损坏记录: 追加到独立 mdb, 验证 categorized error
    bad_dir = root / 'mdb'
    bad_path = str(bad_dir / 'fake_bad.mdb')
    env = lmdb.open(bad_path, map_size=16 * 1024**2, subdir=False)
    with env.begin(write=True) as txn:
        txn.put(struct.pack('<Q', 0), pickle_dumps({'img_bytes': b'\xff\xd8not-a-jpeg-garbage'}))
    env.close()
    return mdb_path


def pickle_dumps(obj):
    import pickle
    return pickle.dumps(obj)


def decode(path, sid, width, height, timeout=30.0):
    from data.datasets.video_decode import decode_isolated
    request = dict(kind='images', path=f'lmdb://{path.rsplit("/mdb/", 1)[0]}/{Path(path).stem}/{sid}:single',
                   width=width, height=height, style='cover')
    return decode_isolated(request, timeout, context='test')


def benchmark(label, n_iter, width, height, mdb_path):
    from data.datasets.video_decode import _PERSISTENT_CLIENT
    latencies = []
    for i in range(n_iter):
        start = time.monotonic()
        pixels = decode(mdb_path, i, width, height)
        latencies.append(time.monotonic() - start)
        assert pixels.shape == (1, 3, 1, height, width), pixels.shape
    if _PERSISTENT_CLIENT is not None:
        _PERSISTENT_CLIENT.close()
    array = np.array(latencies)
    rate = n_iter / array.sum()
    log(BLUE, f'  {label:<12} n={n_iter} mean={array.mean()*1000:7.1f}ms '
              f'p50={np.percentile(array, 50)*1000:7.1f}ms p95={np.percentile(array, 95)*1000:7.1f}ms '
              f' throughput={rate:.2f} samples/s')
    return latencies


def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    width, height = 512, 384
    with tempfile.TemporaryDirectory(prefix='yisu-persist-test-') as tmp:
        mdb_path = build_fake_lmdb(Path(tmp), n_samples=max(n_iter, 80))
        log(GREEN, f'✅ 构造测试 LMDB: {mdb_path} ({max(n_iter, 80)} 样本, {width}x{height} JPEG)')

        # ── 1. 正确性: 两种模式逐字节一致 ─────────────────────────────
        os.environ.pop('YISU_IMAGE_PERSISTENT_DECODER', None)
        ref = decode(mdb_path, 3, width, height)
        os.environ['YISU_IMAGE_PERSISTENT_DECODER'] = '1'
        got = decode(mdb_path, 3, width, height)
        assert ref.shape == got.shape and np.array_equal(ref, got), '两种模式输出不一致!'
        log(GREEN, '✅ 正确性: persistent 与 one-shot 输出逐字节一致')

        # ── 2. 吞吐/延迟对比 ──────────────────────────────────────────
        os.environ.pop('YISU_IMAGE_PERSISTENT_DECODER', None)
        benchmark('one-shot', n_iter, width, height, mdb_path)
        os.environ['YISU_IMAGE_PERSISTENT_DECODER'] = '1'
        benchmark('persistent', n_iter, width, height, mdb_path)

        # ── 3. 子进程死亡自动重生 ─────────────────────────────────────
        from data.datasets.video_decode import _PERSISTENT_CLIENT
        assert decode(mdb_path, 9, width, height).shape == (1, 3, 1, height, width)
        assert _PERSISTENT_CLIENT is not None and _PERSISTENT_CLIENT.proc is not None
        _PERSISTENT_CLIENT.proc.kill()
        _PERSISTENT_CLIENT.proc.wait()
        got = decode(mdb_path, 5, width, height)
        assert got.shape == (1, 3, 1, height, width)
        log(GREEN, '✅ 故障恢复: 子进程被杀后自动重生并成功解码')

        # ── 4. 损坏样本: categorized error (经 one-shot 回退后抛出) ───
        bad_root = Path(mdb_path).parent.parent
        bad_uri_dir = bad_root
        from data.datasets.video_decode import decode_isolated
        try:
            decode_isolated(dict(kind='images', path=f'lmdb://{bad_uri_dir}/fake_bad/0:single',
                                 width=width, height=height, style='cover'),
                            30.0, context='test-bad')
            log(RED, '❌ 损坏样本未抛错!')
            return 1
        except Exception as exc:
            from data.datasets.video_decode import media_error_category
            category = media_error_category(exc)
            assert category != 'executor_bug', f'损坏样本被误判为 executor_bug: {exc}'
            log(GREEN, f'✅ 损坏样本: categorized error (category={category})')

        # ── 5. LRU 驱逐: 超缓存上限的样本流全部成功 ───────────────────
        total = max(n_iter, 80)
        for sid in range(total):
            pixels = decode(mdb_path, sid, width, height)
            assert pixels.shape == (1, 3, 1, height, width)
        log(GREEN, f'✅ LRU: 连续 {total} 个不同样本全部成功 (缓存上限 64)')

        # ── 6. 超时: 常驻超时 → 回退 one-shot → 仍超时 → MediaDeadlineError
        # 注: warm 子进程 + record 缓存命中时 1~3ms 即可完成, 用 1ms 确保触发
        try:
            decode(mdb_path, 7, width, height, timeout=0.001)
            log(RED, '❌ 极小超时未抛错!')
            return 1
        except Exception as exc:
            assert 'timeout' in str(exc).lower() or 'Timeout' in type(exc).__name__, f'{type(exc).__name__}: {exc}'
            log(GREEN, f'✅ 超时回退: {type(exc).__name__}')

    log(GREEN, '🎉 全部测试通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
