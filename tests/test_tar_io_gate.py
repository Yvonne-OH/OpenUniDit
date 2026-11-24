"""CPU-only lease tests; source data and production cache are never accessed."""
import contextlib
import importlib.util
import io
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid


SOURCE = Path(os.environ.get('GIGA_VIDEO_DECODE_TEST_SOURCE',
                  Path(__file__).resolve().parents[1] / 'data/datasets/video_decode.py'))
spec = importlib.util.spec_from_file_location('tar_gate_decode_test', SOURCE)
decode = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = decode
spec.loader.exec_module(decode)


def concurrent_reader(namespace, count, barrier, active, maximum, guard, results):
    os.environ['YISU_TAR_IO_NAMESPACE'] = namespace
    os.environ['YISU_TAR_IO_CONCURRENCY'] = str(count)
    decode._REQUEST_DEADLINE = time.monotonic() + 15
    try:
        barrier.wait(timeout=10)
        with decode._tar_io_lease():
            with guard:
                active.value += 1
                maximum.value = max(maximum.value, active.value)
            try:
                time.sleep(.06)
            finally:
                with guard:
                    active.value -= 1
        results.put(None)
    except BaseException as exc:
        results.put(repr(exc))


def hold_lease(namespace, acquired):
    os.environ['YISU_TAR_IO_NAMESPACE'] = namespace
    os.environ['YISU_TAR_IO_CONCURRENCY'] = '1'
    decode._REQUEST_DEADLINE = time.monotonic() + 15
    with decode._tar_io_lease():
        acquired.set()
        time.sleep(60)


class TarIOGateTests(unittest.TestCase):
    def setUp(self):
        self.namespace = 'tar-gate-test-' + uuid.uuid4().hex
        self.environment = patch.dict(os.environ, {
            'YISU_TAR_IO_NAMESPACE': self.namespace,
            'YISU_TAR_IO_CONCURRENCY': '1',
        })
        self.environment.start()
        self.old_deadline = decode._REQUEST_DEADLINE
        decode._REQUEST_DEADLINE = time.monotonic() + 15
        self.children = []

    def tearDown(self):
        for proc in self.children:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=5)
        for (namespace, _), paths in list(decode._TAR_IO_SLOT_CACHE.items()):
            if namespace.startswith(self.namespace):
                shutil.rmtree(Path(paths[0]).parent, ignore_errors=True)
                decode._TAR_IO_SLOT_CACHE.pop((namespace, len(paths)), None)
        decode._REQUEST_DEADLINE = self.old_deadline
        self.environment.stop()

    def test_default_disabled_keeps_old_behavior(self):
        os.environ.pop('YISU_TAR_IO_CONCURRENCY')
        decode._REQUEST_DEADLINE = 0
        with patch.object(decode, '_tar_io_slot_paths', side_effect=AssertionError('gate must remain disabled')):
            with decode._tar_io_lease():
                pass

    def test_multiprocess_concurrency_and_atomic_inode_creation(self):
        ctx = mp.get_context('spawn')
        workers, slots = 8, 2
        os.environ['YISU_TAR_IO_CONCURRENCY'] = str(slots)
        barrier = ctx.Barrier(workers+1)
        active, maximum, guard = ctx.Value('i', 0), ctx.Value('i', 0), ctx.Lock()
        results = ctx.Queue()
        for _ in range(workers):
            proc = ctx.Process(target=concurrent_reader,
                args=(self.namespace, slots, barrier, active, maximum, guard, results))
            proc.start(); self.children.append(proc)
        barrier.wait(timeout=15)
        self.assertEqual([results.get(timeout=15) for _ in range(workers)], [None]*workers)
        for proc in self.children:
            proc.join(timeout=5)
            self.assertEqual(proc.exitcode, 0)
        self.assertEqual(active.value, 0)
        self.assertEqual(maximum.value, slots)
        paths = decode._tar_io_slot_paths(slots)
        inodes = [os.stat(path).st_ino for path in paths]
        decode._TAR_IO_SLOT_CACHE.clear()
        self.assertEqual([os.stat(path).st_ino for path in decode._tar_io_slot_paths(slots)], inodes)
        results.close(); results.join_thread()

    def test_exception_releases_and_does_not_advance_rng(self):
        rng = decode.random.getstate()
        with self.assertRaisesRegex(RuntimeError, 'synthetic read error'):
            with decode._tar_io_lease():
                raise RuntimeError('synthetic read error')
        with decode._tar_io_lease():
            pass
        self.assertEqual(decode.random.getstate(), rng)

    def test_holder_keeps_slot_until_actual_exit_and_kill_releases(self):
        ctx = mp.get_context('spawn')
        acquired = ctx.Event()
        proc = ctx.Process(target=hold_lease, args=(self.namespace, acquired))
        proc.start(); self.children.append(proc)
        self.assertTrue(acquired.wait(timeout=10))
        decode._REQUEST_DEADLINE = time.monotonic() + .08
        started = time.monotonic()
        with self.assertRaises(decode.MediaDeadlineError):
            with decode._tar_io_lease():
                self.fail('a live holder must not lose its slot')
        self.assertLess(time.monotonic()-started, .3)
        self.assertTrue(proc.is_alive())
        proc.kill(); proc.join(timeout=5)
        self.assertFalse(proc.is_alive())
        decode._REQUEST_DEADLINE = time.monotonic() + 1
        with decode._tar_io_lease():
            pass

    def test_namespaces_are_independent(self):
        with decode._tar_io_lease():
            os.environ['YISU_TAR_IO_NAMESPACE'] = self.namespace + '-virtual-node-1'
            with decode._tar_io_lease():
                pass

    def test_tar_read_is_inside_lease_but_pil_is_outside(self):
        from PIL import Image
        pixels = Image.new('RGB', (4, 3), (12, 34, 56))
        image_bytes = io.BytesIO(); pixels.save(image_bytes, format='PNG')
        blob = image_bytes.getvalue()
        state = {'held': False, 'reads': 0}
        @contextlib.contextmanager
        def fake_lease():
            state['held'] = True
            try:
                yield
            finally:
                state['held'] = False
        actual_open = Image.open
        def checked_pil_open(*args, **kwargs):
            self.assertFalse(state['held'])
            return actual_open(*args, **kwargs)
        actual_file_open = open
        def checked_file_open(*args, **kwargs):
            self.assertTrue(state['held'])
            state['reads'] += 1
            return actual_file_open(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory)/'shard'
            shard.with_suffix('.tar').write_bytes(b'prefix'+blob)
            uri = f'tar://{shard}/6:{len(blob)}'
            with patch.object(decode, '_tar_io_lease', fake_lease), patch.object(Image, 'open', checked_pil_open), patch('builtins.open', checked_file_open):
                result = decode.read_image(uri)
        self.assertEqual(result.shape, (3, 4, 3))
        self.assertEqual(state['reads'], 1)


if __name__ == '__main__':
    unittest.main()
