"""TAR descriptor reuse and killable persistent-image decoder regressions.

Run with the training interpreter: python -m unittest discover -s tests
    -p test_persistent_tar_reader.py -v
No torch imports, CUDA work, or dataset mutation is needed.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / 'data/datasets/video_decode.py'
spec = importlib.util.spec_from_file_location('tar_decoder_under_test', SOURCE)
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class FakeProcess:
    def __init__(self, stranded=False):
        self.stdin, self.stdout, self.stderr = io.StringIO(), io.StringIO(), io.StringIO()
        self.returncode = None
        self.stranded = stranded
        self.killed = 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed += 1

    def wait(self, timeout):
        if self.stranded:
            raise subprocess.TimeoutExpired('decoder', timeout)
        self.returncode = -9
        return self.returncode


class TarFileReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = []
        for index in range(3):
            path = self.root / f'{index}.tar'
            path.write_bytes(bytes(range(32)))
            self.paths.append(str(path))
        self.env = patch.dict(os.environ, {'YISU_TAR_IO_CONCURRENCY': '0',
            'YISU_MEDIA_MAX_FILE_BYTES': '1048576'})
        self.env.start()
        self.readers = []

    def tearDown(self):
        for reader in self.readers:
            reader.close()
        if decoder._TAR_FILE_READER is not None:
            decoder._TAR_FILE_READER[1].close()
            decoder._TAR_FILE_READER = None
        self.env.stop()
        self.tmp.cleanup()

    def reader(self, count=2, buffer=-1):
        reader = decoder._TarFileReader(count, buffer)
        self.readers.append(reader)
        return reader

    def test_disabled_default_closes_every_descriptor(self):
        reader = self.reader(0)
        with patch('builtins.open', wraps=open) as opened:
            self.assertEqual(reader.read(self.paths[0], 0, 4), bytes(range(4)))
            self.assertEqual(reader.read(self.paths[0], 4, 4), bytes(range(4, 8)))
            self.assertEqual(opened.call_count, 2)
        self.assertFalse(reader.handles)

    def test_descriptor_reused_and_lru_evicts_least_recent(self):
        reader = self.reader(2)
        with patch('builtins.open', wraps=open) as opened:
            reader.read(self.paths[0], 0, 1)
            first = reader.handles[self.paths[0]][0]
            reader.read(self.paths[1], 0, 1)
            second = reader.handles[self.paths[1]][0]
            reader.read(self.paths[0], 10, 2)
            reader.read(self.paths[2], 0, 1)
            self.assertEqual(opened.call_count, 3)
        self.assertIs(reader.handles[self.paths[0]][0], first)
        self.assertFalse(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(len(reader.handles), 2)

    def test_close_releases_all_descriptors(self):
        reader = self.reader()
        reader.read(self.paths[0], 0, 1)
        handle = reader.handles[self.paths[0]][0]
        reader.close()
        reader.close()
        self.assertTrue(handle.closed)
        self.assertFalse(reader.handles)

    def test_invalid_ranges_do_not_open_source(self):
        reader = self.reader()
        with patch('builtins.open') as opened:
            for offset, size in [(-1, 1), (0, 0), (0, -1), (0.5, 1), (0, 1.5)]:
                with self.subTest(offset=offset, size=size):
                    with self.assertRaises(decoder.MediaInvalidInputError):
                        reader.read(self.paths[0], offset, size)
            opened.assert_not_called()

    def test_out_of_bounds_evicts_cached_descriptor(self):
        reader = self.reader()
        reader.read(self.paths[0], 0, 1)
        handle = reader.handles[self.paths[0]][0]
        with self.assertRaises(decoder.MediaCorruptError):
            reader.read(self.paths[0], 31, 2)
        self.assertTrue(handle.closed)
        self.assertFalse(reader.handles)

    def test_short_read_evicts_descriptor(self):
        # Mutation intentionally violates the immutable-source contract: the
        # short-read guard must still catch it rather than pass corrupt bytes.
        reader = self.reader(buffer=0)
        reader.read(self.paths[0], 0, 1)
        handle = reader.handles[self.paths[0]][0]
        with open(self.paths[0], 'r+b') as source:
            source.truncate(3)
        with self.assertRaisesRegex(decoder.MediaCorruptError, 'Short TAR read'):
            reader.read(self.paths[0], 2, 5)
        self.assertTrue(handle.closed)
        self.assertFalse(reader.handles)

    def test_io_failure_evicts_descriptor(self):
        reader = self.reader()
        broken = Mock()
        broken.closed = False
        broken.seek.side_effect = OSError('source I/O failed')
        reader.handles[self.paths[0]] = (broken, 32)
        with self.assertRaisesRegex(OSError, 'source I/O failed'):
            reader.read(self.paths[0], 0, 1)
        self.assertFalse(reader.handles)
        self.assertTrue(broken.close.called)

    def test_blob_size_limit_prevents_large_allocation(self):
        with patch.dict(os.environ, {'YISU_MEDIA_MAX_FILE_BYTES': '4'}):
            with self.assertRaises(decoder.MediaCapacityError):
                self.reader().read(self.paths[0], 0, 5)

    def test_buffer_configuration_and_validation(self):
        for size in [-1, 0, 8192]:
            with self.subTest(buffer=size):
                reader = self.reader(buffer=size)
                self.assertEqual(reader.read(self.paths[0], 5, 3), bytes([5, 6, 7]))
        for count, size in [(-1, -1), (1, -2), (1, 1)]:
            with self.assertRaises(ValueError):
                decoder._TarFileReader(count, size)

    def test_reader_options_change_closes_old_descriptors(self):
        with patch.dict(os.environ, {'YISU_TAR_MAX_OPEN_FILES': '2', 'YISU_TAR_READ_BUFFER_BYTES': '-1'}):
            decoder._read_tar_blob(self.paths[0], 0, 1)
            old = decoder._TAR_FILE_READER[1].handles[self.paths[0]][0]
            with patch.dict(os.environ, {'YISU_TAR_MAX_OPEN_FILES': '0'}):
                decoder._read_tar_blob(self.paths[0], 1, 1)
            self.assertTrue(old.closed)


class PersistentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.previous = decoder._STRANDED
        decoder._STRANDED = []
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        decoder._STRANDED = self.previous
        self.tmp.cleanup()

    def test_close_tracks_stranded_child_and_keeps_output_until_reaped(self):
        output = Path(self.tmp.name) / 'active'
        output.mkdir()
        (output / 'pixels.npy').write_bytes(b'pending')
        proc = FakeProcess(stranded=True)
        client = decoder._PersistentDecoderClient()
        client.proc, client._request_directory = proc, str(output)
        client.close()
        client.close()
        self.assertIsNone(client.proc)
        self.assertEqual(decoder._STRANDED, [(proc, str(output))])
        self.assertTrue(proc.stdin.closed and proc.stdout.closed and proc.stderr.closed)
        decoder._reap_stranded()
        self.assertTrue(output.exists())
        proc.returncode = -9
        decoder._reap_stranded()
        self.assertFalse(output.exists())
        self.assertFalse(decoder._STRANDED)

    def test_metadata_stranded_child_without_output_reaps(self):
        proc = FakeProcess(stranded=True)
        client = decoder._PersistentDecoderClient()
        client.proc = proc
        client.close()
        self.assertEqual(decoder._STRANDED, [(proc, None)])
        proc.returncode = -9
        decoder._reap_stranded()
        self.assertFalse(decoder._STRANDED)

    def test_stranded_limit_prevents_replacement_spawn(self):
        decoder._STRANDED = [(FakeProcess(stranded=True), None) for _ in range(decoder._MAX_STRANDED)]
        with patch.object(decoder.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(decoder.MediaResourceError, 'stranded process limit'):
                decoder._PersistentDecoderClient()._start()
            spawn.assert_not_called()

    def test_exited_child_does_not_consume_stranded_slot(self):
        client = decoder._PersistentDecoderClient()
        client.proc = FakeProcess()
        client.close()
        self.assertFalse(decoder._STRANDED)

    def test_broken_pipe_closes_all_parent_pipe_handles(self):
        client = decoder._PersistentDecoderClient()
        proc = FakeProcess()
        client.proc = proc
        with patch.object(proc.stdin, 'write', side_effect=BrokenPipeError('closed')):
            with self.assertRaises(decoder.MediaExecutorBugError):
                client.request({'kind': 'record'}, 1)
        self.assertIsNone(client.proc)
        self.assertTrue(proc.stdin.closed and proc.stdout.closed and proc.stderr.closed)

    def test_invalid_timeout_is_rejected_before_spawn(self):
        with patch.object(decoder.subprocess, 'Popen') as spawn:
            for timeout in [0, -1, float('inf'), float('nan')]:
                with self.assertRaises(ValueError):
                    decoder._PersistentDecoderClient().request({}, timeout)
            spawn.assert_not_called()

    def test_persistent_main_applies_file_output_resource_limit(self):
        with patch.dict(os.environ, {'YISU_MEDIA_MAX_FILE_BYTES': '2048',
                'YISU_MEDIA_MAX_OUTPUT_BYTES': '1024'}), \
                patch.object(resource, 'getrlimit', return_value=(-1, -1)), \
                patch.object(resource, 'setrlimit') as limits, \
                patch.object(decoder.sys, 'stdin', io.StringIO('')):
            decoder.persistent_main()
        limits.assert_any_call(resource.RLIMIT_FSIZE, (2048, -1))

    def test_parent_checks_output_size_before_numpy_load(self):
        def oversized(request, timeout):
            Path(request['_output_path']).write_bytes(b'x' * 65)

        with patch.dict(os.environ, {'YISU_IMAGE_PERSISTENT_DECODER': '1',
                'YISU_IMAGE_TMP_DIR': self.tmp.name, 'YISU_MEDIA_MAX_OUTPUT_BYTES': '64'}), \
                patch.object(decoder, '_PERSISTENT_DISABLED_UNTIL', 0), \
                patch.object(decoder, '_persistent_decode_request', side_effect=oversized), \
                patch.object(decoder.np, 'load') as load, \
                patch.object(decoder.subprocess, 'Popen', side_effect=RuntimeError('one-shot fallback')):
            with self.assertRaisesRegex(RuntimeError, 'one-shot fallback'):
                decoder.decode_isolated({'kind': 'image', 'path': 'unused'}, 3, 'test')
            load.assert_not_called()
        self.assertFalse(list(Path(self.tmp.name).iterdir()))


class PersistentSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        encoded = io.BytesIO()
        Image.fromarray(np.full((16, 16, 3), 73, dtype=np.uint8)).save(encoded, format='PNG')
        self.blob = encoded.getvalue()
        self.tar = self.root / 'shard.tar'
        self.tar.write_bytes(b'padding' + self.blob + self.blob)
        self.path = f'tar://{self.root}/shard/7:{len(self.blob)}'
        self.env = patch.dict(os.environ, {'YISU_TAR_MAX_OPEN_FILES': '2',
            'YISU_TAR_IO_CONCURRENCY': '0', 'YISU_MEDIA_MAX_FILE_BYTES': '1048576',
            'YISU_MEDIA_MAX_OUTPUT_BYTES': '1048576'})
        self.env.start()
        self.client = decoder._PersistentDecoderClient()

    def tearDown(self):
        self.client.close()
        decoder._reap_stranded()
        self.env.stop()
        self.tmp.cleanup()

    def request(self):
        return {'kind': 'images', 'path': self.path, 'width': 16, 'height': 16,
                '_output_path': str(self.root / 'pixels.npy')}

    def scripted_child(self, body):
        self.client.proc = subprocess.Popen(
            [decoder.sys.executable, '-c', 'import os, sys, json, time\nsys.stdin.readline()\n' + body],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=decoder._child_env(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
        )

    def test_real_child_reuses_process_and_preserves_pixels(self):
        self.client.request(self.request(), 10)
        first = np.load(self.root / 'pixels.npy', allow_pickle=False)
        pid = self.client.proc.pid
        self.client.request(self.request(), 10)
        second = np.load(self.root / 'pixels.npy', allow_pickle=False)
        self.assertEqual(self.client.proc.pid, pid)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(second == 73))

    def test_real_child_rejects_oversized_output_and_remains_usable(self):
        with patch.dict(os.environ, {'YISU_MEDIA_MAX_OUTPUT_BYTES': '64'}):
            with self.assertRaises(decoder.MediaCapacityError):
                self.client.request(self.request(), 10)
        self.assertIsNotNone(self.client.proc)
        self.assertIsNone(self.client.proc.poll())
        self.assertFalse((self.root / 'pixels.npy').exists())

    def test_real_child_startup_deadline_is_bounded(self):
        with self.assertRaises(decoder.MediaDeadlineError):
            self.client.request(self.request(), 0.001)
        self.assertIsNone(self.client.proc)

    def test_partial_line_then_stall_cannot_escape_deadline(self):
        self.scripted_child('os.write(sys.stdout.fileno(), b\'{"ok": true, "result":\')\ntime.sleep(30)\n')
        started = time.monotonic()
        with patch.object(decoder.os, 'read', wraps=os.read) as read:
            with self.assertRaises(decoder.MediaDeadlineError):
                self.client.request({'kind': 'record'}, 0.5)
            self.assertGreaterEqual(read.call_count, 1, 'child must emit a partial response before timeout')
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIsNone(self.client.proc)

    def test_truncated_response_is_channel_error(self):
        self.scripted_child('os.write(sys.stdout.fileno(), b\'{"ok": true\')\n')
        with self.assertRaisesRegex(decoder.MediaExecutorBugError, 'complete response'):
            self.client.request({'kind': 'record'}, 5)
        self.assertIsNone(self.client.proc)

    def test_response_byte_limit_terminates_child(self):
        self.scripted_child('os.write(sys.stdout.fileno(), b"x" * 4096)\ntime.sleep(30)\n')
        with patch.dict(os.environ, {'YISU_MEDIA_MAX_RESPONSE_BYTES': '1024'}):
            with self.assertRaisesRegex(decoder.MediaCapacityError, 'response exceeds'):
                self.client.request({'kind': 'record'}, 5)
        self.assertIsNone(self.client.proc)

    def test_large_metadata_response_spans_multiple_reads(self):
        self.scripted_child('print(json.dumps({"ok": True, "result": {"text": "文" * 200000}}), flush=True)\n')
        with patch.dict(os.environ, {'YISU_MEDIA_MAX_RESPONSE_BYTES': str(2 * 1024**2)}):
            result = self.client.request({'kind': 'record'}, 5)
        self.assertEqual(result, {'text': '文' * 200000})

    def test_complete_response_at_exact_limit_is_accepted(self):
        payload = json.dumps({'ok': True, 'result': {'value': 'test'}})
        self.scripted_child(f'print({payload!r}, flush=True)\n')
        with patch.dict(os.environ, {'YISU_MEDIA_MAX_RESPONSE_BYTES': str(len(payload.encode()))}):
            self.assertEqual(self.client.request({'kind': 'record'}, 5), {'value': 'test'})


if __name__ == '__main__':
    unittest.main()
