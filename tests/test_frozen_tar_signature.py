"""Frozen tar startup must only read an existing local/PFS mmap cache.

The fixture uses the real columnar/runtime formats and forbids *all* filesystem
metadata or opens under the dataset roots, including a deliberately missing root.
Run: python -m unittest discover -s tests -p test_frozen_tar_signature.py -v
"""
from contextlib import ExitStack, contextmanager
import builtins
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from data.datasets.image_database import LmdbParquetDataset, _ColumnarIndex
from data.datasets.tar_database import TarParquetDataset


ROOT = "/forbidden_nfs/images"
T2I_ROOT = "/forbidden_nfs/edits"


@contextmanager
def forbid_dataset_io():
    def guarded(fn):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                value = os.fsdecode(path)
                if value.startswith("/forbidden_nfs"):
                    raise AssertionError(f"unexpected remote filesystem access: {fn.__name__}({value})")
            return fn(path, *args, **kwargs)
        return call
    with ExitStack() as stack:
        for module, name in ((os, "stat"), (os, "lstat"), (os, "listdir"),
                             (os, "scandir"), (os, "open"),
                             (builtins, "open"), (io, "open")):
            stack.enter_context(patch.object(module, name, guarded(getattr(module, name))))
        yield


class FrozenTarSignatureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)
        # Construct a cache fixture without touching or creating dataset roots.
        stub = object.__new__(TarParquetDataset)
        stub.freeze_index_signature = True
        stub.tar_root = Path(ROOT)
        stub.database_path = stub.lmdb_root = stub.edit_lmdb_root = ROOT
        stub.sources = {"photo"}
        stub.edit_sources = set()
        stub.edit_split = "train"
        stub.t2i_edit_database_path = T2I_ROOT
        stub.t2i_edit_sources_t2i_mode = {"edit"}
        stub.index_cache_dir = self.cache
        stub.index_cache_force_refresh = False
        stub.buckets = ["256x256"]
        stub.max_upscale_ratio = 1.2
        stub.read_stats = {"image": {}, "edit": {}}
        for label, root, payload in (("tar", ROOT, {"tar:photo": [1, 1]}),
                                     ("t2i", T2I_ROOT, {"t2i:edit": [1, 1]})):
            path = stub._signature_snapshot_path(self.cache, label, Path(root))
            path.write_text(json.dumps(payload))
            os.utime(path, (1, 1))  # Ancient snapshots must still be accepted.
        self.stub = stub
        self.fingerprint = stub._index_fingerprint(True, False, None, None, None)
        stub.index_fingerprint = self.fingerprint
        rows = [{"sample_id": 1, "source": "photo", "width": 256, "height": 256,
                 "caption": "a cat", "is_edit": False,
                 "image_path": f"tar://{ROOT}/photo/shard-00000/512:64",
                 "blob_off": 512, "blob_size": 64},
                {"sample_id": 2, "source": "edit", "width": 256, "height": 256,
                 "caption": "a dog", "is_edit": False,
                 "image_path": f"tar://{T2I_ROOT}/edit/shard-00000/512:64",
                 "blob_off": 512, "blob_size": 64}]
        store = _ColumnarIndex.build_from_rows(rows, [])
        self.index_dir = self.cache / f"index_{self.fingerprint}"
        self.index_dir.mkdir()
        store.save(self.index_dir, self.fingerprint, time.time(), stub.read_stats)
        # Invoke the base builder once to produce a real completed runtime cache.
        LmdbParquetDataset._build_bucket_index(stub, store, None, None, 0.0)
        self.runtime_dir = stub._runtime_cache_dir(None, None, 0.0)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **overrides):
        args = dict(database_path=ROOT, buckets=["256x256"], sources=["photo"],
                    t2i_edit_database_path=T2I_ROOT, t2i_edit_sources_t2i_mode=["edit"],
                    index_cache_dir=str(self.cache), freeze_index_signature=True)
        args.update(overrides)
        with forbid_dataset_io(), patch.dict(os.environ, {"GWP_SHARD_SIGNATURE_TTL": "0"}), \
                patch.object(TarParquetDataset, "_read_parquet", side_effect=AssertionError("parquet scan")), \
                patch.object(TarParquetDataset, "_discover_source_dirs", side_effect=AssertionError("source discovery")):
            return TarParquetDataset(**args)

    def test_real_mmap_startup_does_not_access_remote_roots(self):
        dataset = self.load()
        self.assertEqual(dataset.index_fingerprint, self.fingerprint)
        self.assertEqual(len(dataset.records), 2)
        self.assertEqual(dataset.records[0]["caption"], "a cat")
        self.assertEqual(dataset.records[1]["caption"], "a dog")
        self.assertEqual(list(dataset.bucket_indices["256x256"]), [0, 1])

    def test_missing_index_fails_without_rebuild(self):
        (self.index_dir / "meta.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "冻结索引缓存缺失或无效"):
            self.load()

    def test_corrupt_column_fails_without_rebuild(self):
        (self.index_dir / "w.npy").write_bytes(b"truncated")
        with self.assertRaisesRegex(RuntimeError, "冻结索引缓存缺失或无效"):
            self.load()

    def test_missing_runtime_fails_without_rebuild(self):
        shutil.rmtree(self.runtime_dir)
        with self.assertRaisesRegex(RuntimeError, "冻结运行期 bucket 缓存缺失或无效"):
            self.load()

    def test_all_configured_sources_required(self):
        with self.assertRaisesRegex(RuntimeError, "tar:missing"):
            self.load(sources=["photo", "missing"])

    def test_signature_pair_is_strict(self):
        path = self.stub._signature_snapshot_path(self.cache, "t2i", Path(T2I_ROOT))
        for bad in ([1], [1, -1], [1, True], [1, 1.0], "1,1", None):
            with self.subTest(bad=bad):
                path.write_text(json.dumps({"t2i:edit": bad}))
                with self.assertRaisesRegex(RuntimeError, "t2i:edit"):
                    self.load()

    def test_missing_or_invalid_snapshot_fails_closed(self):
        path = self.stub._signature_snapshot_path(self.cache, "tar", Path(ROOT))
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "冻结索引签名缺失或损坏"):
            self.load()
        path.write_text("[]")
        with self.assertRaisesRegex(RuntimeError, "JSON object"):
            self.load()

    def test_refresh_and_disable_rejected(self):
        with self.assertRaisesRegex(ValueError, "不允许强制刷新"):
            self.load(refresh_index_cache=True)
        with patch.dict(os.environ, {"GWP_DATASET_INDEX_CACHE_DISABLE": "1"}):
            with self.assertRaisesRegex(ValueError, "不允许强制刷新"):
                self.load()

    def test_explicit_freeze_flag_required(self):
        with self.assertRaisesRegex(AssertionError, "unexpected remote filesystem access"):
            self.load(freeze_index_signature=False)
        with self.assertRaisesRegex(ValueError, "必须为 bool"):
            self.load(freeze_index_signature="true")

    def test_adapter_passes_freeze_flag(self):
        from data.lmdb_dataset import GigaLMDBDataset
        cfg = dict(dataset_type="tar_parquet", database_path=ROOT, buckets=["256x256"],
                   image_sources=["photo"], t2i_edit_database_path=T2I_ROOT,
                   t2i_edit_sources_t2i_mode=["edit"], freeze_index_signature=True)
        with forbid_dataset_io():
            dataset = GigaLMDBDataset(cfg, index_cache_dir=str(self.cache))
        self.assertEqual(dataset.index_fingerprint, self.fingerprint)


if __name__ == "__main__":
    unittest.main()
