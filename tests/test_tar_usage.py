"""Pure-stdlib tests for consumption accounting and checkpoint sidecars."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "optimizations" / "tar_usage.py"
SPEC = importlib.util.spec_from_file_location("tar_usage_under_test", MODULE)
tar_usage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tar_usage)
TarUsageTracker = tar_usage.TarUsageTracker


class TarUsageTests(unittest.TestCase):
    def test_counts_occurrences_and_batches_once_per_tar(self):
        tracker = TarUsageTracker()
        self.assertEqual(tracker.record_batch({"image_paths": [
            "tar:///data/a.tar/10:2", "tar:///data/a.tar/20:2",
            "/normal.png", "tar:///data/b.tar/7:3", None,
        ]}), 3)
        tracker.record_batch({"image_paths": ["tar:///data/a.tar/10:2"]})
        state = tracker.state_dict()
        self.assertEqual(state["consumed_samples"], 4)
        self.assertEqual(state["batches"], 2)
        self.assertEqual(state["tars"]["/data/a.tar"], {
            "consumed_samples": 3, "batches": 2, "last_offset": 10,
            "adjacent_pairs": 1, "increasing_offset_pairs": 1,
        })
        self.assertEqual(state["tars"]["/data/b.tar"]["batches"], 1)

    def test_explicit_metadata_is_authoritative_and_not_double_counted(self):
        tracker = TarUsageTracker()
        tracker.record_batch({"image_paths": ["tar://old.tar/1:2"],
                              "__tar_uris__": ["tar://real.tar/5:2"]})
        self.assertEqual(tracker.record_batch({"image_paths": ["tar://old.tar/1:2"],
                                              "__tar_uris__": []}), 0)
        self.assertEqual(set(tracker.state_dict()["tars"]), {"real.tar"})
        self.assertEqual(tracker.consumed_samples, 1)

    def test_ignored_non_tar_and_scalar_metadata(self):
        tracker = TarUsageTracker()
        for batch in ({}, {"image_paths": None}, {"image_paths": []},
                      {"image_paths": ["/plain.jpg", "bos://data/a.jpg", None]},
                      {"image_paths": Path("/plain.png")}):
            self.assertEqual(tracker.record_batch(batch), 0)
        self.assertEqual(tracker.batches, 0)
        self.assertEqual(tracker.record_batch({"image_paths": "tar://relative.tar/0:0"}), 1)

    def test_adjacency_uses_tar_consumption_order_across_batches(self):
        tracker = TarUsageTracker()
        tracker.record_batch({"image_paths": ["tar://a.tar/30:1", "tar://a.tar/30:1"]})
        tracker.record_batch({"image_paths": "/plain.png"})
        tracker.record_batch({"image_paths": ["tar://a.tar/20:1", "tar://a.tar/40:1"]})
        counts = tracker.state_dict()["tars"]["a.tar"]
        self.assertEqual(counts["adjacent_pairs"], 3)
        self.assertEqual(counts["increasing_offset_pairs"], 1)

    def test_malformed_tar_batch_is_transactional(self):
        tracker = TarUsageTracker()
        tracker.record_batch({"image_paths": "tar://a.tar/1:2"})
        before = tracker.state_dict()
        for uri in ("tar://a.tar/no-offset", "tar:///1:2", "tar://a.tar/-1:2",
                    "tar://a.tar/1:-2", "tar://a.tar/1:2:3"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                tracker.record_batch({"image_paths": ["tar://a.tar/2:2", uri]})
            self.assertEqual(tracker.state_dict(), before)
        with self.assertRaises(TypeError):
            tracker.record_batch({"image_paths": [["tar://a.tar/1:2"]]})
        self.assertEqual(tracker.state_dict(), before)

    def test_record_and_state_dict_do_not_access_filesystem(self):
        tracker = TarUsageTracker()
        prefetched = {"image_paths": ["tar:///not-mounted/a.tar/0:10"]}
        self.assertEqual(tracker.consumed_samples, 0)
        with patch("builtins.open", side_effect=AssertionError("unexpected I/O")), \
                patch.object(Path, "open", side_effect=AssertionError("unexpected I/O")), \
                patch.object(tar_usage.os, "stat", side_effect=AssertionError("unexpected I/O")):
            tracker.record_batch(prefetched)
            state = tracker.state_dict()
        self.assertEqual(state["consumed_samples"], 1)
        state["tars"]["/not-mounted/a.tar"]["consumed_samples"] = 100
        self.assertEqual(tracker.consumed_samples, 1)
        self.assertEqual(tracker.state_dict()["tars"]["/not-mounted/a.tar"]["consumed_samples"], 1)

    def test_save_restore_and_resume_match_uninterrupted_consumption(self):
        tracker = TarUsageTracker()
        tracker.record_batch({"image_paths": ["tar:///路径/a.tar/10:2", "tar://b.tar/4:1"]})
        with tempfile.TemporaryDirectory() as directory:
            reference = tracker.save(directory, step=120, rank=3)
            self.assertIn("step-000000000120.rank-000003", reference["path"])
            self.assertEqual(hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest(), reference["sha256"])
            resumed = TarUsageTracker.load(reference)
            self.assertEqual(resumed.state_dict(), tracker.state_dict())
            batch = {"image_paths": ["tar://b.tar/9:1", "tar:///路径/a.tar/2:1"]}
            resumed.record_batch(batch)
            tracker.record_batch(batch)
            self.assertEqual(resumed.state_dict(), tracker.state_dict())
            with Path(reference["tsv_path"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual({row["tar_path"] for row in rows}, {"/路径/a.tar", "b.tar"})
            self.assertEqual({row["step"] for row in rows}, {"120"})
            self.assertEqual({row["rank"] for row in rows}, {"3"})

    def test_empty_snapshot_and_unique_per_save_rank_files(self):
        tracker = TarUsageTracker()
        with tempfile.TemporaryDirectory() as directory:
            references = [tracker.save(directory, step=0, rank=rank) for rank in (0, 1, 0)]
            self.assertEqual(len({ref["path"] for ref in references}), 3)
            self.assertEqual(len(list(Path(directory).iterdir())), 6)
            for reference in references:
                self.assertEqual(TarUsageTracker.load(reference).state_dict(), tracker.state_dict())

    def test_hash_and_reference_identity_validation(self):
        tracker = TarUsageTracker()
        with tempfile.TemporaryDirectory() as directory:
            reference = tracker.save(directory, step=10, rank=0)
            for update in ({"step": 11}, {"rank": 1}, {"sha256": "bad"}):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    TarUsageTracker.load(dict(reference, **update))
            path = Path(reference["path"])
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                TarUsageTracker.load(reference)

    def test_rejects_invalid_state_even_with_matching_hash(self):
        tracker = TarUsageTracker()
        tracker.record_batch({"image_paths": "tar://a.tar/3:1"})
        with tempfile.TemporaryDirectory() as directory:
            reference = tracker.save(directory, step=1, rank=0)
            path = Path(reference["path"])
            original = json.loads(path.read_text())
            mutations = [lambda s: s.update(version=99),
                         lambda s: s.update(consumed_samples=4),
                         lambda s: s.update(last_tar_path="missing.tar"),
                         lambda s: s["tars"]["a.tar"].update(adjacent_pairs=5),
                         lambda s: s["tars"]["a.tar"].update(batches=True)]
            for mutation in mutations:
                state = json.loads(json.dumps(original))
                mutation(state)
                content = json.dumps(state).encode()
                path.write_bytes(content)
                changed_ref = dict(reference, sha256=hashlib.sha256(content).hexdigest())
                with self.assertRaises(ValueError):
                    TarUsageTracker.load(changed_ref)

    def test_failed_atomic_save_leaves_no_partial_reports(self):
        tracker = TarUsageTracker()
        with tempfile.TemporaryDirectory() as directory:
            real_replace = tar_usage.os.replace
            def fail_json(source, destination):
                if str(destination).endswith(".json"):
                    raise OSError("simulated checkpoint failure")
                real_replace(source, destination)
            with patch.object(tar_usage.os, "replace", side_effect=fail_json):
                with self.assertRaisesRegex(OSError, "simulated"):
                    tracker.save(directory, step=1, rank=0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_save_rejects_invalid_step_and_rank_before_writing(self):
        tracker = TarUsageTracker()
        with tempfile.TemporaryDirectory() as directory:
            for step, rank in ((-1, 0), (0, -1), (True, 0), (0, 1.5)):
                with self.subTest(step=step, rank=rank), self.assertRaises(ValueError):
                    tracker.save(directory, step, rank)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
