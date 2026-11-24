"""Regression tests for stage_1 accumulation, worker recovery and sample accounting."""
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock, patch

import numpy as np
import psutil
import torch
import torch.distributed as dist


SOURCE = Path(os.environ.get("GIGA_TRAIN_SOURCE", Path(__file__).resolve().parents[1] / "scripts/train/train_giga_exp.py"))
sys.path.insert(0, str(SOURCE.parents[2]))
TREE = ast.parse(SOURCE.read_text())
HELPERS = {"_data_error_category", "TrainingDataStream", "next_training_batch", "set_lr",
           "_read_checkpoint_metadata", "_read_weight_metadata", "save_pure_weights", "load_pure_weights",
           "atomic_write", "save_torch_weights", "sampler_signature", "restore_data_cursor",
           "checkpoint_distributed", "load_weight_candidates", "fixed_validation_plan",
           "run_fixed_validation", "run_next_batch_validation", "wan_latent_stats",
           "parse_muon_lr", "MuonAdamW", "build_optimizer"}


def identity_collate(batch):
    return batch


def helper_namespace(distributed=None):
    namespace = dict(torch=torch, dist=distributed or SimpleNamespace(all_reduce=lambda *a, **k: None,
                     ReduceOp=SimpleNamespace(MAX=None), is_available=lambda: False), time=time, logging=logging, Path=Path,
                     json=json, re=re, os=os, uuid=uuid, hashlib=hashlib, collate=identity_collate, _dist_rank=lambda: 0,
                     _semantic_metadata=lambda module: module.semantic_metadata(),
                     _warn_semantic_mismatch=lambda *a: None)
    nodes = [n for n in TREE.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in HELPERS]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class FixedSampler:
    def __init__(self, size):
        self.size, self.epoch, self.start_batch = size, 0, 0

    def set_start_batch(self, offset, epoch=0):
        self.start_batch, self.epoch = offset, epoch

    def __iter__(self):
        return iter([[i] for i in range(self.start_batch, self.size)])

    def __len__(self):
        return self.size


class BlockingDataset:
    def __init__(self, pid_file):
        self.pid_file = pid_file

    def __len__(self):
        return 5

    def __getitem__(self, index):
        if index == 0:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            Path(self.pid_file).write_text(str(child.pid))
            time.sleep(60)
        return index


class DataError(RuntimeError):
    category = "corrupt"


class RuntimeTests(unittest.TestCase):
    def ready_resume_fixture(self, enabled=True):
        ns = helper_namespace()
        sampler = FixedSampler(5)
        cursor = {"version": 2, "stream_kind": "ready_first", "epoch": 3,
                  "batch_offset": 1, "consumed_ranges": [[2, 4]], "future_epochs": [],
                  "sampler_signature": ns["sampler_signature"](sampler), "epoch_length": 5}
        saved = {"version": 2, "stream_kind": "ready_first_distributed",
                 "world_size": 1, "rank_cursors": [cursor]}
        engine = SimpleNamespace(giga_data_cursor=saved, giga_ready_pipeline_enabled=enabled)
        return ns, sampler, engine, cursor

    def test_ready_resume_uses_actual_rank_frontier_and_consumed_ranges(self):
        ns, sampler, engine, cursor = self.ready_resume_fixture()
        actual = ns["restore_data_cursor"](engine, sampler, 0, 2, logging.getLogger("test"))
        self.assertEqual(actual, (3, 1))
        self.assertEqual(engine.giga_ready_resume_state["consumed_ranges"], [[2, 4]])
        self.assertIs(engine.giga_ready_resume_state, cursor)

    def test_ready_resume_disabled_is_explicitly_approximate(self):
        ns, sampler, engine, _ = self.ready_resume_fixture(False)
        with self.assertLogs(level="WARNING"):
            actual = ns["restore_data_cursor"](engine, sampler, 3, 2, logging.getLogger("test"))
        self.assertEqual(actual, (1, 1))
        self.assertFalse(hasattr(engine, "giga_ready_resume_state"))

    def test_ready_resume_rejects_corrupt_metadata(self):
        ns, sampler, engine, cursor = self.ready_resume_fixture()
        cursor["batch_offset"] = -1
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            ns["restore_data_cursor"](engine, sampler, 0, 2, logging.getLogger("test"))

    def test_worker_timeout_rebuilds_and_keeps_cursor(self):
        ns = helper_namespace()
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            stream = ns["TrainingDataStream"](BlockingDataset(str(pid_file)), FixedSampler(5),
                     {"num_workers": 1, "dataloader_timeout_seconds": 1.0}, logging.getLogger("test"))
            old_worker = stream.iterator._workers[0].pid
            try:
                with self.assertRaisesRegex(RuntimeError, "DataLoader timed out"):
                    stream.next_batch()
                self.assertTrue(pid_file.exists())
                child_pid = int(pid_file.read_text())
                started = time.monotonic()
                stream.restart()
                self.assertLess(time.monotonic() - started, 10)
                self.assertNotEqual(old_worker, stream.iterator._workers[0].pid)
                self.assertFalse(psutil.pid_exists(old_worker))
                self.assertTrue(not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE)
                self.assertEqual(stream.next_batch(), [1])
                self.assertEqual(stream.batch_offset, 2)
            finally:
                stream.close()

    def test_restart_and_epoch_boundary_without_workers(self):
        ns = helper_namespace()
        stream = ns["TrainingDataStream"](list(range(3)), FixedSampler(3), {"num_workers": 0}, logging.getLogger("test"))
        try:
            self.assertEqual(stream.next_batch(), [0])
            stream.restart()
            self.assertEqual(stream.next_batch(), [1])
            self.assertEqual(stream.next_batch(), [2])
            self.assertEqual(stream.next_batch(), [0])
            self.assertEqual(stream.epoch, 1)
        finally:
            stream.close()

    def test_remote_rank_timeout_rebuilds_healthy_rank(self):
        def reduce(flag, **kwargs):
            if flag.item() == 0 and not getattr(reduce, "seen", False):
                flag.fill_(2)
                reduce.seen = True
        ns = helper_namespace(SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MAX=None), is_available=lambda: False))
        stream = SimpleNamespace(next_batch=lambda: {"ok": True}, restart=Mock())
        self.assertIsNone(ns["next_training_batch"](stream, "cpu", logging.getLogger("test"), 1, 0))
        stream.restart.assert_called_once()

    def test_sample_count_round_trip(self):
        ns = helper_namespace()
        module = torch.nn.Linear(1, 1)
        module.semantic_metadata = lambda: {"test": True}
        engine = SimpleNamespace(module=module, giga_samples_seen=123,
                                 giga_data_cursor={"epoch": 2, "batch_offset": 3, "sampler_signature": "test"},
                                 optimizer=SimpleNamespace(refresh_fp32_params=Mock()))
        def load(model, state, metadata):
            model.load_state_dict(state)
            return len(state), 0, 0
        ns["load_compatible_state_dict"] = load
        with tempfile.TemporaryDirectory() as tmp:
            ns["save_pure_weights"](engine, Path(tmp), 4)
            engine.giga_samples_seen = 0
            self.assertEqual(ns["load_pure_weights"](engine, Path(tmp)), 4)
            self.assertEqual(engine.giga_samples_seen, 123)
            self.assertEqual(engine.giga_data_cursor["epoch"], 2)
            engine.optimizer.refresh_fp32_params.assert_called_once()
            self.assertEqual(ns["_read_checkpoint_metadata"](Path(tmp)/"checkpoints/weights/latest.pt")["samples_seen"], 123)

    def test_atomic_failure_preserves_published_file(self):
        ns = helper_namespace()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"step-00000004.safetensors"
            target.write_bytes(b"previous-complete-checkpoint")
            def interrupted(path):
                path.write_bytes(b"partial")
                raise OSError("injected full disk")
            with self.assertRaises(OSError): ns["atomic_write"](target, interrupted)
            self.assertEqual(target.read_bytes(), b"previous-complete-checkpoint")
            self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_corrupt_latest_falls_back_and_explicit_file_fails(self):
        from safetensors.torch import save_file
        ns = helper_namespace()
        model = torch.nn.Linear(1, 1)
        engine = SimpleNamespace(module=model, optimizer=SimpleNamespace(refresh_fp32_params=Mock()))
        ns["load_compatible_state_dict"] = lambda m, state, meta: (m.load_state_dict(state) and (2, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"checkpoints/weights"
            directory.mkdir(parents=True)
            good = directory/"step-00000004.safetensors"
            save_file(model.state_dict(), str(good))
            bad = directory/"step-00000005.safetensors"
            bad.write_bytes(b"partial")
            (directory/".step-00000006.safetensors.incomplete.tmp").write_bytes(b"partial")
            self.assertEqual(ns["load_pure_weights"](engine, Path(tmp)), 4)
            with self.assertRaises(RuntimeError): ns["load_pure_weights"](engine, Path(tmp), str(bad))
            good.unlink()
            with self.assertRaises(RuntimeError): ns["load_pure_weights"](engine, Path(tmp))

    def test_torch_sidecar_rejects_mixed_publication(self):
        ns = helper_namespace()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"step-00000004.pt"
            ns["save_torch_weights"]({"x":torch.ones(1)}, target, {"step":4, "samples_seen":12})
            self.assertEqual(ns["_read_checkpoint_metadata"](target)["samples_seen"], 12)
            target.write_bytes(b"incomplete-new-file")
            with self.assertRaises(ValueError): ns["_read_checkpoint_metadata"](target)

    def test_cursor_keeps_skips_and_replays_pending_validation(self):
        ns = helper_namespace()
        stream = ns["TrainingDataStream"](list(range(5)), FixedSampler(5), {"num_workers":0}, logging.getLogger("test"))
        self.assertEqual(stream.next_batch(), [0])
        self.assertEqual(stream.next_batch(), [1])  # a consumed but globally skipped slot
        self.assertEqual(stream.next_batch(), [2])
        saved = stream.state_dict()
        engine = SimpleNamespace(giga_data_cursor=saved)
        sampler = FixedSampler(5)
        self.assertEqual(ns["restore_data_cursor"](engine, sampler, 1, 2, logging.getLogger("test")), (0,3))
        stream.next_batch()  # pending validation sample must be replayed after resume
        engine.giga_data_cursor = stream.state_dict(pending_batch=True)
        epoch, offset = ns["restore_data_cursor"](engine, sampler, 1, 2, logging.getLogger("test"))
        resumed = ns["TrainingDataStream"](list(range(5)),sampler,{"num_workers":0},logging.getLogger("test"),epoch,offset)
        try:
            self.assertEqual(resumed.next_batch(), [3])
        finally:
            resumed.close(); stream.close()
        self.assertEqual(ns["restore_data_cursor"](engine, FixedSampler(6),1,2,logging.getLogger("test")),(0,2))

    def test_fixed_validation_repeats_pixels_across_steps(self):
        ns = helper_namespace()
        class Module(torch.nn.Module):
            def forward(self, x, *args): return torch.zeros_like(x)
            def set_moe_metrics_collector(self, value): pass
            def set_moe_denoising_step(self, value): pass
        class Engine(torch.nn.Module):
            def __init__(self): super().__init__(); self.module=Module()
            def zero_optimization_stage(self): return 1
        class Tokens:
            attention_mask=torch.ones(2,2,dtype=torch.bool)
            def to(self, device): return self
        tokenizer=SimpleNamespace(eos_token="END")
        class Tokenizer:
            eos_token="END"
            def __call__(self,*args,**kwargs): return Tokens()
        ns.update(chat_format_prompts=lambda t,p,c,enable_thinking=True:p,
                  encode_text=lambda e,i,l:torch.zeros(2,2,4), wandb=SimpleNamespace(), swanlab=SimpleNamespace())
        vae=SimpleNamespace(dtype=torch.float32,config=SimpleNamespace(scale_factor_spatial=16,z_dim=1,latents_mean=[0.],latents_std=[1.]),
             decode=lambda z:SimpleNamespace(sample=torch.nn.functional.interpolate(z[:,:,0].repeat(1,3,1,1),scale_factor=16,mode="nearest").unsqueeze(2)))
        cfg={"components":{"text_encoder":{}},"scheduler":{"shift":3},"validation":{"num_inference_steps":2,
             "fixed":{"enabled":True,"seed":42,"image_size":[32,32],"prompts":[{"id":"count","prompt":"three apples"},{"id":"text","prompt":"HELLO WORLD"}]}}}
        plan=ns["fixed_validation_plan"](cfg)
        self.assertEqual([p["seed"] for p in plan],[42,43])
        engine=Engine()
        with tempfile.TemporaryDirectory() as tmp:
            rng=torch.random.get_rng_state().clone()
            for step in [100,200]:
                ns["run_fixed_validation"](engine,vae,Tokenizer(),None,"cpu",torch.float32,step,0,cfg,Path(tmp),logging.getLogger("test"),None)
            torch.testing.assert_close(torch.random.get_rng_state(),rng)
            for case in ["count","text"]:
                a=Path(tmp)/"validation_fixed/step-00000100"/f"{case}.png"
                b=Path(tmp)/"validation_fixed/step-00000200"/f"{case}.png"
                self.assertTrue(a.is_file())
                self.assertEqual(a.read_bytes(),b.read_bytes())
                meta=json.loads(b.with_suffix(".json").read_text())
                self.assertEqual(meta["group"],"fixed")
            self.assertTrue(engine.training)


def run_deepspeed_loop():
    import deepspeed
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
    device = torch.device("cuda", local_rank)
    ns = helper_namespace(dist)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.bfloat16))
        def forward(self, x, timestep, text, mask):
            return x * self.weight
        def semantic_metadata(self):
            return {"test": True}

    class TokenBatch:
        def __init__(self, n): self.attention_mask = torch.ones(n, 2, dtype=torch.bool, device=device)
        def to(self, device): return self

    class ToyStream:
        def __init__(self): self.attempts = 0
        def state_dict(self, pending_batch=False):
            return {"epoch":0,"batch_offset":self.attempts-int(pending_batch),"sampler_signature":"test"}
        def next_batch(self):
            sizes = [2, 2, 1, 2, 1, 2, 1, 2, 1]
            i = self.attempts
            self.attempts += 1
            if i == 1 and rank == 0:
                raise DataError("injected bad data")
            n = sizes[i]
            return {"pixel_values": np.ones((n, 1, 1, 1, 1), np.float32), "captions": ["test"] * n}

    model = ToyModel()
    optimizer = ns["build_optimizer"](model, {"learning_rate": 0.01, "muon": {"enabled": False},
                                            "adam": {"enabled": True, "weight_decay": 0.005}})
    ds_cfg = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 2,
              "train_batch_size": 2 * dist.get_world_size(), "zero_allow_untested_optimizer": True,
              "bf16": {"enabled": True}, "zero_optimization": {"stage": 1}, "steps_per_print": 10000}
    engine, ds_optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_cfg)
    locations = [tempfile.mkdtemp(prefix="giga-checkpoint-test-") if rank == 0 else None]
    dist.broadcast_object_list(locations, src=0)
    directory = Path(locations[0])
    if rank == 0:
        from safetensors.torch import save_file
        for n in [4, 5]:
            save_file({"weight": model.weight.detach().cpu()}, str(directory/f"step-{n:08d}.safetensors"),
                      metadata={"samples_seen":"123", "step":str(n)})
    dist.barrier()
    original_metadata = ns["_read_checkpoint_metadata"]
    def rank_specific_read(path):
        if rank == 1 and path.name == "step-00000005.safetensors":
            raise OSError("injected rank-local storage failure")
        return original_metadata(path)
    def load_model(module, state, metadata):
        module.load_state_dict(state)
        return 1, 0, 0
    ns["load_compatible_state_dict"] = load_model
    with patch.dict(ns, {"_read_checkpoint_metadata":rank_specific_read}):
        restored = ns["load_pure_weights"](engine, directory, str(directory))
    assert restored == (4 if dist.get_world_size() > 1 else 5), restored
    assert engine.giga_samples_seen == 123
    dist.barrier()
    if rank == 0:
        __import__("shutil").rmtree(directory)
        print("PASS checkpoint fallback: every rank selected the same readable snapshot",flush=True)
    engine.module.patch_size = (1, 1, 1)
    engine.save_checkpoint = Mock()
    payloads, lr_steps = [], []
    actual_set_lr = ns["set_lr"]
    def tracked_lr(*args):
        lr_steps.append(args[2])
        return actual_set_lr(*args)
    ns["set_lr"] = tracked_lr
    cfg = {"training": {"learning_rate": 0.01, "lr_warmup_steps": 2, "total_steps": 4,
                        "cfg_dropout": 0, "checkpointing_steps": 1000},
           "scheduler": {"shift": 3}, "validation": {"enabled": False}}
    stream = ToyStream()
    from optimizations.mhc_monitoring import TrainingMonitor
    ns.update(monitor=TrainingMonitor(cfg, device, engine.module, mhc_enabled=False),
              engine=engine, ds_optimizer=ds_optimizer, client_optimizer=optimizer,
              cfg=cfg, device=device, dtype=torch.bfloat16, rank=rank, step=0, max_steps=4,
              grad_accum=2, SIGMA_BUCKETS=10, samples_seen=0, pending_batch=None,
              consecutive_batch_skips=0, max_consecutive_batch_skips=4, data_stream=stream,
              vae=SimpleNamespace(encode=lambda pixels: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: pixels))),
              latents_mean=0., latents_std_inv=1., text_encoder=None,
              tokenizer=lambda prompts, **kwargs: TokenBatch(len(prompts)), tc={},
              chat_format_prompts=lambda tokenizer, prompts, flag, enable_thinking=True: prompts,
              encode_text=lambda encoder, inputs, layer: torch.zeros(inputs.attention_mask.shape[0], 2, 4, device=device),
              progress=SimpleNamespace(update=lambda *a: None, set_postfix=lambda **k: None),
              log_every=1, logger=logging.getLogger("test"), wandb_run=None, swanlab_run=True,
              swanlab=SimpleNamespace(log=lambda data, step: payloads.append((step, data))),
              sigma_bucket_ema=[float("nan")]*10, math=__import__("math"), hashlib=__import__("hashlib"),
              checkpoint_dir=Path("/tmp/unused_stage1_regression"), output_dir=Path("/tmp/unused_stage1_regression"),
              save_pure_weights=lambda *a: None, prune_state_checkpoints=lambda *a: None)
    main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in main.body if isinstance(n, ast.While))
    with patch.object(torch, "rand", side_effect=lambda *shape, **kw: torch.full(shape, 0.5, **kw)):
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(SOURCE), "exec"), ns)
    assert engine.global_steps == 4, engine.global_steps
    assert engine.micro_steps == 8, engine.micro_steps
    assert ns["step"] == 4
    assert stream.attempts == 9
    assert ns["samples_seen"] == 12 * dist.get_world_size()
    assert lr_steps == [1, 2, 3, 4], lr_steps
    assert torch.isfinite(model.weight).all()
    assert engine.save_checkpoint.call_args.kwargs["client_state"]["samples_seen"] == ns["samples_seen"]
    assert engine.save_checkpoint.call_args.kwargs["client_state"]["data_cursor"]["batch_offset"] == 9
    if rank == 0:
        assert [s for s, _ in payloads] == [1, 2, 3, 4]
        assert [p["train/samples"] for _, p in payloads] == [3 * dist.get_world_size() * i for i in range(1, 5)]
        assert all(abs(p["train/sigma_mean"] - 0.75) < 1e-6 for _, p in payloads)
        print("PASS real training loop with production AdamW wrapper: 4 updates / 8 valid micro-batches; one globally skipped bad batch; mixed-batch samples and sigma correct", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "--deepspeed" in sys.argv:
        run_deepspeed_loop()
    else:
        unittest.main()
