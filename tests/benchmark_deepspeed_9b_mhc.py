"""Bounded, single-bucket 8-GPU 9B/VAE/mHC/AdamW/ZeRO-1 throughput benchmark.

Run one configuration per process invocation. For equal effective batches,
compare 512px B1/accum2 with B2/accum1, or 256px B2/accum2 with B4/accum1.
Synthetic image/noise tensors are prepared outside timing; CPU tokenization,
frozen text encoding, VAE, full DiT backward and optimizer updates are timed.
With --real-data, use the YAML's mixed bucket sampler, copied mmap index and
real tar/NFS images instead; captions use dynamic padding with max_length=2048.
This mode defaults to 20 timed updates and includes fetch/recovery waiting.
No training main, tracker, validation or checkpoint save is invoked.
CUDA synchronization occurs only at the timed-region boundaries. Per-update
times use default-stream CUDA events, including stream idle gaps; they are not
exclusive kernel times. Global samples/s uses the slowest rank's wall time.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "thirdparty")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("SWANLAB_MODE", "disabled")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_1_5b_qwen35_9b_mhc_tar.yaml")
    parser.add_argument("--image-size", type=int, choices=(256, 512), default=512)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--text-tokens", type=int, choices=(128, 512, 2048), default=None)
    parser.add_argument("--real-data", action="store_true", help="Use actual YAML tar/NFS data and mixed buckets; no synthetic fallback")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(value is not None and value <= 0 for value in (args.micro_batch, args.grad_accum, args.updates)) or args.warmup < 0:
        parser.error("batch, accumulation and timed updates must be positive; warmup must be nonnegative")
    if args.real_data and args.text_tokens is not None:
        parser.error("--text-tokens is synthetic-only; real captions use dynamic padding capped at 2048")
    return args


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def training_helpers():
    spec = importlib.util.spec_from_file_location("bounded_benchmark_helpers", ROOT / "scripts/train/train_giga_exp.py")
    module = importlib.util.module_from_spec(spec)
    # Needed by pickling if a later benchmark mode uses DataLoader workers.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ObservedDataStream:
    """Time only CPU next_batch calls, preserving production recovery semantics."""

    def __init__(self, stream):
        self.stream = stream
        self.reset_observations()

    def reset_observations(self):
        self.waits, self.timeouts, self.errors = [], 0, 0
        self.recovery_seconds = 0.

    def next_batch(self):
        started = time.perf_counter()
        try:
            return self.stream.next_batch()
        except Exception as exc:
            self.errors += 1
            self.timeouts += int("DataLoader timed out" in str(exc))
            raise
        finally:
            self.waits.append(time.perf_counter()-started)

    def restart(self):
        started = time.perf_counter()
        try:
            return self.stream.restart()
        finally:
            self.recovery_seconds += time.perf_counter()-started

    def close(self):
        return self.stream.close()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main():
    args = arguments()
    import deepspeed
    import torch
    import torch.distributed as dist
    import yaml
    from diffusers import AutoencoderKLWan
    from models.model_factory import build_diffusion_model, mhc_config
    from models.text_encoding import (build_tokenizer, build_text_encoder,
                                      chat_format_prompts, encode_text)

    rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 8 and not (args.real_data and world == 64):
        raise ValueError(f"Expected eight ranks, or 64 for a real-data test, got {world}")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    device, dtype = torch.device("cuda", local_rank), torch.bfloat16
    torch.manual_seed(args.seed + rank)
    deepspeed.init_distributed(dist_backend="nccl")
    cfg = yaml.safe_load(args.config.read_text())
    training = cfg["training"]
    args.micro_batch = args.micro_batch or (int(training["batch_size_per_device"]) if args.real_data else (1 if args.image_size == 512 else 2))
    args.grad_accum = args.grad_accum or (int(training.get("gradient_accumulation_steps", 1)) if args.real_data else 2)
    args.updates = args.updates or (20 if args.real_data else 4)
    args.text_tokens = 2048 if args.real_data else (args.text_tokens or 128)
    total_updates = args.warmup + args.updates
    training.setdefault("total_steps", total_updates + 1)
    training["batch_size_per_device"] = args.micro_batch
    training["gradient_accumulation_steps"] = args.grad_accum
    if training.get("muon", {}).get("enabled", True):
        raise ValueError("This benchmark requires AdamW: set training.muon.enabled=false")
    if not mhc_config(cfg)["enabled"]:
        raise ValueError("This benchmark requires mhc.enabled=true")
    bucket_label = "real_mixed" if args.real_data else str(args.image_size)
    output = args.output or (Path(cfg["output_dir"]) / "preflight" /
        f"benchmark_{args.label}_{bucket_label}_b{args.micro_batch}_a{args.grad_accum}_t{args.text_tokens}.json")
    stage, started = "initialization", time.monotonic()
    data_stream = None
    stop_heartbeat = threading.Event()
    heartbeat = None
    if args.real_data:
        def beat():
            while not stop_heartbeat.wait(20):
                print(f"BENCH_HEARTBEAT rank={rank} stage={stage} elapsed={time.monotonic()-started:.1f}s", flush=True)
        heartbeat = threading.Thread(target=beat, daemon=True)
        heartbeat.start()
    try:
        helpers = training_helpers()
        tc, dc = cfg["components"]["text_encoder"], cfg["components"]["wan_vae"]
        tokenizer = build_tokenizer(tc)
        encoder = build_text_encoder(tc, dtype, device).eval().requires_grad_(False)
        model = build_diffusion_model(cfg).to(device=device, dtype=dtype)
        if training.get("torch_compile", False):
            from torch._dynamo import config as dynamo_config
            from torch.fx.experimental import _config as fx_config
            dynamo_config.cache_size_limit = 64
            dynamo_config.assume_static_by_default = False
            fx_config.use_duck_shape = False
            model.compile(dynamic=True)
        source = training.get("init_from")
        if not source:
            raise ValueError("training.init_from must specify the fixed comparison checkpoint")
        snapshot = helpers.load_init_weights(model, source, strict=bool(training.get("init_from_strict", True)))
        model.set_mhc_collect_stats(False)
        vae = AutoencoderKLWan.from_pretrained(dc["pretrained_path"], subfolder=dc.get("subfolder", "vae"),
            torch_dtype=dtype, local_files_only=True).eval().requires_grad_(False).to(device)
        mean, invstd = helpers.wan_latent_stats(vae, device)
        optimizer = helpers.build_optimizer(model, training)
        if optimizer.muon is not None or not isinstance(optimizer.adam, torch.optim.AdamW):
            raise TypeError("Benchmark optimizer must be actual AdamW")
        ds_path = Path(cfg["distributed"]["config"])
        ds_cfg = json.loads((ds_path if ds_path.is_absolute() else ROOT / ds_path).read_text())
        if ds_cfg["zero_optimization"]["stage"] != 1:
            raise ValueError("Benchmark requires DeepSpeed ZeRO stage 1")
        effective_batch = args.micro_batch * args.grad_accum
        ds_cfg.update(train_micro_batch_size_per_gpu=args.micro_batch,
            gradient_accumulation_steps=args.grad_accum, train_batch_size=effective_batch*world,
            steps_per_print=1000000, wall_clock_breakdown=False)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
            optimizer=optimizer, config=ds_cfg)
        engine.train()

        # Per-sample seeds make the source pixels, posterior noise, flow noise
        # and sigma identical when only micro-batch/accumulation grouping changes.
        samples = total_updates * effective_batch
        def random_bank(shape, salt, uniform=False):
            operation = torch.rand if uniform else torch.randn
            return torch.stack([operation(shape, device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(args.seed + rank*1000003 + i*101 + salt))
                for i in range(samples)])
        shift = float(cfg["scheduler"]["shift"])
        if not args.real_data:
            pixels = random_bank((3, 1, args.image_size, args.image_size), 1)
            with torch.no_grad():
                probe = vae.encode(pixels[:1]).latent_dist
                latent_shape = tuple(probe.mean.shape[1:])
            del probe
            posterior_noise = random_bank(latent_shape, 2)
            flow_noise = random_bank(latent_shape, 3).float()
            sigmas = random_bank((), 4, uniform=True).float()
            sigmas = sigmas * shift / (1 + (shift-1)*sigmas)
            prompt = "A detailed photograph of three red apples on a white table, with a blue vase and a distant green forest. "
            prompt *= args.text_tokens // max(len(tokenizer.encode(prompt)), 1) + 8
            prompts = chat_format_prompts(tokenizer, [prompt]*samples,
                tc.get("use_chat_template"), tc.get("enable_thinking", True))
            probe_tokens = tokenizer(prompts[:1], return_tensors="pt", padding="max_length",
                truncation=True, max_length=args.text_tokens)
            if probe_tokens.attention_mask.sum().item() != args.text_tokens:
                raise RuntimeError("Synthetic prompt did not fill the requested valid token length")
            del probe_tokens
        else:
            stage = "real_dataset_mmap_index"
            if os.environ.get("GIGA_INDEX_BACKEND", "mmap") != "mmap":
                raise ValueError("--real-data requires GIGA_INDEX_BACKEND=mmap")
            if cfg["dataset"].get("dataset_type") != "tar_parquet":
                raise ValueError("--real-data requires the real tar_parquet dataset")
            cache_dir = Path(cfg["output_dir"]) / "dataset_index_cache"
            if not cache_dir.is_dir() or not any(cache_dir.glob("index_*/sid.npy")):
                raise FileNotFoundError(f"Copied mmap index is missing from this experiment: {cache_dir}")
            logger = logging.getLogger(f"bounded_real_data_rank_{rank}")
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler())
            data = helpers.GigaLMDBDataset(cfg["dataset"], seed=int(cfg.get("seed", 42)), index_cache_dir=str(cache_dir))
            import numpy as np
            if not data.bucket_indices or not all(isinstance(value, np.memmap) for value in data.bucket_indices.values()):
                raise TypeError("Real dataset did not retain mmap bucket indices")
            sampler = helpers.BucketBatchSampler(data.bucket_indices, args.micro_batch,
                weights=cfg["dataset"].get("bucket_sampling_weights"),
                bucket_batch_sizes=cfg["dataset"].get("bucket_batch_sizes"),
                seed=int(cfg.get("seed", 42)), rank=rank, world_size=world,
                randomize_order_on_start=True, drop_last=True)
            if len(sampler) <= 0:
                raise ValueError("Real dataset has no complete training batches")
            data_stream = ObservedDataStream(helpers.make_training_data_stream(data, sampler, training, logger))
            if training.get("ready_data_pipeline", False):
                status = data_stream.wait_ready(int(training.get("ready_min_batches", 8)),
                    float(training.get("ready_warmup_timeout_seconds", 180)))
                failed = torch.tensor([not status["ready"]], device=device, dtype=torch.int32)
                dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                if failed.item():
                    raise RuntimeError("Real-data benchmark did not reach the configured ready-batch watermark")
            print(f"BENCH_REAL_DATA_READY rank={rank} records={len(data.records)} cache={cache_dir} sampler_steps={len(sampler)}", flush=True)

        losses, update_events, submission_seconds, update_records = [], [], [], []
        old_step = engine.global_steps
        old_skipped = int(getattr(engine, "skipped_steps", 0))
        def update(index, timed):
            nonlocal stage
            stage = f"{'timed' if timed else 'warmup'}_update_{index+1}"
            before = engine.global_steps
            helpers.set_lr(optimizer, engine.optimizer, index+1, training)
            if data_stream is not None:
                data_stream.reset_observations()
            sample_count = text_valid_count = text_padded_count = skipped = 0
            fetch_seconds, shapes = 0., []
            micro = 0
            while micro < args.grad_accum:
                first = index*effective_batch + micro*args.micro_batch
                selection = slice(first, first+args.micro_batch)
                if args.real_data:
                    stage = f"{'timed' if timed else 'warmup'}_update_{index+1}_micro_{micro}_data"
                    fetch_start = time.perf_counter()
                    batch = helpers.next_training_batch(data_stream, device, logger, index+1, micro)
                    fetch_seconds += time.perf_counter()-fetch_start
                    if batch is None:
                        skipped += 1
                        if skipped > int(os.environ.get("GIGA_MAX_CONSECUTIVE_BATCH_SKIPS", "64")):
                            raise RuntimeError("Real-data benchmark exceeded its skipped-batch limit")
                        continue
                    image = torch.from_numpy(batch["pixel_values"]).to(device=device)
                    captions = [str(value).strip() or tokenizer.eos_token for value in batch["captions"]]
                else:
                    image, captions = pixels[selection], prompts[selection]
                stage = f"{'timed' if timed else 'warmup'}_update_{index+1}_micro_{micro}_compute"
                with torch.no_grad():
                    posterior = vae.encode(image.to(dtype)).latent_dist
                    latent_sample = posterior.sample() if args.real_data else posterior.mean + posterior.std*posterior_noise[selection]
                    latents = (latent_sample.float()-mean)*invstd
                    if args.real_data:
                        probability = float(training.get("cfg_dropout", 0.))
                        if probability:
                            if training.get("cpu_cfg_dropout", False):
                                seed = int(cfg.get("seed", 42)) + rank*1000003 + data_stream.epoch*1000000007 + data_stream.batch_offset
                                generator = torch.Generator(device="cpu").manual_seed(seed)
                                drop = (torch.rand(len(captions), generator=generator) < probability).tolist()
                            else:
                                drop = (torch.rand(len(captions), device=device) < probability).tolist()
                            captions = [(tokenizer.eos_token or " ") if empty else caption for empty, caption in zip(drop, captions)]
                        captions = chat_format_prompts(tokenizer, captions, tc.get("use_chat_template"), tc.get("enable_thinking", True))
                    inputs_cpu = tokenizer(captions, return_tensors="pt", padding=True if args.real_data else "max_length",
                        truncation=True, max_length=args.text_tokens)
                    text_valid_count += int(inputs_cpu.attention_mask.sum())
                    inputs = inputs_cpu.to(device)
                    features = encode_text(encoder, inputs, tc.get("layer"))
                    mask = inputs.attention_mask.bool()
                    if features.shape[1] < 2:
                        features = torch.nn.functional.pad(features, (0, 0, 0, 2-features.shape[1]))
                        mask = torch.nn.functional.pad(mask, (0, 2-mask.shape[1]))
                    text_padded_count += features.shape[0]*features.shape[1]
                sample_count += image.shape[0]
                shapes.append({"batch": image.shape[0], "height": image.shape[-2], "width": image.shape[-1], "padded_text_tokens": features.shape[1]})
                if args.real_data:
                    sigma = torch.rand(latents.shape[0], device=device)
                    sigma = sigma*shift/(1+(shift-1)*sigma)
                    noise = torch.randn_like(latents)
                else:
                    sigma, noise = sigmas[selection], flow_noise[selection]
                sigma_view = sigma[:, None, None, None, None]
                xt = (1-sigma_view)*latents + sigma_view*noise
                prediction = engine(xt.to(dtype), sigma*1000, features, mask)
                loss = (prediction.float()-(noise-latents)).square().mean()
                engine.backward(loss)
                engine.step()
                if timed:
                    losses.append(loss.detach())
                del posterior, latent_sample, latents, inputs, inputs_cpu, features, mask, image, sigma, noise, sigma_view, xt, prediction, loss
                micro += 1
            if engine.global_steps != before+1:
                raise RuntimeError(f"Expected one actual optimizer update, got {engine.global_steps-before}")
            if timed:
                record = {"optimizer_update": index+1, "samples": sample_count,
                    "valid_text_tokens": text_valid_count, "padded_text_tokens": text_padded_count,
                    "micro_batches": shapes}
                if data_stream is not None:
                    waits = data_stream.waits
                    record.update(cpu_data_wait_seconds=sum(waits),
                        cpu_data_wait_p50=percentile(waits, .5), cpu_data_wait_p95=percentile(waits, .95),
                        cpu_data_wait_max=max(waits, default=0.), fetch_wall_seconds=fetch_seconds,
                        cpu_data_wait_attempts_seconds=list(waits), skipped_batches=skipped,
                        local_data_errors=data_stream.errors, local_timeouts=data_stream.timeouts,
                        worker_recovery_seconds=data_stream.recovery_seconds)
                update_records.append(record)
            if args.real_data:
                print(f"BENCH_UPDATE rank={rank} update={index+1} phase={'timed' if timed else 'warmup'} samples={sample_count} fetch_s={fetch_seconds:.3f} skips={skipped} timeouts={data_stream.timeouts}", flush=True)

        for index in range(args.warmup):
            update(index, False)
        # Synchronize at region boundaries only, never between measured updates.
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        region_start = time.perf_counter()
        for index in range(args.warmup, total_updates):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            submit_start = time.perf_counter()
            update(index, True)
            submission_seconds.append(time.perf_counter()-submit_start)
            end.record()
            update_events.append((begin, end))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-region_start
        if engine.global_steps != old_step+total_updates:
            raise RuntimeError("DeepSpeed update count differs from the requested warmup+timed updates")
        if int(getattr(engine, "skipped_steps", 0)) != old_skipped:
            raise RuntimeError("An optimizer update was skipped; throughput result is invalid")
        local = {
            "rank": rank, "wall_seconds": elapsed,
            "update_cuda_seconds": [begin.elapsed_time(end)/1000 for begin, end in update_events],
            "update_submission_seconds": submission_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30,
            "gpu_name": torch.cuda.get_device_name(device),
            "global_steps": engine.global_steps,
            "timed_loss_finite": bool(torch.isfinite(torch.stack(losses)).all()),
            "timed_last_loss": float(losses[-1]),
            "updates": update_records,
        }
        if args.real_data:
            waits = [value for record in update_records for value in record["cpu_data_wait_attempts_seconds"]]
            local["real_data"] = {"cache_dir": str(cache_dir), "records": len(data.records),
                "data_stream_end_cursor": data_stream.state_dict(),
                "cpu_data_wait_seconds": sum(waits), "cpu_data_wait_p50": percentile(waits, .5),
                "cpu_data_wait_p95": percentile(waits, .95), "cpu_data_wait_max": max(waits, default=0.),
                "local_timeouts": sum(record["local_timeouts"] for record in update_records),
                "local_data_errors": sum(record["local_data_errors"] for record in update_records),
                "shared_skipped_batches": sum(record["skipped_batches"] for record in update_records)}
        graph = getattr(encoder, "_giga_reference_graph_cache", None)
        if graph is not None:
            local["encoder_graph_stats"] = graph.stats().copy()
        ranks = [None]*world
        dist.all_gather_object(ranks, local)
        if rank == 0:
            global_elapsed = max(result["wall_seconds"] for result in ranks)
            update_times = [max(result["update_cuda_seconds"][i] for result in ranks) for i in range(args.updates)]
            record = {
                "ok": all(result["timed_loss_finite"] for result in ranks), "mode": "real_tar_nfs" if args.real_data else "synthetic_compute",
                "label": args.label, "config": str(args.config.resolve()), "source_checkpoint": str(snapshot),
                "world_size": world, "seed": args.seed, "image_size": "mixed_yaml_buckets" if args.real_data else args.image_size,
                "micro_batch_per_gpu": args.micro_batch, "gradient_accumulation_steps": args.grad_accum,
                "effective_samples_per_update": (sum(record["samples"] for result in ranks for record in result["updates"])/args.updates),
                "text_tokens": "dynamic_max_2048" if args.real_data else args.text_tokens,
                "warmup_updates": args.warmup, "timed_updates": args.updates,
                "actual_optimizer_updates": total_updates, "steady_wall_seconds": global_elapsed,
                "steady_samples_per_second": sum(record["samples"] for result in ranks for record in result["updates"])/global_elapsed,
                "seconds_per_update": global_elapsed/args.updates,
                "update_cuda_seconds_p50": percentile(update_times, .5),
                "update_cuda_seconds_p90": percentile(update_times, .9),
                "update_cuda_seconds_slowest_rank": update_times,
                "timing_note": "Wall time includes CPU tokenization and GPU computation; CUDA event update intervals include stream idle gaps. Synthetic inputs prepared before timing; real-data mode additionally includes actual tar/NFS fetch, production rank coordination, worker recovery and per-rank heartbeat output. No added per-update CUDA synchronization, monitoring, validation or checkpoint save.",
                "model_options": cfg["model"], "text_encoder_options": tc,
                "ranks": ranks, "elapsed_total_seconds": time.monotonic()-started,
            }
            if args.real_data:
                waits = [value for result in ranks for record in result["updates"] for value in record["cpu_data_wait_attempts_seconds"]]
                record["real_data_summary"] = {"dataset_type": cfg["dataset"]["dataset_type"],
                    "database_path": cfg["dataset"]["database_path"], "cache_dir": str(cache_dir),
                    "workers_per_rank": int(training.get("num_workers", 4)),
                    "bucket_batch_sizes": cfg["dataset"].get("bucket_batch_sizes"),
                    "cpu_data_wait_seconds_all_ranks": sum(waits),
                    "cpu_data_wait_p50": percentile(waits, .5), "cpu_data_wait_p95": percentile(waits, .95),
                    "cpu_data_wait_max": max(waits, default=0.),
                    "shared_skipped_batches": max(result["real_data"]["shared_skipped_batches"] for result in ranks),
                    "local_timeouts_all_ranks": sum(result["real_data"]["local_timeouts"] for result in ranks)}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(record, indent=2)+"\n")
            print("EIGHT_GPU_9B_MHC_BENCHMARK_COMPLETE", str(output),
                json.dumps({key: record[key] for key in ("steady_samples_per_second", "seconds_per_update", "update_cuda_seconds_p50", "update_cuda_seconds_p90")}), flush=True)
        dist.barrier()
    except Exception as exc:
        print(f"BENCHMARK_FAILED rank={rank} stage={stage}: {exc}\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        stop_heartbeat.set()
        if heartbeat is not None:
            heartbeat.join(timeout=1)
        try:
            if data_stream is not None:
                data_stream.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
