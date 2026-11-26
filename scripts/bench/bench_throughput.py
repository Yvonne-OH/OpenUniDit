#!/usr/bin/env python3
"""吞吐微基准: 复刻训练计算结构 (冻结文本编码器 + DeepSpeed ZeRO2 + flow matching),
但不启动数据管道/训练循环, 用合成数据测 512x512 下不同 (batch, accum, AC ratio, 预取) 的吞吐。

用法 (torchrun 8 卡):
  torchrun --nnodes 1 --nproc_per_node 8 --master_port 29550 \
    scripts/bench/bench_throughput.py --config configs/stage_1_5b_qwen35_4b_tar.yaml \
    --batch-size 3 --accum 1 --ac-ratio 0.8 --prefetch 0 --steps 16 --warmup 6 \
    --output exp/bench_throughput/bs3_a1_ac0.8_pf0.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import deepspeed
import torch
import torch.distributed as dist
import yaml

from models.model_factory import build_diffusion_model
from models.text_encoding import build_text_encoder, encode_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_1_5b_qwen35_4b_tar.yaml")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accum", type=int, required=True)
    parser.add_argument("--ac-ratio", type=float, required=True)
    parser.add_argument("--prefetch", type=int, required=True, choices=[0, 1])
    parser.add_argument("--text-len", type=int, default=512, help="每个样本的文本 token 数 (固定, 可比)")
    parser.add_argument("--warmup", type=int, default=6, help="预热 optimizer 步数 (不计入统计)")
    parser.add_argument("--steps", type=int, default=16, help="测量 optimizer 步数")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def write_result(path: Path, payload: dict, rank: int):
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"BENCH_RESULT {json.dumps(payload, ensure_ascii=False)}")


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    dist.init_process_group("nccl")
    rank = int(dist.get_rank())
    # torchrun 注入 LOCAL_RANK; 必须先 set_device 再建 device, 否则所有 rank 都落 GPU 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    # 覆盖 AC 配置 (bench 变量)
    cfg["model"]["activation_checkpointing"] = args.ac_ratio > 0
    cfg["model"]["activation_checkpointing_ratio"] = args.ac_ratio
    # bench 不需要监控/验证/保存
    cfg.setdefault("monitoring", {})["enabled"] = False

    micro_bs = args.batch_size
    grad_accum = args.accum
    text_len = args.text_len
    latents_shape = (micro_bs, 48, 1, 64, 64)  # 512x512 → 4096 latent token/样本, 与真实一致

    # 预生成合成数据池 (CPU), 循环使用, 避免 host 端生成噪声混入计时
    torch.manual_seed(42)
    pool_size = grad_accum * (args.warmup + args.steps) + 2
    latent_pool = [torch.randn(*latents_shape) for _ in range(pool_size)]
    vocab = 150000
    ids_pool = [torch.randint(1000, vocab, (micro_bs, text_len)) for _ in range(pool_size)]
    mask_pool = [torch.ones(micro_bs, text_len, dtype=torch.long) for _ in range(pool_size)]

    text_encoder = build_text_encoder(cfg["components"]["text_encoder"], torch.bfloat16, device)

    def encode_inputs(ids, mask):
        inputs = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
        hidden = encode_text(text_encoder, inputs, layer=int(cfg["components"]["text_encoder"].get("layer", -1)))
        return hidden, mask.to(device)

    model = build_diffusion_model(cfg).to(device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.005)
    ds_config = json.loads((ROOT / cfg["distributed"]["config"]).read_text())
    ds_config["train_micro_batch_size_per_gpu"] = micro_bs
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = micro_bs * grad_accum * dist.get_world_size()
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                           optimizer=optimizer, config=ds_config)

    text_stream = torch.cuda.Stream(device) if args.prefetch else None
    shift = float(cfg["scheduler"]["shift"])
    total_steps = args.warmup + args.steps
    pool_idx = 0

    # 文本编码计时 (CUDA events, rank 本地)
    encode_seconds = 0.0
    encode_calls = 0

    def synth_text(i):
        return encode_inputs(ids_pool[i % pool_size], mask_pool[i % pool_size])

    prepared_text = None
    lookahead_i = None  # 预取模式下: 已发起编码的下一 micro 的全局 micro 序号
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    measure_start = None
    measured_updates = 0
    oom = None
    try:
        for update in range(total_steps):
            boundary_samples = 0
            for micro in range(grad_accum):
                idx = pool_idx
                pool_idx += 1
                latents = latent_pool[idx % pool_size].to(device, torch.bfloat16, non_blocking=True)
                if prepared_text is not None:
                    train_stream = torch.cuda.current_stream(device)
                    train_stream.wait_stream(text_stream)
                    text, mask = prepared_text
                    text.record_stream(train_stream)
                    mask.record_stream(train_stream)
                    prepared_text = None
                else:
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    text, mask = synth_text(idx)
                    ev1.record()
                    torch.cuda.synchronize()
                    encode_seconds += ev0.elapsed_time(ev1) / 1000.0
                    encode_calls += 1
                sigma = torch.rand(latents.shape[0], device=device)
                sigma = sigma * shift / (1 + (shift - 1) * sigma)
                noise = torch.randn_like(latents)
                xt = (1 - sigma[:, None, None, None, None]) * latents + sigma[:, None, None, None, None] * noise
                # 预取: 在 side stream 上编码下一 micro 的文本, 与 DiT 前向/反向重叠
                if text_stream is not None:
                    next_i = pool_idx
                    text_stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(text_stream):
                        prepared_text = synth_text(next_i)
                pred = engine(xt.to(torch.bfloat16), sigma * 1000, text, mask)
                diff_sq = (pred.float() - (noise - latents).float()).square()
                loss = diff_sq.flatten(1).mean(1).mean()
                engine.backward(loss)
                engine.step()
                boundary_samples += latents.shape[0]
            measured_updates += 1
            if update + 1 == args.warmup:
                torch.cuda.synchronize()
                measure_start = time.perf_counter()
                measure_samples = 0
                encode_seconds = 0.0  # 重置: 只统计测量窗口
                encode_calls = 0
            if update + 1 > args.warmup:
                measure_samples += boundary_samples
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        write_result(Path(args.output), {
            "name": Path(args.output).stem, "batch_size": micro_bs, "accum": grad_accum,
            "ac_ratio": args.ac_ratio, "prefetch": bool(args.prefetch), "text_len": text_len,
            "oom": True, "error": repr(exc)[:200]}, rank)
        return
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - measure_start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    samples_per_s = measure_samples / elapsed
    world = dist.get_world_size()
    write_result(Path(args.output), {
        "name": Path(args.output).stem, "batch_size": micro_bs, "accum": grad_accum,
        "ac_ratio": args.ac_ratio, "prefetch": bool(args.prefetch), "text_len": text_len,
        "world_size": world, "measured_updates": measured_updates - args.warmup,
        "samples_per_s": round(samples_per_s, 2),
        "global_samples_per_s": round(samples_per_s * world, 2),
        "s_per_update": round(elapsed / max(measured_updates - args.warmup, 1), 4),
        "encode_s_per_micro": round(encode_seconds / max(encode_calls, 1), 4),
        "peak_mem_gib": round(peak_gib, 2), "oom": False,
    }, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
