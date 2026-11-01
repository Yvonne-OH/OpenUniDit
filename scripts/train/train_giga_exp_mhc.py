"""Stage_1 Giga-Exp mHC 训练 (独立 fork, 不触碰 train_giga_exp.py 主分支)。

与 train_giga_exp.py 的差异:
1. 模型为 GigaExpTransformerMHC (configs/mhc_2stream.yaml, mhc.enabled 必须显式 true);
2. 优化器分组: 原模型参数走 AdamW (base lr), mHC 新参数 (mapping/readout) 走独立
   AdamW (mhc_lr), 两组共用 warmup+cosine 调度 (按 _lr_scale 缩放);
3. init_from 加载 mHC 转换权重时使用 strict load (缺键/多键直接报错);
4. logging 每 log_every_n_steps 追加 mhc/* 映射统计 (按 layer 分组, 非全层平均)。

可复用组件 (collate/数据/验证/断点恢复/纯权重保存) 直接 import 自 train_giga_exp,
行为与主分支逐行一致。
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
THIRDPARTY = ROOT / "thirdparty"
if str(THIRDPARTY) not in sys.path:
    sys.path.insert(0, str(THIRDPARTY))

import deepspeed
import torch
import torch.distributed as dist
import swanlab
import wandb
import yaml
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from data.lmdb_dataset import GigaLMDBDataset
from data.samplers.bucket_sampler import BucketBatchSampler
from models.giga_transformer_mhc import GigaExpTransformerMHC
from train_giga_exp import (  # noqa: E402  主分支可复用组件
    SIGMA_BUCKETS,
    _blue_print,
    _data_error_category,
    _loud_phase,
    _semantic_metadata,
    build_text_encoder,
    collate,
    encode_text,
    latest_state_tag,
    load_pure_weights,
    prune_state_checkpoints,
    run_next_batch_validation,
    save_pure_weights,
    setup_logging,
    wan_latent_stats,
)


class AdamWGroups(torch.optim.Optimizer):
    """双组 AdamW: base 参数 (base_lr) + mHC 参数 (mhc_lr), 共享调度 (经 _lr_scale)。

    set_lr (train_giga_exp) 按 group['_lr_scale'] 缩放 base lr, 天然支持两组不同 lr。
    """

    def __init__(self, base_params, mhc_params, adam_cfg, lr, mhc_lr):
        betas = tuple(adam_cfg.get("betas", [0.9, 0.95]))
        eps = float(adam_cfg.get("epsilon", 1e-8))
        wd = float(adam_cfg.get("weight_decay", 0.0))
        self.adam = torch.optim.AdamW(list(base_params), lr=lr, betas=betas, eps=eps, weight_decay=wd)
        self.mhc_adam = torch.optim.AdamW(list(mhc_params), lr=mhc_lr, betas=betas, eps=eps, weight_decay=wd)
        groups = self.adam.param_groups + self.mhc_adam.param_groups
        for g in groups:
            g["_lr_scale"] = g["lr"] / lr
        params = [p for g in groups for p in g["params"]]
        super().__init__(params, {})
        self.param_groups = groups

    def step(self, closure=None):
        self.adam.step(closure)
        return self.mhc_adam.step(closure)

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.mhc_adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"adam": self.adam.state_dict(), "mhc_adam": self.mhc_adam.state_dict()}

    def load_state_dict(self, state_dict):
        if state_dict.get("adam"):
            self.adam.load_state_dict(state_dict["adam"])
        if state_dict.get("mhc_adam"):
            self.mhc_adam.load_state_dict(state_dict["mhc_adam"])


def build_optimizer_mhc(model, cfg):
    """mHC 参数 (mapping/readout) -> mhc_lr 组; 其余 -> base 组。muon 不支持 (第一版)。"""
    if cfg.get("muon", {}).get("enabled"):
        raise NotImplementedError("mHC 第一版训练仅支持 adamw; muon.enabled 必须为 false")
    lr = float(cfg.get("learning_rate", 1e-4))
    mhc_lr = float(cfg.get("mhc_lr", lr))
    base_params, mhc_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (mhc_params if ("mhc" in name or name == "readout") else base_params).append(p)
    if not mhc_params:
        raise ValueError("模型中没有 mHC 参数, 请确认使用 GigaExpTransformerMHC")
    return AdamWGroups(base_params, mhc_params, cfg.get("adam", {}), lr, mhc_lr)


def load_mhc_init_weights(engine, output_dir, checkpoint_path=None):
    """strict 加载 mHC 转换权重 (键集合必须完全一致); 兼容目录自动取最新 step-*。"""
    target = Path(checkpoint_path).expanduser() if checkpoint_path else None
    if target is None:
        weight_dir = output_dir / "checkpoints" / "weights"
        candidates = sorted(weight_dir.glob("step-*.safetensors")) if weight_dir.is_dir() else []
        if not candidates:
            return 0
        target = max(candidates, key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    elif target.is_dir():
        candidates = sorted(target.glob("step-*.safetensors")) + sorted(target.glob("step-*.pt"))
        if not candidates:
            raise FileNotFoundError(f"指定目录没有权重文件: {target}")
        target = max(candidates, key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    if not target.is_file():
        raise FileNotFoundError(f"权重不存在: {target}")
    import re
    from safetensors.torch import load_file
    state = load_file(str(target)) if target.suffix == ".safetensors" else torch.load(str(target), map_location="cpu")
    if target.name.startswith("converted"):
        # 转换权重: 键集合必须与模型完全一致 (strict), 杜绝静默吞键
        engine.module.load_state_dict(state)  # strict=True
        engine.optimizer.refresh_fp32_params()
        step = 0
        logging.getLogger("giga_exp").info("strict 加载 mHC 转换权重: %s (%d tensors)", target, len(state))
    else:
        step = load_pure_weights(engine, output_dir, str(target))
    return step


def print_banner(cfg, rank):
    if rank != 0:
        return
    model = cfg["model"]
    print(f"🧠 Giga-Exp mHC stage_1 | dim={model['dim']} layers={model['layers']} "
          f"n_streams={model['n_streams']} sinkhorn_iters={model.get('sinkhorn_iters', 20)}", flush=True)
    print(f"⚙️  optimizer=adamw+adamw | base_lr={cfg['training']['learning_rate']} "
          f"mhc_lr={cfg['training'].get('mhc_lr')} | mhc.enabled={cfg['mhc']['enabled']} "
          f"init_mode={cfg['mhc'].get('init_mode')}", flush=True)
    print(f"💾 Output: {cfg['output_dir']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mhc_2stream.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text()) if not Path(args.config).is_absolute() \
        else yaml.safe_load(Path(args.config).read_text())
    if not cfg.get("mhc", {}).get("enabled"):
        raise ValueError("mHC 训练脚本要求 configs 显式 mhc.enabled: true")
    deepspeed.init_distributed()
    rank = dist.get_rank()
    print_banner(cfg, rank)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    output_dir = Path(cfg["output_dir"])
    logger = setup_logging(output_dir, rank)
    checkpoint_dir = output_dir / "checkpoints" / "state"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tc, dc = cfg["components"]["text_encoder"], cfg["components"]["wan_vae"]
    dtype = torch.bfloat16 if cfg["training"].get("mixed_precision", "bf16") == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(tc["tokenizer_path"], local_files_only=True)
    text_encoder = build_text_encoder(tc, dtype, device)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(dc["pretrained_path"], subfolder=dc.get("subfolder", "vae"),
                                           torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    if cfg["training"].get("torch_compile"):
        raise NotImplementedError("mHC 第一版关闭 torch.compile (Sinkhorn 循环 eager 保证数值正确)")
    model = GigaExpTransformerMHC(**cfg["model"]).to(device, dtype=dtype)
    latents_mean, latents_std_inv = wan_latent_stats(vae, device)
    for p in (*text_encoder.parameters(), *vae.parameters()):
        p.requires_grad_(False)

    init_from = cfg["training"].get("init_from")
    if init_from and latest_state_tag(checkpoint_dir) is None:
        # 初始化必须在 optimizer 创建前: 对 model strict 加载转换后的 mHC 权重
        _init_model_weights(model, init_from, logger)
        if rank == 0:
            logger.info("🎓 mHC SFT 初始化: 从 %s strict 加载 (step 从 0 开始)", init_from)

    client_optimizer = build_optimizer_mhc(model, cfg["training"])
    ds_config_path = Path(cfg["distributed"]["config"])
    if not ds_config_path.is_absolute():
        ds_config_path = ROOT / ds_config_path
    ds_config = json.loads(ds_config_path.read_text())
    ds_config["train_micro_batch_size_per_gpu"] = int(cfg["training"]["batch_size_per_device"])
    ds_config["gradient_accumulation_steps"] = int(cfg["training"].get("gradient_accumulation_steps", 1))
    ds_config["train_batch_size"] = ds_config["train_micro_batch_size_per_gpu"] * ds_config["gradient_accumulation_steps"] * dist.get_world_size()
    with _loud_phase(rank, logger, f"⚙️ DeepSpeed 引擎初始化 (world_size={dist.get_world_size()})"):
        engine, ds_optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), optimizer=client_optimizer, config=ds_config)

    resume_mode = cfg["training"].get("resume_mode", "auto")
    resume_tag = latest_state_tag(checkpoint_dir) if resume_mode == "auto" else None
    step = 0
    if isinstance(resume_mode, str) and resume_mode not in {"auto", "none", "false", "off"}:
        step = load_mhc_init_weights(engine, output_dir, resume_mode)
        if rank == 0:
            _blue_print(f"🔁 已从 {resume_mode} 恢复纯权重, step={step} (optimizer 状态重置)")
        dist.barrier()
    elif resume_tag:
        try:
            from train_giga_exp import load_training_checkpoint
            load_path, client_state = load_training_checkpoint(engine, checkpoint_dir, resume_tag)
            step = int(client_state.get("step", int(resume_tag.removeprefix("step-"))))
            if rank == 0:
                _blue_print(f"🔁 checkpoint {resume_tag} 恢复完成, step={step}")
        except Exception as exc:
            step = load_mhc_init_weights(engine, output_dir)
            if rank == 0:
                _blue_print(f"⚠️ 完整恢复失败 ({exc}), 降级纯权重恢复: step={step}")
        dist.barrier()
    elif resume_mode == "auto":
        step = load_mhc_init_weights(engine, output_dir)
        if rank == 0:
            _blue_print(f"{'🔁 已恢复纯权重, step=' + str(step) if step else '🆕 未发现 checkpoint, 从 0 开始'}")

    with _loud_phase(rank, logger, "📦 数据集索引加载"):
        data = GigaLMDBDataset(cfg["dataset"], seed=int(cfg.get("seed", 42)),
                               index_cache_dir=str(output_dir / "dataset_index_cache"))
    batch_sampler = BucketBatchSampler(data.bucket_indices, cfg["training"]["batch_size_per_device"],
                                       weights=cfg["dataset"].get("bucket_sampling_weights"),
                                       seed=int(cfg.get("seed", 42)), rank=rank,
                                       world_size=dist.get_world_size(), randomize_order_on_start=True,
                                       drop_last=True)
    steps_per_epoch = len(batch_sampler)
    max_epochs = int(cfg["training"]["max_epochs"])
    grad_accum = int(cfg["training"].get("gradient_accumulation_steps", 1))
    max_steps = (steps_per_epoch * max_epochs + grad_accum - 1) // grad_accum
    cfg["training"]["total_steps"] = max_steps
    epoch, batch_offset = divmod(min(step * grad_accum, steps_per_epoch * max_epochs), steps_per_epoch)
    batch_sampler.set_start_batch(batch_offset, epoch=epoch)
    if rank == 0:
        logger.info("📅 total_steps=%d 从 step=%d (epoch=%d batch=%d) 继续", max_steps, step, epoch, batch_offset)
    loader = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler,
                                         num_workers=cfg["training"].get("num_workers", 4),
                                         prefetch_factor=4, persistent_workers=True, collate_fn=collate)
    iterator = iter(loader)

    wandb_run = swanlab_run = None
    if rank == 0 and cfg["logging"]["wandb"]["enabled"]:
        wandb_dir = output_dir / "wandb"
        wandb_dir.mkdir(exist_ok=True)
        run_file = wandb_dir / "run.json"
        import uuid
        run_id = json.loads(run_file.read_text())["id"] if run_file.exists() else uuid.uuid4().hex
        run_file.write_text(json.dumps({"id": run_id}))
        wandb_run = wandb.init(project=cfg["logging"]["wandb"]["project"], name=cfg["logging"]["wandb"]["name"],
                               id=run_id, resume="allow", dir=str(wandb_dir), config=cfg)
    if rank == 0 and cfg["logging"].get("swanlab", {}).get("enabled"):
        sl_cfg = cfg["logging"]["swanlab"]
        if sl_cfg.get("api_key"):
            os.environ["SWANLAB_API_KEY"] = str(sl_cfg["api_key"])
        swanlab_dir = output_dir / "swanlab"
        swanlab_dir.mkdir(exist_ok=True)
        run_file = swanlab_dir / "run.json"
        import uuid
        run_id = json.loads(run_file.read_text())["id"] if run_file.exists() else uuid.uuid4().hex[:20]
        run_file.write_text(json.dumps({"id": run_id}))
        swanlab_run = swanlab.init(project=sl_cfg["project"], name=sl_cfg["name"], id=run_id,
                                   resume="allow", log_dir=str(swanlab_dir), config=cfg)

    from train_giga_exp import set_lr
    log_every = int(cfg["logging"].get("log_every_n_steps", 1))
    pending_batch = None
    consecutive_batch_skips = 0
    max_consecutive_batch_skips = int(os.environ.get("GIGA_MAX_CONSECUTIVE_BATCH_SKIPS", "64"))
    sigma_bucket_ema = [float("nan")] * SIGMA_BUCKETS
    progress = tqdm(total=max_steps, initial=step, disable=rank != 0, dynamic_ncols=True, desc="🚂 mHC stage_1")
    while step < max_steps:
        step += 1
        micro_losses = []
        micro_bucket_sums = torch.zeros(SIGMA_BUCKETS, device=device)
        micro_bucket_counts = torch.zeros(SIGMA_BUCKETS, device=device)
        last_captions = None
        lr_now = set_lr(client_optimizer, ds_optimizer, step, cfg["training"])
        for micro_idx in range(grad_accum):
            bad_local = 0
            try:
                if pending_batch is not None:
                    batch = pending_batch
                else:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        batch_sampler.set_epoch(batch_sampler.epoch + 1)
                        iterator = iter(loader)
                        batch = next(iterator)
            except Exception as exc:
                category = _data_error_category(exc)
                consecutive_batch_skips += 1
                if category == "executor_bug" or consecutive_batch_skips > max_consecutive_batch_skips:
                    raise
                logger.warning("⏭️ step=%d micro=%d 跳过坏数据 (category=%s): %s", step, micro_idx, category, exc)
                bad_local = 1
            skip_flag = torch.tensor([bad_local], device=device)
            dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
            if skip_flag.item() > 0:
                continue
            consecutive_batch_skips = 0
            pending_batch = None
            pixels = torch.from_numpy(batch["pixel_values"]).to(device)
            captions = batch["captions"]
            last_captions = captions
            with torch.no_grad():
                latents = (vae.encode(pixels.float()).latent_dist.sample() - latents_mean) * latents_std_inv
                prompts = [str(c).strip() or tokenizer.eos_token for c in captions]
                cfg_dropout = float(cfg["training"].get("cfg_dropout", 0.0))
                if cfg_dropout > 0:
                    drop = torch.rand(len(prompts), device=device) < cfg_dropout
                    empty_prompt = tokenizer.eos_token or " "
                    prompts = [empty_prompt if drop[i].item() else prompt for i, prompt in enumerate(prompts)]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                   max_length=tc.get("max_length", 512)).to(device)
                text, mask = encode_text(text_encoder, inputs, tc.get("layer")), inputs.attention_mask.bool()
                if text.shape[1] < 2:
                    text = torch.nn.functional.pad(text, (0, 0, 0, 2 - text.shape[1]))
                    mask = torch.nn.functional.pad(mask, (0, 2 - mask.shape[1]))
            sigma = torch.rand(latents.shape[0], device=device)
            sigma = sigma * cfg["scheduler"]["shift"] / (1 + (cfg["scheduler"]["shift"] - 1) * sigma)
            noise = torch.randn_like(latents)
            xt = (1 - sigma[:, None, None, None, None]) * latents + sigma[:, None, None, None, None] * noise
            pred = engine(xt.to(dtype), sigma * 1000, text, mask)
            diff_sq = (pred.float() - (noise - latents).float()).square()
            loss = diff_sq.mean()
            per_sample_loss = diff_sq.flatten(1).mean(1)
            engine.backward(loss)
            micro_losses.append(diff_sq.mean().detach())
            bins = (sigma.detach() * SIGMA_BUCKETS).long().clamp_(0, SIGMA_BUCKETS - 1)
            micro_bucket_sums.scatter_add_(0, bins, per_sample_loss.detach().float())
            micro_bucket_counts += torch.bincount(bins, minlength=SIGMA_BUCKETS).float()
        if not micro_losses:
            logger.warning("🚫 step=%d 全部 micro batch 跳过", step)
            continue
        engine.step()
        loss_value = torch.stack(micro_losses).mean().item()
        if rank == 0:
            progress.update(1)
            progress.set_postfix(loss=f"{loss_value:.5f}", lr=f"{lr_now:.2e}")
        if step % log_every == 0:
            stats = torch.stack([micro_bucket_sums, micro_bucket_counts])
            dist.all_reduce(stats)
            logger.info("rank=%d step=%d loss=%.6f lr=%.3e time=%s", rank, step, loss_value, lr_now,
                        time.strftime("%H:%M:%S"))
            if rank == 0 and (wandb_run or swanlab_run):
                payload = {"train/loss": loss_value, "train/lr": lr_now,
                           "train/mhc_lr": lr_now * float(client_optimizer.mhc_adam.param_groups[0]["_lr_scale"])}
                for i in range(SIGMA_BUCKETS):
                    if stats[1, i] > 0:
                        bucket_loss = (stats[0, i] / stats[1, i]).item()
                        if math.isfinite(bucket_loss):
                            prev = sigma_bucket_ema[i]
                            sigma_bucket_ema[i] = bucket_loss if not math.isfinite(prev) else 0.9 * prev + 0.1 * bucket_loss
                        payload[f"train/loss_sigma_{i / 10:.1f}-{(i + 1) / 10:.1f}"] = sigma_bucket_ema[i]
                # mHC 映射统计 (按 layer 分组)
                for key, value in engine.module.mhc_stats().items():
                    v = value.item()
                    if math.isfinite(v):
                        payload[f"mhc/{key}"] = v
                if wandb_run:
                    wandb.log(payload, step=step)
                if swanlab_run:
                    swanlab.log(payload, step=step)
        if cfg["validation"].get("enabled", True) and step % int(cfg["validation"].get("every_n_steps", 1000)) == 0:
            try:
                pending_batch = next(iterator)
            except Exception as exc:
                pending_batch = None
                logger.warning("⏭️ step=%d 验证 batch 错误: %s", step, exc)
            else:
                try:
                    run_next_batch_validation(engine, vae, tokenizer, text_encoder, device, dtype,
                                             pending_batch, step, rank, cfg, output_dir, logger,
                                             wandb_run, swanlab_run)
                except Exception as exc:
                    logger.warning("⏭️ step=%d 验证异常: %s", step, repr(exc))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        if step % cfg["training"].get("checkpointing_steps", 1000) == 0 or step == max_steps:
            engine.save_checkpoint(str(checkpoint_dir), tag=f"step-{step}", client_state={
                "step": step, "giga_semantic_metadata": _semantic_metadata(engine.module)})
            save_pure_weights(engine, output_dir, step)
            if rank == 0:
                prune_state_checkpoints(checkpoint_dir, int(cfg["training"].get("state_checkpoints_keep", 2)))
            dist.barrier()

    if rank == 0:
        progress.close()
    if wandb_run:
        wandb.finish()
    if swanlab_run:
        swanlab.finish()


def _init_model_weights(model, init_path, logger):
    """optimizer 创建前的 strict 权重加载 (转换后的 mHC checkpoint)。"""
    path = Path(init_path)
    if path.is_dir():
        candidates = sorted(path.glob("step-*.safetensors")) + sorted(path.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"init_from 目录没有权重文件: {path}")
        path = max(candidates, key=lambda p: int(p.name.split("-")[1].split(".")[0]) if "-" in p.name else 0)
    from safetensors.torch import load_file
    state = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)  # strict=True: 缺键/多键直接报错
    logger.info("strict 加载 mHC 初始化权重: %s (%d tensors)", path, len(state))


if __name__ == "__main__":
    main()
