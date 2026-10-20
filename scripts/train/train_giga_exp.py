"""Stage_1 Giga-Exp T2V/image flow-matching training."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
THIRDPARTY = ROOT / "thirdparty"  # fa3 copied here (no symlink)
if str(THIRDPARTY) not in sys.path:
    sys.path.insert(0, str(THIRDPARTY))

import deepspeed
import torch
import torch.distributed as dist
import wandb
import yaml
from tqdm.auto import tqdm
from diffusers import AutoencoderKLWan
from transformers import AutoModel, AutoTokenizer

from data.lmdb_dataset import GigaLMDBDataset, collate_image_batch
from data.samplers.bucket_sampler import BucketBatchSampler
from models.giga_transformer import GigaExpTransformer


def collate(batch):
    result = collate_image_batch(batch)
    # decode 输出每样本已是 [C,T=1,H,W], collate 后 [B,C,T,H,W], Wan VAE 与模型直接可用, 不再 unsqueeze
    result["pixel_values"] = torch.from_numpy(result["pixel_values"])
    return result


class MuonAdamW(torch.optim.Optimizer):
    """单优化器封装：2D 参数用 Muon，其余用 AdamW。"""

    def __init__(self, matrix_params, non_matrix_params, muon_cfg, adam_cfg, lr):
        self.muon = self.adam = None
        self.matrix_shapes = [tuple(p.shape) for p in matrix_params]
        if matrix_params and muon_cfg.get("enabled", True) and hasattr(torch.optim, "Muon"):
            if any(p.ndim != 2 for p in matrix_params):
                raise ValueError("Muon 只支持 2D 参数")
            self.muon = torch.optim.Muon(matrix_params, lr=lr, momentum=float(muon_cfg.get("momentum", 0.95)))
        elif matrix_params and muon_cfg.get("enabled", True):
            print("⚠️  torch.optim.Muon 不可用, matrix 参数回退 AdamW", flush=True)
        if self.muon is None and matrix_params:
            self.adam = torch.optim.AdamW(
                matrix_params + non_matrix_params, lr=lr,
                betas=tuple(adam_cfg.get("betas", [0.9, 0.95])), eps=float(adam_cfg.get("epsilon", 1e-8)),
                weight_decay=float(adam_cfg.get("weight_decay", 0.0)))
        elif non_matrix_params:
            self.adam = torch.optim.AdamW(
                non_matrix_params, lr=lr,
                betas=tuple(adam_cfg.get("betas", [0.9, 0.95])), eps=float(adam_cfg.get("epsilon", 1e-8)),
                weight_decay=float(adam_cfg.get("weight_decay", 0.0)))
        groups = (self.muon.param_groups if self.muon else []) + (self.adam.param_groups if self.adam else [])
        params = [p for g in groups for p in g["params"]]
        super().__init__(params, {})
        self.param_groups = groups

    @torch.no_grad()
    def _step_partitioned_muon(self):
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        group = self.muon.param_groups[0]
        shard = group["params"][0]
        if shard.grad is None:
            return
        world = dist.get_world_size()
        rank = dist.get_rank()
        gradient_parts = [torch.empty_like(shard.grad) for _ in range(world)]
        dist.all_gather(gradient_parts, shard.grad.contiguous())
        gradient = torch.cat(gradient_parts)
        state = self.muon.state[shard]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(shard)
        momentum_shard = state["momentum_buffer"]
        momentum_parts = [torch.empty_like(momentum_shard) for _ in range(world)]
        dist.all_gather(momentum_parts, momentum_shard.contiguous())
        momentum = torch.cat(momentum_parts)

        offset = 0
        start = rank * shard.numel()
        end = start + shard.numel()
        shard.mul_(1 - group["lr"] * group["weight_decay"])
        coefficient = group["momentum"]
        for shape in self.matrix_shapes:
            size = shape[0] * shape[1]
            grad_matrix = gradient.narrow(0, offset, size).view(shape)
            momentum_matrix = momentum.narrow(0, offset, size).view(shape)
            momentum_matrix.lerp_(grad_matrix, 1 - coefficient)
            update = grad_matrix.lerp(momentum_matrix, coefficient) if group["nesterov"] else momentum_matrix
            update = _zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])
            adjusted_lr = _adjust_lr(group["lr"], group["adjust_lr_fn"], shape)
            overlap_start = max(start, offset)
            overlap_end = min(end, offset + size)
            if overlap_start < overlap_end:
                shard.narrow(0, overlap_start - start, overlap_end - overlap_start).sub_(
                    update.reshape(-1).narrow(0, overlap_start - offset, overlap_end - overlap_start).float() * adjusted_lr
                )
            offset += size
        momentum_shard.copy_(momentum.narrow(0, start, shard.numel()))

    def step(self, closure=None):
        loss = None
        if self.muon is not None:
            if self.muon.param_groups[0]["params"][0].ndim == 1:
                self._step_partitioned_muon()
            else:
                loss = self.muon.step(closure)
        if self.adam is not None:
            current = self.adam.step(closure)
            loss = current if current is not None else loss
        return loss

    def zero_grad(self, set_to_none=True):
        for optimizer in (self.muon, self.adam):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict() if self.muon else None,
                "adam": self.adam.state_dict() if self.adam else None}

    def load_state_dict(self, state_dict):
        if self.muon and state_dict.get("muon"):
            self.muon.load_state_dict(state_dict["muon"])
        if self.adam and state_dict.get("adam"):
            self.adam.load_state_dict(state_dict["adam"])


def build_optimizer(model, cfg):
    matrix, rest = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (matrix if parameter.ndim == 2 else rest).append(parameter)
    lr = float(cfg.get("learning_rate", 1e-4))
    return MuonAdamW(matrix, rest, cfg.get("muon", {}), cfg, lr)


def set_lr(optimizer, engine_optimizer, step, training_cfg):
    """Warmup + cosine; 同时写入 wrapper 与 DeepSpeed ZeroOptimizer 的 param_groups."""
    import math
    base = float(training_cfg.get("learning_rate", 1e-4))
    warmup = int(training_cfg.get("lr_warmup_steps", 0))
    total = int(training_cfg["total_steps"])
    if warmup > 0 and step <= warmup:
        lr = base * step / warmup
    else:
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        lr = base * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    for group in engine_optimizer.param_groups:
        group["lr"] = lr
    return lr


def setup_logging(output_dir, rank):
    run_log_dir = Path(os.environ.get("GIGA_RUN_LOG_DIR", output_dir / "logs" / "manual"))
    run_log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("giga_exp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if rank == 0:
        file_handler = logging.FileHandler(run_log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        (run_log_dir / "run_env.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in sorted(os.environ.items()) if key.startswith(("WORLD", "RANK", "LOCAL", "MASTER", "GIGA_"))) + "\n",
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(run_log_dir / f"train_rank{rank}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def save_pure_weights(engine, output_dir, step):
    if dist.get_rank() != 0:
        return
    weight_dir = output_dir / "checkpoints" / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in engine.module.state_dict().items()}
    target = weight_dir / f"step-{step:08d}.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(state, str(target), metadata={"format": "pt", "step": str(step)})
    except ImportError:
        target = target.with_suffix(".pt")
        torch.save(state, target)
    torch.save(state, weight_dir / "latest.pt")


def prune_state_checkpoints(checkpoint_dir, keep):
    """state 完整 checkpoint 只保留最近 keep 个, 纯 weights 快照不清理。"""
    if keep <= 0:
        return
    tags = sorted((int(p.name.removeprefix("step-")), p) for p in checkpoint_dir.glob("step-*") if p.is_dir())
    for _, path in tags[:-keep]:
        shutil.rmtree(path, ignore_errors=True)
        logging.getLogger("giga_exp").info("🧹 已清理旧 state checkpoint: %s", path.name)


def latest_state_tag(checkpoint_dir):
    tags = []
    for path in checkpoint_dir.glob("step-*"):
        if path.is_dir():
            try:
                tags.append((int(path.name.removeprefix("step-")), path.name))
            except ValueError:
                pass
    return max(tags)[1] if tags else None


def load_compatible_state_dict(model, state):
    """只加载名称和 shape 都匹配的参数，其余参数保留当前初始化值。"""
    current = model.state_dict()
    matched = {key: value for key, value in state.items()
               if key in current and current[key].shape == value.shape}
    result = model.load_state_dict(matched, strict=False)
    ignored = len(state) - len(matched)
    return len(matched), len(result.missing_keys), ignored


def load_pure_weights(engine, output_dir, checkpoint_path=None):
    """兼容加载纯权重；checkpoint_path 可指定权重文件或包含 step-* 权重的目录。"""
    if checkpoint_path:
        target = Path(checkpoint_path).expanduser()
        if target.is_dir():
            candidates = sorted(target.glob("step-*.safetensors")) + sorted(target.glob("step-*.pt"))
            if not candidates:
                raise FileNotFoundError(f"指定目录没有 step-* 权重文件: {target}")
            target = max(candidates, key=lambda path: int(path.name.split("-")[1].split(".")[0]))
        if not target.is_file():
            raise FileNotFoundError(f"指定权重不存在: {target}")
    else:
        weight_dir = output_dir / "checkpoints" / "weights"
        candidates = sorted(weight_dir.glob("step-*.safetensors")) + sorted(weight_dir.glob("step-*.pt")) if weight_dir.is_dir() else []
        if not candidates:
            return 0
        target = max(candidates, key=lambda path: int(path.name.split("-")[1].split(".")[0]))
    step_match = re.search(r"step-(\\d+)", target.name)
    step = int(step_match.group(1)) if step_match else 0
    if target.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(target))
    else:
        state = torch.load(str(target), map_location="cpu")
    matched, missing, unexpected = load_compatible_state_dict(engine.module, state)
    if dist.get_rank() == 0:
        logger = logging.getLogger("giga_exp")
        logger.info("兼容恢复权重: %s, matched=%d, reinitialized=%d, ignored=%d", target, matched, missing, unexpected)
    return step


def load_init_weights(model, init_path):
    """SFT 初始化: 加载纯权重 (safetensors/pt) 到 model, 必须在 optimizer 创建前调用.

    init_path 可以是权重文件, 也可以是 checkpoints/weights 目录 (自动取最新 step-*)."""
    path = Path(init_path)
    if path.is_dir():
        candidates = sorted(path.glob("step-*.safetensors")) + sorted(path.glob("step-*.pt"))
        if not candidates:
            raise FileNotFoundError(f"init_from 目录没有权重文件: {path}")
        path = max(candidates, key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu")
    matched, missing, unexpected = load_compatible_state_dict(model, state)
    logging.getLogger("giga_exp").info("兼容初始化权重: %s, matched=%d, reinitialized=%d, ignored=%d", path, matched, missing, unexpected)
    return path


def build_text_encoder(tc, dtype, device):
    """按 type 构建文本编码器: qwen3_vl 用 ForConditionalGeneration, 其他走 AutoModel。"""
    if tc.get("type") == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(tc["pretrained_path"], torch_dtype=dtype, local_files_only=True)
    else:
        encoder = AutoModel.from_pretrained(tc["pretrained_path"], torch_dtype=dtype, local_files_only=True)
    return encoder.eval().to(device)


def encode_text(text_encoder, inputs, layer=None):
    """文本前向: layer 为 None 取 last_hidden_state, 否则取 hidden_states[layer] (如 -2)。"""
    out = text_encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=layer is not None)
    return out.hidden_states[layer] if layer is not None else out.last_hidden_state


def wan_latent_stats(vae, device):
    """官方 Wan 约定 (diffusers WanPipeline): 模型侧 z_norm = (z_vae - mean) / std;
    decode 前 z_vae = z_norm * std + mean. Wan2.2 config 无 scaling_factor, 键为 z_dim/latents_mean/latents_std."""
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, vae.config.z_dim, 1, 1, 1).float()
    std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=device)).view(1, vae.config.z_dim, 1, 1, 1).float()
    return mean, std_inv


SIGMA_BUCKETS = 10  # sigma 等宽分桶数: [0,0.1) ... [0.9,1.0]


def sigma_bucket_loss_stats(sigma, per_sample_loss):
    """按 sigma 等宽 10 桶汇总 per-sample loss, 跨卡 all_reduce 后返回 (各桶 loss 之和, 各桶样本数)."""
    with torch.no_grad():
        bins = (sigma.detach() * SIGMA_BUCKETS).long().clamp_(0, SIGMA_BUCKETS - 1)
        sums = torch.zeros(SIGMA_BUCKETS, device=sigma.device)
        sums.scatter_add_(0, bins, per_sample_loss.detach().float())
        counts = torch.bincount(bins, minlength=SIGMA_BUCKETS).float()
        stats = torch.stack([sums, counts])
    dist.all_reduce(stats)  # 20 个 float, 单次通信开销可忽略
    return stats[0], stats[1]


@torch.no_grad()
def run_next_batch_validation(engine, vae, tokenizer, text_encoder, device, dtype, batch, step, rank, cfg, output_dir, logger, wandb_run):
    """每卡对 next batch 做真实推理: flow ODE 采样 → VAE 解码 → 保存 INPUT|GEN 对比图."""
    from PIL import Image, ImageDraw
    tc, val_cfg = cfg["components"]["text_encoder"], cfg["validation"]
    pixels = batch["pixel_values"].to(device)  # [B,C,T=1,H,W] in [-1,1]
    prompts = [str(c).strip() or tokenizer.eos_token for c in batch["captions"]]
    negative_prompts = [tokenizer.eos_token] * len(prompts)
    inputs = tokenizer(prompts + negative_prompts, return_tensors="pt", padding=True, truncation=True, max_length=tc.get("max_length", 512)).to(device)
    text_all = encode_text(text_encoder, inputs, tc.get("layer"))
    mask_all = inputs.attention_mask.bool()
    text, neg = text_all[:len(prompts)], text_all[len(prompts):]
    mask, neg_mask = mask_all[:len(prompts)], mask_all[len(prompts):]
    b, _, _, h, w = pixels.shape
    compress = vae.config.scale_factor_spatial
    mean, std_inv = wan_latent_stats(vae, device)
    generator = torch.Generator(device=device).manual_seed(int(val_cfg.get("seed", 42)) + step * 1000 + rank)
    x = torch.randn(b, vae.config.z_dim, 1, h // compress, w // compress, device=device, dtype=torch.float32, generator=generator)
    num_steps = int(val_cfg.get("num_inference_steps", 20))
    engine.eval()
    try:
        for i in range(num_steps):  # σ: 1(噪声) → 0(数据), v = noise - data
            sigma = 1.0 - i / num_steps
            model_input = x.to(dtype)
            timestep = torch.full((b,), sigma * 1000.0, device=device)
            v_all = engine(
                torch.cat((model_input, model_input), 0),
                torch.cat((timestep, timestep), 0),
                torch.cat((neg, text), 0),
                torch.cat((neg_mask, mask), 0),
            ).float()
            v_uncond, v_cond = v_all.chunk(2, 0)
            guidance_scale = float(val_cfg.get("guidance_scale", 3.0))
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            x = x - v / num_steps
    finally:
        engine.train()
    gen = vae.decode(x / std_inv + mean).sample

    def to_uint8(t):  # [C,T,H,W] → HWC uint8
        frame = t[:, 0].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
        return ((frame + 1) * 127.5).round().astype("uint8")

    label_h, gap = 20, 8
    rows = []
    for i in range(min(b, 8)):
        src, out = to_uint8(pixels[i]), to_uint8(gen[i])
        row = Image.new("RGB", (src.shape[1] * 2 + gap, src.shape[0] + label_h), (245, 245, 245))
        row.paste(Image.fromarray(src), (0, label_h))
        row.paste(Image.fromarray(out), (src.shape[1] + gap, label_h))
        draw = ImageDraw.Draw(row)
        draw.text((4, 3), "INPUT", fill=(180, 0, 0))
        draw.text((src.shape[1] + gap + 4, 3), "GEN", fill=(0, 0, 180))
        rows.append(row)
    grid = Image.new("RGB", (rows[0].width, sum(r.height + 4 for r in rows) - 4), (245, 245, 245))
    y = 0
    for r in rows:
        grid.paste(r, (0, y))
        y += r.height + 4
    val_dir = output_dir / "validation" / f"step-{step:08d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    grid.save(val_dir / f"rank{rank}.png")
    logger.info("🖼️ step=%d rank=%d next-batch 推理完成: %d samples → %s", step, rank, b, val_dir / f"rank{rank}.png")
    if wandb_run and rank == 0:
        wandb.log({"val/grid": wandb.Image(grid)}, step=step)


def print_banner(cfg, rank):
    if rank != 0:
        return
    red, reset = "\033[1;31m", "\033[0m"
    model = cfg["model"]
    fa3_file = THIRDPARTY / "flash_attn_interface.py"
    fa3_state = "READY ✅" if fa3_file.exists() else "MISSING ❌"
    print(f"{red}{'=' * 74}{reset}", flush=True)
    print(f"{red}⚠️  ATTENTION BACKEND: FlashAttention 3 (fa3) [{fa3_state}]{reset}", flush=True)
    print(f"{red}    copied from /home/jovyan/pfs/users/zhanqian.wu/thirdparty → {THIRDPARTY}{reset}", flush=True)
    print(f"{red}{'=' * 74}{reset}", flush=True)
    print("🧠 Giga-Exp stage_1 | T2V-only flow matching (T=1, video-ready)", flush=True)
    print(f"📐 DiT: dim={model['dim']} layers={model['layers']} heads={model['heads']} ffn={model['ffn_dim']} head_dim={model['dim'] // model['heads']}", flush=True)
    print(f"🌀 RoPE: axes(t/h/w)={model['rope_axes_dims']} theta={model['rope_theta']} — time axis retained for video", flush=True)
    print(f"📝 Text encoder: {cfg['components']['text_encoder']['pretrained_path']}", flush=True)
    print(f"🎞️  VAE: {cfg['components']['wan_vae']['pretrained_path']} (48ch)", flush=True)
    print(f"🗃️  Dataset: {cfg['dataset']['database_path']}", flush=True)
    print(f"    sources={len(cfg['dataset']['image_sources'])} | buckets={cfg['dataset']['buckets']} | long_edge={cfg['dataset']['long_edge']}", flush=True)
    print(f"    edit_training={cfg['dataset']['edit_training_enabled']} | index_cache={'output/dataset_index_cache (复用)' if not cfg['dataset'].get('refresh_index_cache') else '强制重建'}", flush=True)
    print(f"🎚️  Scheduler: {cfg['scheduler']['type']} shift={cfg['scheduler']['shift']} | valid_shift={cfg['scheduler'].get('validation_shift')}", flush=True)
    print(f"⚙️  Optimizer: {cfg['training']['optimizer']} | lr={cfg['training']['learning_rate']} | warmup={cfg['training'].get('lr_warmup_steps')} | max_epochs={cfg['training']['max_epochs']} | cosine", flush=True)
    print(f"    global_batch={cfg['training']['batch_size_per_device']}×{cfg['distributed'].get('world_size', '?')} GPUs | precision={cfg['training'].get('mixed_precision')}", flush=True)
    print(f"💾 Output: {cfg['output_dir']}", flush=True)
    print(f"📊 Wandb: project={cfg['logging']['wandb']['project']} name={cfg['logging']['wandb']['name']}", flush=True)
    print("📋 Config:", flush=True)
    for key in ("model", "scheduler", "dataset", "training"):
        print(f"   [{key}] {json.dumps(cfg[key], ensure_ascii=False, default=str)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text()) if not Path(args.config).is_absolute() else yaml.safe_load(Path(args.config).read_text())
    deepspeed.init_distributed()
    rank = dist.get_rank()
    print_banner(cfg, rank)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    output_dir = Path(cfg["output_dir"])
    logger = setup_logging(output_dir, rank)
    if rank == 0:
        logger.info("distributed world_size=%s node_rank=%s global_rank=%s local_rank=%s", os.environ.get("WORLD_SIZE", "1"), os.environ.get("GIGA_NODE_RANK", os.environ.get("NODE_RANK", "0")), os.environ.get("RANK", "0"), os.environ.get("LOCAL_RANK", "0"))
    checkpoint_dir = output_dir / "checkpoints" / "state"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tc, dc = cfg["components"]["text_encoder"], cfg["components"]["wan_vae"]
    dtype = torch.bfloat16 if cfg["training"].get("mixed_precision", "bf16") == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(tc["tokenizer_path"], local_files_only=True)
    if len(tokenizer) <= 1:
        raise ValueError(f"tokenizer 词表异常: {tc['tokenizer_path']}")
    text_encoder = build_text_encoder(tc, dtype, device)
    vae = AutoencoderKLWan.from_pretrained(dc["pretrained_path"], subfolder=dc.get("subfolder", "vae"), torch_dtype=torch.float32, local_files_only=True).eval().to(device)
    model = GigaExpTransformer(**cfg["model"]).to(device, dtype=dtype)
    latents_mean, latents_std_inv = wan_latent_stats(vae, device)
    for parameter in (*text_encoder.parameters(), *vae.parameters()):
        parameter.requires_grad_(False)

    # SFT 初始化: 无可恢复 checkpoint 时用 init_from 纯权重载入 (optimizer 状态全新)
    init_from = cfg["training"].get("init_from")
    if init_from and latest_state_tag(checkpoint_dir) is None:
        resolved = load_init_weights(model, init_from)
        if rank == 0:
            logger.info("🎓 SFT 初始化: 从 %s 加载纯权重 (step 从 0 开始)", resolved)

    client_optimizer = build_optimizer(model, cfg["training"])
    ds_config_path = Path(cfg["distributed"]["config"])
    if not ds_config_path.is_absolute():
        ds_config_path = ROOT / ds_config_path
    ds_config = json.loads(ds_config_path.read_text())
    ds_config["train_micro_batch_size_per_gpu"] = int(cfg["training"]["batch_size_per_device"])
    ds_config["gradient_accumulation_steps"] = int(cfg["training"].get("gradient_accumulation_steps", 1))
    ds_config["train_batch_size"] = ds_config["train_micro_batch_size_per_gpu"] * ds_config["gradient_accumulation_steps"] * dist.get_world_size()
    engine, ds_optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), optimizer=client_optimizer,
        config=ds_config,
    )
    # 断点恢复三级策略: 完整 DeepSpeed state (含 optimizer) → 纯权重快照 (step 保留) → 从 0 开始。
    # resume_mode 为文件/目录路径时，跳过自动发现，强制从指定纯权重恢复。
    resume_mode = cfg["training"].get("resume_mode", "auto")
    resume_tag = latest_state_tag(checkpoint_dir) if resume_mode == "auto" else None
    step = 0
    if isinstance(resume_mode, str) and resume_mode not in {"auto", "none", "false", "off"}:
        step = load_pure_weights(engine, output_dir, resume_mode)
        if rank == 0:
            logger.info("🔁 已从指定纯权重恢复: %s, 从 step=%d 继续训练 (optimizer 状态已重置)", resume_mode, step)
        dist.barrier()
    elif resume_tag:
        try:
            load_path, client_state = engine.load_checkpoint(str(checkpoint_dir), tag=resume_tag, load_optimizer_states=True, load_lr_scheduler_states=True)
            if not load_path:
                raise RuntimeError("load_checkpoint 返回空路径")
            step = int(client_state.get("step", int(resume_tag.removeprefix("step-"))))
            logger.info("🔁 已恢复完整 DeepSpeed checkpoint %s (含 optimizer), 从 step=%d 继续训练", resume_tag, step)
        except Exception as exc:
            step = load_pure_weights(engine, output_dir)
            logger.warning("⚠️ 完整 checkpoint 恢复失败 (%s), 降级为纯权重恢复: step=%d (optimizer 状态已重置)", exc, step)
        dist.barrier()
    elif resume_mode == "auto":
        step = load_pure_weights(engine, output_dir)
        if rank == 0:
            if step:
                logger.info("🔁 已恢复纯权重，从 step=%d 继续训练 (optimizer 状态已重置)", step)
            else:
                logger.info("🆕 未发现 checkpoint, 从 step=0 开始训练")

    data = GigaLMDBDataset(cfg["dataset"], seed=int(cfg.get("seed", 42)), index_cache_dir=str(output_dir / "dataset_index_cache"))
    if rank == 0:
        logger.info("dataset ready: records=%d buckets=%s", len(data.records), {key: len(value) for key, value in data.bucket_indices.items()})
    batch_sampler = BucketBatchSampler(data.bucket_indices, cfg["training"]["batch_size_per_device"], weights=cfg["dataset"].get("bucket_sampling_weights"), seed=int(cfg.get("seed", 42)), rank=rank, world_size=dist.get_world_size(), randomize_order_on_start=True, drop_last=True)
    steps_per_epoch = len(batch_sampler)
    if steps_per_epoch <= 0:
        raise ValueError("数据集没有可训练的完整 batch")
    max_epochs = int(cfg["training"]["max_epochs"])
    if max_epochs <= 0:
        raise ValueError("max_epochs 必须为正整数")
    max_steps = steps_per_epoch * max_epochs
    cfg["training"]["total_steps"] = max_steps
    if step > max_steps:
        raise ValueError(f"恢复 step={step} 超过 {max_epochs} epoch 的总步数 {max_steps}")
    epoch, batch_offset = divmod(step, steps_per_epoch)
    batch_sampler.set_start_batch(batch_offset, epoch=epoch)
    if rank == 0:
        logger.info("📅 max_epochs=%d steps_per_epoch=%d total_steps=%d; 从 epoch=%d batch=%d step=%d 继续", max_epochs, steps_per_epoch, max_steps, epoch, batch_offset, step)
    loader = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler, num_workers=cfg["training"].get("num_workers", 4), collate_fn=collate)
    iterator = iter(loader)

    wandb_run = None
    if rank == 0 and cfg["logging"]["wandb"]["enabled"]:
        wandb_dir = output_dir / "wandb"
        wandb_dir.mkdir(exist_ok=True)
        run_file = wandb_dir / "run.json"
        # 新版 wandb 移除了 wandb.util.generate_id, uuid4().hex (32位) 满足 wandb run id 要求
        run_id = json.loads(run_file.read_text())["id"] if run_file.exists() else uuid.uuid4().hex
        run_file.write_text(json.dumps({"id": run_id}))
        wandb_run = wandb.init(project=cfg["logging"]["wandb"]["project"], name=cfg["logging"]["wandb"]["name"], id=run_id, resume="allow", dir=str(wandb_dir), config=cfg)

    sigma_bucket_ema = [float("nan")] * SIGMA_BUCKETS  # rank0: 各 sigma 桶 loss 的 EMA (α=0.1)
    log_every = int(cfg["logging"].get("log_every_n_steps", 1))
    pending_batch = None
    progress = tqdm(total=max_steps, initial=step, disable=rank != 0, dynamic_ncols=True, desc="🚂 stage_1")
    while step < max_steps:
        step += 1
        try:
            batch = pending_batch if pending_batch is not None else next(iterator)
        except StopIteration:
            batch_sampler.set_epoch(batch_sampler.epoch + 1)
            iterator = iter(loader)
            batch = next(iterator)
        pending_batch = None
        pixels = batch["pixel_values"].to(device)
        captions = batch["captions"]
        with torch.no_grad():
            # Wan 官方约定: encode 后标准化 z_norm = (z - mean) * latents_std_inv
            latents = (vae.encode(pixels.float()).latent_dist.sample() - latents_mean) * latents_std_inv
            prompts = [str(c).strip() or tokenizer.eos_token for c in captions]
            cfg_dropout = float(cfg["training"].get("cfg_dropout", 0.0))
            if cfg_dropout > 0:
                drop = torch.rand(len(prompts), device=device) < cfg_dropout
                empty_prompt = tokenizer.eos_token or " "  # Qwen3-VL tokenizer 对空串产出 0 长度序列, 会导致 attention reshape 崩溃
                prompts = [empty_prompt if drop[i].item() else prompt for i, prompt in enumerate(prompts)]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tc.get("max_length", 512)).to(device)
            text, mask = encode_text(text_encoder, inputs, tc.get("layer")), inputs.attention_mask.bool()
        sigma = torch.rand(latents.shape[0], device=device)
        sigma = sigma * cfg["scheduler"]["shift"] / (1 + (cfg["scheduler"]["shift"] - 1) * sigma)
        noise = torch.randn_like(latents)
        xt = (1 - sigma[:, None, None, None, None]) * latents + sigma[:, None, None, None, None] * noise
        lr_now = set_lr(client_optimizer, ds_optimizer, step, cfg["training"])
        pred = engine(xt.to(dtype), sigma * 1000, text, mask)
        diff_sq = (pred.float() - (noise - latents).float()).square()
        loss = diff_sq.mean()
        per_sample_loss = diff_sq.flatten(1).mean(1)  # [B] 每样本 loss, 用于 sigma 分桶
        engine.backward(loss)
        engine.step()
        if rank == 0:
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.5f}", lr=f"{lr_now:.2e}", samples=step * cfg['training']['batch_size_per_device'] * dist.get_world_size())
        if step % log_every == 0:
            data_hash = hashlib.md5("\x1f".join(captions).encode("utf-8")).hexdigest()[:8]  # 本 rank batch 数据指纹
            logger.info("rank=%d step=%d loss=%.6f data=%s lr=%.3e time=%s sigma_mean=%.5f", rank, step, loss.item(), data_hash, lr_now, time.strftime("%H:%M:%S"), sigma.mean().item())
            bucket_sums, bucket_counts = sigma_bucket_loss_stats(sigma, per_sample_loss)
            if rank == 0:
                if wandb_run:
                    payload = {"train/loss": loss.item(), "train/lr": lr_now, "train/sigma_mean": sigma.mean().item(), "train/samples": step * cfg['training']['batch_size_per_device'] * dist.get_world_size()}
                    for i in range(SIGMA_BUCKETS):
                        if bucket_counts[i] <= 0:
                            continue  # 本步该桶无样本, 不上报 (wandb 自动断点显示)
                        bucket_loss = (bucket_sums[i] / bucket_counts[i]).item()
                        if math.isfinite(bucket_loss):  # NaN/Inf 不污染 EMA
                            prev = sigma_bucket_ema[i]
                            sigma_bucket_ema[i] = bucket_loss if not math.isfinite(prev) else 0.9 * prev + 0.1 * bucket_loss
                        payload[f"train/loss_sigma_{i / 10:.1f}-{(i + 1) / 10:.1f}"] = sigma_bucket_ema[i]
                    wandb.log(payload, step=step)
        if cfg["validation"].get("enabled", True) and step % int(cfg["validation"].get("every_n_steps", 1000)) == 0:
            try:
                pending_batch = next(iterator)
            except StopIteration:
                batch_sampler.set_epoch(batch_sampler.epoch + 1)
                iterator = iter(loader)
                pending_batch = next(iterator)
            if rank == 0:
                logger.info("🔍 step=%d 每卡对 next batch 推理", step)
            run_next_batch_validation(engine, vae, tokenizer, text_encoder, device, dtype, pending_batch, step, rank, cfg, output_dir, logger, wandb_run)
        if step % cfg["training"].get("checkpointing_steps", 1000) == 0 or step == max_steps:
            engine.save_checkpoint(str(checkpoint_dir), tag=f"step-{step}", client_state={"step": step})
            save_pure_weights(engine, output_dir, step)
            if rank == 0:
                prune_state_checkpoints(checkpoint_dir, int(cfg["training"].get("state_checkpoints_keep", 2)))
            dist.barrier()

    if rank == 0:
        progress.close()
    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
