"""GWP 6B teacher pipeline — 蒸馏 teacher 推理/编码, 与 student giga_pipeline.py 区分.

职责:
  1. 加载 GWPTransformer6BTeacher (from gigaworld stage1 checkpoint, model_weights.pt)
  2. teacher 文本编码: Qwen3-VL chat-template 约定 (clean_prompt + apply_chat_template,
     add_generation_prompt=True, 取 hidden_states[layer=-2] 有效位) → list[[L,D]]
  3. teacher Euler 去噪: t = 1 - sigma (Z-Image 约定), CFG, 输出取负号对齐 student
     velocity 约定 (teacher 原始输出 = z0 - noise, student 目标 = noise - z0)
  4. Wan VAE 解码 (latent * std + mean)

train_giga_exp_kd.py 复用: build_gwp6b_teacher / load_gwp6b_teacher_checkpoint /
encode_prompts_6b_teacher (文本编码可与 student 共享同一 Qwen3-VL 实例, 省一份 8GB 权重)。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from PIL import Image


# teacher 默认结构 (与 gigaworld stage1_wan_image_qwen3_vl 训练一致, checkpoint 实测确认)
TEACHER_DEFAULTS = {
    "all_patch_size": (2,),
    "all_f_patch_size": (1,),
    "in_channels": 48,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 2560,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": [32, 48, 48],
    "axes_lens": [4096, 512, 512],
}


def clean_prompt_6b_teacher(prompt: str) -> str:
    """对齐 gigaworld trainer._clean_prompt: 剥离残留的对话模板标记。"""
    prompt = str(prompt)
    prompt = re.sub(r"<\|(?:im_start|start)\|>\s*user\s*", "", prompt, flags=re.IGNORECASE)
    prompt = re.split(r"<\|(?:im_start|start)\|>\s*assistant\b", prompt, maxsplit=1, flags=re.IGNORECASE)[0]
    prompt = re.sub(r"<\|[^|>]+\|>", "", prompt)
    return prompt.strip()


def encode_prompts_6b_teacher(
    vl_processor,
    vl_text_encoder,
    prompts: list[str],
    device: torch.device,
    layer: int = -2,
    max_condition_tokens: int = 4064,
) -> list[torch.Tensor]:
    """teacher 约定文本编码 (chat template), 批量右 padding + mask 取有效位。

    与 gigaworld trainer._prompt_embeds 的逐条编码数值等价: Qwen3-VL 为因果 LM,
    右 padding 不影响真实 token 的注意域; 返回 list of [L_valid, D]。
    """
    prompts = [clean_prompt_6b_teacher(p) for p in prompts]
    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts
    ]
    rendered = [
        vl_processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    inputs = vl_processor(text=rendered, padding=True, return_tensors="pt").to(device)
    seq_lens = inputs["attention_mask"].sum(-1)
    if int(seq_lens.max()) > max_condition_tokens:
        raise RuntimeError(
            f"teacher 条件长度 {int(seq_lens.max())} 超过 max_condition_tokens={max_condition_tokens}"
        )
    outputs = vl_text_encoder(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = outputs.hidden_states[layer]
    mask = inputs["attention_mask"].bool()
    encoded = [hidden[i][mask[i]].clone() for i in range(len(prompts))]
    del outputs, hidden, inputs
    return encoded


def resolve_teacher_checkpoint_path(checkpoint: str | Path) -> Path:
    """解析 teacher checkpoint: 目录 → latest 指针 / checkpoint-*/model_weights.pt / 直接 .pt 文件。"""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        latest = path / "latest"
        if latest.is_file():
            pointed = path / latest.read_text(encoding="utf-8").strip()
            if pointed.exists():
                path = pointed
        if path.is_dir():
            weights = path / "model_weights.pt"
            if weights.is_file():
                return weights
            candidates = sorted(
                path.glob("checkpoint-*/model_weights.pt"),
                key=lambda p: int(p.parent.name.split("-")[-1]),
            )
            if candidates:
                return candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(f"teacher checkpoint 不存在: {checkpoint}")
    return path


def load_gwp6b_teacher_checkpoint(transformer, checkpoint: str | Path) -> Path:
    """加载 gigaworld 训练的纯权重 (model_weights.pt / mp_rank 合并态), 键名严格匹配。"""
    path = resolve_teacher_checkpoint_path(checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"teacher checkpoint 必须是 state_dict: {path}")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    result = transformer.load_state_dict(state, strict=True)
    print(f"✅ teacher checkpoint 加载: {path} (missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})")
    return path


def build_gwp6b_teacher(teacher_cfg: dict, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """按 yaml teacher 段构建 6B teacher: 显式结构参数 (缺省用 TEACHER_DEFAULTS)。"""
    from models.giga_transformer_6b_teacher import GWPTransformer6BTeacher, configure_attention_backend

    configure_attention_backend(str(teacher_cfg.get("attention_backend", "flash3")))
    arch = dict(TEACHER_DEFAULTS)
    for key in ("dim", "n_layers", "n_heads", "n_kv_heads", "cap_feat_dim", "in_channels",
                "n_refiner_layers", "norm_eps", "qk_norm", "rope_theta", "t_scale"):
        if key in teacher_cfg:
            arch[key] = teacher_cfg[key]
    arch["axes_dims"] = list(teacher_cfg.get("rope_axes_dims", arch["axes_dims"]))
    arch["axes_lens"] = list(teacher_cfg.get("rope_axes_lens", arch["axes_lens"]))
    # patch_size/f_patch_size 兼容标量 (yaml 里 patch_size: 2 -> 多档元组 (2,))
    patch_size = teacher_cfg.get("patch_size")
    f_patch_size = teacher_cfg.get("f_patch_size")
    arch["all_patch_size"] = tuple(patch_size) if isinstance(patch_size, (list, tuple)) else ((patch_size,) if patch_size is not None else arch["all_patch_size"])
    arch["all_f_patch_size"] = tuple(f_patch_size) if isinstance(f_patch_size, (list, tuple)) else ((f_patch_size,) if f_patch_size is not None else arch["all_f_patch_size"])
    transformer = GWPTransformer6BTeacher(**arch)
    resolved = load_gwp6b_teacher_checkpoint(transformer, teacher_cfg["checkpoint"])
    transformer = transformer.to(device=device, dtype=dtype).eval()
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    return transformer, resolved


class GigaPipeline6BTeacher:
    """teacher-only T2I pipeline (推理验证用); KD 训练请直接用模块级辅助函数。"""

    def __init__(self, transformer, vl_processor, vl_text_encoder, vae, scheduler,
                 text_layer: int = -2, max_condition_tokens: int = 4064):
        self.transformer = transformer
        self.vl_processor = vl_processor
        self.vl_text_encoder = vl_text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.text_layer = text_layer
        self.max_condition_tokens = max_condition_tokens

    @classmethod
    def from_pretrained(
        cls,
        teacher_checkpoint: str | Path,
        qwen3_vl_path: str,
        vae_path: str,
        vae_subfolder: str = "vae",
        attention_backend: str = "flash3",
        text_layer: int = -2,
        max_condition_tokens: int = 4064,
        scheduler_num_train_timesteps: int = 1000,
        scheduler_shift: float = 6.0,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        teacher_arch: dict | None = None,
    ):
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        device = torch.device(device)
        transformer, _ = build_gwp6b_teacher(
            {**(teacher_arch or {}), "checkpoint": teacher_checkpoint, "attention_backend": attention_backend},
            device, dtype,
        )
        vae = AutoencoderKLWan.from_pretrained(
            vae_path, subfolder=vae_subfolder, torch_dtype=torch.float32, local_files_only=True
        ).eval().to(device)
        vl_processor = AutoProcessor.from_pretrained(qwen3_vl_path, local_files_only=True)
        vl_text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen3_vl_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
        ).eval().to(device)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=scheduler_num_train_timesteps, shift=scheduler_shift
        )
        return cls(transformer, vl_processor, vl_text_encoder, vae, scheduler,
                   text_layer=text_layer, max_condition_tokens=max_condition_tokens)

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str], device: torch.device) -> list[torch.Tensor]:
        return encode_prompts_6b_teacher(
            self.vl_processor, self.vl_text_encoder, prompts, device,
            layer=self.text_layer, max_condition_tokens=self.max_condition_tokens,
        )

    @torch.no_grad()
    def denoise(
        self,
        caps: list[torch.Tensor],
        negative_caps: list[torch.Tensor] | None,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """teacher Euler 去噪, 返回归一化 latent [1, C, 1, H/8, W/8]。"""
        from tqdm.auto import tqdm

        z_dim = int(self.vae.config.z_dim)
        compress = int(self.vae.config.scale_factor_spatial)
        latent_shape = (1, z_dim, 1, height // compress, width // compress)
        generator = torch.Generator(device=device).manual_seed(seed)
        sample = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.scheduler.set_begin_index(0)
        model_dtype = next(self.transformer.parameters()).dtype

        for t_step in tqdm(self.scheduler.timesteps, desc="🔎 teacher denoise", unit="step",
                           dynamic_ncols=True, leave=False):
            latent_input = sample.to(dtype=model_dtype)
            # Z-Image 约定: t = 1 - sigma
            timestep = ((1000.0 - t_step) / 1000.0).expand(1)

            output = self.transformer(
                x=[latent_input[0]], t=timestep, cap_feats=caps,
                patch_size=2, f_patch_size=1,
            )
            model_out = output.sample if hasattr(output, "sample") else output[0]
            model_out = torch.stack(model_out) if isinstance(model_out, list) else model_out

            if negative_caps is not None and guidance_scale > 0.0:
                neg_output = self.transformer(
                    x=[latent_input[0]], t=timestep, cap_feats=negative_caps,
                    patch_size=2, f_patch_size=1,
                )
                neg_out = neg_output.sample if hasattr(neg_output, "sample") else neg_output[0]
                neg_out = torch.stack(neg_out) if isinstance(neg_out, list) else neg_out
                model_out = model_out + guidance_scale * (model_out - neg_out)

            # teacher 输出 = z0 - noise, Euler 步按 Z-Image 约定取负
            sample = self.scheduler.step(-model_out.to(torch.float32), t_step, sample, return_dict=False)[0]
        return sample

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """归一化 latent [B,C,1,h,w] → uint8 [B,1,H,W,3]。"""
        device = latents.device
        mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        decoded = self.vae.decode((latents * std + mean).to(next(self.vae.parameters()).dtype), return_dict=False)[0]
        return ((decoded / 2 + 0.5).clamp(0, 1) * 255).byte().permute(0, 2, 3, 4, 1).cpu()

    @torch.no_grad()
    def __call__(
        self,
        prompts,
        height: int = 1024,
        width: int = 1024,
        steps: int = 30,
        guidance_scale: float = 2.5,
        seed: int = 42,
        negative_prompt: str = "",
        device: str | torch.device | None = None,
    ) -> list[Image.Image]:
        device = torch.device(device) if device is not None else next(self.transformer.parameters()).device
        if height % 32 or width % 32:
            raise ValueError("height/width 必须能被 32 整除")
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        caps_list = self.encode_prompts(prompts, device)
        caps_list = [c.to(next(self.transformer.parameters()).dtype) for c in caps_list]
        negative_caps = None
        if guidance_scale > 0.0 and negative_prompt:
            negative_caps = self.encode_prompts([negative_prompt], device)
            negative_caps = [c.to(next(self.transformer.parameters()).dtype) for c in negative_caps]

        images = []
        for index, caps in enumerate(caps_list):
            latent = self.denoise(
                caps=[caps], negative_caps=negative_caps, height=height, width=width,
                num_inference_steps=steps, guidance_scale=guidance_scale,
                seed=seed + index, device=device, dtype=next(self.transformer.parameters()).dtype,
            )
            frames = self.decode(latent)
            images.append(Image.fromarray(frames[0, 0].numpy()))
        return images
