from __future__ import annotations
from pathlib import Path
import yaml
import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer
from models.giga_transformer import GigaExpTransformer


class GigaExpPipeline:
    def __init__(self, transformer, text_encoder, tokenizer, vae, scheduler=None, text_layer=None):
        self.transformer, self.text_encoder, self.tokenizer, self.vae = transformer, text_encoder, tokenizer, vae
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        self.text_layer = text_layer

    @classmethod
    def from_pretrained(cls, text_path, vae_path, transformer_path=None, dtype=torch.bfloat16):
        tokenizer = AutoTokenizer.from_pretrained(Path(text_path).parent / "tokenizer", local_files_only=True)
        text = AutoModel.from_pretrained(text_path, torch_dtype=dtype, local_files_only=True).eval()
        vae = AutoencoderKLWan.from_pretrained(vae_path, subfolder="vae" if (Path(vae_path) / "vae").is_dir() else None, torch_dtype=torch.float32, local_files_only=True).eval()
        config_path = Path(__file__).resolve().parents[1] / "configs/stage_1.yaml"
        config = yaml.safe_load(config_path.read_text())
        text_layer = config["components"]["text_encoder"].get("layer")  # qwen3_vl 约定取 hidden_states[-2]
        model = GigaExpTransformer(**config["model"])
        if transformer_path is not None:
            if Path(transformer_path).suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(transformer_path), device="cpu")
            else:
                state = torch.load(transformer_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        return cls(model.to(dtype=dtype).eval(), text, tokenizer, vae,
                   FlowMatchEulerDiscreteScheduler(num_train_timesteps=config["scheduler"]["num_train_timesteps"],
                                                   shift=config["scheduler"].get("validation_shift", config["scheduler"]["shift"])),
                   text_layer=text_layer)

    @torch.no_grad()
    def encode(self, prompts, device, max_length=512):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        prompts = [str(p).strip() or self.tokenizer.eos_token for p in prompts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        out = self.text_encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=self.text_layer is not None)
        hidden = out.hidden_states[self.text_layer] if self.text_layer is not None else out.last_hidden_state
        return hidden, inputs.attention_mask.bool()

    @torch.no_grad()
    def __call__(self, prompts, height=192, width=192, steps=20, guidance_scale=3.0, seed=42, device=None):
        device = device or next(self.transformer.parameters()).device
        if height % 32 or width % 32:
            raise ValueError("height 和 width 必须能被 32 整除")
        text, mask = self.encode(prompts, device)
        neg, neg_mask = self.encode([""] * text.shape[0], device)
        compress = self.vae.config.scale_factor_spatial
        latent_h, latent_w = height // compress, width // compress
        g = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn((text.shape[0], self.vae.config.z_dim, 1, latent_h, latent_w), generator=g, device=device, dtype=torch.float32)
        self.scheduler.set_timesteps(steps, device=device)
        sigmas = self.scheduler.sigmas
        for i in range(steps):
            sigma = sigmas[i]
            timestep = (sigma * 1000).expand(z.shape[0])
            pred_c = self.transformer(z.to(next(self.transformer.parameters()).dtype), timestep, text, mask).float()
            pred_u = self.transformer(z.to(next(self.transformer.parameters()).dtype), timestep, neg, neg_mask).float()
            z = z + (sigmas[i + 1] - sigma) * (pred_u + guidance_scale * (pred_c - pred_u))
        mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        decoded = self.vae.decode((z * std + mean).to(next(self.vae.parameters()).dtype), return_dict=False)[0]
        return ((decoded / 2 + 0.5).clamp(0, 1) * 255).byte().permute(0, 2, 3, 4, 1).cpu()
