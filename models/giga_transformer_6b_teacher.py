"""GWP 6B teacher transformer (Z-Image 架构) — 蒸馏 teacher 专用, 与 student giga_transformer.py 区分.

来源: gigaworld-1.5- repo models/transformer/transformer_gwp.py (Apache-2.0, Z-Image/HF)。
仅保留本仓库用到的 basic T2V/T2I 路径 (x: list[[C,F,H,W]], cap_feats: list[[L,D]]),
裁剪 omni/siglip/TI2V/controlnet 分支; 权重键名与 gigaworld checkpoint 完全一致
(layers.*/noise_refiner.*/context_refiner.*/t_embedder.*/cap_embedder.*/all_x_embedder.2-1/
 all_final_layer.2-1/x_pad_token/cap_pad_token)。

约定 (对齐 gigaworld stage1 训练):
  - 模型输入 t = 1 - sigma (t=0 纯噪声, t=1 干净数据), 内部 t_embedder(t * t_scale=1000)
  - 模型原始输出 = z0 - noise (student 约定的 velocity = noise - z0 的相反数)
  - latent 为 Wan VAE mean/std 归一化空间 (与 student 相同)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

ATTENTION_BACKEND_AUTO = "auto"
ATTENTION_BACKEND_FLASH3 = "flash3"
ATTENTION_BACKEND_FLASH = "flash"
ATTENTION_BACKEND_NATIVE = "native"
ATTENTION_BACKEND_XFORMERS = "xformers"

_VALID_ATTENTION_BACKENDS = {
    ATTENTION_BACKEND_AUTO,
    ATTENTION_BACKEND_FLASH3,
    ATTENTION_BACKEND_FLASH,
    ATTENTION_BACKEND_NATIVE,
    ATTENTION_BACKEND_XFORMERS,
}


def detect_attention_backend(requested_backend: str = ATTENTION_BACKEND_AUTO) -> tuple[str, str]:
    """按 FlashAttention 3 -> FlashAttention 2 -> PyTorch SDPA -> xFormers 顺序探测。"""
    requested_backend = str(requested_backend).strip().lower()
    if requested_backend not in _VALID_ATTENTION_BACKENDS:
        raise ValueError(f"未知 attention backend: {requested_backend!r}，可选: {sorted(_VALID_ATTENTION_BACKENDS)}")
    if not torch.cuda.is_available():
        if requested_backend not in {ATTENTION_BACKEND_AUTO, ATTENTION_BACKEND_NATIVE}:
            raise RuntimeError(f"手动指定 {requested_backend}，但当前没有 CUDA")
        return ATTENTION_BACKEND_NATIVE, "无 CUDA"

    if requested_backend in {ATTENTION_BACKEND_AUTO, ATTENTION_BACKEND_FLASH3}:
        try:
            from flash_attn_interface import flash_attn_func

            q = torch.zeros(1, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
            flash_attn_func(q, q, q, causal=False)
            return ATTENTION_BACKEND_FLASH3, "flash-attn FA3 探测通过"
        except Exception:
            if requested_backend == ATTENTION_BACKEND_FLASH3:
                raise RuntimeError("手动指定 flash3，但实际 CUDA 探测失败")

    if requested_backend in {ATTENTION_BACKEND_AUTO, ATTENTION_BACKEND_FLASH}:
        try:
            from flash_attn import flash_attn_func

            q = torch.zeros(1, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
            flash_attn_func(q, q, q, causal=False)
            return ATTENTION_BACKEND_FLASH, "flash-attn FA2 探测通过"
        except Exception:
            if requested_backend == ATTENTION_BACKEND_FLASH:
                raise RuntimeError("手动指定 flash，但实际 CUDA 探测失败")

    q = torch.zeros(1, 1, 2, 64, device="cuda", dtype=torch.bfloat16)
    F.scaled_dot_product_attention(q, q, q)
    return ATTENTION_BACKEND_NATIVE, "PyTorch SDPA 探测通过"


GLOBAL_ATTENTION_BACKEND = ATTENTION_BACKEND_NATIVE
GLOBAL_ATTENTION_BACKEND_REASON = "尚未执行启动探测"


def configure_attention_backend(requested_backend: str = ATTENTION_BACKEND_AUTO) -> tuple[str, str]:
    """启动时探测一次，并为当前进程设置全局 attention 后端。"""
    global GLOBAL_ATTENTION_BACKEND, GLOBAL_ATTENTION_BACKEND_REASON
    GLOBAL_ATTENTION_BACKEND, GLOBAL_ATTENTION_BACKEND_REASON = detect_attention_backend(requested_backend)
    GWPSingleStreamAttnProcessor6BTeacher._attention_backend = GLOBAL_ATTENTION_BACKEND
    return GLOBAL_ATTENTION_BACKEND, GLOBAL_ATTENTION_BACKEND_REASON


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class GWPSingleStreamAttnProcessor6BTeacher:
    """Z-Image single stream attention processor (与 transformer_gwp.py 行为一致)。"""

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("GWPSingleStreamAttnProcessor6BTeacher requires PyTorch 2.0+")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # [batch, seq_len] -> [batch, 1, 1, seq_len] 广播到 [batch, heads, seq, seq]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        attention_backend = self._attention_backend
        if attention_backend == ATTENTION_BACKEND_FLASH3 and attention_mask is None:
            from flash_attn_interface import flash_attn_func

            hidden_states = flash_attn_func(query, key, value, causal=False)
        else:
            if attention_backend == ATTENTION_BACKEND_FLASH and attention_mask is not None:
                attention_backend = "flash_varlen"
            elif attention_backend == ATTENTION_BACKEND_FLASH3:
                attention_backend = ATTENTION_BACKEND_NATIVE
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=attention_backend,
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@maybe_allow_in_graph
class GWPTransformerBlock6BTeacher(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=GWPSingleStreamAttnProcessor6BTeacher(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ):
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        scale = scale.unsqueeze(1)
        x = self.norm_final(x) * scale
        return self.linear(x)


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: list[int] = (32, 48, 48),
        axes_lens: list[int] = (4096, 512, 512),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens) == 3, "GWP 6B teacher RoPE requires exactly 3 axes (time, height, width)"
        assert all(length > 0 for length in axes_lens), "RoPE axes_lens must all be positive"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: list[int], end: list[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            if index.min() < 0 or index.max() >= self.axes_lens[i]:
                raise ValueError(
                    f"RoPE axis {i} position range [{index.min().item()}, {index.max().item()}] "
                    f"exceeds configured length {self.axes_lens[i]}"
                )
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class GWPTransformer6BTeacher(ModelMixin, ConfigMixin):
    """GWP 6B teacher (Z-Image 架构): dim=3840, 30 层, 30 头, head_dim=128, FFN=10240.

    输入契约 (对齐 gigaworld 训练/推理):
      x: list of [C, F, H, W] latent (每样本一个, 已按 Wan VAE mean/std 归一化)
      t: [B] 张量, 值为 1 - sigma (0=纯噪声, 1=干净数据)
      cap_feats: list of [L, D] Qwen3-VL hidden (layer -2, 无 batch 维)
    输出: list of [C, F, H, W] (未 patchify 还原), 值为 z0 - noise。
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["GWPTransformerBlock6BTeacher"]
    _repeated_blocks = ["GWPTransformerBlock6BTeacher"]

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=48,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[4096, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                GWPTransformerBlock6BTeacher(1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                GWPTransformerBlock6BTeacher(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                GWPTransformerBlock6BTeacher(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims), f"head_dim {head_dim} != sum(axes_dims) {sum(axes_dims)}"
        assert len(axes_dims) == len(axes_lens) == 3, "GWP 6B teacher requires exactly 3 RoPE axes"
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify_image(self, image: torch.Tensor, patch_size: int, f_patch_size: int):
        """(C, F, H, W) -> (num_patches, patch_dim)。"""
        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F, H, W = image.size()
        if F > self.axes_lens[0]:
            raise ValueError(f"GWP 6B teacher F 轴长度 {F} 超过 axes_lens[0]={self.axes_lens[0]}")
        if F % pF or H % pH or W % pW:
            raise ValueError(f"patchify 输入尺寸必须可整除 patch: {(F, H, W)} / {(pF, pH, pW)}")
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return image, (F, H, W), (F_tokens, H_tokens, W_tokens)

    def _pad_with_ids(self, feat: torch.Tensor, pos_grid_size: tuple, pos_start: tuple, device: torch.device):
        """Pad 到 SEQ_MULTI_OF 对齐, 生成 position ids 与内部 pad mask。"""
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF
        total_len = ori_len + pad_len

        ori_pos_ids = self.create_coordinate_grid(size=pos_grid_size, start=pos_start, device=device).flatten(0, 2)
        if pad_len > 0:
            pad_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                .flatten(0, 2)
                .repeat(pad_len, 1)
            )
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat(
                [
                    torch.zeros(ori_len, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device),
                ]
            )
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)
        return padded_feat, pos_ids, pad_mask, total_len

    def patchify_and_embed(
        self,
        all_image: list[torch.Tensor],
        all_cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        """基本模式: 每样本单图。token 顺序 [x, cap]。"""
        device = all_image[0].device
        all_img_out, all_img_size, all_img_pos_ids, all_img_pad_mask = [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats):
            # Caption: pos 从 (1,0,0) 起, 为 image 区留出 pos0 (Z-Image 预训练约定)
            cap_out, cap_pos_ids, cap_pad_mask, cap_len = self._pad_with_ids(
                cap_feat, (len(cap_feat), 1, 1), (1, 0, 0), device
            )
            all_cap_out.append(cap_out)
            all_cap_pos_ids.append(cap_pos_ids)
            all_cap_pad_mask.append(cap_pad_mask)

            img_patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
            img_out, img_pos_ids, img_pad_mask, img_len = self._pad_with_ids(
                img_patches, (F_t, H_t, W_t), (cap_len + 1, 0, 0), device
            )
            all_img_out.append(img_out)
            all_img_size.append(size)
            all_img_pos_ids.append(img_pos_ids)
            all_img_pad_mask.append(img_pad_mask)

        return all_img_out, all_cap_out, all_img_size, all_img_pos_ids, all_cap_pos_ids, all_img_pad_mask, all_cap_pad_mask

    def _prepare_sequence(
        self,
        feats: list[torch.Tensor],
        pos_ids: list[torch.Tensor],
        inner_pad_mask: list[torch.Tensor],
        pad_token: torch.nn.Parameter,
        device: torch.device = None,
    ):
        """应用 pad token, RoPE embed, pad 到 batch, 生成 attention mask。"""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        feats_cat = torch.cat(feats, dim=0)
        mask = torch.cat(inner_pad_mask).unsqueeze(-1)
        feats_cat = torch.where(mask, pad_token, feats_cat)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))

        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        # attention mask 仅屏蔽 batch 间外部 padding; 内部 SEQ_MULTI_OF padding 保持 attend
        # (与官方 Z-Image 预训练行为一致)
        if all(seq == max_seqlen for seq in item_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(item_seqlens):
                attn_mask[i, :seq_len] = True
        return feats, freqs_cis, attn_mask, item_seqlens

    def _build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: torch.Tensor,
        x_seqlens: list[int],
        cap: torch.Tensor,
        cap_freqs: torch.Tensor,
        cap_seqlens: list[int],
        device: torch.device,
    ):
        """基本模式: [x, cap] 拼接 + batch 间 padding。"""
        bsz = len(x_seqlens)
        unified = [torch.cat([x[i][: x_seqlens[i]], cap[i][: cap_seqlens[i]]]) for i in range(bsz)]
        unified_freqs = [torch.cat([x_freqs[i][: x_seqlens[i]], cap_freqs[i][: cap_seqlens[i]]]) for i in range(bsz)]

        unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
        max_seqlen = max(unified_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        if all(seq == max_seqlen for seq in unified_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(unified_seqlens):
                attn_mask[i, :seq_len] = True
        return unified, unified_freqs, attn_mask

    def _unpatchify_one(self, tokens: torch.Tensor, size: tuple, patch_size: int, f_patch_size: int) -> torch.Tensor:
        pH = pW = patch_size
        pF = f_patch_size
        F, H, W = size
        ori_len = (F // pF) * (H // pH) * (W // pW)
        return (
            tokens[:ori_len]
            .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(self.out_channels, F, H, W)
        )

    def forward(
        self,
        x: list[torch.Tensor],
        t,
        cap_feats: list[torch.Tensor],
        return_dict: bool = True,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        """Flow: patchify -> t_embed -> x_embed -> x_refine -> cap_embed -> cap_refine
        -> build_unified -> main_layers -> final_layer -> unpatchify。"""
        if f_patch_size != 1:
            raise ValueError(f"GWP 6B teacher 仅支持 f_patch_size=1, 收到 {f_patch_size}")
        if patch_size not in self.all_patch_size or f_patch_size not in self.all_f_patch_size:
            raise ValueError(
                f"不支持的 patch 配置: patch_size={patch_size}, f_patch_size={f_patch_size}; "
                f"可用配置={list(zip(self.all_patch_size, self.all_f_patch_size))}"
            )
        if not isinstance(x, list) or not x:
            raise TypeError("x must be a non-empty list of [C,F,H,W] tensors")
        if not isinstance(cap_feats, list) or len(cap_feats) != len(x):
            raise TypeError("cap_feats must be a list with one [L,D] tensor per sample")
        device = x[0].device

        # t = 1 - sigma (Z-Image 预训练约定: t=0 纯噪声, t=1 干净数据)
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])

        # Patchify
        (x, cap_feats, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask) = self.patchify_and_embed(
            x, cap_feats, patch_size, f_patch_size
        )

        # X embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _ = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, device
        )
        for layer in self.noise_refiner:
            x = (
                self._gradient_checkpointing_func(layer, x, x_mask, x_freqs, adaln_input)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(x, x_mask, x_freqs, adaln_input)
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))
        cap_feats, cap_freqs, cap_mask, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, device
        )
        for layer in self.context_refiner:
            cap_feats = (
                self._gradient_checkpointing_func(layer, cap_feats, cap_mask, cap_freqs)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(cap_feats, cap_mask, cap_freqs)
            )

        unified, unified_freqs, unified_mask = self._build_unified_sequence(
            x, x_freqs, x_seqlens, cap_feats, cap_freqs, cap_seqlens, device
        )

        for layer in self.layers:
            unified = (
                self._gradient_checkpointing_func(layer, unified, unified_mask, unified_freqs, adaln_input)
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(unified, unified_mask, unified_freqs, adaln_input)
            )

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)

        # Unpatchify: list of [C, F, H, W]
        x = [self._unpatchify_one(tokens, size, patch_size, f_patch_size) for tokens, size in zip(unified.unbind(dim=0), x_size)]

        return (x,) if not return_dict else Transformer2DModelOutput(sample=x)
