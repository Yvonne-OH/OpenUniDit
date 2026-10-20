"""Giga-Exp image/video-ready flow-matching DiT."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

try:  # FA3 (FlashAttention 3), copied under model_exp/thirdparty
    from flash_attn_interface import flash_attn_func as _fa3_func
    _FA3 = True
except Exception:
    _FA3 = False


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class SpacetimeRoPE(nn.Module):
    """4D axial RoPE over (time, height, width, token-position)."""
    def __init__(self, head_dim, theta=10000.0, axes_dims=(16, 48, 48, 16)):
        super().__init__()
        if sum(axes_dims) != head_dim or any(d % 2 for d in axes_dims):
            raise ValueError("4D rope axes_dims must be even and sum to head_dim")
        self.axes_dims = tuple(axes_dims)
        inv = []
        for dim in axes_dims:
            inv.append(1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)))
        self.register_buffer("inv_freq", torch.cat(inv), persistent=False)

    def forward(self, t, h, w, text_len, device, dtype):
        tt, yy, xx = torch.meshgrid(
            torch.arange(t, device=device), torch.arange(h, device=device),
            torch.arange(w, device=device), indexing="ij")
        image_positions = torch.cat((
            torch.stack((tt.flatten(), yy.flatten(), xx.flatten()), -1),
            torch.zeros((t * h * w, 1), device=device, dtype=torch.long),
        ), -1)
        text_positions = torch.cat((
            torch.zeros((text_len, 3), device=device, dtype=torch.long),
            (torch.arange(text_len, device=device) + 1)[:, None],
        ), -1)
        positions = torch.cat((image_positions, text_positions), 0).float()
        pieces = []
        start = 0
        for axis, dim in enumerate(self.axes_dims):
            freq = self.inv_freq[start:start + dim // 2]
            phase = positions[:, axis:axis + 1] * freq
            pieces.append(torch.cat((phase, phase), -1))
            start += dim // 2
        phase = torch.cat(pieces, -1)
        return phase.cos().to(dtype), phase.sin().to(dtype)


class GigaBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        self.heads = heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim, ffn_dim, bias=False), nn.SiLU(), nn.Linear(ffn_dim, dim, bias=False))
        self.mod = nn.Linear(dim, dim * 6)

    def forward(self, x, cond, cos, sin, attention_mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.mod(cond).chunk(6, -1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        q, k, v = self.qkv(y).chunk(3, -1)
        b, n, _ = q.shape
        if _FA3 and attention_mask is None and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
            # FA3 layout [B, S, H, D]; cos/sin broadcast as [1, S, 1, D]
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            q = q.view(b, n, self.heads, -1)
            k = k.view(b, n, self.heads, -1)
            v = v.view(b, n, self.heads, -1)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            a = _fa3_func(q, k, v, causal=False).reshape(b, n, -1)
        else:
            q = q.view(b, n, self.heads, -1).transpose(1, 2)
            k = k.view(b, n, self.heads, -1).transpose(1, 2)
            v = v.view(b, n, self.heads, -1).transpose(1, 2)
            q = q * cos.unsqueeze(0).unsqueeze(0) + rotate_half(q) * sin.unsqueeze(0).unsqueeze(0)
            k = k * cos.unsqueeze(0).unsqueeze(0) + rotate_half(k) * sin.unsqueeze(0).unsqueeze(0)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask).transpose(1, 2).reshape(b, n, -1)
        x = x + self.out(a) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        return x + self.ff(y) * gate_f[:, None]


class GigaExpTransformer(nn.Module):
    """~0.8B dense DiT: 1792 width, 16 blocks, 14 heads, temporal RoPE retained."""
    def __init__(self, in_channels=48, out_channels=48, text_dim=4096, dim=1792,
                 ffn_dim=4864, heads=14, layers=24, rope_theta=10000.0,
                 rope_axes_dims=(32, 32, 32, 32), patch_size=(1, 1, 1)):
        super().__init__()
        self.patch_size = tuple(patch_size)
        if len(self.patch_size) != 3 or self.patch_size[0] != 1 or any(p < 1 for p in self.patch_size):
            raise ValueError("patch_size 必须是时间维为 1 的三个正整数")
        self.in_channels, self.out_channels, self.dim, self.heads = in_channels, out_channels, dim, heads
        patch_volume = math.prod(self.patch_size)
        self.patch_in = nn.Linear(in_channels * patch_volume, dim)
        self.text_in = nn.Linear(text_dim, dim)
        self.time_in = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rope = SpacetimeRoPE(dim // heads, rope_theta, rope_axes_dims)
        self.blocks = nn.ModuleList([GigaBlock(dim, heads, ffn_dim) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, out_channels * patch_volume)

    def forward(self, x, timestep, text, text_mask=None):
        if x.ndim != 5:
            raise ValueError("x must be [B,C,T,H,W]")
        b, _, t, h, w = x.shape
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError(f"latent 大小 {(t, h, w)} 必须能被 patch_size {self.patch_size} 整除")
        t, h, w = t // pt, h // ph, w // pw
        image = x.reshape(b, self.in_channels, t, pt, h, ph, w, pw)
        image = self.patch_in(image.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, t * h * w, -1))
        text = self.text_in(text)
        attention_mask = None
        if text_mask is not None and not text_mask.bool().all().item():
            valid_keys = torch.cat((torch.ones((b, image.shape[1]), device=x.device, dtype=torch.bool),
                                    text_mask.bool()), dim=1)
            attention_mask = valid_keys[:, None, None, :]
        x = torch.cat((image, text), 1)
        ts = timestep.float().reshape(-1).expand(b)
        half = 128
        freq = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        emb = torch.cat(((ts[:, None] * freq).cos(), (ts[:, None] * freq).sin()), -1)
        cond = self.time_in(emb.to(self.time_in[0].weight.dtype))
        cos, sin = self.rope(t, h, w, text.shape[1], x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cond, cos, sin, attention_mask)
        image = self.out(self.norm(x[:, :t * h * w]))
        image = image.reshape(b, t, h, w, pt, ph, pw, self.out_channels)
        return image.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, self.out_channels, t * pt, h * ph, w * pw)
