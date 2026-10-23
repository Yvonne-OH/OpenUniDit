"""Giga-Exp image/video-ready flow-matching DiT."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

try:  # FA3 (FlashAttention 3), copied under model_exp/thirdparty
    from flash_attn_interface import flash_attn_func as _fa3_func
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen
    _FA3 = True
except Exception:
    _FA3 = False

# FA3 varlen 的 max_seqlen 必须是 host 端常量: 若传随 batch 变化的 n_total, inductor 会按
# 数值区间 (<=1536 / >1536) 反复特化重编译, 每次重编译所有 rank 在梯度 allreduce 上干等数分钟。
# 上界 = 最大桶 (512px -> latent 64x64 -> patch 2x2 -> 1024 图像 token) + max_length(512)。
# 换更大分辨率桶或更长文本时必须同步调大, 否则 forward 里会显式报错。
MAX_VARLEN_SEQLEN = 1536


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
        self.theta = float(theta)
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self, device=None):
        return torch.cat([
            1.0 / (self.theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            for dim in self.axes_dims
        ])

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # dtype 转换后从公式重建，不能用 .float() 恢复已被 bf16/fp16 量化的频率。
        self.inv_freq = self._make_inv_freq(self.inv_freq.device)
        return self

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
            pieces.append(positions[:, axis:axis + 1] * freq)
            start += dim // 2
        phase = torch.cat(pieces, -1)          # [N, head_dim/2] 每轴唯一频率
        phase = torch.cat((phase, phase), -1)  # i ↔ i+half 同频配对, 与 rotate_half 的 chunk(2) 对齐
        return phase.cos(), phase.sin()        # 保持 fp32, 旋转在 block 内以 fp32 完成后再 cast


class GigaBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        self.heads = heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        head_dim = dim // heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim, ffn_dim, bias=False), nn.SiLU(), nn.Linear(ffn_dim, dim, bias=False))
        self.mod = nn.Linear(dim, dim * 6)

    def forward(self, x, cond, cos, sin, attention_mask=None, pack=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.mod(cond).chunk(6, -1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        q, k, v = self.qkv(y).chunk(3, -1)
        b, n, _ = q.shape
        q = self.q_norm(q.view(b, n, self.heads, -1)).view(b, n, -1)
        k = self.k_norm(k.view(b, n, self.heads, -1)).view(b, n, -1)
        # pack 非 None 时表示右 padding 文本 → 走 varlen 只算有效前缀; 否则 mask=None 走 dense FA3
        if _FA3 and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and (attention_mask is None or pack is not None):
            # FA3 layout [B, S, H, D]; cos/sin broadcast as [1, S, 1, D]
            q = q.view(b, n, self.heads, -1)
            k = k.view(b, n, self.heads, -1)
            v = v.view(b, n, self.heads, -1)
            if pack is not None:
                pack_idx, pos_idx, cu_seqlens, max_seqlen = pack
                q = q.reshape(b * n, self.heads, -1).index_select(0, pack_idx)
                k = k.reshape(b * n, self.heads, -1).index_select(0, pack_idx)
                v = v.reshape(b * n, self.heads, -1).index_select(0, pack_idx)
                cos = cos.index_select(0, pos_idx)[:, None, :]
                sin = sin.index_select(0, pos_idx)[:, None, :]
            else:
                cos = cos[None, :, None, :]
                sin = sin[None, :, None, :]
            # RoPE 旋转在 fp32 中完成, 避免 bf16 cos/sin 相位量化; FA3 要求原始 dtype
            qf, kf = q.float(), k.float()
            q = (qf * cos + rotate_half(qf) * sin).to(v.dtype)
            k = (kf * cos + rotate_half(kf) * sin).to(v.dtype)
            if pack is not None:
                a = _fa3_varlen(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
                # pad 位置回填 0: out 无 bias, 0 输入得 0 输出, residual 流不受影响
                a = q.new_zeros(b * n, self.heads, v.shape[-1]).index_copy(0, pack_idx, a).view(b, n, -1)
            else:
                a = _fa3_func(q, k, v, causal=False).reshape(b, n, -1)
        else:
            q = q.view(b, n, self.heads, -1).transpose(1, 2)
            k = k.view(b, n, self.heads, -1).transpose(1, 2)
            v = v.view(b, n, self.heads, -1).transpose(1, 2)
            cos4, sin4 = cos[None, None], sin[None, None]
            qf, kf = q.float(), k.float()
            q = (qf * cos4 + rotate_half(qf) * sin4).to(v.dtype)
            k = (kf * cos4 + rotate_half(kf) * sin4).to(v.dtype)
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
        self.text_dim, self.ffn_dim, self.layers = text_dim, ffn_dim, layers
        patch_volume = math.prod(self.patch_size)
        self.patch_in = nn.Linear(in_channels * patch_volume, dim)
        self.text_in = nn.Linear(text_dim, dim)
        self.time_in = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rope = SpacetimeRoPE(dim // heads, rope_theta, rope_axes_dims)
        self.blocks = nn.ModuleList([GigaBlock(dim, heads, ffn_dim) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, out_channels * patch_volume)

    def semantic_metadata(self):
        return {
            "channels": [int(self.in_channels), int(self.out_channels)],
            "text_dim": int(self.text_dim),
            "dim": int(self.dim),
            "ffn": int(self.ffn_dim),
            "heads": int(self.heads),
            "layers": int(self.layers),
            "patch_size": list(self.patch_size),
            "rope_theta": self.rope.theta,
            "rope_axes": list(self.rope.axes_dims),
            "implementation": {
                "version": "giga_semantics_v1",
                "rope": "fp32_cos_sin_and_rotation",
                "qk_norm": "per_head_rms_norm",
                "rotate_layout": "half_chunk",
            },
        }

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
        n_img = image.shape[1]
        n_total = n_img + text.shape[1]
        attention_mask = None
        pack = None
        if text_mask is not None:
            valid_text = text_mask.bool()
            attention_mask = torch.cat((torch.ones((b, n_img), device=x.device, dtype=torch.bool),
                                        valid_text), dim=1)[:, None, None, :]
            if _FA3:
                # 每样本有效 token 是连续前缀 (image 全有效 + text 右侧 padding), varlen 跳过 padding 计算。
                # 不调用 any/all/max().item(): 避免 torch.compile 下的 GPU→CPU 同步和 graph break。
                lens = n_img + valid_text.sum(-1)  # [b] 每样本有效长度
                cu_seqlens = torch.zeros(b + 1, device=x.device, dtype=torch.int32)
                cu_seqlens[1:] = lens.cumsum(0).to(torch.int32)
                cols = torch.arange(n_total, device=x.device)
                valid2d = cols[None, :] < lens[:, None]  # [b, n_total] 有效前缀掩码
                pack_idx = (torch.arange(b, device=x.device)[:, None] * n_total + cols[None, :])[valid2d]
                pos_idx = cols[None, :].expand(b, -1)[valid2d]  # cos/sin 按 [n_total] 位置索引
                # max_seqlen 用常量上界, FA3 kernel 只编译一次; 越界显式报错, 防止 kernel 静默截断算错
                if n_total > MAX_VARLEN_SEQLEN:
                    raise ValueError(f"n_total={n_total} 超过 MAX_VARLEN_SEQLEN={MAX_VARLEN_SEQLEN}, 请调大上界")
                max_seqlen = MAX_VARLEN_SEQLEN
                pack = (pack_idx, pos_idx, cu_seqlens, max_seqlen)
        x = torch.cat((image, text), 1)
        ts = timestep.float().reshape(-1).expand(b)
        half = 128
        freq = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        emb = torch.cat(((ts[:, None] * freq).cos(), (ts[:, None] * freq).sin()), -1)
        cond = self.time_in(emb.to(self.time_in[0].weight.dtype))
        cos, sin = self.rope(t, h, w, text.shape[1], x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cond, cos, sin, attention_mask, pack)
        image = self.out(self.norm(x[:, :t * h * w]))
        image = image.reshape(b, t, h, w, pt, ph, pw, self.out_channels)
        return image.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, self.out_channels, t * pt, h * ph, w * pw)
