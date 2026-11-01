"""Giga-Exp mHC 变体: 独立于主分支的 giga_transformer.py, 只改造 residual connection。

设计约束 (见 docs/mhc/audit.md §6):
- 原始 attention / FFN / AdaLN / RoPE / patchify / output head 逻辑与权重名完全不变;
- 每个 GigaBlockMHC 的两处 residual add 替换为两个独立 MHCResidual (mhc_attn / mhc_ffn);
- 子层核心 (attn_core / ffn_core) 逐行复制自 GigaBlock.forward 并去掉 residual add,
  gate (gate_a/gate_f) 归属于子层, mHC 不重复加 gate;
- 子层接收单流 [B,N,C] (pre 映射混合), 因此 FA3/varlen/pack/RoPE/mask 路径零改动;
- H = [x, 0, ..., 0] 进模型, readout=[1,0..0] stream0 直出, 初始状态与原模型 forward 等价。

state_dict 命名: 原 291 个张量名 1:1 保留; 新增 blocks.{i}.mhc_attn.mapping.*,
blocks.{i}.mhc_ffn.mapping.*, readout。
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.giga_transformer import (
    MAX_SEQLEN_IMAGE,
    MAX_SEQLEN_VIDEO,
    RMSNorm,
    SpacetimeRoPE,
    _FA3,
    rotate_half,
)
from models.mhc_residual import MHCResidual, read_out, stack_stream


class GigaBlockMHC(nn.Module):
    """GigaBlock 的 mHC 版本: 子层权重名与 GigaBlock 完全一致。"""

    def __init__(self, dim, heads, ffn_dim, n_streams=4, sinkhorn_iters=20,
                 mhc_mapping_fp32=True):
        super().__init__()
        self.heads = heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        head_dim = dim // heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim, ffn_dim, bias=False), nn.SiLU(),
                                nn.Linear(ffn_dim, dim, bias=False))
        self.mod = nn.Linear(dim, dim * 6)
        self.mhc_attn = MHCResidual(dim, n_streams=n_streams, sinkhorn_iters=sinkhorn_iters,
                                    mapping_fp32=mhc_mapping_fp32)
        self.mhc_ffn = MHCResidual(dim, n_streams=n_streams, sinkhorn_iters=sinkhorn_iters,
                                   mapping_fp32=mhc_mapping_fp32)

    # ------------------------------------------------------------------ #
    # 原始子层 (复制自 GigaBlock.forward, 仅去掉 residual add)
    # ------------------------------------------------------------------ #
    def _attn_core(self, x, cond, cos, sin, attention_mask=None, pack=None):
        shift_a, scale_a, gate_a, _, _, _ = self.mod(cond).chunk(6, -1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        q, k, v = self.qkv(y).chunk(3, -1)
        b, n, _ = q.shape
        q = self.q_norm(q.view(b, n, self.heads, -1)).view(b, n, -1)
        k = self.k_norm(k.view(b, n, self.heads, -1)).view(b, n, -1)
        if _FA3 and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and (attention_mask is None or pack is not None):
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
            qf, kf = q.float(), k.float()
            q = (qf * cos + rotate_half(qf) * sin).to(v.dtype)
            k = (kf * cos + rotate_half(kf) * sin).to(v.dtype)
            if pack is not None:
                a = _fa3_varlen(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
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
        # 原: x = x + self.out(a) * gate_a[:, None]; mHC 中 residual add 由 mhc_attn 完成
        return self.out(a) * gate_a[:, None]

    def _ffn_core(self, x, cond):
        _, _, _, shift_f, scale_f, gate_f = self.mod(cond).chunk(6, -1)
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        # 原: return x + self.ff(y) * gate_f[:, None]
        return self.ff(y) * gate_f[:, None]

    def forward(self, h, cond, cos, sin, attention_mask=None, pack=None):
        h = self.mhc_attn(h, lambda s: self._attn_core(s, cond, cos, sin, attention_mask, pack))
        h = self.mhc_ffn(h, lambda s: self._ffn_core(s, cond))
        return h


class GigaExpTransformerMHC(nn.Module):
    """mHC DiT: 输入输出接口与 GigaExpTransformer 完全一致 ([B,C,T,H,W] 像素域 latent)。"""

    def __init__(self, in_channels=48, out_channels=48, text_dim=4096, dim=1792,
                 ffn_dim=4864, heads=14, layers=24, rope_theta=10000.0,
                 rope_axes_dims=(32, 32, 32, 32), patch_size=(1, 1, 1),
                 n_streams=4, sinkhorn_iters=20, mhc_mapping_fp32=True):
        super().__init__()
        self.patch_size = tuple(patch_size)
        if len(self.patch_size) != 3 or self.patch_size[0] != 1 or any(p < 1 for p in self.patch_size):
            raise ValueError("patch_size 必须是时间维为 1 的三个正整数")
        self.in_channels, self.out_channels, self.dim, self.heads = in_channels, out_channels, dim, heads
        self.text_dim, self.ffn_dim, self.layers = text_dim, ffn_dim, layers
        self.n_streams = int(n_streams)
        patch_volume = math.prod(self.patch_size)
        self.patch_in = nn.Linear(in_channels * patch_volume, dim)
        self.text_in = nn.Linear(text_dim, dim)
        self.time_in = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rope = SpacetimeRoPE(dim // heads, rope_theta, rope_axes_dims)
        self.blocks = nn.ModuleList([
            GigaBlockMHC(dim, heads, ffn_dim, n_streams=self.n_streams,
                         sinkhorn_iters=sinkhorn_iters, mhc_mapping_fp32=mhc_mapping_fp32)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, out_channels * patch_volume)
        # read-out: stream 线性组合, 初始 [1,0..0] = stream0 直出 (function-preserving)
        self.readout = nn.Parameter(torch.zeros(self.n_streams))
        with torch.no_grad():
            self.readout[0] = 1.0

    def semantic_metadata(self):
        meta = {
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
                "version": "giga_semantics_v1_mhc",
                "rope": "fp32_cos_sin_and_rotation",
                "qk_norm": "per_head_rms_norm",
                "rotate_layout": "half_chunk",
                "mhc": "v1",
            },
            "mhc": {"n_streams": self.n_streams},
        }
        return meta

    def forward(self, x, timestep, text, text_mask=None):
        if x.ndim != 5:
            raise ValueError("x must be [B,C,T,H,W]")
        b, _, t, h, w = x.shape
        H_px, W_px = h, w  # 像素域尺寸 (后面 h 复用为 hidden 变量)
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError(f"latent 大小 {(t, h, w)} 必须能被 patch_size {self.patch_size} 整除")
        t, gh, gw = t // pt, h // ph, w // pw
        image = x.reshape(b, self.in_channels, t, pt, gh, ph, gw, pw)
        image = self.patch_in(image.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, t * gh * gw, -1))
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
                lens = n_img + valid_text.sum(-1)
                cu_seqlens = torch.zeros(b + 1, device=x.device, dtype=torch.int32)
                cu_seqlens[1:] = lens.cumsum(0).to(torch.int32)
                cols = torch.arange(n_total, device=x.device)
                valid2d = cols[None, :] < lens[:, None]
                pack_idx = (torch.arange(b, device=x.device)[:, None] * n_total + cols[None, :])[valid2d]
                pos_idx = cols[None, :].expand(b, -1)[valid2d]
                bound = MAX_SEQLEN_IMAGE if t == 1 else MAX_SEQLEN_VIDEO
                if n_total > bound:
                    raise ValueError(f"n_total={n_total} 超过上界 {bound}, 请调大对应 MAX_SEQLEN_* 常量")
                max_seqlen = bound
                pack = (pack_idx, pos_idx, cu_seqlens, max_seqlen)
        h = torch.cat((image, text), 1)
        ts = timestep.float().reshape(-1).expand(b)
        half = 128
        freq = 10000 ** (-torch.arange(half, device=h.device, dtype=torch.float32) / half)
        emb = torch.cat(((ts[:, None] * freq).cos(), (ts[:, None] * freq).sin()), -1)
        cond = self.time_in(emb.to(self.time_in[0].weight.dtype))
        cos, sin = self.rope(t, gh, gw, text.shape[1], h.device, h.dtype)
        # mHC 接入点: 唯一的 stream 扩展与收缩
        h = stack_stream(h, self.n_streams)                     # [B, N, S, C]
        for block in self.blocks:
            h = block(h, cond, cos, sin, attention_mask, pack)
        image = self.out(self.norm(read_out(h[:, :t * gh * gw], self.readout)))
        image = image.reshape(b, t, gh, gw, pt, ph, pw, self.out_channels)
        out = image.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, self.out_channels, t * pt, H_px, W_px)
        return out

    def mhc_stats(self):
        """收集全模型 mHC 映射统计 (logging 用): {blocks.{i}.mhc_attn/pre_mean: tensor, ...}。"""
        stats = {}
        for i, block in enumerate(self.blocks):
            for name in ("mhc_attn", "mhc_ffn"):
                for key, value in (getattr(block, name).last_stats or {}).items():
                    stats[f"blocks.{i}.{name}/{key}"] = value
        return stats
