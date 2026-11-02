"""Giga-Exp mHC 变体: 独立于主分支的 giga_transformer.py, 只改造 residual connection。

设计约束 (见 docs/mhc/audit.md §6):
- 原始 attention / FFN / AdaLN / RoPE / patchify / output head 逻辑与权重名完全不变;
- 每个 GigaBlockMHC 的两处 residual add 替换为两个独立 MHCResidual (mhc_attn / mhc_ffn);
- 子层核心 (attn_core / ffn_core) 逐行复制自 GigaBlock.forward 并去掉 residual add,
  gate (gate_a/gate_f) 归属于子层, mHC 不重复加 gate;
- 子层接收单流 [B,N,C] (pre 映射混合), 因此 FA3/varlen/pack/RoPE/mask 路径零改动;
- H = [x, 0, ..., 0] 进模型, readout=[1,0..0] stream0 直出, 初始状态与原模型 forward 等价。

mod_adaln=True 时 (yaml model 下显式开启, 默认 False 保持上述约束):
- blocks.{i}.mod 替换为 mod_img / mod_txt 双分支 (MiniMax H3 风格):
  image 前缀用 mod_img 调制, text 后缀用 mod_txt, 两分支独立参数独立梯度;
- 旧 ckpt 恢复时由 models.giga_transformer.apply_mod_adaln_migration 把旧 mod.*
  拷贝初始化到两分支 (拷贝复用而非权重共享), step0 forward 与 shared AdaLN 等价。

state_dict 命名: 原 291 个张量名 1:1 保留 (mod_adaln=False); 新增 blocks.{i}.mhc_attn.mapping.*,
blocks.{i}.mhc_ffn.mapping.*, readout; mod_adaln=True 时 blocks.{i}.mod.* → mod_img/mod_txt。
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
    _fa3_func,
    _fa3_varlen,
    TextRefiner,
    rotate_half,
)
from models.mhc_residual import MHCResidual, read_out, stack_stream


class GigaBlockMHC(nn.Module):
    """GigaBlock 的 mHC 版本: 子层权重名与 GigaBlock 完全一致。"""

    def __init__(self, dim, heads, ffn_dim, n_streams=4, sinkhorn_iters=20,
                 mhc_mapping_fp32=True, mod_adaln=False):
        super().__init__()
        self.heads = heads
        self.mod_adaln = bool(mod_adaln)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        head_dim = dim // heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim, ffn_dim, bias=False), nn.SiLU(),
                                nn.Linear(ffn_dim, dim, bias=False))
        if self.mod_adaln:
            # dual-branch AdaLN: 旧 ckpt 恢复时由 apply_mod_adaln_migration 拷贝初始化
            self.mod_img = nn.Linear(dim, dim * 6)
            self.mod_txt = nn.Linear(dim, dim * 6)
        else:
            self.mod = nn.Linear(dim, dim * 6)
        self.mhc_attn = MHCResidual(dim, n_streams=n_streams, sinkhorn_iters=sinkhorn_iters,
                                    mapping_fp32=mhc_mapping_fp32)
        self.mhc_ffn = MHCResidual(dim, n_streams=n_streams, sinkhorn_iters=sinkhorn_iters,
                                   mapping_fp32=mhc_mapping_fp32)

    # ------------------------------------------------------------------ #
    # 原始子层 (复制自 GigaBlock.forward, 仅去掉 residual add)
    # ------------------------------------------------------------------ #
    def _attn_core(self, x, cond, cos, sin, attention_mask=None, pack=None, n_img=None):
        if self.mod_adaln:
            sa_i, sca_i, ga_i, _, _, _ = self.mod_img(cond).chunk(6, -1)
            sa_t, sca_t, ga_t, _, _, _ = self.mod_txt(cond).chunk(6, -1)
            img, txt = x[:, :n_img], x[:, n_img:]
            y = torch.cat((self.norm1(img) * (1 + sca_i[:, None]) + sa_i[:, None],
                           self.norm1(txt) * (1 + sca_t[:, None]) + sa_t[:, None]), 1)
        else:
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
        out = self.out(a)
        if self.mod_adaln:
            return torch.cat((out[:, :n_img] * ga_i[:, None],
                              out[:, n_img:] * ga_t[:, None]), 1)
        return out * gate_a[:, None]

    def _ffn_core(self, x, cond, n_img=None):
        if self.mod_adaln:
            _, _, _, sf_i, scf_i, gf_i = self.mod_img(cond).chunk(6, -1)
            _, _, _, sf_t, scf_t, gf_t = self.mod_txt(cond).chunk(6, -1)
            img, txt = x[:, :n_img], x[:, n_img:]
            y = torch.cat((self.norm2(img) * (1 + scf_i[:, None]) + sf_i[:, None],
                           self.norm2(txt) * (1 + scf_t[:, None]) + sf_t[:, None]), 1)
            f = self.ff(y)
            return torch.cat((f[:, :n_img] * gf_i[:, None],
                              f[:, n_img:] * gf_t[:, None]), 1)
        _, _, _, shift_f, scale_f, gate_f = self.mod(cond).chunk(6, -1)
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        # 原: return x + self.ff(y) * gate_f[:, None]
        return self.ff(y) * gate_f[:, None]

    def forward(self, h, cond, cos, sin, attention_mask=None, pack=None, n_img=None):
        if self.mod_adaln and (n_img is None or not 0 < n_img < h.shape[1]):
            raise ValueError(f"mod_adaln 需要 0 < n_img < seq_len, got n_img={n_img}, seq_len={h.shape[1]}")
        h = self.mhc_attn(h, lambda s: self._attn_core(s, cond, cos, sin, attention_mask, pack, n_img))
        h = self.mhc_ffn(h, lambda s: self._ffn_core(s, cond, n_img))
        return h


class GigaExpTransformerMHC(nn.Module):
    """mHC DiT: 输入输出接口与 GigaExpTransformer 完全一致 ([B,C,T,H,W] 像素域 latent)。"""

    def __init__(self, in_channels=48, out_channels=48, text_dim=4096, dim=1792,
                 ffn_dim=4864, heads=14, layers=24, rope_theta=10000.0,
                 rope_axes_dims=(32, 32, 32, 32), patch_size=(1, 1, 1),
                 n_streams=4, sinkhorn_iters=20, mhc_mapping_fp32=True, mod_adaln=False,
                 text_trans=False):
        super().__init__()
        self.patch_size = tuple(patch_size)
        if len(self.patch_size) != 3 or self.patch_size[0] != 1 or any(p < 1 for p in self.patch_size):
            raise ValueError("patch_size 必须是时间维为 1 的三个正整数")
        self.in_channels, self.out_channels, self.dim, self.heads = in_channels, out_channels, dim, heads
        self.text_dim, self.ffn_dim, self.layers = text_dim, ffn_dim, layers
        self.n_streams = int(n_streams)
        self.mod_adaln = bool(mod_adaln)
        # Text Refiner (与 GigaExpTransformer 同款, MOSS-Image 风格): 恒等零初始化残差块,
        # 只精炼 text_in 投影后的文本 token; text_trans 可为 bool 或 dict({enabled, layers, heads})。
        refiner_cfg = None
        if isinstance(text_trans, dict):
            refiner_cfg = text_trans if text_trans.get("enabled", True) else None
        elif text_trans:
            refiner_cfg = {}
        self.text_trans = None if refiner_cfg is None else {
            "layers": int(refiner_cfg.get("layers", 2)),
            "heads": int(refiner_cfg.get("heads", heads)),
        }
        if self.text_trans is not None:
            self.text_refiner = TextRefiner(dim, self.text_trans["heads"], self.text_trans["layers"])
        patch_volume = math.prod(self.patch_size)
        self.patch_in = nn.Linear(in_channels * patch_volume, dim)
        self.text_in = nn.Linear(text_dim, dim)
        self.time_in = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rope = SpacetimeRoPE(dim // heads, rope_theta, rope_axes_dims)
        self.blocks = nn.ModuleList([
            GigaBlockMHC(dim, heads, ffn_dim, n_streams=self.n_streams,
                         sinkhorn_iters=sinkhorn_iters, mhc_mapping_fp32=mhc_mapping_fp32,
                         mod_adaln=self.mod_adaln)
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
            "mod_adaln": self.mod_adaln,
            "text_trans": self.text_trans,
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
        if self.text_trans is not None:
            # Text Refiner: 恒等初始化的残差自注意力+FFN, 只精炼文本 token, padding mask 透传
            text = self.text_refiner(text, text_mask)
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
            h = block(h, cond, cos, sin, attention_mask, pack, n_img)
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
