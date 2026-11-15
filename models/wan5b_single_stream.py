# -*- coding: utf-8 -*-
"""Wan2.2 TI2V-5B 单流改造版 (stage_1_wan5b)。

设计 (见 configs/stage_1_wan5b_tar.yaml 头注释):
- 主干 30 层取自 Wan2.2 TI2V-5B DiT (dim 3072 / heads 24 / ffn 14336), 权重由
  tools/convert_wan22_5b.py 从 diffusers checkpoint 转换初始化;
- 单流: 去掉每层 cross-attn (attn2), 文本 token 经 text_in (Qwen3-VL 2560→dim) +
  Text Refiner (MOSS 风格恒等初始化) 后 concat 进主序列, 与 image token 同一自注意力;
- 调制保持 Wan 原生结构: 全局 cond6 = mod_proj(time_in(timestep)) + 每层 scale_shift_table
  (非本代码库 shared-AdaLN 的 per-block Linear, 保真映射 Wan condition_embedder);
- head 保持 Wan 结构: shift/scale = (head_table + cond6[:, :2dim]).chunk(2) 调制 LayerNorm;
- patchify (1,1,1): patch_in/out 由 Wan (1,2,2) conv/proj 权重空间求和转换 (DC 保持);
- 4D RoPE 沿用本代码库 SpacetimeRoPE (time,h,w,text_position), Wan RoPE 无参数直接替换。

与 GigaExpTransformer 的 forward API 完全一致 (x, timestep, text, text_mask), 训练管线零改动。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.giga_transformer import (
    RMSNorm,
    GigaExpTransformer,
    rotate_half,
    _FA3,
    _fa3_func,
    _fa3_varlen,
    MAX_SEQLEN_IMAGE,
    MAX_SEQLEN_VIDEO,
)


class WanGigaBlock(nn.Module):
    """Wan2.2 TI2V-5B transformer block, 单流化 (无 cross-attn)。

    与 GigaBlock 的差异:
    - norm1: LayerNorm(affine=False) (Wan ckpt 无参数); norm2: LayerNorm(affine=True) (Wan ckpt 有 w/b);
    - qkv/out/ffn 全部带 bias (Wan), q/k norm 是 across-heads RMSNorm(dim) (Wan
      'rms_norm_across_heads', weight [dim]), 非 per-head RMSNorm(head_dim);
    - FFN 激活 tanh-GELU (Wan), 非 SiLU;
    - 调制: (scale_shift_table + cond6).chunk(6) (Wan condition_embedder 结构),
      非 per-block mod Linear。
    """

    def __init__(self, dim: int, ffn_dim: int, heads: int, eps: float = 1e-6,
                 zero_init_residual: bool = False):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(dim, eps=eps)   # across-heads (weight [dim]), 与 Wan 对齐
        self.k_norm = RMSNorm(dim, eps=eps)
        self.out = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)  # Wan norm2 带 affine (ckpt 有 weight/bias)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim, bias=True),
        )
        self.scale_shift_table = nn.Parameter(torch.zeros(6 * dim))
        # ReZero 残差门: 预训练块恒 1 (与 Wan 原版逐位一致); 新增块 0 初始化
        # (step0 输出恒等, 不破坏预训练前向; 可训练, 随训练自动增长)
        self.res_scale = nn.Parameter(torch.zeros(1) if zero_init_residual else torch.ones(1))
        self.last_moe_aux = None

    def _attend(self, y, cos, sin, attention_mask, pack):
        """qkv -> RoPE -> attention, 与 GigaBlock._attend 逐行等价 (qk norm 维度不同)。"""
        q, k, v = self.qkv(y).chunk(3, -1)
        b, n, _ = q.shape
        # across-heads RMSNorm: 直接对 [b, n, dim] 归一化 (与 Wan diffusers 对齐)
        q = self.q_norm(q)
        k = self.k_norm(k)
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
        return self.out(a)

    def forward(self, x, cond6, cos, sin, attention_mask=None, pack=None, n_img=None, denoising_step=None):
        """x [b, n_img+n_text, dim]; cond6 [b, 6·dim] 全局 Wan 调制向量。"""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (self.scale_shift_table + cond6).chunk(6, -1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        x = x + self._attend(y, cos, sin, attention_mask, pack) * gate_a[:, None] * self.res_scale
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        self.last_moe_aux = None
        return x + self.ff(y) * gate_f[:, None] * self.res_scale


class Wan5BSingleStream(GigaExpTransformer):
    """Wan2.2 TI2V-5B 初始化的单流 DiT (patchify 1,1,1 + Qwen3-VL 文本 + 4D RoPE)。

    复用父类的 patch_in / text_in / time_in / text_refiner / rope / out;
    块替换为 WanGigaBlock, 调制与 head 保持 Wan 原生结构。
    """

    def __init__(self, in_channels: int = 48, out_channels: int = 48, text_dim: int = 2560,
                 dim: int = 3072, ffn_dim: int = 14336, heads: int = 24, layers: int = 30,
                 pretrained_layers: int | None = None,
                 rope_theta: float = 10000.0, rope_axes_dims=(16, 48, 48, 16),
                 patch_size=(1, 1, 1), text_trans=False, eps: float = 1e-6, **kwargs):
        # 父类 layers=0 只建 embed/text_refiner/rope/out, Wan 块在下方重建。
        # adaln_type 由 yaml 经 **kwargs 传入 (Wan 结构用 shared), 缺省时父类推导同为 shared。
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, text_dim=text_dim,
            dim=dim, ffn_dim=ffn_dim, heads=heads, layers=0,
            rope_theta=rope_theta, rope_axes_dims=rope_axes_dims,
            patch_size=patch_size, text_trans=text_trans, **kwargs,
        )
        self.layers = layers
        # 深度增长: 前 pretrained_layers 层与 Wan2.2 5B 预训练权重一一对应 (res_scale=1),
        # 其余层 ReZero 零初始化 (res_scale=0, step0 恒等, 不破坏预训练前向)。
        self.pretrained_layers = layers if pretrained_layers is None else int(pretrained_layers)
        self.eps = eps
        self.blocks = nn.ModuleList(
            WanGigaBlock(dim, ffn_dim, heads, eps=eps,
                         zero_init_residual=i >= self.pretrained_layers)
            for i in range(layers))
        # Wan head 结构: norm_out(affine=False) + 全局 shift/scale 调制 + proj_out
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.mod_proj = nn.Linear(dim, dim * 6)
        self.head_table = nn.Parameter(torch.zeros(2, dim))

    def semantic_metadata(self):
        meta = super().semantic_metadata()
        meta.update({
            "class": "wan5b_single_stream",
            "pretrained_layers": int(self.pretrained_layers),
            "implementation": {
                "version": "wan22_ti2v_5b_single_stream_v1",
                "rope": "fp32_cos_sin_and_rotation",
                "qk_norm": "across_head_rms_norm",
                "rotate_layout": "half_chunk",
                "modulation": "wan_condition_embedder_scale_shift_table",
            },
        })
        return meta

    def forward(self, x, timestep, text, text_mask=None, feature_layers=None):
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
                lens = n_img + valid_text.sum(-1)  # [b] 每样本有效长度
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
        x = torch.cat((image, text), 1)
        ts = timestep.float().reshape(-1).expand(b)
        half = 128
        freq = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        emb = torch.cat(((ts[:, None] * freq).cos(), (ts[:, None] * freq).sin()), -1)
        # Wan 调制: cond6 = mod_proj(time_in(timestep)); 文本调制成分 (text_embedder) 单流化后移除
        cond6 = self.mod_proj(self.time_in(emb.to(self.time_in[0].weight.dtype)))
        cos, sin = self.rope(t, h, w, text.shape[1], x.device, x.dtype)
        self.moe_aux_losses = []
        features = [] if feature_layers is not None else None
        for block_idx, block in enumerate(self.blocks):
            x = block(x, cond6, cos, sin, attention_mask, pack, n_img)
            if features is not None and block_idx in feature_layers:
                features.append(x)
        # Wan head: shift/scale 取 cond6 前 2 块 (与 blocks 共享同一 modulation, 与 diffusers 一致)
        shift, scale = (self.head_table[None] + cond6.view(b, 6, self.dim)[:, :2]).chunk(2, 1)
        shift, scale = shift.reshape(b, self.dim), scale.reshape(b, self.dim)
        image = self.out(self.norm(x[:, :n_img]) * (1 + scale[:, None]) + shift[:, None])
        image = image.reshape(b, t, h, w, pt, ph, pw, self.out_channels)
        out = image.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, self.out_channels, t * pt, h * ph, w * pw)
        if features is not None:
            return out, features
        return out
