"""Giga-Exp image/video-ready flow-matching DiT."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

from .moe import MoEFFN

try:  # FA3 (FlashAttention 3), copied under model_exp/thirdparty
    from flash_attn_interface import flash_attn_func as _fa3_func
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen
    _FA3 = True
except Exception:
    _FA3 = False

# FA3 varlen 的 max_seqlen 必须是 host 端常量: 若传随 batch 变化的 n_total, inductor 会按
# 数值区间 (<=1536 / >1536) 反复特化重编译, 每次重编译所有 rank 在梯度 allreduce 上干等数分钟。
# 按 t==1 图像 / t>1 视频分两档常量, kernel 各编译一次; 越界在 forward 里显式报错。
# 图像档: 2K 图 (2048/8=256 -> latent 256x256 -> patch 2x2 -> 16384 token) + 2048 文本。
# 视频档: 15s@24fps=361 帧 -> VAE 时间 /4 -> 91 latent 帧; 1280x720 -> latent 160x90 ->
#         patch 2x2 -> 每帧 80x45=3600 token, 91 帧 = 327600; + 4096 文本。
MAX_SEQLEN_IMAGE = 16384 + 2048
MAX_SEQLEN_VIDEO = 91 * 3600 + 4096


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


def apply_mod_adaln_migration(current, state):
    """mod-adaln checkpoint 迁移: 把旧 shared-AdaLN ckpt 的 blocks.*.mod.* 拷贝初始化到
    mod_img/mod_txt 双分支 (拷贝复用而非权重共享, 加载后两分支独立接收梯度)。

    - current: 模型期望的 state_dict (只读, 判断模型是否双分支);
    - state:   checkpoint state_dict (就地修改: 补齐 mod_img/mod_txt, 移除已拷贝的旧 mod.*);
    - 返回 {"model_dual": bool, "copied": [层号], "missing": [层号], "orphan_dual": [层号]}
      copied:       旧 mod.* 已拷贝到双分支的层 (step0 forward 与 shared AdaLN 逐位等价);
      missing:      期望双分支但 ckpt 既无 mod_img/mod_txt 也无旧 mod.* (保持随机初始化);
      orphan_dual:  模型为 shared mod 但 ckpt 是双分支 (mod.* 无法恢复, 保持随机初始化)。
    """
    result = {"model_dual": False, "copied": [], "missing": [], "orphan_dual": []}
    if any(".mod_img." in key for key in current):
        result["model_dual"] = True
        layers = sorted({int(key.split(".")[1]) for key in current if ".mod_img." in key})
        for i in layers:
            if f"blocks.{i}.mod_img.weight" in state and f"blocks.{i}.mod_txt.weight" in state:
                continue  # 新格式 ckpt, 直接加载, 无需迁移
            old = {part: state.get(f"blocks.{i}.mod.{part}") for part in ("weight", "bias")}
            if old["weight"] is None:
                result["missing"].append(i)
                continue
            for branch in ("mod_img", "mod_txt"):
                for part in ("weight", "bias"):
                    key = f"blocks.{i}.{branch}.{part}"
                    if key in current and key not in state and old[part] is not None:
                        state[key] = old[part]
            state.pop(f"blocks.{i}.mod.weight", None)
            state.pop(f"blocks.{i}.mod.bias", None)
            result["copied"].append(i)
    else:
        result["orphan_dual"] = sorted({int(key.split(".")[1]) for key in state if ".mod_img." in key})
    return result


def _adaln_time_curve(state, num_t=512):
    """按 ckpt 的 time_in 权重重建 t∈[0,1000] 上的条件曲线 c(t) [num_t, dim]。

    与 GigaExpTransformer.forward 的 timestep embedding 完全一致
    (freq = 10000^(-arange(128)/128), emb = cat(cos, sin), time_in = Linear-SiLU-Linear)。"""
    w0, b0 = state.get("time_in.0.weight"), state.get("time_in.0.bias")
    w1, b1 = state.get("time_in.2.weight"), state.get("time_in.2.bias")
    if w0 is None or w1 is None:
        return None
    half = w0.shape[1] // 2
    ts = torch.linspace(0.0, 1000.0, num_t)
    freq = 10000.0 ** (-torch.arange(half, dtype=torch.float32) / half)
    emb = torch.cat(((ts[:, None] * freq).cos(), (ts[:, None] * freq).sin()), -1)
    x = F.silu(F.linear(emb, w0.float(), b0.float()))
    return F.linear(x, w1.float(), b1.float())


def apply_adaln_basis_migration(current, state):
    """basis (r=64 共享基) ckpt 迁移: 把旧全秩 AdaLN ckpt 低秩转换到共享基结构。

    数学: 调制只依赖 timestep, 输入永远落在曲线 c(t) 张成的低维子空间上
    (实测 56 条调制曲线有效秩 ~32-64)。取曲线协方差 Σ = cᵀc 的 top-r 特征向量作
    共享基 P (r×d, 即 adaln_basis.weight), 每分支系数头 A_i = W_i Pᵀ
    (即 blocks.{i}.mod_{img,txt}.weight [6d, r]), 则 A_i·P·c(t) = W_i·Π_r·c(t) ≈ W_i·c(t),
    在曲线上的误差只取决于被丢弃的曲线尾部能量 (实测 r=64 全模型 ~3.2% RMS)。
    注意必须用曲线主方向而非 W_all 的 SVD: 后者按矩阵 Frobenius 能量截断,
    对 t 流形上的实际误差是错位的。bias 逐字保留。

    - current: basis 模型期望的 state_dict (须含 adaln_basis.weight [r, dim]);
    - state:   checkpoint state_dict (就地修改);
    - 返回 {"converted": bool, "source": "basis"|"dual"|"shared"|None,
            "rank": int, "energy": float (t 曲线能量捕获率), "layers": [层号]}
      source=None 表示模型不是 basis (调用方应回退 apply_mod_adaln_migration) 或
      ckpt 无任何可转换的全秩 mod 权重 (随机初始化)。"""
    basis_key = "adaln_basis.weight"
    if basis_key not in current:
        return {"converted": False, "source": None, "rank": None, "energy": None, "layers": []}
    rank, dim = current[basis_key].shape
    if state.get(basis_key) is not None and state[basis_key].shape == (rank, dim):
        return {"converted": False, "source": "basis", "rank": rank, "energy": 1.0, "layers": []}
    # 旧 shared ckpt: 先按拷贝迁移到 dual 双分支, 再统一走低秩转换
    source = "dual"
    if not any(f"blocks.0.{b}.weight" in state for b in ("mod_img", "mod_txt")):
        if not any(key.startswith("blocks.") and ".mod.weight" in key for key in state):
            return {"converted": False, "source": None, "rank": rank, "energy": None, "layers": []}
        source = "shared"
        apply_mod_adaln_migration(current, state)
    full_keys = [(key, w) for key, w in state.items()
                 if (".mod_img.weight" in key or ".mod_txt.weight" in key)
                 and w.ndim == 2 and w.shape[1] == dim]
    if not full_keys:
        return {"converted": False, "source": None, "rank": rank, "energy": None, "layers": []}
    dtype = full_keys[0][1].dtype
    curve = _adaln_time_curve(state)
    if curve is not None and curve.shape[1] == dim:
        cov = curve.T @ curve
        evals, evecs = torch.linalg.eigh(cov)
        proj = evecs[:, -rank:].T.flip(0).contiguous()        # P: [r, dim] 曲线 top-r 主方向
        energy = (evals[-rank:].sum() / evals.sum().clamp_min(1e-12)).item()
    else:  # 无 time_in (理论上不该发生): 退化为堆叠 SVD
        w_all = torch.cat([w.float() for _, w in full_keys], 0)
        svd = torch.linalg.svd(w_all, full_matrices=False)
        proj = svd[2][:rank].contiguous()
        energy = (svd[1][:rank].pow(2).sum() / svd[1].pow(2).sum()).item()
    state[basis_key] = proj.to(dtype)
    layers = set()
    for key, w in full_keys:
        state[key] = (w.float() @ proj.T).to(dtype)           # A_i: [6d, r]
        layers.add(int(key.split(".")[1]))
    return {"converted": True, "source": source, "rank": rank, "energy": energy,
            "layers": sorted(layers)}


class TextRefinerBlock(nn.Module):
    """文本侧残差自注意力+FFN block (MOSS-Image Text Refiner)。

    - 只处理文本 token (text_in 投影后), 不与图像 token 交互, 无 timestep AdaLN;
    - 残差支路输出层 (attn.out_proj / ffn 末层) 零初始化 → block 初始为恒等映射,
      step0 推理与原版逐位一致, 旧 ckpt (无 text_refiner.*) 可无损热启动平滑增量学习;
    - padding 位置的输出会被主 DiT 的 attention mask / FA3 varlen 丢弃, 不影响数值。
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        # 恒等初始化: 残差支路输出层权重+bias 置零, 初始 forward = identity
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x, key_padding_mask=None):
        # x: [B, L_txt, D]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TextRefiner(nn.Module):
    def __init__(self, dim, num_heads, num_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(TextRefinerBlock(dim, num_heads) for _ in range(num_layers))

    def forward(self, text_hidden, text_mask=None):
        # text_mask: bool [B, L], True=有效 token; MHA key_padding_mask 语义相反 (True=padding)
        key_padding_mask = ~text_mask if text_mask is not None else None
        for blk in self.blocks:
            text_hidden = blk(text_hidden, key_padding_mask)
        return text_hidden


class GigaBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, mod_adaln=False, adaln_rank=None, moe=None):
        super().__init__()
        self.heads = heads
        self.mod_adaln = bool(mod_adaln)
        self.adaln_rank = None if adaln_rank is None else int(adaln_rank)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        head_dim = dim // heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out = nn.Linear(dim, dim, bias=False)
        # moe 非 None 时用 top-2 MoE FFN 替换 dense FFN (结构见 models/moe.py);
        # aux loss 暂存 self.last_moe_aux, 由 GigaExpTransformer.forward 收集进 moe_aux_losses
        self.ff = (MoEFFN(dim, ffn_dim, **moe) if moe is not None
                   else nn.Sequential(nn.Linear(dim, ffn_dim, bias=False), nn.SiLU(), nn.Linear(ffn_dim, dim, bias=False)))
        self.last_moe_aux = None
        if self.mod_adaln:
            # dual-branch AdaLN (MiniMax H3 风格): image token 用 mod_img, text token 用 mod_txt。
            # 旧 ckpt 恢复时由 apply_mod_adaln_migration 把旧 mod.* 拷贝初始化到两分支
            # (拷贝复用而非权重共享), step0 forward 与 shared AdaLN 完全等价, 无损热启动。
            # adaln_rank 非 None 时为 basis 模式 (r=64 共享基): 调制输入是 transformer 级
            # adaln_basis 投影后的 r 维向量 z=P·cond, 每分支只存 Linear(r, 6·dim) 系数头;
            # 旧全秩 ckpt 由 apply_adaln_basis_migration 做 W_i ≈ (W_i Pᵀ)·P 低秩转换后加载。
            mod_in = dim if self.adaln_rank is None else self.adaln_rank
            self.mod_img = nn.Linear(mod_in, dim * 6)
            self.mod_txt = nn.Linear(mod_in, dim * 6)
        else:
            self.mod = nn.Linear(dim, dim * 6)

    def _attend(self, y, cos, sin, attention_mask, pack):
        """qkv -> RoPE -> attention, 返回 [b, n, dim]; shared/dual 两条路径共用。"""
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
        return a

    def forward(self, x, cond, cos, sin, attention_mask=None, pack=None, n_img=None, denoising_step=None):
        if self.mod_adaln:
            if n_img is None or not 0 < n_img < x.shape[1]:
                raise ValueError(f"mod_adaln 需要 0 < n_img < seq_len, got n_img={n_img}, seq_len={x.shape[1]}")
            return self._forward_dual(x, cond, cos, sin, attention_mask, pack, n_img, denoising_step)
        # shared AdaLN 路径 (与原实现逐行等价)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.mod(cond).chunk(6, -1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        x = x + self.out(self._attend(y, cos, sin, attention_mask, pack)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        return x + self._ff(y, attention_mask, n_img, denoising_step, gate_f[:, None]) * gate_f[:, None]

    def _ff(self, y, attention_mask, n_img=None, denoising_step=None, gate=None):
        """FFN 分派: dense 直通; MoE 需 valid mask/模态/去噪步做统计, aux 暂存 last_moe_aux。"""
        if not isinstance(self.ff, MoEFFN):
            self.last_moe_aux = None
            return self.ff(y)
        # valid mask: [B,1,1,S] bool → flatten; 模态 0=image / 1=text (与 moe_stats 对齐)
        f, aux = self.ff(y, valid_mask=attention_mask, modality=self._modality_tags(y.shape[1], n_img, gate),
                         denoising_step=denoising_step, residual_gate=gate)
        self.last_moe_aux = aux
        return f

    def _modality_tags(self, n_total, n_img, gate):
        if n_img is None:
            return None
        # [B,S] 展平序: 每个样本都是 image 前缀 + text 后缀, 按 batch 平铺
        tags = torch.cat((torch.zeros(n_img, dtype=torch.long, device=gate.device),
                          torch.ones(n_total - n_img, dtype=torch.long, device=gate.device)))
        return tags.repeat(gate.shape[0])

    def _forward_dual(self, x, cond, cos, sin, attention_mask, pack, n_img, denoising_step=None):
        """dual-branch AdaLN: image 前缀用 mod_img, text 后缀用 mod_txt, 两分支梯度独立。"""
        b = x.shape[0]
        sa_i, sca_i, ga_i, sf_i, scf_i, gf_i = self.mod_img(cond).chunk(6, -1)
        sa_t, sca_t, ga_t, sf_t, scf_t, gf_t = self.mod_txt(cond).chunk(6, -1)
        img, txt = x[:, :n_img], x[:, n_img:]
        y = torch.cat((self.norm1(img) * (1 + sca_i[:, None]) + sa_i[:, None],
                       self.norm1(txt) * (1 + sca_t[:, None]) + sa_t[:, None]), 1)
        a = self.out(self._attend(y, cos, sin, attention_mask, pack))
        x = torch.cat((img + a[:, :n_img] * ga_i[:, None],
                       txt + a[:, n_img:] * ga_t[:, None]), 1)
        img, txt = x[:, :n_img], x[:, n_img:]
        y = torch.cat((self.norm2(img) * (1 + scf_i[:, None]) + sf_i[:, None],
                       self.norm2(txt) * (1 + scf_t[:, None]) + sf_t[:, None]), 1)
        # per-token AdaLN gate 向量 (image 段 gf_i, text 段 gf_t), 供 MoE 统计/gating
        gate = torch.cat((gf_i[:, None, :].expand(b, n_img, -1),
                          gf_t[:, None, :].expand(b, x.shape[1] - n_img, -1)), 1) \
            if isinstance(self.ff, MoEFFN) else None
        f = self._ff(y, attention_mask, n_img, denoising_step, gate)
        return torch.cat((img + f[:, :n_img] * gf_i[:, None],
                          txt + f[:, n_img:] * gf_t[:, None]), 1)


class GigaExpTransformer(nn.Module):
    """~0.8B dense DiT: 1792 width, 16 blocks, 14 heads, temporal RoPE retained."""
    def __init__(self, in_channels=48, out_channels=48, text_dim=4096, dim=1792,
                 ffn_dim=4864, heads=14, layers=24, rope_theta=10000.0,
                 rope_axes_dims=(32, 32, 32, 32), patch_size=(1, 1, 1), mod_adaln=False,
                 text_trans=False, adaln_type=None, adaln_rank=64, moe=None):
        super().__init__()
        self.patch_size = tuple(patch_size)
        if len(self.patch_size) != 3 or self.patch_size[0] != 1 or any(p < 1 for p in self.patch_size):
            raise ValueError("patch_size 必须是时间维为 1 的三个正整数")
        self.in_channels, self.out_channels, self.dim, self.heads = in_channels, out_channels, dim, heads
        self.text_dim, self.ffn_dim, self.layers = text_dim, ffn_dim, layers
        # AdaLN 类型三选一 (yaml model.adaln_type 显式注明; 缺省按旧 mod_adaln bool 推导):
        #   shared : 每层一个 Linear(dim, 6·dim), img/text 共用 (原版)
        #   dual   : 每层 mod_img/mod_txt 两个全秩 Linear (MiniMax H3 风格)
        #   basis  : 全模型共享 Linear(dim, r) 投影 + 每层双分支 Linear(r, 6·dim) 系数头。
        #            调制只依赖 timestep, 全部 56 条调制曲线在 t 流形上秩 ~32-64,
        #            r=64 时事后低秩转换误差 ~3.2% RMS, adaLN 参数 793M → ~34M。
        #            旧 shared/dual 全秩 ckpt 由 apply_adaln_basis_migration 转换后加载。
        if adaln_type is None:
            adaln_type = "dual" if mod_adaln else "shared"
        if adaln_type not in ("shared", "dual", "basis"):
            raise ValueError(f"adaln_type 必须是 shared/dual/basis, got {adaln_type!r}")
        self.adaln_type = adaln_type
        self.adaln_rank = int(adaln_rank) if adaln_type == "basis" else None
        self.mod_adaln = adaln_type in ("dual", "basis")
        self.adaln_basis = (nn.Linear(dim, self.adaln_rank, bias=False)
                            if adaln_type == "basis" else None)
        # Text Refiner (MOSS-Image 风格): text_in 投影后的文本 token 先过恒等初始化的残差
        # 自注意力+FFN 再 concat 进主 DiT; text_trans 可为 bool 或 dict({enabled, layers, heads})。
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
        # MoE FFN (models/moe.py): None=dense; dict 直接传给 MoEFFN (num_experts/top_k/调度/aux 权重)。
        # 分层 MoE: moe.layers 指定哪些层用 MoE, 其余层保持 dense FFN (Attention 恒为 dense)。
        #   - 缺省/None      : 全部层 MoE
        #   - dict{first,last}: 跳过前 first 层与后 last 层, 中间全部 MoE (Dense→MoE→Dense 分布)
        #   - list[int]      : 显式层号 (如 [3, 5, 7])
        # 该键会随 semantic_metadata 落入 checkpoint, resume 时指纹不一致会告警。
        self.moe_cfg = dict(moe) if moe else None
        if self.moe_cfg is not None:
            layers_spec = self.moe_cfg.get("layers")
            if layers_spec is None:
                moe_layer_ids = set(range(layers))
            elif isinstance(layers_spec, dict):
                first = int(layers_spec.get("first", 0))
                last = int(layers_spec.get("last", 0))
                if first < 0 or last < 0 or first + last > layers:
                    raise ValueError(f"moe.layers first/last 非法: {layers_spec!r} (共 {layers} 层)")
                moe_layer_ids = set(range(first, layers - last))
            else:
                moe_layer_ids = {int(i) for i in layers_spec}
                bad = moe_layer_ids - set(range(layers))
                if bad:
                    raise ValueError(f"moe.layers 层号越界: {sorted(bad)} (共 {layers} 层)")
            self.moe_layer_ids = sorted(moe_layer_ids)
            if not self.moe_layer_ids:
                raise ValueError("moe.layers 为空: 没有任何层使用 MoE")
        else:
            self.moe_layer_ids = None
        self.moe_routing_step = 0
        self._moe_denoising_step = None
        self._moe_stats_collector = None
        self.moe_aux_losses = []
        block_moe_cfg = ({k: v for k, v in self.moe_cfg.items() if k != "layers"}
                         if self.moe_cfg is not None else None)
        self.blocks = nn.ModuleList([GigaBlock(dim, heads, ffn_dim, mod_adaln=self.mod_adaln,
                                               adaln_rank=self.adaln_rank,
                                               moe=block_moe_cfg if self.moe_layer_ids is not None and i in set(self.moe_layer_ids) else None)
                                     for i in range(layers)])
        # MoE 统计收集器按层命名 (纯数字, moe_visualization 按数值排序画 per-layer 曲线)
        for idx, block in enumerate(self.blocks):
            if isinstance(block.ff, MoEFFN):
                block.ff._stats_layer = str(idx)
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, out_channels * patch_volume)

    def set_moe_routing_step(self, step: int):
        """同步 RoutingSchedule 步数 (A 强制均衡 → B 渐进 → C 学习), 训练循环每 step 调用。"""
        self.moe_routing_step = int(step)
        for block in self.blocks:
            if isinstance(block.ff, MoEFFN):
                block.ff.set_routing_step(step)

    def set_moe_denoising_step(self, value):
        """validation ODE 步序号 (0..N-1), 供 collector 做 denoising-step 用量与相邻步 stability 统计。"""
        self._moe_denoising_step = value

    def set_moe_metrics_collector(self, collector):
        """挂载/卸载 MoEStatsCollector (validation 前挂载 full 档, 结束后传 None 卸载)。"""
        self._moe_stats_collector = collector
        for block in self.blocks:
            if isinstance(block.ff, MoEFFN):
                block.ff.set_stats_collector(collector)

    def cast_router_fp32(self):
        """模型 .to(bf16) 会把 gate 一并降精度; router 要求全 fp32, 加载/搬设备后调用回浮。"""
        for block in self.blocks:
            if isinstance(block.ff, MoEFFN):
                block.ff.router.float()

    @property
    def has_moe(self) -> bool:
        return any(isinstance(block.ff, MoEFFN) for block in self.blocks)

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
            "mod_adaln": self.mod_adaln,
            "adaln_type": self.adaln_type,
            "adaln_rank": self.adaln_rank,
            "text_trans": self.text_trans,
            "moe": self.moe_cfg,
            "implementation": {
                "version": "giga_semantics_v1",
                "rope": "fp32_cos_sin_and_rotation",
                "qk_norm": "per_head_rms_norm",
                "rotate_layout": "half_chunk",
            },
        }

    def attach_kd_projectors(self, teacher_dim: int, num_layers: int = 3):
        """训练期 Feature KD projector: 每层独立 Linear (dim → teacher_dim, 无 bias)。

        - 只在 KD 训练时由训练脚本挂载; forward 推理路径不引用, 推理结构/FLOPs 完全不变
        - 保存的纯权重多出 kd_projectors.* 键, 推理侧按名称匹配加载自动忽略
        - 不可零初始化 (零向量 normalize 后 cos 退化), 保持 Linear 默认随机初始化
        """
        self.kd_projectors = nn.ModuleList(
            nn.Linear(self.dim, teacher_dim, bias=False) for _ in range(num_layers)
        )

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
                # 每样本有效 token 是连续前缀 (image 全有效 + text 右侧 padding), varlen 跳过 padding 计算。
                # 不调用 any/all/max().item(): 避免 torch.compile 下的 GPU→CPU 同步和 graph break。
                lens = n_img + valid_text.sum(-1)  # [b] 每样本有效长度
                cu_seqlens = torch.zeros(b + 1, device=x.device, dtype=torch.int32)
                cu_seqlens[1:] = lens.cumsum(0).to(torch.int32)
                cols = torch.arange(n_total, device=x.device)
                valid2d = cols[None, :] < lens[:, None]  # [b, n_total] 有效前缀掩码
                pack_idx = (torch.arange(b, device=x.device)[:, None] * n_total + cols[None, :])[valid2d]
                pos_idx = cols[None, :].expand(b, -1)[valid2d]  # cos/sin 按 [n_total] 位置索引
                # max_seqlen 用常量上界(图像/视频分档), FA3 kernel 每档只编译一次;
                # 越界显式报错, 防止 kernel 静默截断算错
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
        cond = self.time_in(emb.to(self.time_in[0].weight.dtype))
        if self.adaln_basis is not None:
            # basis 模式: 全模型一次投影 z = P·cond, blocks 的双分支系数头只吃 r 维 z
            cond = self.adaln_basis(cond.to(self.adaln_basis.weight.dtype))
        cos, sin = self.rope(t, h, w, text.shape[1], x.device, x.dtype)
        # MoE: 每 forward 重置 aux 收集 (训练循环在 backward 前读取 moe_aux_losses)
        self.moe_aux_losses = []
        if self.has_moe:
            for block in self.blocks:
                block.last_moe_aux = None
        features = [] if feature_layers is not None else None
        for block_idx, block in enumerate(self.blocks):
            x = block(x, cond, cos, sin, attention_mask, pack, n_img, denoising_step=self._moe_denoising_step)
            if block.last_moe_aux is not None:
                self.moe_aux_losses.append(block.last_moe_aux)
            if features is not None and block_idx in feature_layers:
                features.append(x)  # [B, n_img+n_text, D] 全序列, KD 侧再切 image 前缀
        image = self.out(self.norm(x[:, :t * h * w]))
        image = image.reshape(b, t, h, w, pt, ph, pw, self.out_channels)
        out = image.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, self.out_channels, t * pt, h * ph, w * pw)
        if features is not None:
            return out, features  # features 顺序 = feature_layers 升序 (blocks 遍历序)
        return out
