"""2-Stream/4-Stream mHC (Manifold-Constrained Hyper-Connections) residual 模块。

数学定义 (per token, H 形状 [B, N, S, C], S = n_streams):

    H_pre  = sigma(logits_pre)                    # [B, N, S]
    H_post = 2 * sigma(logits_post)               # [B, N, S]
    H_res  = Sinkhorn(logits_res)                 # [B, N, S, S] 双随机
    H_out  = H_res @ H_in + (H_post)^T F(H_pre @ H_in)

其中 F 为原始 block 子层 (attention 或 FFN, 含原 AdaLN gate), 接收单流
[B, N, C] (pre 映射把 S 个 stream 加权混合为 1 个 stream)。

动态映射由 RMSNorm(vec(H)) 经独立线性投影产生; 迁移场景使用
function-preserving 初始化 (零投影权重 + 极端 logits bias), 与论文原始
初始化方式是两种独立 init_mode, 不混用。

实现约定:
- 映射 / Sinkhorn 全程 fp32 (可配置), 与 hidden dtype 的转换只发生在
  与 H 相乘的 einsum 处;
- Sinkhorn 在 log 域迭代 (logsumexp 归一化), 数值稳定且可微;
- 本模块不知道 attention/FFN 内部细节, 通过 block_fn 回调解耦。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def sinkhorn_log_domain(logits: torch.Tensor, n_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """log 域 Sinkhorn-Knopp: 交替行/列 logsumexp 归一化, 返回近似双随机矩阵。

    logits: [.., S, S] (任意批维度), 返回 exp 后行和/列和接近 1。
    最后一次是列归一化, 因此列和精确为 1, 行和随迭代收敛 (测试目标 < 1e-5)。
    """
    m = logits.float()
    for _ in range(n_iters):
        m = m - torch.logsumexp(m, dim=-1, keepdim=True)  # 行归一化
        m = m - torch.logsumexp(m, dim=-2, keepdim=True)  # 列归一化
    return torch.exp(m) + eps * 0  # eps 预留: 保持签名稳定, 当前不加平移


class MHCResidual(nn.Module):
    """mHC 残差映射: 替换原始 block 的一处 residual add。

    forward(h, block_fn):
        h: [B, N, S, C]
        block_fn: callable, [B, N, C] -> [B, N, C] (原始子层, 含 AdaLN gate)
        返回 [B, N, S, C]

    新增参数: mapping 投影 Linear(S*C -> 2S + S*S) + bias。
    """

    def __init__(self, dim: int, n_streams: int = 4, sinkhorn_iters: int = 20,
                 eps: float = 1e-6, mapping_fp32: bool = True):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.mapping_fp32 = mapping_fp32
        out_dim = 2 * n_streams + n_streams * n_streams  # pre + post + res logits
        self.mapping = nn.Linear(n_streams * dim, out_dim)
        # 训练/监控用: forward 时缓存最近的映射统计 (不入 state_dict)
        self.last_stats: dict | None = None
        self._init_function_preserving()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _init_function_preserving(self):
        """迁移初始化: 零投影权重 + 极端 logits bias。

        目标: H_pre≈[1,0..0], H_post≈[1,0..0], H_res≈I, 使得对 H=[x,0..0]:
            H_out = [x + F(x), 0, ..., 0]
        - pre:  sigma(b)=[1,0..0]     -> b_pre = [+LP, -LP, ...]
        - post: 2*sigma(b)=[1,0..0]   -> b_post = [0, -LP, ...]  (2*sigma(0)=1)
        - res:  Sinkhorn(exp(LP*I)) 第一轮行归一化即精确 I
        LP=12 时 sigma(-12)≈6.1e-6, 2*sigma(-12)≈1.2e-5, fp32 下 function-preserving
        相对误差 ~1e-5 (LP=8 时 ~1.7e-4, 实测超 1e-4 目标); 代价是 pre/post 对应
        bias 坐标的 sigmoid 梯度饱和至 ~1e-5 量级 (res 路径经 log 域 Sinkhorn
        归一化, 梯度不受幅度影响), Test D 专项监测可学习性。
        """
        s = self.n_streams
        lp = 12.0
        nn.init.zeros_(self.mapping.weight)
        bias = torch.zeros(s * s + 2 * s)
        bias[:s] = torch.tensor([lp] + [-lp] * (s - 1))              # pre
        bias[s:2 * s] = torch.tensor([0.0] + [-lp] * (s - 1))        # post
        res = torch.full((s, s), -lp)
        res.fill_diagonal_(lp)
        bias[2 * s:] = res.flatten()
        self.mapping.bias.copy_(bias)

    @torch.no_grad()
    def init_paper_mode(self, scale: float = 0.02):
        """论文式动态映射初始化 (小随机投影, 非迁移等价), 独立实验模式用。

        与 function-preserving 互斥; 使用时必须放弃 forward 等价性验收。
        """
        nn.init.normal_(self.mapping.weight, std=scale)
        nn.init.zeros_(self.mapping.bias)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _mappings(self, h: torch.Tensor):
        """H [B,N,S,C] -> pre [B,N,S], post [B,N,S], res [B,N,S,S] (fp32)."""
        b, n, s, c = h.shape
        vec = h.reshape(b, n, s * c)                                  # vec(H)
        work_dtype = torch.float32 if self.mapping_fp32 else vec.dtype
        vec = F.rms_norm(vec.float(), (vec.shape[-1],)).to(work_dtype)
        # mapping_fp32 时显式把权重升到 fp32: 模型整体 bf16 下映射计算仍是真 fp32
        if self.mapping_fp32 and self.mapping.weight.dtype != torch.float32:
            w_map, b_map = self.mapping.weight.float(), self.mapping.bias.float()
        else:
            w_map, b_map = self.mapping.weight, self.mapping.bias
        logits = F.linear(vec, w_map, b_map).float()                  # [B,N,2S+S*S]
        pre = torch.sigmoid(logits[..., :s])
        post = 2.0 * torch.sigmoid(logits[..., s:2 * s])
        res = sinkhorn_log_domain(logits[..., 2 * s:].reshape(b, n, s, s),
                                  n_iters=self.sinkhorn_iters)
        return pre, post, res

    def forward(self, h: torch.Tensor, block_fn) -> torch.Tensor:
        if h.ndim != 4:
            raise ValueError(f"mHC 输入必须是 [B,N,S,C], 实际 {tuple(h.shape)}")
        b, n, s, c = h.shape
        if s != self.n_streams:
            raise ValueError(f"stream 维 {s} != n_streams {self.n_streams}")

        pre, post, res = self._mappings(h)                            # fp32
        # 与 hidden 相乘处才 cast, 保证 F 输入 dtype 与原模型一致
        compute_dtype = h.dtype
        pre_mixed = torch.einsum("bns,bnsc->bnc", pre.to(compute_dtype), h)
        f_out = block_fn(pre_mixed)                                   # [B,N,C]
        out = torch.einsum("bnij,bnjc->bnic", res.to(compute_dtype), h) \
            + torch.einsum("bns,bnc->bnsc", post.to(compute_dtype), f_out)

        if self.last_stats is None:
            self.last_stats = {}
        with torch.no_grad():
            self.last_stats = {
                "pre_mean": pre.mean().detach(),
                "post_mean": post.mean().detach(),
                "res_diag_mean": res.diagonal(dim1=-2, dim2=-1).mean().detach(),
                "res_offdiag_mean": (res.sum(-1) - res.diagonal(dim1=-2, dim2=-1)).mean().detach(),
                "sinkhorn_row_error": (res.sum(-1) - 1.0).abs().amax().detach(),
                "sinkhorn_col_error": (res.sum(-2) - 1.0).abs().amax().detach(),
            }
        return out


def stack_stream(x: torch.Tensor, n_streams: int) -> torch.Tensor:
    """[B,N,C] -> [B,N,S,C], 初始 H = [x, 0, ..., 0] (function-preserving 约定)。"""
    if n_streams < 1:
        raise ValueError("n_streams 必须 >= 1")
    if n_streams == 1:
        return x.unsqueeze(2)
    zeros = torch.zeros(*x.shape[:-1], n_streams - 1, x.shape[-1],
                        device=x.device, dtype=x.dtype)
    return torch.cat((x.unsqueeze(2), zeros), dim=2)


def read_out(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[B,N,S,C] x weight[S] -> [B,N,C]。weight 初始 [1,0..0] 时等价 stream0 直出。"""
    return torch.einsum("bnsc,s->bnc", h, weight.to(h.dtype))
