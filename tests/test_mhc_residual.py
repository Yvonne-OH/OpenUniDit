"""mHC 残差模块单元测试: shape / Sinkhorn / function-preserving / 梯度。

运行: python -m pytest tests/test_mhc_residual.py -v  (或 python tests/test_mhc_residual.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mhc_residual import MHCResidual, read_out, sinkhorn_log_domain, stack_stream


def _block_fn(c):
    return torch.nn.Linear(c, c, bias=False)


def test_stack_and_read_out():
    x = torch.randn(2, 5, 8)
    h = stack_stream(x, 4)
    assert h.shape == (2, 5, 4, 8)
    assert torch.equal(h[:, :, 0], x)
    assert torch.all(h[:, :, 1:] == 0)
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.equal(read_out(h, w), x)


def test_shape_variants():
    for b, n, c in [(1, 7, 16), (3, 1, 32), (2, 13, 8)]:
        m = MHCResidual(c, n_streams=4)
        h = stack_stream(torch.randn(b, n, c), 4)
        out = m(h, _block_fn(c))
        assert out.shape == h.shape


def test_sinkhorn_double_stochastic():
    """分级验收 (实测收敛率, 不伪造):
    - scale=1 (mHC 训练早期 logits 的实际量级): 100 轮行/列和 < 1e-5;
    - 20 轮 (论文默认): 报告实际误差, 行和放宽到 1e-3;
    - 近恒等 logits (function-preserving 初始化的实际输入): 2 轮即精确。
    随机大幅值 logits (scale>=3) 收敛慢, 见 docs/mhc/test_report.md。
    """
    torch.manual_seed(0)
    logits = torch.randn(8, 4, 4, 4)
    res = sinkhorn_log_domain(logits, n_iters=20)
    assert torch.isfinite(res).all() and (res >= 0).all()
    row20 = (res.sum(-1) - 1).abs().max().item()
    col20 = (res.sum(-2) - 1).abs().max().item()
    print(f"sinkhorn fp32 n=20: row_err={row20:.2e} col_err={col20:.2e}")
    assert row20 < 1e-3 and col20 < 1e-5, f"20 轮误差超标: row={row20} col={col20}"
    res100 = sinkhorn_log_domain(logits, n_iters=100)
    row100 = (res100.sum(-1) - 1).abs().max().item()
    col100 = (res100.sum(-2) - 1).abs().max().item()
    print(f"sinkhorn fp32 n=100: row_err={row100:.2e} col_err={col100:.2e}")
    assert row100 < 1e-5 and col100 < 1e-5


def test_sinkhorn_identitiy_init_exact():
    s = 4
    lp = 8.0
    logits = torch.full((1, 1, s, s), -lp)
    logits[..., range(s), range(s)] = lp
    res = sinkhorn_log_domain(logits, n_iters=2)
    assert torch.allclose(res, torch.eye(s), atol=1e-5)


def test_function_preserving_exact_structure():
    """H=[x,0..0] + fp32 下, 单块输出应近似 [x+F(x),0..0]。"""
    torch.manual_seed(1)
    b, n, c = 2, 6, 32
    x = torch.randn(b, n, c)
    m = MHCResidual(c, n_streams=4)
    f = _block_fn(c)
    h = stack_stream(x, 4)
    out = m(h, f)
    expect0 = x + f(x)
    rel = (out[:, :, 0] - expect0).norm() / (expect0.norm() + 1e-12)
    print(f"stream0 相对误差={rel.item():.3e}")
    assert rel.item() < 1e-4, f"function-preserving 失败: rel={rel.item()}"
    # 其余 stream 应接近 0 (2*sigma(-8)≈6.7e-4 * F 量级)
    tail = out[:, :, 1:].abs().max().item()
    assert tail < 1e-2 * (expect0.abs().max().item() + 1), f"尾部 stream 未近零: {tail}"


def test_gradients_finite_and_nonzero():
    torch.manual_seed(2)
    c = 32
    m = MHCResidual(c, n_streams=4)
    f = _block_fn(c)
    h = stack_stream(torch.randn(2, 5, c), 4).requires_grad_(True)
    out = m(h, f)
    out.sum().backward()
    assert torch.isfinite(h.grad).all()
    g = m.mapping.weight.grad
    assert g is not None and torch.isfinite(g).all()
    # 零权重 + 极端 bias 下 res 分支经 Sinkhorn 仍有梯度; pre/post 分支因 sigmoid
    # 饱和梯度很小但 res 路径 (log-domain) 非零
    assert g.abs().max() > 0, "mapping 梯度全零"
    print(f"mapping.weight.grad max|g|={g.abs().max().item():.3e}")
    assert torch.isfinite(m.mapping.bias.grad).all()


def test_sinkhorn_backward():
    logits = torch.randn(2, 3, 3, 3, requires_grad=True)
    out = sinkhorn_log_domain(logits, n_iters=5).sum()
    out.backward()
    assert torch.isfinite(logits.grad).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all mhc_residual tests passed")
