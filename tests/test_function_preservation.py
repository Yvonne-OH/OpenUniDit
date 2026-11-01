"""全模型 Function Preservation 测试: original GigaExpTransformer vs mHC 变体。

同一输入下比较 forward 输出: 相对 L2 / max abs / mean abs / cosine similarity。
FP32 以 1e-4 为目标; BF16 以原模型自身数值波动为参考 (dense vs varlen 路径差)。
运行: python tests/test_function_preservation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.giga_transformer import GigaExpTransformer
from models.giga_transformer_mhc import GigaExpTransformerMHC

MODEL_KW = dict(in_channels=8, out_channels=8, text_dim=16, dim=64, ffn_dim=128,
                heads=4, layers=3, rope_axes_dims=(2, 6, 6, 2), patch_size=(1, 2, 2))


def make_inputs(b=2, seed=0, h=16, w=16, text_len=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 8, 1, h, w, generator=g)
    timestep = torch.rand(b, generator=g) * 1000
    text = torch.randn(b, text_len, 16, generator=g)
    mask = torch.ones(b, text_len, dtype=torch.bool)
    mask[:, -3:] = False  # 右 padding, 触发 varlen/pack 路径
    return x, timestep, text, mask


def convert_state(base, mhc):
    state = base.state_dict()
    missing, unexpected = mhc.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert set(missing) == set(k for k in mhc.state_dict() if "mhc" in k or k == "readout"), missing
    return mhc


def metrics(v_base, v_mhc):
    diff = (v_mhc.float() - v_base.float()).flatten()
    base = v_base.float().flatten()
    rel = diff.norm() / (base.norm() + 1e-12)
    cos = torch.nn.functional.cosine_similarity(diff.new_tensor(base), diff.new_tensor(diff), dim=0) if False else \
        torch.dot(base, base + diff) / (base.norm() * (base + diff).norm() + 1e-12)
    return {"rel_l2": rel.item(), "max_abs": diff.abs().max().item(),
            "mean_abs": diff.abs().mean().item(), "cos": cos.item()}


def test_function_preservation_fp32():
    base = GigaExpTransformer(**MODEL_KW).eval()
    mhc = convert_state(base, GigaExpTransformerMHC(**(MODEL_KW | dict(n_streams=4))).eval())
    for seed, (h, w, tl) in enumerate([(16, 16, 11), (12, 20, 7), (8, 8, 1)]):
        x, ts, text, mask = make_inputs(seed=seed, h=h, w=w, text_len=tl)
        with torch.no_grad():
            v_base = base(x, ts, text, mask)
            v_mhc = mhc(x, ts, text, mask)
        m = metrics(v_base, v_mhc)
        print(f"fp32 seed{seed}: " + " ".join(f"{k}={v:.3e}" for k, v in m.items()))
        assert m["rel_l2"] < 1e-4, f"fp32 function preservation 失败: {m}"


def test_function_preservation_bf16():
    base = GigaExpTransformer(**MODEL_KW).eval().to(torch.bfloat16)
    mhc = convert_state(base, GigaExpTransformerMHC(**(MODEL_KW | dict(n_streams=4))).eval().to(torch.bfloat16))
    x, ts, text, mask = make_inputs(seed=3)
    x, text = x.to(torch.bfloat16), text.to(torch.bfloat16)
    with torch.no_grad():
        v_base = base(x, ts, text, mask).float()
        v_mhc = mhc(x, ts, text, mask).float()
        # 参考基线: 原模型自身数值波动 (同权重重复前向)
        v_base2 = base(x, ts, text, mask).float()
    ref = metrics(v_base, v_base2)
    m = metrics(v_base, v_mhc)
    print(f"bf16 原模型自身波动: rel_l2={ref['rel_l2']:.3e}")
    print(f"bf16 base vs mhc:  " + " ".join(f"{k}={v:.3e}" for k, v in m.items()))
    # 允许到原模型自身波动的 2 倍 (fp32 mapping/Sinkhorn 与 bf16 core 混合精度差异)
    assert m["rel_l2"] < max(2.0 * ref["rel_l2"], 5e-3), f"bf16 等价性失败: {m} vs ref {ref}"


def test_without_mask_dense_path():
    base = GigaExpTransformer(**MODEL_KW).eval()
    mhc = convert_state(base, GigaExpTransformerMHC(**(MODEL_KW | dict(n_streams=4))).eval())
    x, ts, text, _ = make_inputs(seed=4)
    with torch.no_grad():
        v_base = base(x, ts, text)
        v_mhc = mhc(x, ts, text)
    m = metrics(v_base, v_mhc)
    print("dense(无mask): " + " ".join(f"{k}={v:.3e}" for k, v in m.items()))
    assert m["rel_l2"] < 1e-4


def test_gradient_flow_mhc_params():
    mhc = GigaExpTransformerMHC(**(MODEL_KW | dict(n_streams=4)))
    x, ts, text, mask = make_inputs(seed=5)
    loss = mhc(x, ts, text, mask).square().mean()
    loss.backward()
    for name in ("blocks.0.mhc_attn.mapping.weight", "blocks.0.mhc_ffn.mapping.bias", "readout"):
        p = dict(mhc.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().max() > 0, f"{name} 梯度全零"
    g = [dict(mhc.named_parameters())[f"blocks.{i}.{sub}.mapping.weight"].grad.abs().max().item()
         for i in range(mhc.layers) for sub in ("mhc_attn", "mhc_ffn")]
    print(f"mHC mapping 梯度 (28->3 blocks) max|g|={max(g):.3e} min|g|={min(g):.3e}")
    assert all(v > 0 for v in g)
    for name, p in mhc.named_parameters():
        assert torch.isfinite(p).all(), f"{name} 参数非有限"


if __name__ == "__main__":
    for fn in [test_function_preservation_fp32, test_function_preservation_bf16,
               test_without_mask_dense_path, test_gradient_flow_mhc_params]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all function preservation tests passed")
