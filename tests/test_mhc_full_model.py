"""mHC 全模型测试: shape / 配置覆盖 (batch, token, timestep, mask, stream 数)。

运行: python tests/test_mhc_full_model.py
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
                heads=4, layers=2, rope_axes_dims=(2, 6, 6, 2), patch_size=(1, 2, 2))


def _build(n_streams=4, **overrides):
    kw = {**MODEL_KW, **overrides}
    base = GigaExpTransformer(**kw).eval()
    mhc = GigaExpTransformerMHC(**kw, n_streams=n_streams).eval()
    missing, unexpected = mhc.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == set(k for k in mhc.state_dict() if "mhc" in k or k == "readout")
    return base, mhc


def test_shapes_and_batches():
    """覆盖不同 batch / 分辨率 / 文本长度 / timestep / mask。"""
    cases = [
        dict(b=1, h=16, w=16, tl=11), dict(b=3, h=16, w=16, tl=7),
        dict(b=2, h=12, w=20, tl=1), dict(b=2, h=8, w=24, tl=33),
        dict(b=4, h=16, w=16, tl=5),
    ]
    base, mhc = _build()
    for i, c in enumerate(cases):
        g = torch.Generator().manual_seed(i)
        x = torch.randn(c["b"], 8, 1, c["h"], c["w"], generator=g)
        ts = (torch.rand(c["b"], generator=g) * 1000)
        text = torch.randn(c["b"], c["tl"], 16, generator=g)
        mask = torch.rand(c["b"], c["tl"], generator=g) > 0.3
        out = mhc(x, ts, text, mask)
        assert out.shape == (c["b"], 8, 1, c["h"], c["w"]), (c, out.shape)
        with torch.no_grad():
            v_base = base(x, ts, text, mask)
        rel = ((out - v_base).norm() / (v_base.norm() + 1e-12)).item()
        print(f"case{i} {c}: rel_l2={rel:.2e}")
        assert rel < 1e-4
    # timestep 边界
    x = torch.randn(2, 8, 1, 16, 16)
    text = torch.randn(2, 5, 16)
    for ts in [torch.zeros(2), torch.full((2,), 1000.0)]:
        out = mhc(x, ts, text, torch.ones(2, 5, dtype=torch.bool))
        assert out.shape == (2, 8, 1, 16, 16) and torch.isfinite(out).all()
    print("PASS shapes/batches/timesteps")


def test_video_latents_t2():
    """T=2 (视频档) latent 前向, RoPE 时间轴与 MAX_SEQLEN 分档逻辑。"""
    base, mhc = _build()
    x = torch.randn(1, 8, 2, 16, 16)
    ts = torch.tensor([500.0])
    text = torch.randn(1, 9, 16)
    mask = torch.ones(1, 9, dtype=torch.bool)
    v_base, v_mhc = base(x, ts, text, mask), mhc(x, ts, text, mask)
    rel = ((v_mhc - v_base).norm() / (v_base.norm() + 1e-12)).item()
    print(f"T=2 rel_l2={rel:.2e}")
    assert rel < 1e-4


def test_stream_counts():
    for s in (1, 2, 4):
        base, mhc = _build(n_streams=s)
        x = torch.randn(1, 8, 1, 16, 16)
        ts = torch.tensor([250.0])
        text = torch.randn(1, 6, 16)
        v_base = base(x, ts, text)
        v_mhc = mhc(x, ts, text)
        rel = ((v_mhc - v_base).norm() / (v_base.norm() + 1e-12)).item()
        print(f"n_streams={s} rel_l2={rel:.2e}")
        assert rel < 1e-4
    assert GigaExpTransformerMHC(**MODEL_KW, n_streams=2).n_streams == 2


def test_stats_and_param_count():
    mhc = GigaExpTransformerMHC(**MODEL_KW, n_streams=4)
    x = torch.randn(1, 8, 1, 16, 16)
    mhc(x, torch.tensor([100.0]), torch.randn(1, 6, 16))
    stats = mhc.mhc_stats()
    assert len(stats) == mhc.layers * 2 * 6  # 每块 2 模块 × 6 指标
    assert all(torch.isfinite(v) for v in stats.values())
    base = GigaExpTransformer(**MODEL_KW)
    added = sum(p.numel() for p in mhc.parameters()) - sum(p.numel() for p in base.parameters())
    print(f"added params={added} ({added / sum(p.numel() for p in base.parameters()) * 100:.3f}%)")
    assert added > 0


if __name__ == "__main__":
    for fn in [test_shapes_and_batches, test_video_latents_t2, test_stream_counts,
               test_stats_and_param_count]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all full model tests passed")
