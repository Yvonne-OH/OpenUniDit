"""Checkpoint 转换测试: 小模型 round-trip (original -> convert -> reload -> forward 一致)。

真实 280k checkpoint 的转换由 tools/convert_checkpoint_to_mhc.py 执行,
本测试用小配置验证转换逻辑本身。
运行: python tests/test_checkpoint_conversion.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.giga_transformer import GigaExpTransformer
from models.giga_transformer_mhc import GigaExpTransformerMHC
from tools.convert_checkpoint_to_mhc import convert

SMALL_MODEL = dict(in_channels=8, out_channels=8, text_dim=16, dim=64, ffn_dim=128,
                   heads=4, layers=3, rope_theta=10000.0,
                   rope_axes_dims=(2, 6, 6, 2), patch_size=(1, 2, 2))


def _make_config(tmp: Path) -> Path:
    cfg = {
        "model": {**SMALL_MODEL, "n_streams": 4, "sinkhorn_iters": 20, "mhc_mapping_fp32": True},
        "mhc": {"enabled": True, "n_streams": 4, "sinkhorn_iters": 20,
                "mapping_mode": "dynamic", "init_mode": "function_preserving", "init_eps": 1e-6},
    }
    path = tmp / "mhc_small.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_conversion_round_trip():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = GigaExpTransformer(**SMALL_MODEL).eval()
        src = tmp / "original.safetensors"
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in base.state_dict().items()}, str(src),
                  metadata={"format": "pt", "step": "123"})

        cfg_path = _make_config(tmp)
        out = tmp / "mhc_converted.safetensors"
        report = convert(src, cfg_path, out)

        # 报告检查
        assert report["original_tensor_count"] == len(base.state_dict())
        assert report["unexpected_keys"] == []
        assert report["shared_weights"] == []
        assert report["added_param_count"] > 0
        assert all("mhc" in k or k == "readout" for k in report["missing_keys"])
        assert (tmp / "conversion_report.json").is_file()

        # 重新加载 mHC checkpoint (strict) 后 forward 必须与原模型一致
        mhc = GigaExpTransformerMHC(**{**SMALL_MODEL, "n_streams": 4}).eval()
        from safetensors.torch import load_file
        mhc.load_state_dict(load_file(str(out)))  # strict=True
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 8, 1, 16, 16, generator=g)
        ts = torch.rand(2, generator=g) * 1000
        text = torch.randn(2, 7, 16, generator=g)
        mask = torch.ones(2, 7, dtype=torch.bool); mask[:, -2:] = False
        with torch.no_grad():
            v_base = base(x, ts, text, mask)
            v_mhc = mhc(x, ts, text, mask)
        rel = ((v_mhc - v_base).norm() / (v_base.norm() + 1e-12)).item()
        print(f"round-trip forward rel_l2={rel:.3e}")
        assert rel < 1e-4


def test_conversion_rejects_mismatched_checkpoint():
    """缺参数的 checkpoint 必须报错, 不允许静默吞掉。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = GigaExpTransformer(**SMALL_MODEL).eval()
        state = {k: v for k, v in base.state_dict().items() if not k.startswith("blocks.1.")}
        src = tmp / "broken.safetensors"
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in state.items()}, str(src))
        try:
            convert(src, _make_config(tmp), tmp / "out.safetensors")
        except RuntimeError as e:
            assert "missing" in str(e).lower() or "不匹配" in str(e)
            print("PASS 拒绝不完整 checkpoint:", str(e)[:80])
            return
        raise AssertionError("不完整 checkpoint 未被拒绝")


def test_conversion_rejects_disabled_mhc():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = GigaExpTransformer(**SMALL_MODEL).eval()
        src = tmp / "original.safetensors"
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in base.state_dict().items()}, str(src))
        cfg = {"model": {**SMALL_MODEL}, "mhc": {"enabled": False}}
        cfg_path = tmp / "disabled.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        try:
            convert(src, cfg_path, tmp / "out.safetensors")
        except ValueError as e:
            print("PASS 拒绝未开启 mhc 的配置")
            return
        raise AssertionError("enabled=false 未被拒绝")


if __name__ == "__main__":
    for fn in [test_conversion_round_trip, test_conversion_rejects_mismatched_checkpoint,
               test_conversion_rejects_disabled_mhc]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all checkpoint conversion tests passed")
