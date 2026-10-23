"""varlen FA3 与 masked-SDPA 数值一致性验证 + torch.compile (dynamic) 冒烟测试.

运行: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) \
      CUDA_VISIBLE_DEVICES=0 python scripts/test/test_varlen_fa3.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "thirdparty")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import models.giga_transformer as G

torch.manual_seed(0)
dev = "cuda:0"


def build(layers=2):
    torch.manual_seed(0)
    m = G.GigaExpTransformer(in_channels=48, out_channels=48, text_dim=64, dim=256, ffn_dim=512,
                             heads=4, layers=layers, rope_theta=10000.0,
                             rope_axes_dims=(16, 16, 16, 16), patch_size=(1, 2, 2))
    return m.to(dev, dtype=torch.bfloat16)


x = torch.randn(2, 48, 1, 24, 16, device=dev, dtype=torch.bfloat16)  # latent 24x16 -> 96 image tokens
ts = torch.tensor([500.0, 250.0], device=dev)
text = torch.randn(2, 20, 64, device=dev, dtype=torch.bfloat16)
mask = torch.ones(2, 20, dtype=torch.bool, device=dev)
mask[1, 7:] = False  # sample1 右 padding


def run(m, fa3):
    G._FA3 = fa3
    m.zero_grad(set_to_none=True)
    out = m(x, ts, text, mask)
    loss = out.square().mean()
    loss.backward()
    return out.detach(), loss.detach()


m = build()
out_var, loss_var = run(m, True)
out_sdpa, loss_sdpa = run(m, False)
diff = (out_var - out_sdpa).abs().max().item()
rel = diff / out_sdpa.abs().max().item()
print(f"[mask]  varlen vs SDPA: max_abs_diff={diff:.3e} rel={rel:.3e} "
      f"loss={loss_var.item():.6f} vs {loss_sdpa.item():.6f}")
grads = [p.grad for p in m.parameters() if p.grad is not None]
print(f"[mask]  grads: n={len(grads)} finite={all(torch.isfinite(g).all().item() for g in grads)}")
assert rel < 3e-2, "varlen 与 SDPA image 输出偏差过大"
assert len(grads) > 0 and all(torch.isfinite(g).all().item() for g in grads)

mask_full = torch.ones_like(mask)
G._FA3 = True
a = m(x, ts, text, mask_full)
G._FA3 = False
b = m(x, ts, text, mask_full)
print(f"[full]  dense FA3 vs SDPA: max_abs_diff={(a - b).abs().max().item():.3e}")

G._FA3 = True
a = m(x, ts, text, None)
G._FA3 = False
b = m(x, ts, text, None)
print(f"[none]  dense FA3 vs SDPA: max_abs_diff={(a - b).abs().max().item():.3e}")

G._FA3 = True
mc = build()
mc.compile(dynamic=True)
for L2, v0, v1 in ((20, 15, 9), (37, 37, 9), (128, 64, 100)):  # 变长 text, 动态 shape
    t2 = torch.randn(2, L2, 64, device=dev, dtype=torch.bfloat16)
    mk = torch.zeros(2, L2, dtype=torch.bool, device=dev)
    mk[0, :v0] = True
    mk[1, :v1] = True
    o = mc(x, ts, t2, mk)
    o.square().mean().backward()
    print(f"[compile] L={L2} valid=({v0},{v1}) ok, out absmax={o.abs().max().item():.4f}")
print("ALL OK")
