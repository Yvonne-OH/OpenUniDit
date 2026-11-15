# -*- coding: utf-8 -*-
"""Wan2.2 TI2V-5B (diffusers) transformer 权重 → Wan5BSingleStream 单流模型初始化权重。

映射关系 (源 key 见 configs/stage_1_wan5b_tar.yaml 头注释):
- patch_embedding  Conv3d(48→3072,(1,2,2)) kernel 空间求和 → patch_in (patch 1,1,1, DC 保持);
- blocks.i.attn1.to_q/k/v  cat → blocks.i.qkv (单流自注意力, bias 保留);
- blocks.i.attn1.norm_q/norm_k (across-heads RMSNorm) → q_norm/k_norm;
- blocks.i.attn2.*  丢弃 (单流无 cross-attn);
- blocks.i.norm2 (affine LayerNorm) / ffn.net.0.proj / ffn.net.2  直载;
- blocks.i.scale_shift_table [1,6,dim] → [6·dim];
- condition_embedder.time_embedder → time_in; time_proj → mod_proj (全局 cond6);
- scale_shift_table [1,2,dim] → head_table (Wan head 调制);
- proj_out [1*2*2*48, dim] 空间求和 → out [48, dim] (patch 1,1,1);
- condition_embedder.text_embedder 丢弃 (单流文本走 text_in, 随机初始化);
- text_in / text_refiner 不在映射内, 由训练脚本 load_compatible_state_dict 保留随机初始化。

用法:
  python tools/convert_wan22_5b.py \
      --ckpt /mnt/nfs/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/transformer \
      --out /home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/stage_1_wan5b/wan22_5b_init.safetensors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file, safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_mapping(n_layers: int):
    """返回 [(src_key_or_none, dst_key, transform_fn), ...]; transform_fn(tensor)->tensor。"""
    mapping = []
    I = lambda x: x
    # 全局
    mapping += [
        ("patch_embedding.weight", "patch_in.weight",
         lambda t: t.reshape(t.shape[0], t.shape[1], -1, t.shape[-2], t.shape[-1]).sum((2, 3, 4))),
        ("patch_embedding.bias", "patch_in.bias", I),
        ("condition_embedder.time_embedder.linear_1.weight", "time_in.0.weight", I),
        ("condition_embedder.time_embedder.linear_1.bias", "time_in.0.bias", I),
        ("condition_embedder.time_embedder.linear_2.weight", "time_in.2.weight", I),
        ("condition_embedder.time_embedder.linear_2.bias", "time_in.2.bias", I),
        ("condition_embedder.time_proj.weight", "mod_proj.weight", I),
        ("condition_embedder.time_proj.bias", "mod_proj.bias", I),
        ("scale_shift_table", "head_table", lambda t: t.squeeze(0)),
        ("proj_out.weight", "out.weight",
         # 192 = pt*ph*pw*C (C 最内层, 与 Wan unpatchify 的 view(...,pt,ph,pw,-1) 一致)
         lambda t: t.reshape(1, 2, 2, t.shape[0] // 4, t.shape[1]).sum((0, 1, 2))),
        ("proj_out.bias", "out.bias", lambda t: t.reshape(1, 2, 2, t.shape[0] // 4).sum((0, 1, 2))),
    ]
    # 每块
    for i in range(n_layers):
        s = f"blocks.{i}."
        d = f"blocks.{i}."
        mapping += [
            (s + "attn1.to_q.weight", d + "qkv.weight", None),  # cat 在下方统一处理
            (s + "attn1.norm_q.weight", d + "q_norm.weight", I),
            (s + "attn1.norm_k.weight", d + "k_norm.weight", I),
            (s + "attn1.to_out.0.weight", d + "out.weight", I),
            (s + "attn1.to_out.0.bias", d + "out.bias", I),
            (s + "norm2.weight", d + "norm2.weight", I),
            (s + "norm2.bias", d + "norm2.bias", I),
            (s + "ffn.net.0.proj.weight", d + "ff.0.weight", I),
            (s + "ffn.net.0.proj.bias", d + "ff.0.bias", I),
            (s + "ffn.net.2.weight", d + "ff.2.weight", I),
            (s + "ffn.net.2.bias", d + "ff.2.bias", I),
            (s + "scale_shift_table", d + "scale_shift_table", lambda t: t.reshape(-1)),
        ]
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="/mnt/nfs/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/transformer")
    parser.add_argument("--out", default="/home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/stage_1_wan5b/wan22_5b_init.safetensors")
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    from models.wan5b_single_stream import Wan5BSingleStream

    # meta 模型: 只取期望 key/shape 做校验, 不占内存
    with torch.device("meta"):
        model = Wan5BSingleStream(dim=3072, ffn_dim=14336, heads=24, layers=args.layers,
                                  text_dim=2560, patch_size=(1, 1, 1),
                                  rope_axes_dims=(16, 48, 48, 16),
                                  text_trans={"layers": 2, "heads": 24})
    expected = dict(model.state_dict())

    ckpt = Path(args.ckpt)
    index = json.load(open(ckpt / "diffusion_pytorch_model.safetensors.index.json"))
    weight_map = index["weight_map"]
    shards = {}
    def get(key: str) -> torch.Tensor:
        f = weight_map[key]
        if f not in shards:
            shards[f] = safe_open(ckpt / f, framework="pt", device="cpu")
        return shards[f].get_tensor(key)

    out_dict = {}
    n_params = 0
    for src, dst, fn in build_mapping(args.layers):
        if fn is None:  # qkv: q/k/v cat
            layer = dst.split(".")[1]
            pairs = (("weight", torch.cat([get(f"blocks.{layer}.attn1.to_{n}.weight") for n in "qkv"], 0)),
                     ("bias", torch.cat([get(f"blocks.{layer}.attn1.to_{n}.bias") for n in "qkv"], 0)))
            for name, tensor in pairs:
                key = f"blocks.{layer}.qkv.{name}"
                tensor = tensor.contiguous()
                assert key in expected and tuple(expected[key].shape) == tuple(tensor.shape), \
                    f"{key}: 期望 {tuple(expected[key].shape)} 实得 {tuple(tensor.shape)}"
                out_dict[key] = tensor
                n_params += tensor.numel()
            continue
        tensor = fn(get(src)).contiguous()
        assert dst in expected, f"目标 key 不在模型中: {dst}"
        assert tuple(expected[dst].shape) == tuple(tensor.shape), \
            f"{dst}: 期望 {tuple(expected[dst].shape)} 实得 {tuple(tensor.shape)}"
        out_dict[dst] = tensor
        n_params += tensor.numel()

    dtype = getattr(torch, args.dtype)
    out_dict = {k: v.to(dtype) for k, v in out_dict.items()}

    missing = [k for k in expected if k not in out_dict]
    print(f"转换 {len(out_dict)} 个张量, {n_params/1e9:.3f}B 参数 (期望模型 {sum(p.numel() for p in expected.values())/1e9:.3f}B)")
    print(f"未覆盖 (保持随机初始化, {len(missing)} 个):")
    for key in sorted(missing)[:8]:
        print(f"  {key} {tuple(expected[key].shape)}")
    if len(missing) > 8:
        print(f"  ... 共 {len(missing)} 个 (应全部为 text_in.*/text_refiner.*)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_dict, out_path)
    print(f"已保存 {out_path} ({out_path.stat().st_size/1024**3:.2f} GiB, {args.dtype})")


if __name__ == "__main__":
    main()
