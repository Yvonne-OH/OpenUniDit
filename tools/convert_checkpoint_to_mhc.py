"""原始 Giga-Exp checkpoint -> mHC checkpoint 转换脚本。

用法:
    python tools/convert_checkpoint_to_mhc.py \
        --checkpoint <原模型 .safetensors> \
        --config configs/mhc_2stream.yaml \
        --output <输出 .safetensors 路径>

职责 (失败即报错退出, 不静默):
1. 原模型 291 参数完整加载, 逐 key 核对 shape, 缺失/多余立即报错;
2. 新增 mHC 参数按 function-preserving 初始化 (零投影权重 + 极端 bias, readout=[1,0..0]);
3. 输出 conversion_report.json: 转换前后参数统计 / missing / unexpected / 共享权重检查;
4. 原始 checkpoint 只读, 不覆盖; optimizer state 不迁移 (第一版仅权重, 日志明示)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.giga_transformer import GigaExpTransformer, apply_mod_adaln_migration
from models.giga_transformer_mhc import GigaExpTransformerMHC


def numel(state) -> int:
    return sum(v.numel() for v in state.values())


def convert(checkpoint_path: Path, config_path: Path, output_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if not cfg.get("mhc", {}).get("enabled"):
        raise ValueError("configs/mhc_2stream.yaml 必须显式 mhc.enabled: true")
    model_cfg = dict(cfg["model"])
    mhc_cfg = cfg["mhc"]
    model_cfg.setdefault("n_streams", mhc_cfg.get("n_streams", 4))
    model_cfg.setdefault("sinkhorn_iters", mhc_cfg.get("sinkhorn_iters", 20))

    # ---- 1. 原模型侧: 加载并建立命名/shape 基准 ---- #
    from safetensors.torch import load_file, save_file
    original = load_file(str(checkpoint_path), device="cpu")
    base_keys = {"in_channels", "out_channels", "text_dim", "dim", "ffn_dim",
                 "heads", "layers", "rope_theta", "rope_axes_dims", "patch_size"}
    base_model = GigaExpTransformer(**{k: v for k, v in cfg["model"].items() if k in base_keys})
    base_ref = base_model.state_dict()
    missing_in_ckpt = sorted(set(base_ref) - set(original))
    unexpected_in_ckpt = sorted(set(original) - set(base_ref))
    shape_mismatch = {k: (tuple(base_ref[k].shape), tuple(original[k].shape))
                      for k in set(base_ref) & set(original) if base_ref[k].shape != original[k].shape}
    if missing_in_ckpt or unexpected_in_ckpt or shape_mismatch:
        raise RuntimeError(
            f"原 checkpoint 与模型不匹配: missing={missing_in_ckpt} "
            f"unexpected={unexpected_in_ckpt} shape_mismatch={shape_mismatch}")
    # 共享权重检查: state_dict 中同名张量若 data_ptr 相同则存在 sharing
    ptr_map = {}
    shared = []
    for k, v in base_model.named_parameters():
        ptr = v.data_ptr()
        if ptr in ptr_map:
            shared.append((ptr_map[ptr], k))
        ptr_map[ptr] = k
    del base_model, base_ref

    # ---- 2. mHC 模型: 载入原参数 (strict), mHC 新参数保持初始化 ---- #
    mhc_model = GigaExpTransformerMHC(**model_cfg)
    mhc_state = mhc_model.state_dict()
    # mod_adaln 开启时先把旧 blocks.*.mod.* 拷贝初始化到 mod_img/mod_txt 双分支
    # (拷贝复用而非权重共享), 再参与 strict 核对; 缺失且无旧 mod.* 的层保持随机初始化
    migration = apply_mod_adaln_migration(mhc_state, original)
    if migration["copied"] or migration["missing"] or migration["orphan_dual"]:
        print(f"⚠️ mod-adaln 迁移: copied_layers={migration['copied']} "
              f"missing_layers={migration['missing']} orphan_dual={migration['orphan_dual']}")
    mhc_new_keys = sorted(set(mhc_state) - set(original))
    load_result = mhc_model.load_state_dict(original, strict=False)
    # strict=False 后必须人工核对: missing 只允许 mhc 新增键, unexpected 必须为空
    if load_result.unexpected_keys:
        raise RuntimeError(f"unexpected keys (不允许): {load_result.unexpected_keys}")
    if set(load_result.missing_keys) != set(mhc_new_keys):
        raise RuntimeError(f"原模型参数存在 missing (不允许): "
                           f"{sorted(set(load_result.missing_keys) - set(mhc_new_keys))}")

    # function-preserving 初始化已在模块构造时完成; 这里复核关键不变量
    with torch.no_grad():
        assert torch.all(mhc_model.readout == torch.tensor([1.0] + [0.0] * (mhc_model.n_streams - 1)))
        for block in mhc_model.blocks:
            for mhc in (block.mhc_attn, block.mhc_ffn):
                assert torch.all(mhc.mapping.weight == 0), "function-preserving 要求投影权重为 0"

    # ---- 3. 保存 ---- #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_state = {k: v.detach().cpu().contiguous() for k, v in mhc_model.state_dict().items()}
    semantic = mhc_model.semantic_metadata()
    semantic["mhc"]["init_mode"] = mhc_cfg.get("init_mode", "function_preserving")
    semantic["mhc"]["converted_from"] = str(checkpoint_path)
    semantic["mhc"]["source_step"] = _source_step(checkpoint_path)
    save_file(final_state, str(output_path),
              metadata={"format": "pt", "step": str(semantic["mhc"]["source_step"]),
                        "giga_semantic_metadata": json.dumps(semantic, sort_keys=True)})

    # ---- 4. 回读校验: 保存后重新加载必须与内存态完全一致 ---- #
    reloaded = load_file(str(output_path), device="cpu")
    for k in final_state:
        if not torch.equal(reloaded[k], final_state[k]):
            raise RuntimeError(f"round-trip 校验失败: {k}")
    if set(reloaded) != set(final_state):
        raise RuntimeError("round-trip 键集合不一致")

    report = {
        "source_checkpoint": str(checkpoint_path),
        "source_step": semantic["mhc"]["source_step"],
        "config": str(config_path),
        "output": str(output_path),
        "original_param_count": numel(original),
        "mhc_param_count": numel(final_state),
        "added_param_count": numel(final_state) - numel(original),
        "added_ratio": (numel(final_state) - numel(original)) / numel(original),
        "original_tensor_count": len(original),
        "mhc_tensor_count": len(final_state),
        "added_keys": mhc_new_keys,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "shared_weights": shared,
        "optimizer_state": "not migrated (第一版仅权重; mHC 新参数 optimizer 状态全新)",
        "function_preserving_check": {
            "readout": [1.0] + [0.0] * (mhc_model.n_streams - 1),
            "mapping_weight_all_zero": True,
            "n_streams": mhc_model.n_streams,
            "sinkhorn_iters": model_cfg.get("sinkhorn_iters", 20),
        },
    }
    report_path = output_path.parent / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _source_step(path: Path) -> int:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    try:
        return int(meta.get("step", 0))
    except (TypeError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="原模型 .safetensors")
    parser.add_argument("--config", default="configs/mhc_2stream.yaml")
    parser.add_argument("--output", required=True, help="输出 mHC .safetensors")
    args = parser.parse_args()
    report = convert(Path(args.checkpoint), ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config), Path(args.output))
    print(json.dumps({k: v for k, v in report.items() if k != "added_keys"}, ensure_ascii=False, indent=2))
    print(f"新增 {len(report['added_keys'])} 个张量, +{report['added_param_count']/1e6:.3f}M params "
          f"(+{report['added_ratio']*100:.4f}%), report -> {report['output'].rsplit('/', 1)[0]}/conversion_report.json")


if __name__ == "__main__":
    main()
