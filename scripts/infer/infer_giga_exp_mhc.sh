#!/usr/bin/env bash
# mHC 对比实验推理入口: 与 train_giga_exp_mhc.sh / configs/mhc_2stream.yaml 配套。
# 只改 CONFIG 与输出后缀, prompt 清单/采样参数/采样循环全部复用 infer_giga_exp.sh
# (模型构建由 infer_giga_exp.py 按 mHC 配置自动分发到 GigaExpTransformerMHC)。
# 用法: bash infer_giga_exp_mhc.sh
# 可选环境变量 (透传 infer_giga_exp.sh, GPU/BATCH_SIZE/STEPS/CFGS/SHIFTS 均生效):
#   CHECKPOINT  显式指定权重文件 (缺省取 mHC output_dir/checkpoints/weights 最新 step-*)
#   OUTPUT_DIR  显式指定输出目录 (缺省 <ckpt 名>_mhc)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── conda 环境 (推理依赖: torch/transformers/diffusers, 与 infer_giga_exp_bos.sh 一致) ──
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh

# CUDA 803 修复: 预加载与驱动同版本的 libcuda (与 train_giga_exp_mhc.sh 一致)
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}"
if [[ -f "$LIBCUDA" && ":${LD_PRELOAD:-}:" != *":$(basename "$LIBCUDA"):*" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

CONFIG="${CONFIG:-configs/mhc_2stream.yaml}"

# ── 解析 checkpoint (与 infer_giga_exp.sh 同一逻辑, 提前到此以拼 _mhc 输出目录) ──
# 启动时一次性选定, 避免推理期间训练写入新权重导致对比不一致。
CKPT_PATH="$(python - "$ROOT" "$CONFIG" "${CHECKPOINT:-}" <<'PY' | grep '^/'
import sys
from pathlib import Path
import yaml
root, config, checkpoint = sys.argv[1:]
config = Path(config)
if not config.is_absolute():
    config = Path(root) / config
cfg = yaml.safe_load(config.read_text())
target = Path(checkpoint) if checkpoint else Path(cfg["output_dir"]) / "checkpoints" / "weights"
if target.is_dir():
    candidates = list(target.glob("step-*.safetensors")) + list(target.glob("step-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"目录里没有 step-* 权重文件: {target}")
    target = max(candidates, key=lambda p: int(p.stem.split("-")[1]))
if not target.is_file():
    raise FileNotFoundError(f"权重不存在: {target}")
print(target.resolve())
PY
)"
CKPT_BASE="$(basename "$CKPT_PATH")"
CKPT_BASE="${CKPT_BASE%.*}"

# ── 输出目录: infer_giga_exp.sh 默认规则 (<ckpt 名>) 基础上加 _mhc 后缀, 与主分支区分 ──
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/pretrain_v2/vis_token_exp/${CKPT_BASE}_mhc}"

printf 'mHC 推理: config=%s\ncheckpoint=%s\n输出目录=%s\n' "$CONFIG" "$CKPT_PATH" "$OUTPUT_DIR"

export CONFIG
export CHECKPOINT="$CKPT_PATH"
export OUTPUT_DIR
exec bash "${SCRIPT_DIR}/infer_giga_exp.sh"
