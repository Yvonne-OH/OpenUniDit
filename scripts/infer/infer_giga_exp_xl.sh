#!/usr/bin/env bash
# stage_1 XL (~2B, patch 1x1x1) 本地权重推理入口: 与 train_giga_exp_stage_1_xl.sh /
# configs/stage_1_xl.yaml 配套。
# checkpoint 缺省取本地训练路径 (<config output_dir>/checkpoints/weights) 下 step 最大的
# step-*.safetensors; prompt 清单/采样循环全部复用 infer_giga_exp.sh。
# 用法: bash infer_giga_exp_xl.sh
# 可选环境变量 (透传 infer_giga_exp.sh: GPU/BATCH_SIZE/STEPS/CFGS/SHIFTS/HEIGHT/WIDTH 均生效):
#   CONFIG      模型配置      (默认 <model_exp>/configs/stage_1_xl.yaml)
#   CHECKPOINT  显式指定权重  (缺省取 xl output_dir/checkpoints/weights 最新 step-*)
#   OUTPUT_DIR  推理输出目录  (缺省 <model_exp>/outputs/pretrain_xl/<ckpt 名>)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── conda 环境 (推理依赖: torch/transformers/diffusers, 与 infer_giga_exp_bos.sh 一致) ──
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh

# CUDA 803 修复: 预加载与驱动同版本的 libcuda (与 train_giga_exp_stage_1_xl.sh 一致)
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}"
if [[ -f "$LIBCUDA" && ":${LD_PRELOAD:-}:" != *":$(basename "$LIBCUDA"):*" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

CONFIG="${CONFIG:-${ROOT}/configs/stage_1_xl.yaml}"

# ── 解析 checkpoint (与 infer_giga_exp.sh 同一逻辑, 提前到此以拼 _xl 输出目录) ──
# 启动时一次性选定, 避免推理期间训练写入新权重导致各 prompt 组对比不一致。
# grep '^/' 过滤平台 ftelasticagent 钩子向 stdout 打印的提示行。
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
CKPT_NAME="$(basename "$CKPT_PATH")"
CKPT_BASE="${CKPT_NAME%.*}"

# ── 输出目录: 与 stage_1 / mHC 主实验区分, 单独放 outputs/pretrain_xl/<ckpt 名> ──
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/pretrain_xl/${CKPT_BASE}}"

# XL 训练 scheduler.validation_shift=3.0、验证 guidance=3.0: 采样默认对齐训练,
# 需要扫其它组合时用 SHIFTS/CFGS (空格分隔) 覆盖。
export SHIFTS="${SHIFTS:-3}"
export CFGS="${CFGS:-3 6 9}"

printf 'XL 推理: config=%s\ncheckpoint=%s (%s)\n输出目录=%s\n采样: shift=[%s] cfg=[%s] steps=%s size=%s\n' \
  "$CONFIG" "$CKPT_PATH" "$CKPT_NAME" "$OUTPUT_DIR" \
  "${SHIFTS}" "${CFGS}" "${STEPS:-40}" "${HEIGHT:-512}x${WIDTH:-512}"

export CONFIG CHECKPOINT="$CKPT_PATH" OUTPUT_DIR
exec bash "${SCRIPT_DIR}/infer_giga_exp.sh"
