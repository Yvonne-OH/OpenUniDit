#!/usr/bin/env bash
# 从 BOS 拉取 giga_exp_image/stage_1_28layers 最新权重并推理, 输出目录加 _5090 后缀
# 用法: bash infer_giga_exp_bos.sh
# 可选环境变量 (透传 infer_giga_exp.sh 的 CONFIG/GPU/BATCH_SIZE/... 均生效):
#   BUCKET         BOS 桶名        (默认 gigaai-5090)
#   BOS_PREFIX     BOS 权重目录    (默认 users/zhanqian.wu/exp/giga_exp_image/stage_1_28layers/checkpoints/weights)
#   LOCAL_CKPT_DIR 本地权重缓存目录 (默认 <model_exp>/outputs/bos_ckpt/giga_exp_image/stage_1_28layers)
#   OUTPUT_DIR     推理输出目录    (默认 <model_exp>/outputs/pretrain_v2/vis_token_exp/<step 名>_5090)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── conda 环境 (含 baidubce 与推理依赖) ──
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh

BUCKET="${BUCKET:-gigaai-5090}"
BOS_PREFIX="${BOS_PREFIX:-users/zhanqian.wu/exp/giga_exp_image/stage_1_28layers/checkpoints/weights}"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-${MODEL_ROOT}/outputs/bos_ckpt/giga_exp_image/stage_1_28layers}"

# 1) 查询 BOS 上 step 最大的 step-*.safetensors
BOS_INFO="$(python - "$BUCKET" "$BOS_PREFIX" <<'PY'
import sys
sys.path.insert(0, "/home/jovyan/pfs/users/zhanqian.wu/tools/bos_tools")
from bos_utils import get_client
bucket, prefix = sys.argv[1:3]
client = get_client()
r = client.list_objects(bucket, prefix=prefix.rstrip("/") + "/", max_keys=1000)
cands = [(o.key, o.size) for o in r.contents
         if o.key.endswith(".safetensors") and "/step-" in o.key]
if not cands:
    sys.exit(f"BOS 上没有 step-*.safetensors: bos://{bucket}/{prefix}/")
key, size = max(cands, key=lambda x: int(x[0].rsplit("step-", 1)[1].split(".", 1)[0]))
print(key, size)
PY
)"
read -r BOS_KEY BOS_SIZE <<< "$BOS_INFO"
CKPT_NAME="$(basename "$BOS_KEY")"
printf 'BOS 最新权重: bos://%s/%s (%s)\n' "$BUCKET" "$BOS_KEY" \
  "$(numfmt --to=iec --suffix=B "$BOS_SIZE" 2>/dev/null || echo "${BOS_SIZE}B")"

# 2) 下载到本地 (已存在且大小一致则跳过, 断点重跑安全)
LOCAL_CKPT="${LOCAL_CKPT_DIR}/${CKPT_NAME}"
if [[ -f "$LOCAL_CKPT" && "$(stat -c%s "$LOCAL_CKPT")" == "$BOS_SIZE" ]]; then
  echo "本地已存在同名同大小权重, 跳过下载: $LOCAL_CKPT"
else
  mkdir -p "$LOCAL_CKPT_DIR"
  echo "下载 → $LOCAL_CKPT ..."
  T0=$SECONDS
  python - "$BUCKET" "$BOS_KEY" "$LOCAL_CKPT" <<'PY'
import sys
sys.path.insert(0, "/home/jovyan/pfs/users/zhanqian.wu/tools/bos_tools")
from bos_utils import get_client, download_resumable
bucket, key, path = sys.argv[1:4]
download_resumable(get_client(), bucket, key, path)
PY
  GOT="$(stat -c%s "$LOCAL_CKPT")"
  if [[ "$GOT" != "$BOS_SIZE" ]]; then
    echo "下载不完整: 本地 $GOT != BOS $BOS_SIZE" >&2
    exit 1
  fi
  echo "下载完成 ($((SECONDS - T0))s)"
fi

# 3) 推理: 输出目录在 checkpoint 名基础上加 _5090 后缀 (OUTPUT_DIR 已显式指定时原样使用)
CKPT_BASE="${CKPT_NAME%.*}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${MODEL_ROOT}/outputs/pretrain_v2/vis_token_exp/${CKPT_BASE}_5090"
fi

echo "开始推理: checkpoint=$LOCAL_CKPT → output=$OUTPUT_DIR"
CHECKPOINT="$LOCAL_CKPT" OUTPUT_DIR="$OUTPUT_DIR" bash "${SCRIPT_DIR}/infer_giga_exp.sh"
