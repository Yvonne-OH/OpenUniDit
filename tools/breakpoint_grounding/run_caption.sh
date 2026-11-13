#!/bin/bash
# breakpoint-grounding-55m stage 2: Qwen3-VL-4B caption (8 GPU 全量, 只给过滤通过行打标)
# 日志: tools/breakpoint_grounding/logs/caption_<ts>.log
cd "$(dirname "$0")" || exit 1
mkdir -p logs
TS=$(date +%m%d_%H%M%S)
nohup python caption_qwen3vl.py --num-gpus 8 --batch-size 24 \
  > "logs/caption_${TS}.log" 2>&1 &
echo "🚀 caption_qwen3vl PID $! → logs/caption_${TS}.log"
