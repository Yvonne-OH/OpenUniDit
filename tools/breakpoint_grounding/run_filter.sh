#!/bin/bash
# breakpoint-grounding-55m stage 1: 质量指标 + aesthetic 打分 (8 GPU 全量)
# 日志: tools/breakpoint_grounding/logs/filter_<ts>.log
cd "$(dirname "$0")" || exit 1
mkdir -p logs
TS=$(date +%m%d_%H%M%S)
nohup python filter_quality.py --num-gpus 8 --workers-per-gpu 24 \
  > "logs/filter_${TS}.log" 2>&1 &
echo "🚀 filter_quality PID $! → logs/filter_${TS}.log"
