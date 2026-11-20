#!/bin/bash
# Pixabay 持续采集循环：词表断点续跑（已达标词自动跳过），
# 免费 API 100 请求/小时由脚本内 37s/页 节流保证。每轮结束 sleep 300s。
cd /home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools || exit 1
OUT=/mnt/nfs/data/zhanqian.wu/unified_images
LOG=$OUT/_meta/pixabay_loop.log
echo "===== pixabay loop started $(date '+%F %T') pid=$$ =====" >> "$LOG"
while true; do
  echo "===== round start $(date '+%F %T') =====" >> "$LOG"
  python3 unified_image_collector.py --platform pixabay \
    --queries-file queries_full.txt --per-query 20 --pages 3 \
    --workers 8 --out "$OUT" >> "$LOG" 2>&1
  echo "===== round done $(date '+%F %T') =====" >> "$LOG"
  sleep 300
done
