#!/bin/bash
# B站持续采集循环：每轮抓 popular+每周必看+入站必刷+排行榜，
# 已采集 bvid 自动跳过，榜单轮换后自动增量入库。每轮结束 sleep 1h。
cd /home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools || exit 1
OUT=/mnt/nfs/data/zhanqian.wu/unified_images
LOG=$OUT/_meta/bili_loop.log
echo "===== bili loop started $(date '+%F %T') pid=$$ =====" >> "$LOG"
while true; do
  echo "===== round start $(date '+%F %T') =====" >> "$LOG"
  python3 unified_image_collector.py --platform bilibili --bili-source all \
    --out "$OUT" --per-query 100000 --pages 40 --workers 6 \
    >> "$LOG" 2>&1
  echo "===== round done $(date '+%F %T') =====" >> "$LOG"
  sleep 3600
done
