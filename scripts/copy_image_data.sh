#!/usr/bin/env bash
# 将 stage_1_5b_qwen35_4b_mhc_tar.yaml (L79-121) 列出的数据集复制到 image_data
# 特性: 断点续传(done 清单 + copy_tree.py 跳过同大小文件) / 进度 / 速度 / 心跳
# 依赖: python3 (无需 rsync)
#
# 用法:
#   nohup bash copy_image_data.sh > /dev/null 2>&1 &      # 后台运行, 中断后重跑同一命令即续传
#   tail -f $STATE_DIR/STATUS                              # 查看实时进度/速度/心跳
#   bash copy_image_data.sh --status                       # 一次性查看状态
#   bash copy_image_data.sh --retry-failed                 # 只重试失败的数据集
set -uo pipefail

SRC=/mnt/nfs/data/zhanqian.wu
DEST=/home/jovyan/pfs/data/public/image_data
STATE_DIR=/home/jovyan/pfs/users/zhanqian.wu/exp/copy_image_data_state
HEARTBEAT_SEC=5
STALL_LIMIT=24         # 连续 N 个心跳速度<1MB/s 则告警(24x5s=2分钟)
WORKERS=${WORKERS:-32}  # 每个数据集的并行复制线程数, 可用 WORKERS=64 bash ... 覆盖

DONE="$STATE_DIR/done.txt"
FAILED="$STATE_DIR/failed.txt"
STATUS="$STATE_DIR/STATUS"
LOG="$STATE_DIR/copy.log"
HB_PID="$STATE_DIR/heartbeat.pid"

IMAGES=(
  4KLSDB Aesthetic-4K HumanCaption-10M LAION_Aesthetics_1024 Poster100K
  anime_captions bg_photo_concepts blip3o-long-caption-wds blip3o-long-caption
  caption3o-xl-v4 ffhq_captioned_1024 fine_t2i pexels-advertisement-photography
  pexels_aesth_2_1024 qwen-synth-characters-fused sa1b_en anyword_3m
  pixelprose_cc12m pixelprose_commonpool pixelprose_redcaps monet_cc12m
  monet_commoncatalog_cc_by monet_coyo GenPoster100K Text-Render-2M
  moss_synth_sa1b_10h FLUX-Reason-6M JourneyDB moss_synth moss_synth_eval100k
  h3image_synth_webcap_10h moss_synth_300k_48h
)
EDITS=(
  HQ-Edit GPT-UltraEdit GPT-HQEdit pico-banana-400k GPT-OmniEdit
  CrispEdit SEED-Data-Edit
)

mkdir -p "$DEST" "$STATE_DIR"
touch "$DONE" "$FAILED"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# ---------- 状态查看 ----------
if [[ "${1:-}" == "--status" ]]; then
  cat "$STATUS" 2>/dev/null || echo "(尚未运行)"
  exit 0
fi

# ---------- 构建任务列表: "子库|数据集名" ----------
TASKS=()
for d in "${IMAGES[@]}"; do TASKS+=("unified_images_tar|$d"); done
for d in "${EDITS[@]}";  do TASKS+=("unified_edit_images_tar|$d"); done
TOTAL=${#TASKS[@]}

is_done()   { grep -qxF "$1" "$DONE"; }
mark_done() { echo "$1" >> "$DONE"; }
mark_fail() { grep -qxF "$1" "$FAILED" || echo "$1" >> "$FAILED"; }

# ---------- 心跳: 每 HEARTBEAT_SEC 读 current.txt 写 STATUS ----------
# 主进程与心跳通过文件通信(心跳是 fork 出的子进程, 看不到父进程变量)
CUR="$STATE_DIR/current.txt"   # 格式: idx|total|name|plog
TOTAL_G=$TOTAL
heartbeat() {
  local slow=0 start=$SECONDS
  while :; do
    sleep "$HEARTBEAT_SEC"
    local now ts mb done_n idx tot name plog l1 sp v u dn_list
    now=$((SECONDS - start))
    ts=$(date '+%F %T')
    done_n=$(wc -l < "$DONE")
    idx=0; tot=$TOTAL_G; name="无"; plog=""
    if [[ -f "$CUR" ]]; then
      IFS='|' read -r idx tot name plog < "$CUR"
    fi
    # 从快照第一行提取速度, 换算成 MB/s 判断是否停滞
    mb=-1
    if [[ -n "$plog" && -f "$plog" ]]; then
      l1=$(sed -n '1p' "$plog")
      sp=$(grep -aoE '[0-9.]+(B|KB|MB|GB|TB)/s' <<<"$l1" | head -1)
      if [[ -n "$sp" ]]; then
        v=$(grep -oE '^[0-9.]+' <<<"$sp")
        u=$(grep -oE '(B|KB|MB|GB|TB)$' <<<"$sp")
        case "$u" in
          B)  mb=$(awk -v x="$v" 'BEGIN{print x/1048576}') ;;
          KB) mb=$(awk -v x="$v" 'BEGIN{print x/1024}') ;;
          MB) mb=$v ;;
          *)  mb=9999 ;;
        esac
      fi
    fi
    if (( $(awk -v a="$mb" 'BEGIN{print (a>=0 && a<1)?1:0}') )); then
      slow=$((slow+1))
    else
      slow=0
    fi
    {
      echo -e "\033[36m===== $ts 心跳(每${HEARTBEAT_SEC}s) 已运行 $((now/60))分$((now%60))秒 =====\033[0m"
      echo -e "总体进度 : \033[32m$done_n\033[0m/\033[36m$TOTAL_G\033[0m 个数据集完成"
      echo -e "当前任务 : [\033[36m$idx\033[0m/$tot] \033[1m$name\033[0m"
      if [[ -n "$plog" && -f "$plog" ]]; then
        sed 's/^/  /' "$plog"
      else
        echo "  (空闲)"
      fi
      (( slow >= STALL_LIMIT )) && echo -e "\033[33m!! 告警: 速度<1MB/s 已持续 $((slow*HEARTBEAT_SEC))s, 疑似卡住\033[0m"
      echo -e "\033[32m已完成($done_n)\033[0m : $(sort "$DONE" | paste -sd', ' -)"
      [[ -s "$FAILED" ]] && echo -e "\033[31m失败\033[0m     : $(sort "$FAILED" | paste -sd', ' -)"
    } > "$STATUS"
    (( slow >= STALL_LIMIT )) && log "WARN $name 速度停滞(${slow}x心跳<1MB/s)"
  done
}

# 清理旧心跳进程
[[ -f "$HB_PID" ]] && kill "$(cat "$HB_PID")" 2>/dev/null
heartbeat &
echo $! > "$HB_PID"
trap 'kill $(cat "$HB_PID") 2>/dev/null' EXIT

log "========== 开始复制: $TOTAL 个数据集 -> $DEST =========="

# ---------- 主循环 ----------
idx=0
for t in "${TASKS[@]}"; do
  sub=${t%%|*}; name=${t##*|}
  [[ "${1:-}" == "--retry-failed" ]] && ! grep -qxF "$name" "$FAILED" && continue
  if is_done "$name"; then
    log "SKIP $name (已完成)"
    idx=$((idx+1))
    continue
  fi
  idx=$((idx+1))
  src="$SRC/$sub/$name"
  dst="$DEST/$name"
  plog="$STATE_DIR/rsync_${name}.log"
  if [[ ! -d "$src" ]]; then
    log "ERROR 源不存在: $src"
    mark_fail "$name"; continue
  fi
  echo "$idx|$TOTAL|$name|$plog" > "$CUR"
  log "START [$idx/$TOTAL] $name  ($src)"
  # python 复制: 已存在且同大小的文件跳过(断点续传), 原子写防半文件
  python3 "$(dirname "${BASH_SOURCE[0]}")/copy_tree.py" \
          "$src" "$dst" "$plog" "$WORKERS" >> "$plog" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    mark_done "$name"
    sed -i "/^$name\$/d" "$FAILED" 2>/dev/null
    log "DONE  [$idx/$TOTAL] $name  (rc=0)"
  else
    mark_fail "$name"
    log "FAIL  [$idx/$TOTAL] $name  rc=$rc, 重跑脚本自动续传"
  fi
done
echo "0|$TOTAL|已完成|" > "$CUR"

log "========== 结束: 完成 $(wc -l < "$DONE")/$TOTAL, 失败 $(wc -l < "$FAILED") =========="
echo "ALL_FINISHED $(date '+%F %T')" >> "$STATUS"
