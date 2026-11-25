#!/usr/bin/env bash
# yisu CPU 节点上的数据集拷贝 worker
# 用法: bash yisu_copy_worker.sh <group_file> <out_dir>
# group_file 每行: <源子目录>|<数据集名>   out_dir 在 PFS 上, 存 run.log/progress/exit_code
set -u
GROUP_FILE="$1"
OUT="$2"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "JOB_START $(date -u +%FT%TZ) host=$(hostname) group=$GROUP_FILE"

SCRIPT_DIR=/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/scripts
SRC=/mnt/nfs/data/zhanqian.wu
DEST=/home/jovyan/pfs/data/public/image_data
WORKERS=32

total=0
ok=0
fail=0
while IFS='|' read -r sub name; do
  [[ -z "$sub" || "$sub" == \#* ]] && continue
  total=$((total+1))
  src="$SRC/$sub/$name"
  dst="$DEST/$name"
  plog="$OUT/progress_${name}.log"
  if [[ ! -d "$src" ]]; then
    echo "[ERROR] 源不存在: $src"
    fail=$((fail+1))
    continue
  fi
  echo "[$(date '+%F %T')] START $name ($src -> $dst)"
  python3 "$SCRIPT_DIR/copy_tree.py" "$src" "$dst" "$plog" "$WORKERS"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date '+%F %T')] DONE $name"
    ok=$((ok+1))
  else
    echo "[$(date '+%F %T')] FAIL $name rc=$rc (重跑自动续传)"
    fail=$((fail+1))
  fi
done < "$GROUP_FILE"

echo "GROUP_SUMMARY total=$total ok=$ok fail=$fail"
printf "%s" "$(( fail > 0 ? 1 : 0 ))" > "$OUT/exit_code"
echo "JOB_END exit_code=$(( fail > 0 ? 1 : 0 ))"
exit $(( fail > 0 ? 1 : 0 ))
