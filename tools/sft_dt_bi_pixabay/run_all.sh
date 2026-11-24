#!/usr/bin/env bash
# run_all.sh — sft_dt_bi_pixabay 全流程: manifest → caption (8卡) → lmdb → tar
# 用法: bash run_all.sh [步骤...]   # 无参 = 全部; 可选: manifest caption lmdb tar verify
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh
cd "$DIR"

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(manifest caption lmdb tar verify)
WORK=/home/jovyan/pfs/users/zhanqian.wu/exp/sft_dt_bi_pixabay
LMDB_ROOT=/mnt/nfs/data/zhanqian.wu/unified_images_lmdb
TAR_ROOT=/mnt/nfs/data/zhanqian.wu/unified_images_tar
NAME=sft_dt_bi_pixabay
mkdir -p "$WORK"

has_step() { [[ " ${STEPS[*]} " == *" $1 "* ]]; }

if has_step manifest; then
  echo "== [1/4] manifest =="
  python3 build_manifest.py --out "$WORK/manifest.parquet"
fi

if has_step caption; then
  echo "== [2/4] Qwen3-VL caption =="
  python3 caption_manifest.py --num-gpus 8
fi

if has_step lmdb; then
  echo "== [3/4] build lmdb =="
  python3 build_lmdb.py
fi

if has_step tar; then
  echo "== [4/4] lmdb -> tar =="
  # --no-parallel-envs: 走 convert_dataset 主流程, 产出 shard-NNNNN 命名
  # (parallel-envs 会产出 {env_stem}-NNNNN 命名, image 库读取端只 glob shard-*.parquet)
  python3 ../lmdb_to_tar.py \
    --src "$LMDB_ROOT" --dst "$TAR_ROOT" \
    --staging "$WORK/tar_staging" \
    --datasets "$NAME" --no-parallel-envs
fi

if has_step verify; then
  echo "== verify =="
  python3 verify_dataset.py
fi
