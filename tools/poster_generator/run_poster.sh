#!/bin/bash
# 海报广告自动生成器启动脚本 (用前 7 张卡渲染, 第 8 张卡跑 Qwen3-VL)
set -euo pipefail
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

python poster_generator.py "$@"
