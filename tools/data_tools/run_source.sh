#!/usr/bin/env bash
# 通用单数据集启动脚本 (默认自动 8 卡, 脚本内部自动 torchrun 重启)
# 前台阻塞运行 (兼容自定义 task 环境, 不放后台), 日志同时写文件
# 用法:   bash run_source.sh <source> [额外参数...]
# 示例:   bash run_source.sh fine_t2i --stride 2
#         bash run_source.sh blip3o-long-caption --shards blip3o_0000 blip3o_0001
# 单卡:   追加 --ngpu 1
# 日志:   logs/generate_<source>_<时间>.log ; 断点续跑: 重跑同一命令自动跳过已完成
set -e -o pipefail
SOURCE="$1"; shift || true
if [ -z "$SOURCE" ]; then
  echo "用法: bash run_source.sh <source> [额外参数...]"; exit 1
fi
DIR="$(cd "$(dirname "$0")" && pwd)"
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh
cd "$DIR"
# 防 PATH 污染: 确认 python 来自 gwp 环境, 否则 torchrun/-m 会用错 torch
case "$(command -v python)" in
  */env/gwp/*) ;;
  *) echo "❌ python 不在 gwp 环境 ($(command -v python)), 检查 gwp.sh"; exit 1 ;;
esac
mkdir -p "$DIR/logs"
LOG="$DIR/logs/generate_${SOURCE}_$(date +%m%d_%H%M%S).log"
echo "🚀 启动 ${SOURCE} (前台阻塞) -> ${LOG}"
# 前台 + tee: 任务环境能看到实时输出, 日志同时落盘; pipefail 保证异常码透传
python generate_prompts.py --source "$SOURCE" "$@" 2>&1 | tee "$LOG"
