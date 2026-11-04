#!/usr/bin/env bash
# 启动 4KLSDB (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" 4KLSDB "$@"
