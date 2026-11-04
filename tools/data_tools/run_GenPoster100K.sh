#!/usr/bin/env bash
# 启动 GenPoster100K (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" GenPoster100K "$@"
