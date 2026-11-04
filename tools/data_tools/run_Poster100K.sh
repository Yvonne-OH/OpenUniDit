#!/usr/bin/env bash
# 启动 Poster100K (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" Poster100K "$@"
