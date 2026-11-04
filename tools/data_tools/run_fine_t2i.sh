#!/usr/bin/env bash
# 启动 fine_t2i (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" fine_t2i "$@"
