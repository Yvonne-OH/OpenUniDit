#!/usr/bin/env bash
# 启动 ffhq_captioned_1024 (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" ffhq_captioned_1024 "$@"
