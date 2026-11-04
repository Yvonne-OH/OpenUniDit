#!/usr/bin/env bash
# 启动 monet_cc12m (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" monet_cc12m "$@"
