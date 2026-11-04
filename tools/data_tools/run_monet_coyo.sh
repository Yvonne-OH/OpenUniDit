#!/usr/bin/env bash
# 启动 monet_coyo (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" monet_coyo "$@"
