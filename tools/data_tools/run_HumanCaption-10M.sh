#!/usr/bin/env bash
# 启动 HumanCaption-10M (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" HumanCaption-10M "$@"
