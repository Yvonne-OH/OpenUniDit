#!/usr/bin/env bash
# 启动 qwen-synth-characters-fused (默认 8 卡)
exec "$(dirname "$0")/run_source.sh" qwen-synth-characters-fused "$@"
