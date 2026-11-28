#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CONFIG="${CONFIG:-${1:-configs/stage_1_5b_qwen35_9b_fsdp2_tar.yaml}}"
exec bash scripts/train/train_giga_exp_stage_1_5b_qwen35_9b_tar.sh "$CONFIG"
