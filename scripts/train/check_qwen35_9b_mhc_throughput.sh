#!/usr/bin/env bash
# Bounded benchmarks only: no formal training/checkpoints/SwanLab writes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh
MODE="${1:-compute}"
CONFIG="${CONFIG:-configs/stage_1_5b_qwen35_9b_mhc_tar.yaml}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export YISU_IMAGE_TIMEOUT_SECONDS=48 YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=120
export YISU_IMAGE_PERSISTENT_DECODER=0

if [[ "$MODE" == real64 ]]; then
  : "${MASTER_ADDR:?set MASTER_ADDR to the address of node zero}"
  NODE_INDEX="${NODE_RANK:-${RANK:?set NODE_RANK to the node index from 0 to 7}}"
  [[ "$NODE_INDEX" =~ ^[0-7]$ ]] || { echo "NODE_RANK must be 0..7" >&2; exit 2; }
  MULTINODE=(torchrun --nnodes=8 --nproc_per_node=8 --node_rank "$NODE_INDEX"
    --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29549}"
    tests/benchmark_deepspeed_9b_mhc.py --config "$CONFIG" --label optimized_real64
    --real-data --warmup 16 --updates 64)
  if [[ "${GIGA_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${MULTINODE[@]}"
    printf '\n'
    exit 0
  fi
fi
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER}"
OUT="$(python - "$CONFIG" <<'PY' | grep '^/'
import sys,yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg['output_dir'])
PY
)"
mkdir -p "$OUT/preflight"
export TORCHINDUCTOR_CACHE_DIR="$OUT/compile_cache/inductor"
export TRITON_CACHE_DIR="$OUT/compile_cache/triton"
python - "$CONFIG" "$OUT/preflight/gpu_reference.yaml" "$MODE" <<'PY'
import os,subprocess,sys,yaml
from pathlib import Path
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
if len(memory.splitlines())!=8 or any(int(value.strip())>1024 for value in memory.splitlines()):
    raise SystemExit('需要8张空闲GPU；请勿与现有训练并行运行此基准。')
if sys.argv[3] != 'compute':
    sys.exit(0)
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
cfg['mhc']['routing_backend']='eager'
cfg['components']['text_encoder']['reference_gdn_cuda_graph']=False
cfg['components']['text_encoder']['frozen_inference_mode']=False
for service in ('wandb','swanlab'):
    tracker=cfg.get('logging',{}).get(service,{})
    tracker['enabled']=False
    tracker.pop('api_key',None)
fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
with os.fdopen(fd,'w') as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)
PY
run() {
  torchrun --standalone --nproc_per_node=8 tests/benchmark_deepspeed_9b_mhc.py "$@"
}
case "$MODE" in
  compute)
    torchrun --standalone --nproc_per_node=2 tests/test_stage1_training_runtime.py --deepspeed
    run --config "$OUT/preflight/gpu_reference.yaml" --label reference --image-size 512 --micro-batch 1 --grad-accum 2 --text-tokens 128 --warmup 2 --updates 8
    run --config "$CONFIG" --label optimized --image-size 512 --micro-batch 1 --grad-accum 2 --text-tokens 128 --warmup 2 --updates 8
    run --config "$CONFIG" --label optimized_large --image-size 512 --micro-batch 2 --grad-accum 1 --text-tokens 128 --warmup 2 --updates 8
    ;;
  stress)
    run --config "$CONFIG" --label longest_text_large --image-size 512 --micro-batch 2 --grad-accum 1 --text-tokens 2048 --warmup 1 --updates 2
    run --config "$CONFIG" --label longest_text_small_bucket --image-size 256 --micro-batch 4 --grad-accum 1 --text-tokens 2048 --warmup 1 --updates 2
    ;;
  real)
    run --config "$CONFIG" --label optimized_real --real-data --warmup 16 --updates 64
    ;;
  real64)
    "${MULTINODE[@]}"
    ;;
  *) echo "Usage: $0 [compute|stress|real|real64]" >&2; exit 2 ;;
esac
echo "结果目录: $OUT/preflight"
