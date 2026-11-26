#!/usr/bin/env bash
# 吞吐基准矩阵: 512x512, 8 卡 ZeRO2, 固定 text_len=512
# 依次跑 (bs, accum, AC ratio, prefetch) 组合, 结果写 exp/bench_throughput/*.json
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
gwp=/home/jovyan/pfs/users/zhanqian.wu/env/gwp
if [[ "${CONDA_PREFIX:-}" != "$gwp" && "$(command -v python || true)" != "$gwp/bin/python" ]]; then
  source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh || { echo "gwp init failed" >&2; exit 1; }
fi
export PATH="$gwp/bin:$PATH" PYTHON_EXEC="$gwp/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/giga_bench_${UID}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/giga_bench_${UID}/inductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

out_dir=exp/bench_throughput
mkdir -p "$out_dir"
port=29560

# 与训练 launcher 相同: 注入宿主机 libcuda, 否则 CUDA init 报 error 803
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')
libcuda="/usr/lib/x86_64-linux-gnu/libcuda.so.$driver"
if [[ -f "$libcuda" ]]; then
  export LD_PRELOAD="$libcuda${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# name bs accum ac_ratio prefetch
configs=(
  "bs2_a1_ac0.8_pf0  2 1 0.8 0"
  "bs3_a1_ac0.8_pf0  3 1 0.8 0"
  "bs4_a1_ac0.8_pf0  4 1 0.8 0"
  "bs3_a1_ac0.5_pf0  3 1 0.5 0"
  "bs3_a1_ac0_pf0    3 1 0   0"
  "bs2_a2_ac0.8_pf0  2 2 0.8 0"
  "bs3_a2_ac0.8_pf0  3 2 0.8 0"
  "bs4_a2_ac0.8_pf0  4 2 0.8 0"
  "bs2_a1_ac0.8_pf1  2 1 0.8 1"
  "bs3_a1_ac0.8_pf1  3 1 0.8 1"
  "bs3_a2_ac0.8_pf1  3 2 0.8 1"
  "bs4_a2_ac0.8_pf1  4 2 0.8 1"
  "bs4_a1_ac0.5_pf0  4 1 0.5 0"
  "bs4_a2_ac0.5_pf1  4 2 0.5 1"
)

for entry in "${configs[@]}"; do
  read -r name bs accum ratio pf <<< "$entry"
  if [[ -s "$out_dir/$name.json" ]]; then
    echo "SKIP $name (已存在)"
    continue
  fi
  echo "=== RUN $name (bs=$bs accum=$accum ac=$ratio pf=$pf) $(date '+%H:%M:%S') ==="
  port=$((port + 1))
  mkdir -p "$out_dir/logs"
  timeout 900 "$PYTHON_EXEC" -m torch.distributed.run --nnodes 1 --nproc_per_node 8 \
    --master_addr 127.0.0.1 --master_port "$port" --node_rank 0 \
    scripts/bench/bench_throughput.py \
    --config configs/stage_1_5b_qwen35_4b_tar.yaml \
    --batch-size "$bs" --accum "$accum" --ac-ratio "$ratio" --prefetch "$pf" \
    --text-len 512 --warmup 6 --steps 16 \
    --output "$out_dir/$name.json" > "$out_dir/logs/$name.log" 2>&1
  rc=$?
  if [[ ! -s "$out_dir/$name.json" ]]; then
    printf '{"name": "%s", "oom": false, "error": "no result (rc=%s)"}\n' "$name" "$rc" > "$out_dir/$name.json"
    echo "--- $name 失败 (rc=$rc), 日志尾部:"
    grep -E 'Error|OutOfMemory|Traceback' "$out_dir/logs/$name.log" | head -5
  fi
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
cat "$out_dir"/*.json | "$PYTHON_EXEC" -c "
import json, sys
for line in sys.stdin.read().split('}'):
    line = line.strip().lstrip('{')
    if not line: continue
    d = json.loads('{' + line + '}')
    if d.get('oom'):
        print(f\"{d['name']}: OOM\")
    elif d.get('error'):
        print(f\"{d['name']}: ERROR {d['error']}\")
    else:
        print(f\"{d['name']}: {d['samples_per_s']}/s (global {d['global_samples_per_s']}/s, {d['s_per_update']}s/upd, peak {d['peak_mem_gib']}GiB)\")
"
