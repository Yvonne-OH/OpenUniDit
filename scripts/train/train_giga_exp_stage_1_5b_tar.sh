#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jovyan/pfs/users/zhanqian.wu/code/model_exp"
cd "$ROOT"

# 修复 CUDA Error 803: ldconfig 优先命中 cuda-12.8/compat 的旧版 libcuda (570.86.10),
# 与驱动 (nvidia-smi 所报版本) 不匹配; 预加载 /usr/lib 下与驱动同版本的 libcuda。
# 不用 head -1: pipefail 下 head 提前关管道会让 nvidia-smi 偶发 SIGPIPE(141) 直接杀死脚本
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.${DRIVER_VER}"
if [[ -f "$LIBCUDA" && ":${LD_PRELOAD:-}:" != *":$(basename "$LIBCUDA"):*" ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

# 图像样本加载超时预算: DecodeBudget 将样本预算均分给最多 3 个候选。
# NFS tar seek 单读 p50 ~10ms / p99 ~1.5s (远好于 PFS LMDB 随机 get 的 0.5~4.5s),
# 沿用 LMDB 版预算: 样本预算 120s → 每候选 min(48, 120/3)=40s 真实读取窗口。
export YISU_IMAGE_TIMEOUT_SECONDS=48
export YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS=120
# 常驻解码子进程已关闭 (=0): 存储 stall 时常驻通道 select 烧满候选超时;
# 关闭后每次尝试直接一次性解码, fork+tar open 仅毫秒级。存储稳定后可改回 1 提速。
export YISU_IMAGE_PERSISTENT_DECODER=0

# NCCL 只在 WARN 及以上时输出: 平时静默, 通信异常(超时/降级/连接失败)时能看到原因
export NCCL_DEBUG=WARN
# expandable_segments: 显存按需扩展虚拟段, 缓解碎片 (5B: dim 4096, b=1+grad_accum=2)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 打印每次 dynamo 重编译的具体 guard 失败原因, 用于定位反复重编译的元凶
export TORCH_LOGS=recompiles

# stage_1 5B tar 数据版 (attention 加重型): 默认 configs/stage_1_5b_tar.yaml (NFS unified tar 分片)
CONFIG="${CONFIG:-${1:-configs/stage_1_5b_tar.yaml}}"
GPUS="${NPROC_PER_NODE:-${GPUS:-8}}"

# 启动前预检: tar 根目录必须已挂载且非空, 避免 8 个 rank 全部在 FileNotFoundError 上失败后重试
TAR_ROOT="$(python - "$CONFIG" <<'PY' | grep '^/'
import sys, yaml
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_absolute():
    path = Path.cwd() / path
print(yaml.safe_load(path.read_text())["dataset"]["database_path"])
PY
)"
if [[ ! -d "$TAR_ROOT" ]] || [[ -z "$(ls -A "$TAR_ROOT" 2>/dev/null)" ]]; then
  echo "❌ tar 数据集根目录不可用: $TAR_ROOT (NFS 未挂载或为空)" >&2
  exit 1
fi
echo "✅ tar 数据集根目录就绪: $TAR_ROOT"

# 单机多卡: 设置 NPROC_PER_NODE 或 GPUS。
# 多机多卡: 每台机器设置 WORLD_SIZE=机器数、RANK=机器编号、MASTER_ADDR、MASTER_PORT、NPROC_PER_NODE。
# 注意: shell 启动时的 WORLD_SIZE 表示机器数；torchrun 启动 worker 后会自行注入全局进程 WORLD_SIZE。
EXTERNAL_RANK="${RANK:-${NODE_RANK:-}}"
EXTERNAL_WORLD_SIZE="${WORLD_SIZE:-}"
if [[ -n "$EXTERNAL_RANK" ]]; then
  WORLD_SIZE="${EXTERNAL_WORLD_SIZE:-1}"
  RANK="$EXTERNAL_RANK"
else
  # 避免继承上层残留的 WORLD_SIZE=8 将单机误判为 8 台机器。
  WORLD_SIZE=1
  RANK=0
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

OUTPUT_DIR="$(python - "$CONFIG" <<'PY' | grep '^/'
# 平台给每个 python 进程注入的 ftelasticagent 钩子会往 stdout 打一行
# "Module imported successfully: ...", 命令替换会把它一起捕获进 OUTPUT_DIR,
# 导致 logs/compile_cache 全写进 model_exp 下名字带换行的畸形目录
# (历史日志曾全部丢在那里)。只保留以 / 开头的行 = yaml 里的 output_dir。
import sys, yaml
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_absolute():
    path = Path.cwd() / path
print(yaml.safe_load(path.read_text())["output_dir"])
PY
)"
if [[ -n "${GIGA_RUN_ID:-}" ]]; then
  RUN_ID="$GIGA_RUN_ID"
elif (( WORLD_SIZE > 1 )); then
  RUN_ID="multi_${MASTER_ADDR}_${MASTER_PORT}_n${WORLD_SIZE}_$(basename "$CONFIG" .yaml)"
else
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
RUN_LOG_DIR="$OUTPUT_DIR/logs/$RUN_ID"
mkdir -p "$RUN_LOG_DIR"
export GIGA_RUN_LOG_DIR="$RUN_LOG_DIR"
export GIGA_NODE_RANK="$RANK"

# torch.compile/Triton 编译缓存放共享存储 (OUTPUT_DIR 在 PFS 上, 所有机器可见):
# 重启/换机零重编译。首次启用时把本机 /tmp 的 warm 缓存一次性搬入, 保留已完成的 warmup。
COMPILE_CACHE_DIR="$OUTPUT_DIR/compile_cache"
if [[ ! -e "$COMPILE_CACHE_DIR/.ready" ]]; then
  mkdir -p "$COMPILE_CACHE_DIR/inductor" "$COMPILE_CACHE_DIR/triton"
  [[ -d /tmp/torchinductor_root ]] && cp -a /tmp/torchinductor_root/. "$COMPILE_CACHE_DIR/inductor/" || true
  [[ -d "$HOME/.triton/cache" ]] && cp -a "$HOME/.triton/cache/." "$COMPILE_CACHE_DIR/triton/" || true
  touch "$COMPILE_CACHE_DIR/.ready"
fi
export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE_DIR/inductor"
export TRITON_CACHE_DIR="$COMPILE_CACHE_DIR/triton"

CMD=(torchrun \
  --nproc_per_node "$GPUS" \
  --nnodes "$WORLD_SIZE" \
  --node_rank "$RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  scripts/train/train_giga_exp.py \
  --config "$CONFIG")

printf 'command: %q ' "${CMD[@]}" | tee "$RUN_LOG_DIR/command.txt"
printf '\n' | tee -a "$RUN_LOG_DIR/command.txt"

# Keep the launcher output in this run's log directory while preserving exit status.
set +e
"${CMD[@]}" 2>&1 | tee "$RUN_LOG_DIR/console.log"
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
