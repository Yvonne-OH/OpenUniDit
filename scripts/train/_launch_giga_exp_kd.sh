#!/usr/bin/env bash
set -uo pipefail

# 自动定位 conda.sh（按优先级探测常见安装路径，命中即 source）
for _conda_sh in \
  /opt/conda/etc/profile.d/conda.sh \
  /mnt/pfs/users/zhanqian.wu/miniconda3/etc/profile.d/conda.sh \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  /etc/profile.d/conda.sh; do
  if [ -f "$_conda_sh" ]; then
    source "$_conda_sh"
    break
  fi
done

# 修复 CUDA Error 803：ldconfig 默认解析到 /usr/local/cuda-12.8/compat 的旧 libcuda（570.86.10），
# 与内核驱动 570.172.08 不匹配；优先加载系统驱动库
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# 激活 gwp 环境（通过路径指定，无需注册到 conda envs）
conda activate /home/jovyan/pfs/users/zhanqian.wu/env/gwp
export PATH="/home/jovyan/pfs/users/zhanqian.wu/env/gwp/bin:$PATH"

cd /home/jovyan/pfs/users/zhanqian.wu/code/model_exp
bash /home/jovyan/pfs/users/zhanqian.wu/code/model_exp/scripts/train/train_giga_exp_kd.sh
