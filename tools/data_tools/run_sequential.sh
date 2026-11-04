#!/usr/bin/env bash
# 小数据集顺序处理: 每个数据集 8 卡跑完再跑下一个, 失败自动跳过继续
# 用法:   bash run_sequential.sh            (前台阻塞, 兼容自定义 task 环境)
# 日志:   每个数据集各自 logs/generate_<source>_<时间>.log, 控制台实时汇总
# 续跑:   直接重跑, 每个数据集内部断点续跑 (自动跳过 enchanced_prompt 已完成样本)
# 总量:   ~5,559,005 条, 按实测 8 卡 ~17 条/s 约 91 小时
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
source /home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh
cd "$DIR"
mkdir -p "$DIR/logs"

SOURCES=(
  LAION_Aesthetics_1024        # 764,824
  monet_cc12m                  # 794,446
  anime_captions               # 674,076
  bg_photo_concepts            # 567,520
  monet_commoncatalog_cc_by    # 380,883
  pexels_aesth_2_1024          # 1,099,166
  pixelprose_commonpool        # 261,479
  GenPoster100K                # 205,302
  pixelprose_redcaps           # 159,152
  4KLSDB                       # 175,398 (正在跑的部分会自动断点续跑跳过)
  ffhq_captioned_1024          # 140,000
  Poster100K                   # 112,033
  qwen-synth-characters-fused  # 84,426
  caption3o-xl-v4              # 58,094
  Aesthetic-4K                 # 29,970
  pexels-advertisement-photography  # 25,834
  blip3o-long-caption          # 24,378
  monet_coyo                   # 2,024
)

for s in "${SOURCES[@]}"; do
  LOG="$DIR/logs/generate_${s}_$(date +%m%d_%H%M%S).log"
  echo "=== [$(date '+%m-%d %H:%M:%S')] ▶️  开始 ${s} -> ${LOG}"
  # 前台阻塞运行 (脚本内部自动 8 卡 torchrun), 跑完再进入下一个
  python generate_prompts.py --source "$s" > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== [$(date '+%m-%d %H:%M:%S')] ❌ ${s} 退出码 ${rc}, 跳过继续"
  else
    echo "=== [$(date '+%m-%d %H:%M:%S')] ✅ ${s} 完成"
  fi
done
echo "=== [$(date '+%m-%d %H:%M:%S')] 🏁 全部顺序任务结束"
