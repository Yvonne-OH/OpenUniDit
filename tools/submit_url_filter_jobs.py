#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成并提交 yisu CPU 任务: url_dataset_image_filter (wukong / danqing 各一个)。

用法: python3 submit_url_filter_jobs.py [--dry-run]
要求环境变量: SKILL, AIHC_CREDENTIALS (与 yisu-submit skill 约定一致)
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

SKILL = os.environ["SKILL"]
CRED = os.environ["AIHC_CREDENTIALS"]
TOOLS = "/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools"
RUNS = "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/url_filter_runs"
BOS_CRED = "/home/jovyan/pfs/users/zhanqian.wu/.bos_center_cred"
BOS_PREFIX = "users/zhanqian.wu/unified_images_ori"
TS = time.strftime("%Y%m%d-%H%M%S")

# 每数据集拆 N 个分片任务 (按源文件 fi %% N 分配, 互不重叠), 单任务线程适中靠并行分片提速
SHARDS = {"wukong": 6, "danqing": 12}
SRC = {
    "wukong": "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/Noah-Wukong-100M/data",
    "danqing": "/home/jovyan/pfs/users/zhanqian.wu/pfs_data/DanQing100M/data",
}
THREADS = 96


def build_cmd(dataset: str, src: str, job_name: str, shard_idx: int, num_shards: int) -> str:
    out = f"{RUNS}/{job_name}"
    return f"""set -u
OUT={out}
exec > >(tee -a "$OUT/run.log") 2>&1
echo "JOB_START $(date -u +%FT%TZ) host=$(hostname)"
python3 -c "import cv2, PIL, pyarrow, baidubce.protocol, tqdm, requests" 2>/dev/null || \\
    pip install -q opencv-python-headless pillow pyarrow baidubce tqdm requests
# 容器内 baidubce 可能是 py2 坏版本: 强制用 PFS 上 vendored 的可用版本
export PYTHONPATH=/home/jovyan/pfs/users/zhanqian.wu/pfs_data/pylibs
python3 -c "import baidubce.protocol" || {{ echo "baidubce broken even with PYTHONPATH"; exit 3; }}
cd {TOOLS}
python3 url_dataset_image_filter.py \\
    --dataset {dataset} \\
    --src-dir {src} \\
    --staging /dev/shm/url_filter/{dataset}_s{shard_idx} \\
    --state-path "$OUT/state/{dataset}_s{shard_idx}_state.json" \\
    --bos-prefix {BOS_PREFIX} \\
    --cred-file {BOS_CRED} \\
    --threads {THREADS} \\
    --num-shards {num_shards} \\
    --shard-idx {shard_idx} \\
    --part-size 1000
rc=$?
printf "%s\\n" "$rc" > "$OUT/exit_code"
echo "JOB_END exit_code=$rc"
exit "$rc"
"""


def make_request(dataset: str, job_name: str, shard_idx: int, num_shards: int) -> dict:
    r = json.loads(Path(f"{SKILL}/assets/request.json").read_text())
    r["name"] = job_name
    r["priority"] = "high"
    # 纯 CPU 任务: 直接删除 GPU 资源项 (quantity=0 会被 endpoint 拒绝)
    r["jobSpec"]["resources"] = [x for x in r["jobSpec"]["resources"]
                                 if x["name"] != "baidu.com/h800_80g_cgpu"]
    r["jobSpec"]["resources"][0]["quantity"] = 16   # cpu
    r["jobSpec"]["resources"][1]["quantity"] = 64   # memory GB
    r["jobSpec"]["command"] = "bash -lc " + shlex.quote(
        build_cmd(dataset, SRC[dataset], job_name, shard_idx, num_shards))
    return r


def main() -> int:
    dry = "--dry-run" in sys.argv
    # --only <dataset> [dataset2 ...] 只提交指定数据集 (默认全部)
    if "--only" in sys.argv:
        only = [x for x in sys.argv[sys.argv.index("--only") + 1:] if x in SHARDS]
    else:
        only = list(SHARDS)
    for dataset in only:
        n_shards = SHARDS[dataset]
        for shard in range(n_shards):
            spec_name = f"url-filter-{dataset}-s{shard}-{TS}"
            work = Path(f"{RUNS}/{spec_name}")
            work.mkdir(parents=True, exist_ok=True)
            # 分片继承旧任务断点 (若存在 {dataset}_s{shard}_seed_state.json), 避免重扫
            seed = Path(f"{RUNS}/{dataset}_s{shard}_seed_state.json")
            if seed.exists():
                (work / "state").mkdir(parents=True, exist_ok=True)
                (work / "state" / f"{dataset}_s{shard}_state.json").write_text(seed.read_text())
                print(f"[{spec_name}] seeded state from {seed}")
            req_path = work / "request.json"
            req_path.write_text(json.dumps(
                make_request(dataset, spec_name, shard, n_shards), ensure_ascii=False, indent=2))
            dup = subprocess.run(
                ["python3", f"{SKILL}/scripts/aihc.py", "--credentials", CRED,
                 "find", "--name", spec_name], capture_output=True, text=True)
            if "1" in dup.stdout and not dry:
                print(f"[{spec_name}] 同名任务可能已存在, 跳过提交")
                continue
            if dry:
                print(f"[{spec_name}] dry-run ok")
                continue
            sub = subprocess.run(
                ["python3", f"{SKILL}/scripts/aihc.py", "--credentials", CRED,
                 "submit", "--request", str(req_path), "--output-dir", str(work)],
                capture_output=True, text=True)
            ok = '"submitted": true' in sub.stdout
            job_id = ""
            if ok:
                job_id = json.loads(sub.stdout).get("jobId", "")
            print(f"[{spec_name}] submit rc={sub.returncode} ok={ok} job={job_id} "
                  + ("" if ok else sub.stdout.strip()[:200] + sub.stderr.strip()[:200]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
