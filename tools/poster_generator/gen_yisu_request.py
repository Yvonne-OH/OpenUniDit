# -*- coding: utf-8 -*-
"""生成 yisu 8 卡海报合成任务的 request.json (参数走环境变量, 别在 heredoc 里拼)。"""
import json
import os
import shlex
from pathlib import Path

SKILL = os.environ["SKILL"]
JOB_NAME = os.environ["JOB_NAME"]
RUN_NAME = os.environ["RUN_NAME"]
NUM_REQUESTS = os.environ.get("NUM_REQUESTS", "20000")
SHARD_SIZE = os.environ.get("SHARD_SIZE", "500")
MAX_HOURS = os.environ.get("MAX_HOURS", "")  # 空则不限时
RESUME = os.environ.get("RESUME", "1") == "1"

TOOL_DIR = "/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools/poster_generator"
PY = "/home/jovyan/pfs/users/zhanqian.wu/env/gwp/bin/python"
OUT_DIR = f"/home/jovyan/pfs/users/zhanqian.wu/pfs_data/poster_ads/{RUN_NAME}"
ENV_SH = "/home/jovyan/pfs/users/zhanqian.wu/code/env/gwp.sh"

r = json.loads(Path(f"{SKILL}/assets/request.json").read_text())
r["name"] = JOB_NAME
r["priority"] = "high"
r["jobSpec"]["resources"][0]["quantity"] = 8   # 8 x H800
r["jobSpec"]["resources"][1]["quantity"] = 16  # cpu
r["jobSpec"]["resources"][2]["quantity"] = 64  # memory GB
# Qwen3-VL ckpt 是指向 /mnt/nfs 的软链, 需要挂 nfs
r["datasources"].append({"type": "hostpath", "sourcePath": "/mnt/nfs",
                         "mountPath": "/mnt/nfs", "name": "nfs",
                         "options": {"readOnly": False}})

extra = ""
if RESUME:
    extra += " --resume"
if MAX_HOURS:
    extra += f" --max-hours {MAX_HOURS}"

cmd = f"""set -u
OUT={OUT_DIR}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "JOB_START $(date -u +%FT%TZ) host=$(hostname)"
source {ENV_SH}
export HF_ENDPOINT=https://hf-mirror.com
cd {TOOL_DIR}
{PY} poster_generator.py \\
    --num-requests {NUM_REQUESTS} --split train --num-posters 2 \\
    --run-name {RUN_NAME} --shard-size {SHARD_SIZE}{extra}
rc=$?
printf "%s\\n" "$rc" > "$OUT/exit_code"
echo "JOB_END exit_code=$rc"
exit "$rc"
"""
r["jobSpec"]["command"] = "bash -lc " + shlex.quote(cmd)

out = Path(os.environ["WORK"]) / "request.json"
out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
print("written", out)
print("job:", JOB_NAME, "priority:", r["priority"], "gpu:", r["jobSpec"]["resources"][0]["quantity"])
