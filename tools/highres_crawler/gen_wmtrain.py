import json, os, shlex
from pathlib import Path

SKILL = "/home/jovyan/pfs/users/boyuan.wang/misc/aihc/yisu-submit"
TOOL = "/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools/highres_crawler"
OUT = "/home/jovyan/pfs/users/zhanqian.wu/exp/highres_crawl/wm_train"
NAME = "highres-wmtrain-" + os.environ["WM_TS"]

r = json.loads(Path(f"{SKILL}/assets/request.json").read_text())
r["name"] = NAME
r["jobSpec"]["resources"] = [x for x in r["jobSpec"]["resources"] if "h800" not in x["name"]]
r["jobSpec"]["resources"][0]["quantity"] = 16
r["jobSpec"]["resources"][1]["quantity"] = 64

cmd = f"""set -u
exec > >(tee -a {OUT}/run.log) 2>&1
echo "JOB_START $(date -u +%FT%TZ) host=$(hostname)"
mkdir -p {OUT}
cd {TOOL}
export PYTHONPATH={TOOL}/pylibs:${{PYTHONPATH:-}}
python3 -c "import torch" 2>/dev/null || pip install torch torchvision onnxruntime -q || true
python3 -u train_watermark.py
rc=$?
printf "%s\\n" "$rc" > {OUT}/exit_code
echo "JOB_END exit_code=$rc"
exit "$rc"
"""
r["jobSpec"]["command"] = "bash -lc " + shlex.quote(cmd)

d = Path("/tmp/wmtrain_work"); d.mkdir(exist_ok=True)
(d / "request.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
print("生成", NAME, "->", d / "request.json")
