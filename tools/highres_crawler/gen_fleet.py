import json, os, shlex
from pathlib import Path

SKILL = "/home/jovyan/pfs/users/boyuan.wang/misc/aihc/yisu-submit"
TS = os.environ["FLEET_TS"]
N = int(os.environ.get("FLEET_N", "8"))
SUFFIX = os.environ.get("FLEET_SUFFIX", "")
PAGES = os.environ.get("FLEET_PAGES", "60")
SECS = os.environ.get("FLEET_SECONDS", "0")   # 0=不限时, 爬完输入自然退出
BASE = "/home/jovyan/pfs/users/zhanqian.wu/exp/highres_crawl"
TOOL = "/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools/highres_crawler"

tpl = json.loads(Path(f"{SKILL}/assets/request.json").read_text())
tpl["jobSpec"]["resources"] = [x for x in tpl["jobSpec"]["resources"] if "h800" not in x["name"]]
tpl["jobSpec"]["resources"][0]["quantity"] = 16   # cpu
tpl["jobSpec"]["resources"][1]["quantity"] = 64   # mem GB

names = []
for i in range(N):
    r = json.loads(json.dumps(tpl))
    name = f"highres-fleet-{TS}{SUFFIX}-s{i}"
    out = f"{BASE}/fleet-{TS}-s{i}"
    names.append(name)
    r["name"] = name
    cmd = f"""set -u
OUT={out}
exec > >(tee -a "$OUT/run.log") 2>&1
echo "JOB_START $(date -u +%FT%TZ) host=$(hostname)"
cd {TOOL}
python3 -c "import PIL" 2>/dev/null || pip install pillow -q || true
export PYTHONPATH={TOOL}/pylibs:${{PYTHONPATH:-}}
python3 -u crawl_highres.py --mode both \\
  --urls "$OUT/inputs/urls.txt" --sites "$OUT/inputs/sites.txt" \\
  --out "$OUT" --seconds {SECS} --pages-per-site {PAGES} \
  --wm-model {TOOL}/wm_model.onnx \\
  --bos-prefix users/zhanqian.wu/highres_images/fleet-{TS}-s{i}
rc=$?
printf "%s\\n" "$rc" > "$OUT/exit_code"
echo "JOB_END exit_code=$rc"
exit "$rc"
"""
    r["jobSpec"]["command"] = "bash -lc " + shlex.quote(cmd)
    d = Path(f"/tmp/fleet_{TS}/s{i}"); d.mkdir(parents=True, exist_ok=True)
    (d / "request.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print("生成", name, "->", d / "request.json")
print("FLEET_NAMES=" + ",".join(names))
