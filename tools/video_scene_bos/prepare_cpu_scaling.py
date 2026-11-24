import datetime,json,shlex,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
baseline=root/'cluster_runs/video-cpu-high-20261124-171255'
name='video-cpu-scale-'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
work=root/'cluster_runs'/name;work.mkdir(exist_ok=False)
for file in ['time_frames_bos.py','cpu_benchmark.py','cpu_scaling.py']:shutil.copyfile(root/file,work/file)
r=json.loads((baseline/'request.json').read_text());r['name']=name
cmd='\n'.join(['set -u','OUT='+shlex.quote(str(work)), 'exec > >(tee -a "$OUT/run.log") 2>&1',
 'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
 'export PYTHONPATH='+shlex.quote(str(baseline/'deps'))+':${PYTHONPATH:-}',
 'python3 -u "$OUT/cpu_scaling.py"','rc=$?','printf "%s\\n" "$rc" > "$OUT/exit_code"','exit "$rc"'])
r['jobSpec']['command']='bash -lc '+shlex.quote(cmd)
(work/'request.json').write_text(json.dumps(r,indent=2)+'\n')
(root/'latest_cpu_scaling.json').write_text(json.dumps({'name':name,'work':str(work)})+'\n')
print(json.dumps({'name':name,'work':str(work),'cpu':16,'gpu':0,'priority':r['priority']}))
