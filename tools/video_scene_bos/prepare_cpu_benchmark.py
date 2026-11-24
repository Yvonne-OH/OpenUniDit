"""Prepare one immutable, bounded AIHC CPU benchmark; does not submit."""
import datetime,json,os,shlex,shutil
from pathlib import Path
ROOT=Path('/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools/video_scene_bos')
SKILL=Path('/home/jovyan/pfs/users/boyuan.wang/misc/aihc/yisu-submit')
name='video-cpu-high-'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
work=ROOT/'cluster_runs'/name
work.mkdir(parents=True,exist_ok=False)
private=Path('/home/jovyan/pfs/users/zhanqian.wu/.private_aihc');private.mkdir(mode=0o700,exist_ok=True);private.chmod(0o700)
cred=json.loads((ROOT/'.bos_credentials.json').read_text())
fd=os.open(private/'aihc_credentials.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600) if not (private/'aihc_credentials.json').exists() else None
if fd is not None:
 with os.fdopen(fd,'w') as f:json.dump({'accesskey':cred['ak'],'secretkey':cred['sk']},f)
bosfd=os.open(private/'bos_credentials.json',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
with os.fdopen(bosfd,'w') as f:json.dump(cred,f)
shutil.copyfile(ROOT/'time_frames_bos.py',work/'time_frames_bos.py')
shutil.copyfile(ROOT/'cpu_benchmark.py',work/'cpu_benchmark.py')
# Fixed bounded source prepared outside the timed worker phase.
import itertools,pyarrow.parquet as pq
source=Path('/mnt/nfs/users/boyuan.wang/yisu_data_lake/manifests/train/boyuan.wang/20261013_all_2_10s_captioned_usable_delta/manifest.parquet')
rows=list(itertools.islice((r for b in pq.ParquetFile(source).iter_batches(batch_size=128) for r in b.to_pylist()),1000))
(work/'sample1000.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
cmd='\n'.join(['set -u','OUT='+shlex.quote(str(work)),
 'exec > >(tee -a "$OUT/run.log") 2>&1','echo "JOB_START $(date -u +%FT%TZ) host=$(hostname)"',
 'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
 'python3 -c "import cv2,numpy,pyarrow,baidubce" || python3 -m pip install --target "$OUT/deps" "numpy<2" "opencv-python-headless==4.10.0.84" pyarrow bce-python-sdk',
 'export PYTHONPATH="$OUT/deps:${PYTHONPATH:-}"',
 'python3 -u "$OUT/cpu_benchmark.py" --work "$OUT"','rc=$?',
 'printf "%s\\n" "$rc" > "$OUT/exit_code"','echo "JOB_END exit_code=$rc"','exit "$rc"'])
r=json.loads((SKILL/'assets/request.json').read_text())
r['name']=name;r['priority']='high'
r['jobSpec']['resources']=[{'name':'cpu','quantity':16},{'name':'memory','quantity':64},{'name':'sharedMemory','quantity':10}]
r['jobSpec']['command']='bash -lc '+shlex.quote(cmd)
r['datasources'].append({'type':'hostpath','sourcePath':'/mnt/nfs','mountPath':'/mnt/nfs','name':'nfs','options':{'readOnly':True}})
(work/'request.json').write_text(json.dumps(r,indent=2)+'\n')
(ROOT/'latest_cpu_benchmark.json').write_text(json.dumps({'work':str(work),'name':name})+'\n')
print(json.dumps({'work':str(work),'name':name,'priority':'high','cpus':16,'memory_gb':64,'gpus':0,'videos':len(rows)}))
