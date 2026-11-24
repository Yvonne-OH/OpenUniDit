import json,shutil,subprocess,sys
from pathlib import Path
work=Path(__file__).resolve().parent
baseline=Path('/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools/video_scene_bos/cluster_runs/video-cpu-high-20261124-171255')
results=[]
for workers in [16,32]:
 phase=work/f'w{workers}';phase.mkdir(exist_ok=False)
 for name in ['time_frames_bos.py','cpu_benchmark.py']:shutil.copyfile(work/name,phase/name)
 shutil.copyfile(baseline/'sample1000.jsonl',phase/'sample1000.jsonl')
 rc=subprocess.call([sys.executable,'-u',str(phase/'cpu_benchmark.py'),'--work',str(phase),'--workers',str(workers)])
 r=json.loads((phase/'benchmark_report.json').read_text())
 assert rc==0 and r['processed']==1000 and r['kept']==1726 and r['failed']==0, r
 r['speedup_vs_8_workers']=round(r['videos_per_second']/3.446,3)
 results.append(r)
 (work/'scaling_report.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'results':results}),flush=True)
