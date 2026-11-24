"""Run eight CPU workers over one bounded source and record end-to-end throughput."""
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
if args.workers<1:p.error('workers must be positive')
work=Path(args.work)
root=Path('/home/jovyan/pfs/users/zhanqian.wu/code/model_exp/tools/video_scene_bos')
start=time.monotonic();workers=[]
for i in range(args.workers):
 log=(work/f'worker_{i}.log').open('w')
 cmd=[sys.executable,'-u',str(work/'time_frames_bos.py'),'--source',str(work/'sample1000.jsonl'),
      '--credentials','/home/jovyan/pfs/users/zhanqian.wu/.private_aihc/bos_credentials.json','--num-shards',str(args.workers),'--shard-index',str(i),
      '--cpu-threads','1','--work-dir',str(work/'checkpoints')]
 workers.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
while any(proc.poll() is None for proc,_ in workers):
 completed=sum(1 for log in work.glob('worker_*.log') for line in log.read_text(errors='replace').splitlines() if line.startswith('{"row":'))
 state={'elapsed_seconds':round(time.monotonic()-start,2),'attempted_videos':completed,'active_workers':sum(proc.poll() is None for proc,_ in workers)}
 (work/'progress.json').write_text(json.dumps(state)+'\n')
 print(json.dumps(state),flush=True)
 time.sleep(10)
codes=[]
for proc,log in workers:codes.append(proc.wait());log.close()
elapsed=time.monotonic()-start
summaries=[]
for log in sorted(work.glob('worker_*.log')):
 for line in reversed(log.read_text(errors='replace').splitlines()):
  try:r=json.loads(line)
  except ValueError:continue
  if 'work_dir' in r:summaries.append(r);break
report={'elapsed_seconds':round(elapsed,2),'worker_exit_codes':codes,'worker_summaries':len(summaries),'cpus':16,'workers':args.workers,'gpu_requested':0,'python_version':sys.version.split()[0]}
for key in ['processed','failed','skipped','kept','duplicates','candidates','rejected_frames']:
 report[key]=sum(x.get(key,0) for x in summaries)
report['videos_per_second']=round(report['processed']/elapsed,3)
report['images_per_second']=round(report['kept']/elapsed,3)
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
report['child_cpu_seconds']=round(usage.ru_utime+usage.ru_stime,2)
report['average_cpu_cores_used']=round(report['child_cpu_seconds']/elapsed,2)
report['estimated_full_4524495_videos_hours']=round(4524495/max(report['videos_per_second'],1e-9)/3600,2)
(work/'benchmark_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
sys.exit(int(any(codes) or len(summaries)!=args.workers))
