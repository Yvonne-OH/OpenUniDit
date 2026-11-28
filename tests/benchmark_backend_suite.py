"""Two-node suite orchestrator: bounded, sequential, and failure-isolated cases.
Run once per node under the platform-injected node RANK/WORLD_SIZE.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def atomic_json(path, value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def wait_files(paths, seconds):
    end=time.monotonic()+seconds
    while not all(p.exists() for p in paths):
        if time.monotonic()>end:raise TimeoutError('Missing node rendezvous files: '+str([str(p) for p in paths if not p.exists()]))
        time.sleep(1)


def kill_group(pid):
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:return
    time.sleep(3)
    try:os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text());out=Path(manifest['output']);out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK',os.environ.get('NODE_RANK','0')))
    nodes=int(os.environ.get('WORLD_SIZE',os.environ.get('NNODES','1')))
    if nodes!=2 or rank not in (0,1):raise ValueError(f'Requires two platform nodes, got nodes={nodes}, node_rank={rank}')
    os.environ.update(LOCAL_WORLD_SIZE='8',GIGA_CODE_REVISION=manifest['code_revision'])
    startup={'node_rank':rank,'hostname':socket.gethostname(),'master':os.environ.get('MASTER_ADDR'),
             'code_revision':manifest['code_revision'],'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)}
    if len(startup['gpu_inventory'].strip().splitlines())!=8:raise ValueError('Expected eight visible GPUs per node')
    atomic_json(out/f'node-{rank}.json',startup)
    (out/f'topology-node-{rank}.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
    summaries=[]
    for index,case in enumerate(manifest['cases']):
        directory=out/case['name'];directory.mkdir(exist_ok=True)
        atomic_json(directory/f'ready-{rank}.json',{'host':socket.gethostname(),'time':time.time()})
        wait_files([directory/f'ready-{r}.json' for r in range(nodes)],600)
        base=[sys.executable,'-m','torch.distributed.run','--nnodes','2','--nproc-per-node','8','--node_rank',str(rank),
              '--master_addr',os.environ['MASTER_ADDR'],'--master_port',str(28600+index),'--max_restarts','0']
        command=base+case['argv']
        log=directory/f'node-{rank}.log';started=time.time();timed_out=False
        with log.open('w') as handle:
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=process.wait(timeout=case.get('timeout_seconds',900))
            except subprocess.TimeoutExpired:
                timed_out=True;kill_group(process.pid);rc=process.wait(timeout=30)
        # Only signal descendants in this case's private process group.
        kill_group(process.pid)
        for _ in range(30):
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not active:break
            time.sleep(1)
        tail=log.read_text(errors='replace')[-20000:]
        result={'node_rank':rank,'returncode':rc,'timed_out':timed_out,'elapsed_seconds':time.time()-started,
                'oom':'out of memory' in tail.lower(),'gpu_processes_remaining':active}
        atomic_json(directory/f'done-{rank}.json',result)
        wait_files([directory/f'done-{r}.json' for r in range(nodes)],case.get('timeout_seconds',900)+120)
        rows=[json.loads((directory/f'done-{r}.json').read_text()) for r in range(nodes)]
        okay=all(r['returncode']==0 and not r['gpu_processes_remaining'] for r in rows)
        benchmark=directory/'result.json'
        if benchmark.exists():
            b=json.loads(benchmark.read_text());okay=okay and bool(b.get('finite',True))
        entry={'case':case['name'],'ok':okay,'nodes':rows,'result':str(benchmark) if benchmark.exists() else None}
        summaries.append(entry)
        if rank==0:
            atomic_json(out/'suite-progress.json',{'completed':len(summaries),'total':len(manifest['cases']),'cases':summaries})
            print('SUITE_CASE',json.dumps(entry),flush=True)
        if any(r['gpu_processes_remaining'] for r in rows):raise RuntimeError('Previous case left GPU processes; refusing overlap')
        if case.get('required') and not okay:raise RuntimeError('Required correctness preflight failed')
    atomic_json(out/f'suite-done-{rank}.json',{'completed':len(summaries),'time':time.time()})
    if rank==0:atomic_json(out/'suite-result.json',{'completed':len(summaries),'cases':summaries,'all_cases_passed':all(c['ok'] for c in summaries)})
if __name__=='__main__':main()
