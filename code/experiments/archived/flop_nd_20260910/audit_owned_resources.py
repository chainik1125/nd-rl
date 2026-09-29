"""Read-only process, GPU and lease snapshot for this research sprint."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import common

root=common.ROOT
own_pid=os.getpid()
research_words=('train_lm','symbolic_general_train','symbolic_general_bc','evaluate_lm','symbolic_general_eval',
 'run_pipeline','remaining_work','watch','audit_main','audit_online','analyze_lm','final_analysis','final_comparators')
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==own_pid:continue
    try:
        parts=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
        scripts=[Path(s) for s in parts[:5] if '\n' not in s and s.endswith('.py')]
        if not scripts:continue
        cwd=(p/'cwd').resolve()
        ours=cwd==root or any(s.is_absolute() and root in s.parents for s in scripts)
        if ours and any(any(w in s.name for w in research_words) for s in scripts):
            processes.append({'pid':int(p.name),'scripts':[str(s) for s in scripts],'cwd':str(cwd)})
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
query=subprocess.run(['nvidia-smi','-i','3','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True,capture_output=True,timeout=15)
if query.returncode:raise RuntimeError(query.stderr)
gpu_rows=[]
for line in query.stdout.splitlines():
    if not line.strip():continue
    pid,uuid,memory=[s.strip() for s in line.split(',')]
    gpu_rows.append({'pid':int(pid),'gpu_uuid':uuid,'used_memory':memory})
locks_text=Path('/proc/locks').read_text().splitlines()
lease_paths=[Path('/tmp/olmo_single_gpu_3.lock'),root/'leases/gpu_3.lock',
 Path('/data/users/dmitry/nd-rl/proof-fragments-20260910-1800/leases/gpu_3.lock'),Path('/tmp/crra_sse_gpu_3.lock')]
leases=[]
for p in lease_paths:
    if not p.exists():continue
    stat=p.stat();matching=[]
    for line in locks_text:
        fields=line.split()
        if len(fields)<6:continue
        dev_inode=fields[5].split(':')
        if len(dev_inode)!=3:continue
        try:same=int(dev_inode[0],16)==os.major(stat.st_dev) and int(dev_inode[1],16)==os.minor(stat.st_dev) and int(dev_inode[2])==stat.st_ino
        except ValueError:continue
        if same:matching.append({'type':fields[1],'mode':fields[3],'pid':fields[4]})
    leases.append({'path':str(p),'active_kernel_locks':matching})
job_states=[]
for p in sorted((root/'jobs').glob('*/completed.json')):
    d=json.loads(p.read_text());job_states.append({'job':p.parent.name,'status':d['status'],'exit_code':d['exit_code'],'completed_utc':d['utc']})
receipt={'status':'passed' if not processes else 'owned_research_processes_remain',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Only this sprint-owned training/evaluation/controller scripts, local GPU 3 and its shared lease files. Read-only; no processes were stopped or locks acquired.',
 'owned_research_processes':processes,'gpu3_compute_processes':gpu_rows,'lease_files':leases,'gpu_job_completions':job_states,
 'code_sha256':common.sha(__file__),'new_paid_spending_usd':0}
common.write_json(root/'reports/owned_resources_v1.json',receipt)
print(json.dumps({'status':receipt['status'],'owned_research_processes':len(processes),'gpu3_compute_processes':len(gpu_rows),'active_lease_locks':sum(len(r['active_kernel_locks']) for r in leases),'utc':receipt['utc']}))
