"""Read-only campaign health check; save a dated numeric monitoring record."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root
    state=json.loads((root/'campaign.json').read_text())
    now=time.time();entries=[];alerts=[]
    alive=psutil.pid_exists(state['pid'])
    if state['state']=='running' and not alive: alerts.append('Campaign supervisor is not alive')
    if state['state']=='running' and now-state['updated']>120: alerts.append('Campaign state is stale')
    for job in state['jobs']:
        entry={k:job.get(k) for k in ['id','state','attempts','devices','pid','exit_code']}
        progress=root/'runs'/job['id']/'progress.json'
        if progress.exists():
            data=json.loads(progress.read_text())
            entry.update(step=data['step'],total_steps=data['total_steps'],best_val=data.get('best_val'),
                         progress_age_seconds=now-progress.stat().st_mtime)
        if job['state']=='running':
            entry['process_alive']=psutil.pid_exists(job['pid'])
            if not entry['process_alive']: alerts.append(f"{job['id']}: worker PID missing")
            if entry.get('progress_age_seconds',0)>1800: alerts.append(f"{job['id']}: no progress for 30 minutes")
        if job['state']=='failed': alerts.append(f"{job['id']}: failed; inspect {job['log']}")
        entries.append(entry)
    devices=sorted({str(gpu) for job in state['jobs'] for gpu in job.get('devices',[])})
    usage=(subprocess.check_output(['nvidia-smi','--id='+','.join(devices),
        '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
        if devices else 'No GPUs assigned yet')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result=dict(checked_utc=stamp,state=state['state'],complete=state['complete'],failed=state['failed'],
                supervisor_alive=alive,alerts=alerts,gpu_usage=usage,jobs=entries)
    directory=root/'monitor';directory.mkdir(exist_ok=True)
    (directory/f'{stamp}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'},indent=2))
    for entry in entries:
        if entry['state'] in ['running','failed']: print(json.dumps(entry))


if __name__=='__main__': main()
