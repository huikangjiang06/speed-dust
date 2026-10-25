"""Run the released 30-cell seeded campaign on selected CUDA devices."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from .train import atomic_json


def jobs():
    result = []
    for population in [None, 256, 1024, 4096, 16384]:
        for tokens in ['1M', '10M']:
            for seed in [42, 43, 44]:
                method = 'backprop' if population is None else 'dust'
                name = f'{method}-{tokens}' + (f'-p{population}' if population else '') + f'-s{seed}'
                result.append(dict(id=name, method=method, tokens=tokens, seed=seed,
                                   population=population, gpus=4 if population in [4096,16384] else 1))
    return result


def selected_gpus(value=None):
    selected = value or os.environ.get('DUST_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES')
    if selected:
        devices = [device.strip() for device in selected.split(',') if device.strip()]
    else:
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
            devices = [line.strip() for line in output.splitlines() if line.strip()]
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('Select devices with --gpus or set CUDA_VISIBLE_DEVICES') from exc
    if len(devices) != len(set(devices)) or len(devices) < 4:
        raise ValueError('The released multi-GPU configurations require four unique visible GPU IDs')
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', help='Comma-separated GPU IDs; defaults to DUST_GPUS, CUDA_VISIBLE_DEVICES, or nvidia-smi.')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    root = args.root.resolve()
    gpus = selected_gpus(args.gpus)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / 'campaign.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    for entry in old.get('jobs', []):
        pid = entry.get('pid')
        if entry.get('state') == 'running' and pid:
            try:
                p = psutil.Process(pid)
                if abs(p.create_time() - entry.get('process_created',0)) < 1 and p.status() != psutil.STATUS_ZOMBIE:
                    raise RuntimeError(f'Previous worker {pid} still running; do not launch a duplicate')
            except psutil.NoSuchProcess:
                pass
    plan = jobs()
    previous = {j['id']:j for j in old.get('jobs',[])}
    for j in plan:
        j.update(state='pending', attempts=previous.get(j['id'],{}).get('attempts',0))
        result_path = root / 'runs' / j['id'] / 'result.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('complete'):
            j['state']='complete'
    src=Path(__file__).resolve().parents[1]
    python=sys.executable
    active={}
    stopping=False

    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
        # These are exclusively the worker process groups created below.
        for proc, handle, entry in active.values():
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass

    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    base_env=os.environ.copy()
    base_env.setdefault('HF_HOME', str(root/'cache/huggingface'))
    base_env.update(TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'),
                    TORCH_HOME=str(root/'cache/torch'), TRITON_CACHE_DIR=str(root/'cache/triton'),
                    CUDA_CACHE_PATH=str(root/'cache/cuda'), PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    for directory in ['tmp','cache/torch','cache/triton','cache/cuda','logs','runs']:
        (root/directory).mkdir(parents=True,exist_ok=True)
    started=time.time()
    print(json.dumps(dict(event='campaign_start',pid=os.getpid(),jobs=len(plan),source=str(src))),flush=True)
    while True:
        for name,(proc,handle,j) in list(active.items()):
            code=proc.poll()
            if code is None: continue
            handle.close()
            result=root/'runs'/name/'result.json'
            complete=result.exists() and json.loads(result.read_text()).get('complete')
            j.update(state='complete' if complete else ('pending' if j['attempts']<2 and not stopping else 'failed'),
                     exit_code=code, finished=time.time())
            del active[name]
            print(json.dumps(dict(event='worker_exit',id=name,code=code,state=j['state'])),flush=True)
        used={gpu for _,_,j in active.values() for gpu in j['devices']}
        free=[gpu for gpu in gpus if gpu not in used]
        if not stopping:
            for j in plan:
                if j['state']!='pending': continue
                if len(free)<j['gpus']: break
                devices=free[:j['gpus']];free=free[j['gpus']:]
                worker=['-m','replication.train','--method',j['method'],'--tokens',j['tokens'],
                        '--seed',str(j['seed']),'--data-dir',str(root/'data'),
                        '--output',str(root/'runs'/j['id'])]
                if j['population']: worker+=['--population',str(j['population'])]
                cmd=([python,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={j["gpus"]}']
                     if j['gpus']>1 else [python])+worker
                env=base_env.copy();env['CUDA_VISIBLE_DEVICES']=','.join(map(str,devices))
                logfile=root/'logs'/(j['id']+'.log')
                handle=logfile.open('a')
                proc=subprocess.Popen(cmd,cwd=src,env=env,stdin=subprocess.DEVNULL,
                                      stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(state='running',devices=devices,pid=proc.pid,started=time.time(),
                         process_created=psutil.Process(proc.pid).create_time(),attempts=j['attempts']+1,
                         command=cmd,log=str(logfile))
                active[j['id']]=(proc,handle,j)
                print(json.dumps(dict(event='worker_start',id=j['id'],pid=proc.pid,devices=devices)),flush=True)
        done=sum(j['state']=='complete' for j in plan)
        failed=sum(j['state']=='failed' for j in plan)
        state='running' if active or any(j['state']=='pending' for j in plan) else ('complete' if done==len(plan) else 'failed')
        if stopping: state='stopping' if active else 'stopped'
        atomic_json(state_path,dict(pid=os.getpid(),source=str(src),started=started,updated=time.time(),
                                    state=state,complete=done,failed=failed,jobs=plan))
        if not active and (stopping or all(j['state'] in ['complete','failed'] for j in plan)):
            break
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
