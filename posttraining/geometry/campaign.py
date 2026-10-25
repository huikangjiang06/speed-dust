"""Resumable orchestration for geometry preparation and matched training runs."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..configuration import load_config, save_config


METHODS=('backprop','dust','geometry_dust')


def selected_gpus(value=None):
    selected=value or os.environ.get('DUST_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES')
    if selected:
        devices=[device.strip() for device in selected.split(',') if device.strip()]
    else:
        try:
            output=subprocess.check_output(
                ['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True)
            devices=[line.strip() for line in output.splitlines() if line.strip()]
        except (OSError,subprocess.CalledProcessError) as exc:
            raise RuntimeError('Select devices with --gpus or set CUDA_VISIBLE_DEVICES') from exc
    if len(devices)!=len(set(devices)) or not devices:
        raise ValueError('Provide at least one unique visible GPU ID')
    return devices


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def environment(gpu,root):
    env=os.environ.copy()
    cache=root/'cache'
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HOME=env.get('HF_HOME',str(root/'cache/huggingface')),
        HF_DATASETS_CACHE=env.get('HF_DATASETS_CACHE',str(cache/'datasets')),
        XDG_CACHE_HOME=env.get('XDG_CACHE_HOME',str(cache)),
        TMPDIR=env.get('TMPDIR',str(root/'tmp')),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    return env


def invoke(root,name,module,args,gpu=None,wait=True):
    gpu=gpu or selected_gpus()[0]
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m',module,*map(str,args)]
    with (logs/(name+'.log')).open('a') as log:
        process=subprocess.Popen(command,env=environment(gpu,root),stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=not wait)
    record=dict(pid=process.pid,gpu=gpu,command=command,cwd=str(Path.cwd()),started_at=stamp(),log=str(logs/(name+'.log')))
    write_json(root/'processes'/(name+'.json'),record)
    if wait:
        code=process.wait();record.update(exit_code=code,finished_at=stamp())
        write_json(root/'processes'/(name+'.json'),record)
        if code:
            raise RuntimeError(f'{name} failed ({code}); see {record["log"]}')
    return process


def offline(base,root,smoke=False,gpus=None):
    gpus=gpus or selected_gpus()
    root.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(base);cfg.training.device='cuda:0';cfg.geometry.enabled=False;cfg.geometry.bank_path=None
    cfg.data.prepared_path=str(root/'data')
    cfg.dust.noise.scales_file=None
    if smoke:
        cfg.data.train_examples=4;cfg.data.validation_examples=4;cfg.data.test_examples=4
        cfg.training.max_steps=2;cfg.training.eval_interval=1;cfg.training.checkpoint_interval=1
        cfg.training.save_pretrained=False
        cfg.geometry.max_iter=3;cfg.geometry.calibration_tokens=256
        cfg.geometry.cache_shard_rows=512
        cfg.description='Real SmolLM2 smoke only: four conversations, three full-data EM iterations, two training updates. Not production geometry.'
    config_path=root/'base.json';save_config(config_path,cfg)
    if not (root/'data/manifest.json').exists():
        invoke(root,'prepare_data','posttraining.sft_data',['--config',config_path],gpu=gpus[0])
    if not (root/'baseline_scales/scales.json').exists():
        invoke(root,'baseline_scales','posttraining.calibrate',['--config',config_path,'--stage','scales',
            '--output',root/'baseline_scales','--batches','4'],gpu=gpus[0])
    cfg.dust.noise.scales_file=str(root/'baseline_scales/scales.json');save_config(config_path,cfg)
    args=['--config',config_path,'--output',root/'activation_cache']
    if (root/'activation_cache/manifest.json').exists():
        args.append('--resume')
    invoke(root,'cache_activations','posttraining.geometry.cache',args,gpu=gpus[0])
    fitting_started=time.monotonic();processes=[]
    for worker,gpu in enumerate(gpus):
        processes.append(invoke(root,f'fit_worker_{worker}','posttraining.geometry.gmm',
            ['--config',config_path,'--cache',root/'activation_cache','--output',root/'fits',
             '--worker',worker,'--workers',len(gpus)],gpu=gpu,wait=False))
    errors=[]
    for i,process in enumerate(processes):
        code=process.wait()
        path=root/'processes'/f'fit_worker_{i}.json';record=json.loads(path.read_text())
        record.update(exit_code=code,finished_at=stamp());write_json(path,record)
        if code:
            errors.append(i)
    write_json(root/'fitting_wall.json',dict(wall_seconds=time.monotonic()-fitting_started,workers=len(gpus),failures=errors))
    if errors:
        raise RuntimeError(f'Fitting workers failed: {errors}')
    invoke(root,'calibrate_gmm','posttraining.geometry.calibrate',['--config',config_path,
        '--cache',root/'activation_cache','--fits',root/'fits','--output',root/'bank'],gpu=gpus[0])
    for method in METHODS:
        selected=copy.deepcopy(cfg)
        selected.geometry.enabled=method=='geometry_dust'
        selected.geometry.bank_path=str(root/'bank/manifest.json') if selected.geometry.enabled else None
        save_config(root/(method+'.json'),selected)
    write_json(root/'offline_complete.json',dict(complete=True,smoke_only=smoke,finished_at=stamp(),config=str(config_path)))
    return cfg


def launch(root,smoke=False,gpus=None):
    gpus=gpus or selected_gpus()
    if len(gpus)<len(METHODS):
        raise ValueError(f'Launching the three matched methods requires at least {len(METHODS)} GPUs')
    if not (root/'offline_complete.json').exists():
        raise ValueError('Complete and verify the offline stages before launching training')
    processes=[]
    for method,gpu in zip(METHODS,gpus):
        output=root/'runs'/method
        if (output/'result.json').exists():
            continue
        # Reusing this command may resume a stopped run, never duplicate a live PID.
        record_path=root/'processes'/('train_'+method+'.json')
        if record_path.exists():
            record=json.loads(record_path.read_text())
            proc=Path('/proc')/str(record['pid'])/'cmdline'
            if proc.exists() and str(output).encode() in proc.read_bytes():
                raise RuntimeError(f'{method} is already running as PID {record["pid"]}')
        args=['--config',root/(method+'.json'),'--method',method,'--output',output]
        if (output/'last.pt').exists():
            args.append('--resume')
        if smoke and method=='geometry_dust':
            args+=['--stop-after',1]
        processes.append((method,invoke(root,'train_'+method,'posttraining.sft',args,gpu=gpu,wait=False)))
    write_json(root/'launch.json',dict(started_at=stamp(),methods=list(METHODS),smoke_only=smoke,monitor_interval_seconds=3600))
    if smoke:
        for method,process in processes:
            code=process.wait()
            if code:
                raise RuntimeError(f'Real smoke failed: {method}; see training log')
        invoke(root,'resume_geometry_dust','posttraining.sft',['--config',root/'geometry_dust.json',
            '--method','geometry_dust','--output',root/'runs/geometry_dust','--resume'],gpu=gpus[2])
        for method in METHODS:
            result=json.loads((root/'runs'/method/'result.json').read_text())
            if not result['complete'] or result['step']!=2:
                raise AssertionError('Incomplete real-model smoke run')
        write_json(root/'smoke_complete.json',dict(complete=True,finished_at=stamp(),methods=list(METHODS),
            geometry_resume_exercised=True,scope='four-conversation engineering fixture; not full-data experiment'))


def check(root):
    report=dict(checked_at=stamp(),methods={})
    for method in METHODS:
        output=root/'runs'/method;entry={}
        for filename in ('progress','result'):
            path=output/(filename+'.json')
            if path.exists():
                value=json.loads(path.read_text())
                entry[filename]={key:item for key,item in value.items() if key not in ('history','geometry')}
        process_path=root/'processes'/('train_'+method+'.json')
        if process_path.exists():
            process=json.loads(process_path.read_text());entry['pid']=process['pid'];entry['gpu']=process['gpu']
            proc=Path('/proc')/str(process['pid'])/'cmdline'
            entry['alive']=proc.exists() and str(output).encode() in proc.read_bytes()
        report['methods'][method]=entry
    write_json(root/'status.json',report)
    with (root/'hourly_checks.jsonl').open('a') as out:
        out.write(json.dumps(report)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','offline','launch','check','watch'])
    parser.add_argument('--config');parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpus',help='Comma-separated GPU IDs; defaults to DUST_GPUS, CUDA_VISIBLE_DEVICES, or nvidia-smi.')
    args=parser.parse_args();root=args.root.resolve()
    gpus=selected_gpus(args.gpus) if args.stage in ('offline','smoke','launch') else None
    if args.stage in ('offline','smoke'):
        if not args.config:
            parser.error('--config is required for offline/smoke')
        offline(load_config(args.config),root,smoke=args.stage=='smoke',gpus=gpus)
    if args.stage in ('launch','smoke'):
        launch(root,smoke=args.stage=='smoke',gpus=gpus)
    if args.stage=='check':
        check(root)
    if args.stage=='watch':
        # Durable hourly observations; never changes processes or hyperparameters.
        while True:
            time.sleep(3600)
            report=check(root)
            if all(entry.get('result',{}).get('complete') for entry in report['methods'].values()):
                break


if __name__=='__main__':
    main()
