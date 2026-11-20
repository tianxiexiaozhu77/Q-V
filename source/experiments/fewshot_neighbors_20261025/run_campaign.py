#!/usr/bin/env python3
"""Run fixed two-model neighbor comparisons on idle physical GPUs 6 and 7 only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu, release_gpu
sys.path.insert(0,str(HERE))
from prepare_protocol import load_protocol

ENVS={'BiomedCLIP':ROOT/'baselines/protocols/alpharad_binary/BiomedCLIP/local_setup/python_local.sh',
      'AlphaRAD':ROOT/'AlphaRAD/local_setup/python_local.sh'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def env():
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=HERE/'protocol_v1/catalog.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',type=int,default=[6,7])
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if not args.gpus or not set(args.gpus)<={6,7} or len(set(args.gpus))!=len(args.gpus):
        parser.error('Use distinct authorized physical GPU IDs 6 and/or 7')
    args.protocol=args.protocol.resolve(); args.output=args.output.resolve()
    load_protocol(args.protocol,include_test_labels=False)
    sources={str(HERE/name):sha(HERE/name) for name in (
        'run_campaign.py','cache_features.py','evaluate.py','neighbor_methods.py',
        'prepare_protocol.py','summarize.py')}
    for p in [args.protocol,HERE/'EXPERIMENT_PLAN.txt',HERE/'METHODS_PLAN.json',
              ROOT/'experiments/fewshot_native_query/methods.py',
              ROOT/'experiments/fewshot_native_query/simnl_adapter.py',
              ROOT/'experiments/chexchonet_stability_20261024/cluster_bootstrap.py',
              ROOT/'AlphaRAD/local_setup/run_pretrained.py']:
        sources[str(p)]=sha(p)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'logs').mkdir()
    state=dict(status='running',started_at=now(),controller_pid=os.getpid(),gpus=args.gpus,
        protocol=str(args.protocol),sources=sources,smoke_only=args.smoke_only,
        jobs={m:dict(stage='smoke',status='pending') for m in ENVS})
    write(args.output/'FROZEN_RUN_PLAN.json',dict(created_at=now(),sources=sources,
        models=list(ENVS),seeds=list(range(42,47)),tasks=['slvh','dlv','composite_slvh_dlv'],
        method_conditions=18,task_seed_branch_results=540,protocol=str(args.protocol),
        test_labels_opened_for_fitting=False,all_results_retained=True))
    active={}; stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def save():
        state['updated_at']=now();write(args.output/'status.json',state)
    def start(model,stage,cmd,runenv,gpu=None,lease=None):
        log=(args.output/'logs'/f'{model}_{stage}.log').open('w')
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  env=runenv,start_new_session=True)
        except BaseException:
            log.close()
            if lease is not None: release_gpu(lease)
            raise
        active[model]=dict(proc=proc,log=log,gpu=gpu,lease=lease,stage=stage)
        state['jobs'][model].update(status='running',stage=stage,pid=proc.pid,gpu=gpu,command=cmd,stage_started_at=now())
        print('Started',model,stage,'GPU',gpu,'PID',proc.pid,flush=True)
    try:
        while active or any(j['status']=='pending' for j in state['jobs'].values()):
            if stopping: raise InterruptedError('Stopping owned child processes')
            for model,item in list(active.items()):
                code=item['proc'].poll()
                if code is None:continue
                item['log'].close()
                if item['lease'] is not None:release_gpu(item['lease'])
                del active[model]
                stage=item['stage'];job=state['jobs'][model]
                folder=args.output/(model+'_smoke' if stage=='smoke' else model)
                receipt=folder/('evaluation/status.json' if stage=='evaluate' else 'cache/status.json')
                want='smoke_complete' if stage=='smoke' else 'complete'
                ok=code==0 and receipt.is_file() and json.loads(receipt.read_text())['status']==want
                job.setdefault('stages',{})[stage]=dict(returncode=code,success=ok,finished_at=now())
                if ok and stage=='smoke' and not args.smoke_only:job.update(status='pending',stage='full')
                elif ok and stage=='full':job.update(status='pending',stage='evaluate')
                else:job['status']='complete' if ok else 'failed'
                print(model,stage,job['status'],flush=True)
            for model,job in state['jobs'].items():
                if job['status']=='pending' and job['stage']=='evaluate':
                    cmd=['bash',str(ENVS[model]),str(HERE/'evaluate.py'),'--model',model,
                         '--protocol',str(args.protocol),'--cache',str(args.output/model/'cache'),
                         '--output',str(args.output/model/'evaluation')]
                    start(model,'evaluate',cmd,env())
            for gpu in args.gpus:
                if any(v['gpu']==gpu for v in active.values()):continue
                model=next((m for m,j in state['jobs'].items() if j['status']=='pending' and j['stage'] in ('smoke','full')),None)
                if model is None:break
                lease,info=reserve_gpu(gpu)
                if lease is None:
                    state.setdefault('last_gpu_checks',{})[str(gpu)]=info;continue
                stage=state['jobs'][model]['stage']
                cache=args.output/(model+'_smoke' if stage=='smoke' else model)/'cache'
                cmd=['bash',str(ENVS[model]),str(HERE/'cache_features.py'),'--model',model,
                    '--protocol',str(args.protocol),'--output',str(cache),'--gpu',str(gpu),
                    '--batch-size','8' if model=='AlphaRAD' else '16','--workers','2',
                    '--smoke','16' if stage=='smoke' else '0']
                start(model,stage,cmd,{**env(),'CUDA_VISIBLE_DEVICES':info['uuid'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID'},gpu,lease)
                state['jobs'][model]['gpu_check']=info
            save()
            if active or any(j['status']=='pending' for j in state['jobs'].values()):time.sleep(2)
        ok=all(j['status']=='complete' for j in state['jobs'].values())
        if ok and not args.smoke_only:
            state['jobs']['summary']=dict(stage='summary',status='pending')
            state['status']='paired_bootstrap'
            start('summary','summary',[sys.executable,str(HERE/'summarize.py'),'--run',str(args.output),
                '--protocol',str(args.protocol),'--replicates','2000','--seed','20261024'],env())
            save()
            while active['summary']['proc'].poll() is None:
                if stopping:raise InterruptedError('Stopping summary')
                time.sleep(2)
            item=active.pop('summary');code=item['proc'].returncode;item['log'].close()
            results=args.output/'RESULTS.json'
            ok=code==0 and results.exists() and json.loads(results.read_text())['status']=='complete'
            state['jobs']['summary'].update(status='complete' if ok else 'failed',returncode=code,finished_at=now())
        changed=[p for p,h in sources.items() if not Path(p).is_file() or sha(p)!=h]
        state.update(status='complete' if ok and not changed else 'failed',finished_at=now(),changed_sources=changed)
        save();print('Campaign',state['status'],str(args.output),flush=True)
        return 0 if state['status']=='complete' else 1
    except BaseException as exc:
        for item in active.values():
            try:os.killpg(item['proc'].pid,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.monotonic()+15
        while any(i['proc'].poll() is None for i in active.values()) and time.monotonic()<deadline:time.sleep(.2)
        for item in active.values():
            if item['proc'].poll() is None:os.killpg(item['proc'].pid,signal.SIGKILL)
            item['proc'].wait();item['log'].close()
            if item['lease'] is not None:release_gpu(item['lease'])
        state.update(status='interrupted' if stopping else 'failed',error=repr(exc),finished_at=now());save()
        raise

if __name__=='__main__':sys.exit(main())
