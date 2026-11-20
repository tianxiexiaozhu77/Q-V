#!/usr/bin/env python3
"""Run all declared mechanism comparisons on idle physical GPUs 6 and 7."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
MEDICAL=HERE.parents[1]
sys.path.insert(0,str(MEDICAL/'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu, release_gpu
MODELS=('AlphaRAD','RadZero','CARZero','MedKLIP','MAVL','KAD')
ENVS={m:MEDICAL/(f'{m}/local_setup/python_local.sh' if m=='AlphaRAD' else 'RadZero_i2cl/local_setup/python_local.sh' if m=='RadZero' else f'baselines/{m}/local_setup/python_local.sh') for m in MODELS}
SCRIPTS={m:HERE/('infer_alpha.py' if m=='AlphaRAD' else 'infer_radzero.py' if m=='RadZero' else 'infer_baselines.py') for m in MODELS}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():
    return datetime.now(timezone.utc).isoformat()
def write(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=HERE/'protocol_v1/catalog.json')
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--gpus',type=int,nargs='+',default=[6,7])
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--ready-dir',type=Path,help='Optional implementation-ready markers, one file named per model')
    args=p.parse_args()
    if not args.gpus or not set(args.gpus)<={6,7} or len(set(args.gpus))!=len(args.gpus):
        p.error('Only distinct physical GPUs 6 and 7 are authorized')
    if len(set(args.models))!=len(args.models):
        p.error('Duplicate models')
    args.output=args.output.resolve(); args.protocol=args.protocol.resolve()
    protocol=json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'logs').mkdir()
    state=dict(status='running',created_at=now(),controller_pid=os.getpid(),gpus=args.gpus,
        protocol=str(args.protocol),protocol_sha256=sha(args.protocol),models=args.models,
        jobs={m:dict(status='pending',stage='smoke') for m in args.models})
    plan=dict(created_at=now(),models=args.models,protocol=str(args.protocol),protocol_sha256=sha(args.protocol),
        support_seeds=[42,43,44,45,46],tasks=['slvh','dlv','composite_slvh_dlv'],eta=.3,views=['identity'],
        primary_contrast={m:'Q - R (readout only)' if m in ('AlphaRAD','RadZero') else 'Q - F (fixed visual attention outputs)' for m in args.models},
        all_models_tasks_seeds_retained=True,scope='exploratory expansion on previously evaluated CheXchoNet; no performance-based hyperparameter changes',
        code_frozen_per_job_before_inference=True,statistics_after_all_jobs=True,controller_source_sha256=sha(__file__))
    write(args.output/'FROZEN_RUN_PLAN.json',plan)
    active={}
    try:
        while active or any(j['status']=='pending' for j in state['jobs'].values()):
            for gpu,(proc,lease,log,m,stage,snapshot) in list(active.items()):
                code=proc.poll()
                if code is None: continue
                log.close(); release_gpu(lease); del active[gpu]
                folder=args.output/(m+'_smoke' if stage=='smoke' else m)
                sp=folder/'status.json'
                expected='smoke_complete' if stage=='smoke' else 'complete'
                changed=[f for f,digest in snapshot.items() if sha(f)!=digest]
                ok=code==0 and sp.exists() and json.loads(sp.read_text()).get('status')==expected and not changed
                job=state['jobs'][m]
                job.setdefault('stages',{})[stage]=dict(returncode=code,success=ok,finished_at=now(),status_file=str(sp),changed_sources=changed)
                job.update(status='pending' if ok and stage=='smoke' and not args.smoke_only else 'complete' if ok else 'failed',stage='full' if ok and stage=='smoke' and not args.smoke_only else stage)
                print(m,stage,job['status'],flush=True)
            for gpu in args.gpus:
                if gpu in active: continue
                m=next((m for m,j in state['jobs'].items() if j['status']=='pending' and (not args.ready_dir or (args.ready_dir/m).exists())),None)
                if m is None: continue
                if not SCRIPTS[m].exists(): raise FileNotFoundError(SCRIPTS[m])
                if sha(args.protocol)!=state['protocol_sha256']: raise ValueError('Frozen protocol changed')
                lease,info=reserve_gpu(gpu)
                if lease is None:
                    state.setdefault('last_gpu_checks',{})[str(gpu)]=info
                    continue
                j=state['jobs'][m]; stage=j['stage']
                folder=args.output/(m+'_smoke' if stage=='smoke' else m)
                command=['bash',str(ENVS[m]),str(SCRIPTS[m]),'--model',m,'--protocol',str(args.protocol),'--output',str(folder),'--batch-size','8','--workers','2','--smoke','16' if stage=='smoke' else '0']
                # Per-job snapshots freeze implemented modules before any predictions.
                dependencies=[SCRIPTS[m],HERE/('radzero_intervention.py' if m=='RadZero' else 'baseline_intervention.py' if m!='AlphaRAD' else 'infer_alpha.py')]
                snapshot={str(f):sha(f) for f in dependencies}
                env={**os.environ,'CUDA_VISIBLE_DEVICES':info['uuid'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
                log=(args.output/'logs'/f'{m}_{stage}.log').open('w')
                try: proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=MEDICAL,env=env,start_new_session=True)
                except BaseException:
                    log.close();release_gpu(lease);raise
                active[gpu]=(proc,lease,log,m,stage,snapshot)
                j.update(status='running',pid=proc.pid,gpu=gpu,gpu_check=info,command=command,started_at=now(),source_sha256=snapshot)
                print('Started',m,stage,'GPU',gpu,'PID',proc.pid,flush=True)
            write(args.output/'status.json',state)
            if active or any(j['status']=='pending' for j in state['jobs'].values()): time.sleep(5)
        ok=all(j['status']=='complete' for j in state['jobs'].values())
        if ok and not args.smoke_only:
            state['status']='patient_bootstrap';write(args.output/'status.json',state)
            command=[sys.executable,str(HERE/'summarize.py'),'--run',str(args.output),'--protocol',str(args.protocol),'--models',*args.models]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            with (args.output/'logs/bootstrap.log').open('w') as log:
                code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,cwd=MEDICAL,env=env)
            state['bootstrap_returncode']=code;ok=code==0
        state.update(status='complete' if ok else 'failed',finished_at=now())
        write(args.output/'status.json',state)
        return 0 if ok else 1
    except BaseException:
        state.update(status='controller_interrupted_workers_not_terminated',updated_at=now())
        write(args.output/'status.json',state)
        raise
if __name__=='__main__':
    sys.exit(main())
