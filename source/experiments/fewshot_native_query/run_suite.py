#!/usr/bin/env python3
"""Run the fixed first-round suite on checked idle physical GPUs 4--7."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from cache_native_common import MEDICAL,write_json,sha

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(MEDICAL/'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu,release_gpu
ENVS={
 'BiomedCLIP':MEDICAL/'baselines/protocols/alpharad_binary/BiomedCLIP/local_setup/python_local.sh',
 'AlphaRAD':MEDICAL/'AlphaRAD/local_setup/python_local.sh',
 'RadZero':MEDICAL/'RadZero_i2cl/local_setup/python_local.sh'}

def summarize(output,state):
    records=[]
    lines=['固定支持集：公平比较与机制消融',f"状态：{state['status']}；GPU：{state['gpus']}",
        '单位：%；增益：百分点。AUPRC为PR曲线梯形面积。',
        '所有方法同一支持/测试图；可调参数仅由共享支持集5折OOF选择。',
        'SimNL为特征级二元OvR适配版，其他基于CLIP的方法也是明确的OvR移植，并非作者原场景完整复现。',
        '已有测试集曾用于方法探索；单次支持抽样，不能视为独立验证。',
        'Alpha/Rad全局特征方法与查询局部特征不同；注入位置的受控比较看同特征正负差分修正分数。','',
        '零权重时raw margin与原生float32概率的数值并列可产生不足0.001点的差异，不能计作适配收益；见RUN_LOCAL说明。','',
        '| 模型 | 数据集 | 方法 | AUROC | Δ | AUPRC | Δ | PG | Δ |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for key,job in state['jobs'].items():
        path=Path(job.get('result_file',str(output/key/'evaluation/result.json')))
        if job['status']!='complete' or not path.exists():
            lines.append(f"| {job['model']} | {job['dataset']} | {job['status']} | — | — | — | — | — | — |")
            continue
        result=json.loads(path.read_text())
        for method,item in result['methods'].items():
            met=item['metrics'];delta=item['delta_pp'];sp=result['spatial'].get(method,{})
            row={'model':result['model'],'dataset':result['dataset'],'method':method,'name':item['name'],
                'support':result['support_count'],'test':result['test_count'],
                'auroc':met['macro_auroc'],'auprc':met['macro_auprc'],
                'delta_auroc_pp':delta.get('macro_auroc'),'delta_auprc_pp':delta.get('macro_auprc'),
                'pointing_game':sp.get('pointing_game'),'delta_pg_pp':sp.get('delta_pp'),
                'result_file':str(path)}
            records.append(row)
            val=lambda x:'—' if x is None else f'{100*x:.2f}'
            diff=lambda x:'—' if x is None else f'{x:+.2f}'
            lines.append(f"| {row['model']} | {row['dataset']} | {item['name']} | {val(row['auroc'])} | {diff(row['delta_auroc_pp'])} | {val(row['auprc'])} | {diff(row['delta_auprc_pp'])} | {val(row['pointing_game'])} | {diff(row['delta_pg_pp'])} |")
    (output/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    write_json(output/'RESULTS.json',{'status':state['status'],'records':records})
    if records:
        with (output/'COMPARISON.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=list(ENVS),choices=list(ENVS))
    p.add_argument('--datasets',nargs='+',default=['rsna_pneumonia','chestxdet10'],choices=['rsna_pneumonia','chestxdet10'])
    p.add_argument('--gpus',nargs='+',type=int,default=[4,5,6,7])
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--protocol-dir',type=Path,default=ROOT/'protocol_v1')
    a=p.parse_args()
    if not set(a.gpus)<=set(range(4,8)) or len(a.gpus)!=len(set(a.gpus)):
        raise ValueError('Only distinct GPUs 4--7 are authorized')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    jobs={f'{m}_{d}':{'model':m,'dataset':d,'status':'pending'} for d in a.datasets for m in a.models}
    state={'status':'running','controller_pid':os.getpid(),'started_at':datetime.now().isoformat(),
        'gpus':a.gpus,'limit':a.limit,'jobs':jobs,
        'source_sha256':{str(x):sha(x) for x in ROOT.glob('*.py')},
        'protocols':{d:sha(a.protocol_dir/f'{d}.json') for d in a.datasets}}
    active={}
    try:
        while any(j['status']=='pending' for j in jobs.values()) or active:
            for gpu,entry in list(active.items()):
                proc,lease,handle,key=entry
                code=proc.poll()
                if code is None:continue
                handle.close();release_gpu(lease);active.pop(gpu)
                result=a.output/key/'evaluation/result.json'
                jobs[key].update(status='complete' if code==0 and result.exists() else 'failed',returncode=code,
                                 finished_at=datetime.now().isoformat())
                print(key,jobs[key]['status'],flush=True)
            for gpu in a.gpus:
                if gpu in active:continue
                key=next((k for k,j in jobs.items() if j['status']=='pending'),None)
                if key is None:break
                lease,info=reserve_gpu(gpu)
                if lease is None:
                    state.setdefault('last_gpu_checks',{})[str(gpu)]=info;continue
                job=jobs[key];directory=a.output/key;directory.mkdir()
                command=['bash',str(ENVS[job['model']]),str(ROOT/'run_unit.py'),'--model',job['model'],
                    '--protocol',str((a.protocol_dir/f"{job['dataset']}.json").resolve()),'--output',str(directory),
                    '--gpu',str(gpu),'--limit',str(a.limit)]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=info['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                    TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                handle=(directory/'run.log').open('w')
                try:proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=MEDICAL)
                except BaseException:handle.close();release_gpu(lease);raise
                job.update(status='running',pid=proc.pid,gpu=gpu,gpu_check=info,command=command)
                active[gpu]=(proc,lease,handle,key);print('Started',key,'GPU',gpu,'PID',proc.pid,flush=True)
            write_json(a.output/'status.json',state);summarize(a.output,state)
            if active or any(j['status']=='pending' for j in jobs.values()):time.sleep(10)
        state.update(status='complete' if all(j['status']=='complete' for j in jobs.values()) else 'failed',
                     finished_at=datetime.now().isoformat())
        write_json(a.output/'status.json',state);summarize(a.output,state)
    except BaseException:
        # Do not silently kill already running, explicitly authorized GPU workers.
        state['status']='controller_interrupted_workers_not_terminated'
        write_json(a.output/'status.json',state)
        raise
    return 0 if state['status']=='complete' else 1

if __name__=='__main__':sys.exit(main())
