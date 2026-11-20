#!/usr/bin/env python3
"""GPU4-7 locked scheduler: smoke all scopes, then full paired query transfer."""
import argparse
import csv
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from evaluate_baseline_buffered import sha,write
from run_all_verified_baselines import MODEL_ROOTS,ready
from run_pretrained import reserve_gpu,release_gpu

ROOT=Path(__file__).resolve().parents[1]
MODELS=['CheXzero','CARZero','MaCo','BioViL-T','MAVL','KAD','MedKLIP','BiomedCLIP']
STOP=False
def now():return datetime.now(timezone.utc).isoformat()
def stop(signum,frame):
    global STOP
    STOP=True

class AdoptedWorker:
    """Monitor only a previously recorded worker; do not restart live inference."""
    def __init__(self,attempt):
        self.pid=int(attempt['pid']);self.output=Path(attempt['output'])
        proc=Path(f'/proc/{self.pid}/cmdline')
        if proc.exists():
            cmd=proc.read_bytes().split(b'\0')
            if str(self.output).encode() not in cmd or not any(b'evaluate_query_transfer.py' in x for x in cmd):
                raise RuntimeError('Recorded PID no longer belongs to this inference worker')
    def poll(self):
        proc=Path(f'/proc/{self.pid}/stat')
        if proc.exists() and proc.read_text().split(') ',1)[1].split()[0]!='Z':return None
        status=self.output/'status.json'
        return 0 if status.is_file() and json.loads(status.read_text()).get('status')=='complete' else 1

def report(output,catalog,state):
    records=[]
    for model,job in state['models'].items():
        for name,e in catalog['datasets'].items():
            path=Path(job['full_output'])/name/'result.json' if job.get('full_output') else None
            if path and path.is_file():
                r=json.loads(path.read_text());records.append({**r,'result_file':str(path)})
            else:records.append(dict(model=model,dataset=name,task=e['task'],rows=e['count'],
                 support_count=e['support_count'],status=job['status']))
    counts={s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})}
    result=dict(status=state['status'],updated_at=now(),counts=counts,records=records,
       campaign=str(output),eta=.3,scope=catalog['scope'],
       note='Few-shot without gradient training; original frozen supports/test; native query norms/scoring. Four arms. Architecture-specific query feature choices documented per model. No test-driven eta or support selection.')
    write(output/'RESULTS.json',result)
    lines=['8个baseline：查询修正＋原图/中心裁剪平均','',
      f'状态：{state["status"]}；结果数：{counts}',
      '固定η=0.3；沿用既有支持集/测试集；网络全冻结。无需梯度训练的少样本适配，不是纯零样本。',
      '支持图只用原视图；正查询保留原范数、负查询保持原值；双视图平均原生分数。',
      'center80在各模型原生预处理后的输入上裁剪80%并放大；热图先逆变换再按覆盖区域平均。',
      '全部四组结果存JSON/CSV；以下为原始与完整方案。单位%，增益为百分点。',
      'MedKLIP/CARZero为原生查询交互空间的实验性迁移；MAVL每病9个概念共享该疾病修正方向。',
      '不支持项沿用已核实baseline范围，不以少类别替代完整任务。','',
      '| 模型 | 数据集 | 状态 | 测试/支持 | AUROC 原始→改进 | Δ | AUPRC 原始→改进 | Δ | PG/Dice 原始→改进 | Δ |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    for r in records:
        branches=r.get('branches',{});a=branches.get('original',{});b=branches.get('tta_center_refined',{})
        fields=[]
        for key in ('macro_auroc','macro_auprc','pointing_game' if r['task']=='grounding' else 'dice'):
            if a.get(key) is not None and b.get(key) is not None:
                fields.extend([f'{100*a[key]:.2f} → {100*b[key]:.2f}',f'{100*(b[key]-a[key]):+.2f}'])
            else:fields.extend(['—','—'])
        lines.append(f'| {r["model"]} | {r["dataset"]} | {r["status"]} | {r["rows"]}/{r.get("support_count","—")} | '+' | '.join(fields)+' |')
        for branch,metrics in branches.items():
            csvrows.append(dict(model=r['model'],dataset=r['dataset'],task=r['task'],rows=r['rows'],
              support=r.get('support_count'),branch=branch,status=r['status'],
              **{k:metrics.get(k) for k in ('macro_auroc','macro_auprc','pointing_game','dice')}))
        if r.get('reason') or r.get('error'):lines.append(f'说明：{r["model"]}/{r["dataset"]}: {r.get("reason",r.get("error"))}')
    lines+=['','保留限制：CheXpert596图12类，骨折仅1位阳性测试患者；QaTa Dice沿用测试阈值扫描。',
            '已有phase6测试集曾用于方法探索；本轮固定参数迁移，不以测试成绩选择配置。',
            '完整出处、各模型特征接入方式、原始输出一致性检查见各任务result.json。']
    (output/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    fields=['model','dataset','task','rows','support','branch','status','macro_auroc','macro_auprc','pointing_game','dice']
    with (output/'RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csvrows)
    # Separate files: never overwrite the original baseline or phase6 reports.
    for ext in ('txt','json','csv'):
        dest=ROOT/f'QUERY_REFINEMENT_RESULTS.{ext}'
        tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_bytes((output/f'RESULTS.{ext}').read_bytes());tmp.replace(dest)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--gpus',nargs='+',type=int,default=[4,5,6,7])
    p.add_argument('--catalog',type=Path,default=ROOT/'evaluation/query_transfer_20261022_v1/catalog.json')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--smoke-size',type=int,default=8,
                   help='Use the archived batch size for numeric parity, avoiding batch2 vs batch8 cuDNN differences')
    p.add_argument('--resume',action='store_true',help='Adopt live recorded workers and retry failed model smoke checks')
    args=p.parse_args()
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or set(args.gpus)-{4,5,6,7}:p.error('Only physicalGPU4-7 authorized')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=args.resume)
    (args.output/'logs').mkdir(exist_ok=args.resume);catalog=json.loads(args.catalog.read_text())
    if args.resume:
        state=json.loads((args.output/'status.json').read_text())
        previous=int(state['controller_pid'])
        oldstat=Path(f'/proc/{previous}/stat')
        if oldstat.exists() and oldstat.read_text().split(') ',1)[1].split()[0]!='Z':raise RuntimeError('Previous controller must exit before resuming')
        if state['catalog_sha256']!=sha(args.catalog) or state['gpus']!=args.gpus:raise ValueError('Resume scope changed')
        state.setdefault('controller_history',[]).append({'pid':previous,'resumed_at':now()})
        state.update(status='running',controller_pid=os.getpid())
        for job in state['models'].values():
            if job['status']=='failed':job['status']='pending'
    else:
        state=dict(status='running',started_at=now(),controller_pid=os.getpid(),gpus=args.gpus,
          catalog=str(args.catalog),catalog_sha256=sha(args.catalog),
          models={m:dict(status='pending',attempts=[]) for m in args.models})
    active={};signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(model,stage,gpu,lease,info):
        job=state['models'][model];attempt=len(job['attempts'])
        out=args.output/f'{model}_{stage}_{attempt}'
        cmd=['bash',str(MODEL_ROOTS[model]/'local_setup/python_local.sh'),
             str(ROOT/'local_setup/evaluate_query_transfer.py'),'--model',model,
             '--model-root',str(MODEL_ROOTS[model]),'--catalog',str(args.catalog),
             '--output',str(out),'--batch-size',str(args.batch_size)]
        if stage=='smoke':cmd+=['--smoke',str(args.smoke_size)]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=info['uuid'],PYTHONUNBUFFERED='1',
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        logfile=args.output/'logs'/f'{model}_{stage}_{attempt}.log';log=logfile.open('w')
        proc=subprocess.Popen(cmd,cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[gpu]=dict(model=model,stage=stage,proc=proc,lease=lease,info=info,log=log)
        job.update(status='running_'+stage,**{stage+'_output':str(out)})
        job['attempts'].append(dict(stage=stage,pid=proc.pid,output=str(out),log=str(logfile),
                  command=cmd,gpu=info,started_at=now()))
        print('START',model,stage,'GPU',gpu,'PID',proc.pid,flush=True)
    try:
        if args.resume:
            for model,job in state['models'].items():
                if not job['status'].startswith('running_'):continue
                attempt=job['attempts'][-1];info=attempt['gpu'];gpu=int(info['physical_index'])
                proc=AdoptedWorker(attempt)
                # These are proven task-owned workers. The preceding controller
                # has exited; adopt its UUID lock without demanding an idle GPU.
                lease=Path(info['lock_path']).open('a+')
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                active[gpu]=dict(model=model,stage=attempt['stage'],proc=proc,lease=lease,
                                info=info,log=Path(attempt['log']).open('a'))
                print('ADOPT',model,attempt['stage'],'GPU',gpu,'PID',proc.pid,flush=True)
        while not STOP:
            for gpu,item in list(active.items()):
                code=item['proc'].poll()
                if code is None:continue
                model=item['model'];job=state['models'][model];item['log'].close()
                job['attempts'][-1].update(returncode=code,finished_at=now())
                if code==0 and item['stage']=='smoke':
                    launch(model,'full',gpu,item['lease'],item['info'])
                else:
                    release_gpu(item['lease']);del active[gpu]
                    job['status']='complete' if code==0 else 'failed'
                    print('FINISH',model,item['stage'],code,flush=True)
            for model,job in state['models'].items():
                if job['status']!='pending':continue
                if not ready(model):raise RuntimeError('Original model readiness check failed: '+model)
                if not (ROOT/model/'local_setup/query_refinement_adapter.py').is_file():continue
                for gpu in args.gpus:
                    if gpu in active:continue
                    lease,info=reserve_gpu(gpu);job['last_gpu_check']=info
                    if lease is None:continue
                    launch(model,'smoke',gpu,lease,info);break
            if all(j['status'] in ('complete','failed') for j in state['models'].values()):
                state['status']='complete' if all(j['status']=='complete' for j in state['models'].values()) else 'completed_with_failures'
                state['finished_at']=now()
            state['updated_at']=now();write(args.output/'status.json',state);report(args.output,catalog,state)
            if state['status']!='running':break
            time.sleep(10)
        if STOP:
            # Leave owned workers running; explicit stop must separately target them.
            state['status']='controller_stopped_workers_continue';write(args.output/'status.json',state)
    except BaseException as e:
        state.update(status='controller_error_workers_continue',error=repr(e),traceback=traceback.format_exc())
        write(args.output/'status.json',state);raise
    finally:
        # Exiting controller never terminates somebody else's process or a live worker.
        for item in active.values():
            if item['proc'].poll() is not None:release_gpu(item['lease'])
        report(args.output,catalog,state)

if __name__=='__main__':main()
