#!/usr/bin/env python3
"""Merge fixed protocol choices, never select the best score across attempts."""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

from run_all_verified_baselines import MODELS, ROOT, report
from evaluate_baseline import sha, write


def refresh(args):
    original = json.loads((args.original / 'status.json').read_text())
    binary = json.loads((args.binary / 'status.json').read_text())
    delegated = json.loads((args.delegated / 'status.json').read_text()) if args.delegated else None
    state = {'status': 'running', 'models': {}, 'selection_rule':
             'BiomedCLIP/BioViL-T: verified presence-vs-absence binary protocol on every task; other six: native released model protocol. No metric-dependent selection.',
             'source_runs': {'native': str(args.original), 'binary': str(args.binary)}}
    if delegated:
        state['source_runs']['gpu45'] = str(args.delegated)
    for name in MODELS:
        source = binary if name in ('BiomedCLIP', 'BioViL-T') else original
        if delegated and name in delegated['models']:
            source = delegated
        state['models'][name] = copy.deepcopy(source['models'][name])
    if all(j['status'] == 'complete' for j in state['models'].values()):
        state['status'] = 'complete'
    catalog = json.loads(args.catalog.read_text())
    report(ROOT, catalog, state)
    extra = ['\n固定的评分协议与复测说明：',
        '- BiomedCLIP、BioViL-T 正式分类使用逐疾病有病/无病文本的二元softmax；不跨疾病归一化。',
        '- 依据AlphaRAD _zeroshot_binary_helper以及BioViL-T官方正/负提示分类示例确定，不逐数据集挑选高分版本。',
        '- 首轮单正提示cosine评估保存在 '+str(args.original)+'，仅供诊断，未作为这两个模型的正式分类结果。',
        '- 已有AlphaRAD/RadZero参考结果和本轮baseline对齐测试身份、标签、数量；RSNA/SIIM的PNG/DICOM解码流程并非像素完全相同。',
        '- QaTa采用共享原图热图minmax后处理；旧AlphaRAD先在token上归一化。阈值float精度也有记录，因此不宣称逐操作一致。',
        '- KAD空间结果迁移作者512可视化脚本的最后层attention至224模型；论文文字描述多层平均。Node21单query及词形也与作者ChestXDet10示例不同，见KAD_GROUNDING_PROTOCOL_AUDIT.txt；不作为论文定位数值的精确复现。',
        '- MaCo最近邻热图存在并列峰值；本轮统一PG取首个argmax，作者PG接受任一并列最大像素命中。未按分数择优切换规则，详见MACO_GROUNDING_PROTOCOL_AUDIT.txt。',
        '- unsupported表示本次未核实对应查询资源或原生任务实现；不等于该模型架构绝对无法支持新疾病。',
        '- 每模型预处理、提示与评分来源见对应manifest/adapter_metadata；不能直接视为AlphaRAD论文原表的逐项精确复现。']
    with (ROOT / 'RESULTS.txt').open('a') as stream:
        stream.write('\n'.join(extra)+'\n')
    data = json.loads((ROOT/'RESULTS.json').read_text())
    data.update(selection_policy=state['selection_rule'], source_runs=state['source_runs'],
                protocol_notes=extra[1:])
    write(ROOT/'RESULTS.json',data)
    write(ROOT/'CAMPAIGN_STATUS.json',state)
    lines=['公开预训练baseline固定测试范围评估', '',
           '更新：'+datetime.now().isoformat(timespec='seconds'),
           '总体状态：'+state['status'], '正式对照表：'+str(ROOT/'RESULTS.txt'),
           '运行说明：'+str(ROOT/'RUN_LOCAL.txt'), '',
           '| 模型 | 正式评估状态 | 当前任务/进度 |', '|---|---|---|']
    for name, job in state['models'].items():
        status=job['status']; detail=''
        base = ROOT/'protocols/alpharad_binary' if name in ('BiomedCLIP','BioViL-T') else ROOT
        filename='READY.gpu45.json' if delegated and name in delegated['models'] else 'READY.json'
        ready=base/name/'local_setup'/filename
        if not ready.exists(): ready=base/name/'local_setup/READY.json'
        if status=='waiting_for_verified_assets' and ready.is_file() and json.loads(ready.read_text()).get('ready'):
            status='已就绪，等待GPU'
        output=job.get('full_output') or job.get('smoke_output')
        if output:
            path=Path(output)/'progress.json'
            if path.is_file():
                value=json.loads(path.read_text());detail=f"{value['dataset']} {value['images_completed']}/{value['images_total']}"
        lines.append(f'| {name} | {status} | {detail} |')
    clip_done = all(state['models'][name]['status'] == 'complete' for name in ('BiomedCLIP', 'BioViL-T'))
    lines += ['', '11个数据集、13个评估范围。原始baseline不使用支持样本，不训练。',
              ('BiomedCLIP、BioViL-T正/负文本对照协议已完成；首轮正提示相似度结果仅保留诊断。' if clip_done else
               'BiomedCLIP、BioViL-T正在补齐正/负文本对照协议；首轮正提示相似度结果保留为诊断。'),
              '队列共享GPU UUID占用锁，仅使用已经授权且空闲的物理4–7号卡。',
              '当前测试图/标签固定，各模型原生预处理及公开查询资源的差异见RUN_LOCAL和独立审计。']
    target=ROOT/'STATUS.txt';temp=target.with_suffix('.tmp')
    temp.write_text('\n'.join(lines)+'\n');temp.replace(target)
    if state['status']=='complete' and delegated:
        finalize_delegation(args, original, delegated, catalog)
    return state['status']


def finalize_delegation(args, original, delegated, catalog):
    """Close only an idle controller; no inference subprocess is interrupted."""
    receipt=args.original/'DELEGATION_COMPLETE.json'
    if receipt.exists(): return
    local=[j for n,j in original['models'].items() if n not in delegated['models']]
    if not all(j['status']=='complete' for j in local):
        raise RuntimeError('Refuse controller handoff while a local job is unfinished')
    pid=original['controller_pid'];proc=Path('/proc')/str(pid)
    if proc.exists():
        cmd=(proc/'cmdline').read_bytes().decode().split('\0')
        if str(args.original) not in cmd or not any(v.endswith('/run_matched_baselines.py') for v in cmd):
            raise RuntimeError('Controller ownership check failed')
        write(args.original/'PRE_HANDOFF_STATUS.json',original)
        os.kill(pid,signal.SIGTERM)
        deadline=time.time()+30
        while proc.exists() and time.time()<deadline:
            try:
                if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
            except FileNotFoundError:break
            time.sleep(.5)
        if proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise RuntimeError('Idle controller did not exit; no force termination applied')
    exit_state=json.loads((args.original/'status.json').read_text())
    write(args.original/'CONTROLLER_HANDOFF_EXIT.json',exit_state)
    combined=copy.deepcopy(original)
    for name,job in delegated['models'].items():
        if job['status']!='complete': raise RuntimeError('Delegated model unfinished')
        combined['models'][name]=copy.deepcopy(job)
        combined['models'][name]['source_run']=str(args.delegated)
        path=ROOT/name/'local_setup/READY.gpu45.json'
        normal=path.with_name('READY.json')
        if path.exists() and not normal.exists():path.rename(normal)
    combined.update(status='complete',finished_at=datetime.now().astimezone().isoformat(),gpus=[4,5,6,7],delegated_run=str(args.delegated),
                    handoff_note='CARZero/CheXzero ran on GPU4/5; original idle controller stopped only after all inference completed.')
    write(args.original/'status.json',combined)
    report(args.original,catalog,combined)
    write(receipt,{'status':'complete','controller_pid':pid,'inference_processes_interrupted':False,
                  'delegated_run':str(args.delegated),'readiness_restored':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--delegated',type=Path)
    p.add_argument('--catalog',type=Path,default=ROOT/'evaluation/matched_phase6_20261022/catalog.json')
    p.add_argument('--once',action='store_true')
    args=p.parse_args();args.original=args.original.resolve();args.binary=args.binary.resolve()
    if args.delegated:args.delegated=args.delegated.resolve()
    while True:
        try:
            state=refresh(args)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            # Atomic status replacement or an idle controller handoff can be
            # briefly invisible through NFS attribute caches. Keep last report.
            print('Retrying status read:', repr(error), flush=True)
            if args.once:raise
            time.sleep(10)
            continue
        if state=='complete' or args.once:break
        time.sleep(30)


if __name__=='__main__':
    main()
