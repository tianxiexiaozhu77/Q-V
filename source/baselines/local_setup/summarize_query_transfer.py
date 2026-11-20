#!/usr/bin/env python3
"""Readable original vs fixed query+center comparison, including signed changes."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
MODELS=['BiomedCLIP','BioViL-T','CheXzero','MedKLIP','MAVL','KAD','MaCo','CARZero']
NAMES={'chest14':'ChestXray14','rsna_pneumonia':'RSNA','siim_pneumothorax':'SIIM（1704）',
 'chestxdet10':'ChestXDet10 分类','chestxdet10_gt':'ChestXDet10 定位','node21':'Node21 定位',
 'qatacovid_seg':'QaTaCOVID19 分割','shenzhenxray':'Shenzhen','montgomery':'Montgomery',
 'covidkaggle':'CovidKaggle','chexpert':'CheXpert','openi':'OpenI','siim_radzero_holdout':'SIIM（1694）'}
KEYS=('macro_auroc','macro_auprc','pointing_game','dice')

def value(x):return f'{100*x:.2f}'
def delta(x):
    pp=100*x
    return f'{pp:+.5f}' if 0<abs(pp)<.005 else f'{pp:+.2f}'
def pattern(record):
    a,b=record['branches']['original'],record['branches']['tta_center_refined']
    ds=[b[k]-a[k] for k in ('macro_auroc','macro_auprc')]
    return '两项上升' if min(ds)>0 else '两项下降' if max(ds)<0 else '一升一降或持平'

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args()
    run=args.run.resolve();source=run/'RESULTS.json'
    data=json.loads(source.read_text());audit=json.loads((run/'FINAL_AUDIT.json').read_text())
    state=json.loads((run/'status.json').read_text())
    if data['status']!='complete' or audit['status']!='passed':raise ValueError('Require completed audited campaign')
    complete=[r for r in data['records'] if r['status']=='complete']
    cls=[r for r in complete if r['task']=='classification'];spatial=[r for r in complete if r['task']!='classification']
    finish=datetime.fromisoformat(state['finished_at']).astimezone(ZoneInfo('Asia/Shanghai'))
    summary=[];unique_summary=[]
    for m in MODELS:
        allrows=[r for r in cls if r['model']==m]
        for target,rows in ((summary,allrows),(unique_summary,[r for r in allrows if r['dataset']!='siim_radzero_holdout'])):
            item={'model':m,'classification_scope_count':len(rows),'datasets':[r['dataset'] for r in rows]}
            for k in ('macro_auroc','macro_auprc'):
                a=sum(r['branches']['original'][k] for r in rows)/len(rows)
                b=sum(r['branches']['tta_center_refined'][k] for r in rows)/len(rows)
                item[k]={'original_percent':100*a,'refined_percent':100*b,'delta_pp':100*(b-a)}
            item['both_up']=sum(pattern(r)=='两项上升' for r in rows)
            item['both_down']=sum(pattern(r)=='两项下降' for r in rows)
            item['mixed_or_tied']=len(rows)-item['both_up']-item['both_down']
            target.append(item)
    lines=['8个baseline：查询修正＋原图/中心裁剪平均，最终结果汇总','',
      f'推理完成（北京时间）：{finish:%Y-%m-%d %H:%M:%S}；独立审计passed，0错误。',
      f'完成89项（74分类、15空间任务），15项原协议不支持；固定η=0.3。',
      '对比对象是同一支持/测试划分下的原始baseline与完整方案tta_center_refined，不是论文分数。',
      'AUROC、AUPRC、PG、Dice单位为%；涨跌单位为百分点，按未四舍五入值计算。',
      'AUPRC为PR曲线梯形面积。仅给出描述性点估计，不表示统计显著。','',
      '一、各模型分类平均（按本轮实际评估范围等权平均）',
      'SIIM1704和1694分别计入两项，因此是评估范围平均，不是独立数据集平均。',
      'MedKLIP/MAVL缺少OpenI、Shenzhen、Montgomery，平均覆盖范围不同，不能用于严格横向排名。',
      '| 模型 | 范围数 | 原始AUROC | 改进AUROC | AUROC涨跌 | 原始AUPRC | 改进AUPRC | AUPRC涨跌 | 双升/双降/混合 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in summary:
        a=row['macro_auroc'];b=row['macro_auprc']
        lines.append(f'| {row["model"]} | {row["classification_scope_count"]} | {a["original_percent"]:.2f} | {a["refined_percent"]:.2f} | {a["delta_pp"]:+.2f} | {b["original_percent"]:.2f} | {b["refined_percent"]:.2f} | {b["delta_pp"]:+.2f} | {row["both_up"]}/{row["both_down"]}/{row["mixed_or_tied"]} |')
    totals={k:sum(r[k] for r in summary) for k in ('both_up','both_down','mixed_or_tied')}
    lines+=['',f'分类74项：{totals["both_up"]}项两项都升，{totals["both_down"]}项两项都降，{totals["mixed_or_tied"]}项混合。',
      'BiomedCLIP、KAD分类所有范围双升；其他模型有下降或混合项，不能说8个模型全部数据集均提高。','',
      '二、逐模型、逐数据集分类完整明细（74项）']
    for m in MODELS:
        lines+=['',m,'| 数据集/范围 | 图片数 | 原始AUROC | 改进AUROC | AUROC涨跌 | 原始AUPRC | 改进AUPRC | AUPRC涨跌 | 变化 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in cls:
            if r['model']!=m:continue
            a,b=r['branches']['original'],r['branches']['tta_center_refined']
            fields=[]
            for k in ('macro_auroc','macro_auprc'):fields += [value(a[k]),value(b[k]),delta(b[k]-a[k])]
            lines.append(f'| {NAMES[r["dataset"]]} | {r["rows"]} | '+' | '.join(fields)+f' | {pattern(r)} |')
    lines+=['','三、定位/分割完整明细（15项）',
      '| 模型 | 数据集/任务 | 指标 | 原始 | 改进 | 涨跌（百分点） |',
      '|---|---|---|---:|---:|---:|']
    for m in MODELS:
        for r in spatial:
            if r['model']!=m:continue
            k='pointing_game' if r['task']=='grounding' else 'dice'
            a,b=(r['branches'][branch][k] for branch in ('original','tta_center_refined'))
            lines.append(f'| {m} | {NAMES[r["dataset"]]} | {"PG" if k=="pointing_game" else "Dice"} | {value(a)} | {value(b)} | {delta(b-a)} |')
    lines+=['','KAD/QaTa Dice仅+0.00027个百分点，应理解为近乎持平，不是有实质幅度的提升。',
      '空间指标沿用已核实baseline热图实现及旧PG/Dice协议；Dice仍包含测试阈值扫描。','',
      '四、不重复计算SIIM的分类均值（辅助汇总，SIIM只计1704张主范围）',
      '| 模型 | 不同分类数据集数 | AUROC原始→改进 | 涨跌 | AUPRC原始→改进 | 涨跌 |',
      '|---|---:|---|---:|---|---:|']
    for r in unique_summary:
        a,b=r['macro_auroc'],r['macro_auprc']
        lines.append(f'| {r["model"]} | {r["classification_scope_count"]} | {a["original_percent"]:.2f} → {a["refined_percent"]:.2f} | {a["delta_pp"]:+.2f} | {b["original_percent"]:.2f} → {b["refined_percent"]:.2f} | {b["delta_pp"]:+.2f} |')
    lines+=['','五、不支持项（共15项，没有伪造结果）']
    for r in data['records']:
        if r['status']=='unsupported':lines.append(f'- {r["model"]}/{NAMES[r["dataset"]]}：{r.get("reason","沿用原baseline能力限制")}')
    lines+=['','完整四组消融结果仍保留在QUERY_REFINEMENT_RESULTS.json/csv。',
      '本轮只比较original和tta_center_refined，没有按测试结果挑选最佳分支。',
      '使用有标签支持集，网络冻结、无反向传播；这是少样本适配，与纯零样本的信息预算不同。',
      '沿用此前固定测试集，phase6曾用于探索；CheXpert仍为596图12类，骨折仅1个阳性测试患者。',
      f'来源：{source}',f'来源SHA256：{hashlib.sha256(source.read_bytes()).hexdigest()}',
      f'审计：{run/"FINAL_AUDIT.json"}']
    text='\n'.join(lines)+'\n'
    (ROOT/'QUERY_REFINEMENT_SUMMARY.txt').write_text(text)
    (run/'COMPARISON_SUMMARY.txt').write_text(text)
    artifact={'classification_scope_summary':summary,'unique_classification_dataset_summary':unique_summary,
              'classification_pattern_counts':totals,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (run/'COMPARISON_SUMMARY.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    fieldnames=['model','dataset','task','test_images','support_images','status',
      *[f'{k}_{suffix}' for k in KEYS for suffix in ('original_percent','refined_percent','delta_pp')],'change','reason']
    csvrows=[]
    for r in data['records']:
        out=dict(model=r['model'],dataset=r['dataset'],task=r['task'],test_images=r['rows'],
          support_images=r.get('support_count',''),status=r['status'],reason=r.get('reason',''))
        if r['status']=='complete':
            a,b=r['branches']['original'],r['branches']['tta_center_refined']
            for k in KEYS:
                if k in a and a[k] is not None:
                    out.update({k+'_original_percent':100*a[k],k+'_refined_percent':100*b[k],k+'_delta_pp':100*(b[k]-a[k])})
            if r['task']=='classification':out['change']=pattern(r)
            else:
                k='pointing_game' if r['task']=='grounding' else 'dice'
                change=100*(b[k]-a[k])
                out['change']='近乎持平' if abs(change)<.005 else '上升' if change>0 else '下降'
        csvrows.append(out)
    for dest in (ROOT/'QUERY_REFINEMENT_COMPARISON.csv',run/'COMPARISON.csv'):
        with dest.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(csvrows)
    print(json.dumps({'summary':str(ROOT/'QUERY_REFINEMENT_SUMMARY.txt'),'csv':str(ROOT/'QUERY_REFINEMENT_COMPARISON.csv'),
          'counts':data['counts'],'classification_pattern_counts':totals},ensure_ascii=False))

if __name__=='__main__':main()
