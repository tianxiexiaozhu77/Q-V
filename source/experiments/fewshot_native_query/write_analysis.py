#!/usr/bin/env python3
"""Generate a complete descriptive first-round report; never choose a deployed method."""
import argparse
import json
from pathlib import Path
from cache_native_common import sha,write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--summary',type=Path,required=True)
a=p.parse_args();out=a.summary.resolve()
catalog=json.loads((out/'catalog.json').read_text())
if catalog['status']!='complete' or len(catalog['jobs'])!=6:
    raise ValueError('Report requires all six complete formal combinations')
results=[];artifacts={};audits=[]
for job in catalog['jobs'].values():
    path=Path(job['result_file']);r=json.loads(path.read_text());results.append(r)
    if r['status']!='complete' or len(r['methods'])!=11:
        raise ValueError('Incomplete result')
    audit=path.parent/'independent_metric_audit.json'
    if not audit.exists() or json.loads(audit.read_text())['status']!='passed':
        raise ValueError('Independent metric audit is not complete: '+str(path))
    audits.append(str(audit))
    for directory in (path.parent,Path(r['cache'])):
        for artifact in directory.iterdir():
            if artifact.is_file() and artifact.suffix in ('.json','.npz'):
                artifacts[str(artifact)]=sha(artifact)
results.sort(key=lambda r:({'BiomedCLIP':0,'AlphaRAD':1,'RadZero':2}[r['model']],r['dataset']))
fmt=lambda x:f'{100*x:.2f}'
pair=lambda r,k:fmt(r['methods'][k]['metrics']['macro_auroc'])+' / '+fmt(r['methods'][k]['metrics']['macro_auprc'])
lines=['首轮同支持集比较与机制消融：完成报告','',
       '6/6 模型×数据集组合完成，每组11个分类分支，共66个分类结果；AlphaRAD另有5个空间分支。',
       '测试范围：RSNA 10支持/5327测试；ChestXDet10 50支持/492分类测试，定位为其中409张。',
       '全部骨干冻结；Tip/GDA/原型及查询修正无梯度训练，SimNL为可训练的特征级OvR移植。',
       '固定eta=0.3与中心裁剪0.8不在本轮搜索；可调比较法仅在同一支持集五折OOF选参。',
       '同一支持集、同一测试集、相同类别顺序；各底座保留原生预处理。',
       'AUPRC使用PR曲线梯形面积。以下单元格为AUROC / AUPRC，单位%。','',
       '同为单视图的少样本分类比较（Q为正负差分修正原生查询）：','',
       '| 模型 | 数据集 | 原始 | 原型分类器 | 原型混合 | Tip-Adapter OvR | GDA KS OvR | SimNL特征OvR | Q |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|']
comparators=['prototype_score_only','prototype_score_mix_oof','tip_adapter_ovr_oof','gda_ovr_ks_oof','simnl_feature_ovr']
wins={m:0 for m in ('macro_auroc','macro_auprc')}
all_score_wins={m:0 for m in wins}
for r in results:
    keys=['original']+comparators+['query']
    lines.append('| '+r['model']+' | '+r['dataset']+' | '+' | '.join(pair(r,k) for k in keys)+' |')
    for metric in wins:
        if r['methods']['query']['metrics'][metric] > max(r['methods'][k]['metrics'][metric] for k in comparators):
            wins[metric]+=1
        if r['methods']['query']['metrics'][metric] > max(r['methods'][k]['metrics'][metric]
                for k in comparators+['class_local_difference_score_oof']):
            all_score_wins[metric]+=1
lines+=['',f"在这次固定支持抽样中，单视图Q的AUROC在{wins['macro_auroc']}/6组高于表列全部少样本分类对照；AUPRC为{wins['macro_auprc']}/6组。",
        f"如果把下表的同特征分数机制对照也纳入，Q高于全部对照的组数为AUROC {all_score_wins['macro_auroc']}/6、AUPRC {all_score_wins['macro_auprc']}/6。",
        '这是描述性比较，不等于统计显著性，也不能将不同移植版的表现写成作者原方法在标准协议下的结论。','',
        '机制消融（相同支持图）：','',
        '| 模型 | 数据集 | 原始 | 仅阳性原型改查询 | 同类局部差分改分数 | 正负差分改查询Q | 仅双视图V | Q+V |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
for r in results:
    keys=['original','positive_query','class_local_difference_score_oof','query','view','query_view']
    lines.append('| '+r['model']+' | '+r['dataset']+' | '+' | '.join(pair(r,k) for k in keys)+' |')
lines+=['','特别需要保留的反例：']
for r in results:
    q=r['methods']['query']['metrics'];c=r['methods']['class_local_difference_score_oof']['metrics']
    if c['macro_auroc']>q['macro_auroc'] or c['macro_auprc']>q['macro_auprc']:
        lines.append(f"  {r['model']} / {r['dataset']}：同特征分数修正 {pair(r,'class_local_difference_score_oof')}，"
                     f"查询修正 {pair(r,'query')}。不能声称修改原生查询普遍优于修改最终分数。")
lines+=['','空间能力（AlphaRAD × ChestXDet10，409张；分类分数对照没有适配后的空间头，故不伪造PG）：','',
        '| 分支 | PG % | 相对原始变化/百分点 |','|---|---:|---:|']
for r in results:
    if r['spatial']:
        for key,item in r['spatial'].items():
            lines.append(f"| {r['methods'][key]['name']} | {fmt(item['pointing_game'])} | {item['delta_pp']:+.2f} |")
lines+=['','支持集内部选择（所有候选记录在各 evaluation/fit_info.json）：','',
        '| 模型 | 数据集 | 方法 | 支持集OOF选择 |','|---|---|---|---|']
for r in results:
    fit=json.loads((Path(r['cache']).parent/'evaluation/fit_info.json').read_text())
    for k,info in fit.items():
        if 'selected_parameters' in info and info['selected_parameters']:
            selection=json.dumps(info['selected_parameters'],ensure_ascii=False,sort_keys=True)
        elif 'chosen_epoch' in info:selection='epoch='+str(info['chosen_epoch'])
        else:continue
        lines.append(f"| {r['model']} | {r['dataset']} | {r['methods'][k]['name']} | {selection} |")
lines+=['','必须保留的解释边界：',
 '1. AlphaRAD/RadZero的全局特征对照与查询方法使用的类局部统计不同；同类局部分数对照才更接近注入位置的受控比较。',
 '2. BiomedCLIP共享线性嵌入空间中的查询修正可展开为分数项组合，不单独证明非线性注意力机制。',
 '3. 本轮既没有多次支持抽样，也没有新医院独立测试；现有两个测试集曾参与方法探索。上述胜出次数不能替代置信区间。',
 '4. Q+V比单视图方法多一次视觉前向，不能将其全部优势归于查询修正；在某些组合上它低于Q。',
 '5. Tip/GDA是二元OvR移植；SimNL是无图像增强/GPT描述的特征级OvR移植，详见SOURCES文件，不称作者精确复现。',
 '6. RSNA使用已知患者ID分组；ChestXDet10只有图像/哈希分组，不能声明患者级完全排重。',
 '7. 原始float32概率与raw margin的数值并列可能造成微小差异；BiomedCLIP ChestXDet Tip选alpha=0，其不足0.001个百分点差异不计作适配收益。',
 '8. SimNL的epoch=0仍利用支持缓存，不能解释成完全零样本原始模型。',
 '9. 本轮运行期间曾补充仅作用于校验/元数据的检查，保留各控制器启动来源hash和各worker记录的运行末磁盘hash；不要将末尾hash误当作执行时源码快照。','',
 '完整逐行结果：RESULTS.txt / COMPARISON.csv。每类指标、预测数组和选择记录路径见catalog.json。',
 '运行方法及环境：../RUN_LOCAL.txt；完整复跑：bash experiments/fewshot_native_query/run.sh。']
(out/'ANALYSIS.txt').write_text('\n'.join(lines)+'\n')
write_json(out/'ARTIFACT_MANIFEST.json',{'status':'complete','classification_results':66,
    'spatial_branches':5,'independent_audits':audits,'artifacts_sha256':artifacts,
    'note':'Artifact checksums at finalization; existing source provenance retained separately.'})
print(out/'ANALYSIS.txt')
