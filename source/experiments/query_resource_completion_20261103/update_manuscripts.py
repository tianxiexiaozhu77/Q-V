#!/usr/bin/env python3
"""Update bilingual prose from verified six-scope completion and prior snapshots."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = Path(__file__).resolve().parent
PAPER = ROOT / 'latex_ours'
PRIOR = BATCH / 'integration/prior/latex_ours'


def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new)


def main():
    doc = json.loads((ROOT / 'baselines/paper_draft_zh/verified_results.json').read_text())
    assert doc['summary']['counts'] == {'complete': 148, 'unsupported': 9}
    assert doc['resource_completion_verified_source']
    rows = [r for r in doc['records'] if r['status'] == 'complete' and r['task'] == 'classification']
    assert len(rows) == 130
    counts = Counter(r['change'] for r in rows)
    unique = [r for r in rows if r['dataset'] != ('siim_pneumothorax' if r['model'] == 'RadZero' else 'siim_radzero_holdout')]
    assert len(unique) == 120
    uc = Counter(r['change'] for r in unique)
    means = doc['summary']['classification_means_unique_datasets']
    assert all(m['count'] == 10 for m in means.values())
    audit = json.loads((PAPER / 'build/CLASSIFICATION_TABLE_AUDIT.json').read_text())
    effect = '/'.join(f'{audit["query_plus_view_minus_query_pp"][m]:+.2f}' for m in ['macro_auroc','macro_auprc'])
    deltas = {model: '/'.join(f'{means[model]["delta_pp"][m]:+.2f}' for m in ['macro_auroc','macro_auprc']) for model in ['MedKLIP','MAVL']}
    up, down, mixed = (counts[k] for k in ['all_primary_up','all_primary_down','mixed_or_tied'])
    uup, udown, umixed = (uc[k] for k in ['all_primary_up','all_primary_down','mixed_or_tied'])
    updates = {}
    for filename, lang in [('main.tex','en'),('main_zh.tex','zh')]:
        text = (PRIOR / filename).read_text()
        if lang == 'en':
            pairs = [
                ('Original uses the original query vectors $q_c$', "Original uses the task's unadapted query vectors $q_c$"),
                ('V uses the original query vectors $q_c$', 'V uses the same unadapted query vectors $q_c$'),
                ('96 of 124 classification comparisons', f'{up} of 130 classification comparisons'),
                ('142 paired evaluations','148 paired evaluations'),
                ('contains 124 classification','contains 130 classification'),
                ('Of the 142 main comparisons','Of the 148 main comparisons'),
                ('remaining 127 comparisons','remaining 133 comparisons'),
                ('and seven additional AlphaRAD/RadZero evaluations completing the supported classification coverage.', 'seven additional AlphaRAD/RadZero evaluations, and six MedKLIP/MAVL evaluations with extended disease descriptions.'),
                ('and seven additional AlphaRAD/RadZero classification evaluations.', 'seven additional AlphaRAD/RadZero classification evaluations, and six MedKLIP/MAVL evaluations with extended disease descriptions.'),
                ('Across all 124 classification comparisons, 96 improve both metrics, 16 decline in both, and 12 have mixed signs. Removing duplicate SIIM ranges yields 114 tasks (89 joint improvements, 14 joint declines, and 11 mixed), corresponding to 94 model--dataset combinations after grouping the cardiac tasks.', f'Across all 130 classification comparisons, {up} improve both metrics, {down} decline in both, and {mixed} have mixed signs. Removing duplicate SIIM ranges yields 120 tasks ({uup} joint improvements, {udown} joint declines, and {umixed} mixed), corresponding to 100 model--dataset combinations after grouping the cardiac tasks.'),
                ("Negative transfer is material. MedKLIP's mean AUROC decreases by 0.19 pp and MAVL's mean AUPRC by 1.04 pp.", f"The mean AUROC/AUPRC changes are {deltas['MedKLIP']} pp for MedKLIP and {deltas['MAVL']} pp for MAVL. Performance declines remain at dataset level."),
                ('Performance declines remain at dataset level. At dataset level,','Performance declines remain at dataset level:'),
                ('Across the 94 model--dataset combinations, the mean Q+V--Q change is +0.25/+0.12 pp.',f'Across the 100 model--dataset combinations, the mean Q+V--Q change is {effect} pp.'),
                ('Different dataset coverage prevents interpreting these averages as a common-benchmark ranking.', 'Each model now covers the same ten datasets; the within-backbone comparison retains the different SIIM protocol for RadZero.'),
                (r'\paragraph{Query construction for cardiac tasks.}',r'\paragraph{Query construction for added categories.}'),
                ('For CheXchoNet, MedKLIP appends descriptions of the three cardiac tasks to its existing query context; MAVL retains nine concept query slots per task and fills them with the same task description. The Original and query-adapted branches use the same descriptions, query structure, model weights, and prediction heads to ensure consistent comparison settings. The exact descriptions and a CSLV example are provided in Appendix D.2.', 'For CheXchoNet, MedKLIP appends descriptions of the three cardiac tasks to its existing query context; MAVL retains nine query slots per task and fills them with the same task description. We use the same construction for tuberculosis on Shenzhen/Montgomery and for Calcified Granuloma and Granuloma on OpenI, appending only the missing categories for each dataset. Descriptions are fixed before evaluation. Original, Q, V, and Q+V share the extended query context, model weights, and prediction heads. The exact descriptions and construction examples are provided in Appendix D.2.'),
            ]
        else:
            pairs = [
                ('Original 使用原始查询向量 $q_c$', 'Original 使用当前任务中未经支持集适配的查询向量 $q_c$'),
                ('V 使用原始查询向量 $q_c$', 'V 使用同一未经适配的查询向量 $q_c$'),
                ('在 124 项分类比较中的 96 项同时提高',f'在 130 项分类比较中的 {up} 项同时提高'),
                ('和 142 项配对评价','和 148 项配对评价'),
                ('包含 124 项分类比较','包含 130 项分类比较'),
                ('在 142 项主要比较中','在 148 项主要比较中'),
                ('其余 127 项比较','其余 133 项比较'),
                ('以及用于补齐可支持分类范围的七项 AlphaRAD/RadZero 评价。','补齐的七项 AlphaRAD/RadZero 评价，以及通过扩展疾病描述完成的六项 MedKLIP/MAVL 评价。'),
                ('以及补齐的七项 AlphaRAD/RadZero 分类评价。','补齐的七项 AlphaRAD/RadZero 分类评价，以及通过扩展疾病描述完成的六项 MedKLIP/MAVL 评价。'),
                ('在全部 124 项分类比较中，96 项双指标提高，16 项双指标下降，12 项一升一降。去掉 SIIM 重复范围后，得到 114 项任务：89 项双指标提高、14 项双指标下降、11 项一升一降；合并心脏任务后，对应 94 个模型--数据集组合。',f'在全部 130 项分类比较中，{up} 项双指标提高，{down} 项双指标下降，{mixed} 项一升一降。去掉 SIIM 重复范围后，得到 120 项任务：{uup} 项双指标提高、{udown} 项双指标下降、{umixed} 项一升一降；合并心脏任务后，对应 100 个模型--数据集组合。'),
                ('负迁移不可忽略。MedKLIP 的平均 AUROC 下降 0.19 个百分点，MAVL 的平均 AUPRC 下降 1.04 个百分点。',f'MedKLIP 和 MAVL 的平均 AUROC/AUPRC 变化分别为 {deltas["MedKLIP"]} 和 {deltas["MAVL"]} 个百分点。性能下降仍然存在。'),
                ('在 94 个模型--数据集组合中，Q+V 相对 Q 的平均变化为 +0.25/+0.12 个百分点。',f'在 100 个模型--数据集组合中，Q+V 相对 Q 的平均变化为 {effect} 个百分点。'),
                ('由于数据集覆盖范围不同，这些均值不用于共同基准上的跨模型排名。','所有模型均覆盖相同的十个数据集；该比较用于分析各底座内部的适配效果，RadZero 保留了不同的 SIIM 协议。'),
                (r'\paragraph{心脏任务的查询构建。}',r'\paragraph{新增类别的查询构建。}'),
                ('对于 CheXchoNet，MedKLIP 将三个心脏任务的描述追加到原有查询上下文；MAVL 为每个任务保留九个概念查询槽位，并使用同一条任务描述填充。Original 与查询适配分支使用相同的描述、查询结构、模型权重及预测头，以保证比较设置一致。具体描述及 CSLV 构建示例见附录 D.2。','对于 CheXchoNet，MedKLIP 将三个心脏任务的描述追加到原有查询上下文；MAVL 为每个任务保留九个查询槽位，并使用同一条任务描述填充。对于 Shenzhen/Montgomery 的肺结核，以及 OpenI 的钙化肉芽肿和肉芽肿，本文采用相同方式，仅追加当前数据集缺少的类别。描述在评价前固定。Original、Q、V、Q+V 共用相同的扩展查询上下文、模型权重及预测头。具体描述与构建示例见附录 D.2。'),
            ]
        for old,new in pairs: text=replace(text,old,new)
        updates[PAPER/filename]=text
    text=(PRIOR/'sec/supplementary.tex').read_text()
    pairs=[
        ('Original uses the original query vectors and one original view.', "Original uses the task's unadapted query vectors and one original view."),
        ('all 142 completed paired evaluations: 124 classification','all 148 completed paired evaluations: 130 classification'),
        ('1,064 branch/metric values','1,112 branch/metric values'),
        ('124 classification scopes become 114','130 classification scopes become 120'),
        ('then 94 model--dataset combinations','then 100 model--dataset combinations'),
        ('Model coverage differs, precluding a common-benchmark ranking from the aggregated means.','All ten models cover these ten classification datasets; the retained SIIM protocol differs for RadZero.'),
        ('the 142 main evaluations','the 148 main evaluations'),
        ('the 142-task main-evaluation count','the 148-task main-evaluation count'),
        ('Of the 142 main comparisons','Of the 148 main comparisons'),
        ('remaining 127 comparisons','remaining 133 comparisons'),
        ('and seven additional AlphaRAD/RadZero evaluations completing the supported classification coverage.','seven additional AlphaRAD/RadZero evaluations, and six MedKLIP/MAVL evaluations with extended disease descriptions.'),
        ("The fifteen NA entries have explicit resource limitations: BiomedCLIP, CheXzero, and CARZero lack a verified implementation of the model's spatial response output in the current evaluation (three spatial tasks each); MedKLIP and MAVL lack the required tuberculosis resources for Shenzhen/Montgomery and the Calcified Granuloma/Granuloma resources for OpenI.","The nine NA entries concern spatial prediction: BiomedCLIP, CheXzero, and CARZero lack a verified implementation of the model's spatial response output in the current evaluation (three spatial tasks each). All classification scopes are completed; MedKLIP and MAVL use the fixed description extensions detailed in Appendix D.2 for Shenzhen, Montgomery, and OpenI."),
        ('Tables are generated from saved numerical records. No additional model inference was performed to prepare this document.','Table generation reads saved numerical records and does not run model inference.'),
    ]
    for old,new in pairs:text=replace(text,old,new)
    desc=json.loads((BATCH/'query_descriptions.json').read_text())['descriptions']
    paragraph=r'''\paragraph{Additional disease-query descriptions.}
Six fixed-support evaluations extend MedKLIP and MAVL to Shenzhen, Montgomery, and OpenI. Both models use the same study-defined descriptions in Table~\ref{tab:supp-added-disease-prompts}, fixed before these evaluations. Each dataset reuses the existing support/test identities, labels, and class order: Shenzhen has 10 supports and 652 test images, Montgomery has 10 and 128, and OpenI has 90 and 7,294. OpenI retains all 18 labels.

The frozen Bio\_ClinicalBERT encoder produces one CLS embedding per added description, using maximum token length 64. For tuberculosis, MedKLIP appends one query to its original 75; for OpenI, it appends Calcified Granuloma and Granuloma in the dataset's label order, giving 77 queries. MAVL retains the original 75 diseases and their nine slots each, appending nine copies of each new description embedding, giving 76 or 77 diseases. Its eight concept slots and one description slot receive the same sentence for each added disease, as in the cardiac extension. The existing query tensors, pretrained weights, and prediction heads are retained. Query self-attention can change existing-class outputs when the context is extended, so Original, Q, V, and Q+V are all evaluated with the same extended context. All other settings remain fixed: $\eta=0.3$, crop-side ratio 0.8, and equal averaging of the two views' probabilities.

\begin{table*}[t]
\centering
\caption{\textbf{Fixed additional disease descriptions.} These sentences are defined for this study and shared by MedKLIP, MAVL, and all four inference branches.}
\label{tab:supp-added-disease-prompts}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lp{0.77\textwidth}@{}}
\toprule
Disease & Description \\
\midrule
'''
    for name in ['Tuberculosis','Calcified Granuloma','Granuloma']:
        paragraph+=name+' & '+desc[name]+r' \\'+'\n'
    paragraph+=r'''\bottomrule
\end{tabular*}
\end{table*}

'''
    text=replace(text,r'\paragraph{CheXpert expansion.}',paragraph+r'\paragraph{CheXpert expansion.}')
    updates[PAPER/'sec/supplementary.tex']=text
    text=(PRIOR/'supplement_data/README.txt').read_text()
    text=replace(text,'142 completed tasks, four branches, 1,064 metric values.','148 completed tasks, four branches, 1,112 metric values.')
    text=replace(text,'127 subsequent comparisons','133 subsequent comparisons')
    updates[PAPER/'supplement_data/README.txt']=text
    for path,text in updates.items():
        before=(PRIOR/path.relative_to(PAPER)).read_text()
        assert path.read_text() in (before,text), ('unexpected concurrent edit',str(path))
    for path,text in updates.items():path.write_text(text)
    report={'classification_changes':dict(counts),'deduplicated_changes':dict(uc),'model_delta_pp':{m:means[m]['delta_pp'] for m in ['MedKLIP','MAVL']},'query_plus_view_minus_query_pp':audit['query_plus_view_minus_query_pp']}
    (BATCH/'integration/MANUSCRIPT_UPDATE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
