#!/usr/bin/env python3
"""Synchronize manuscript claims after all seven completion runs are verified."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'latex_ours'
VERIFIED = ROOT / 'baselines/paper_draft_zh/verified_results.json'


def replace_once(text, before, after):
    assert text.count(before) == 1, (before, text.count(before))
    return text.replace(before, after, 1)


def main():
    updates = {}
    verified = json.loads(VERIFIED.read_text())
    assert verified['summary']['counts']['complete'] == 142
    assert verified['summary']['complete_task_counts']['classification'] == 124
    assert verified['completion_verified_source']
    records = [r for r in verified['records'] if r['status'] == 'complete' and r['task'] == 'classification']
    all_counts = Counter(r['change'] for r in records)
    unique = [r for r in records if r['dataset'] !=
              ('siim_pneumothorax' if r['model'] == 'RadZero' else 'siim_radzero_holdout')]
    unique_counts = Counter(r['change'] for r in unique)
    assert len(unique) == 114
    summary = verified['summary']['classification_means_unique_datasets']
    assert sum(s['count'] for s in summary.values()) == 94
    audit = json.loads((PAPER / 'build/CLASSIFICATION_TABLE_AUDIT.json').read_text())
    effect = audit['query_plus_view_minus_query_pp']
    delta = {model: '/'.join(f'{summary[model]["delta_pp"][metric]:+.2f}'
                            for metric in ('macro_auroc', 'macro_auprc'))
             for model in ('AlphaRAD', 'RadZero')}
    qv_delta = '/'.join(f'{effect[metric]:+.2f}' for metric in ('macro_auroc', 'macro_auprc'))
    rad_q = 100 * summary['RadZero']['means']['query_refined']['macro_auprc']
    rad_qv = 100 * summary['RadZero']['means']['tta_center_refined']['macro_auprc']
    up, down, mixed = (all_counts[k] for k in ('all_primary_up', 'all_primary_down', 'mixed_or_tied'))
    uup, udown, umixed = (unique_counts[k] for k in ('all_primary_up', 'all_primary_down', 'mixed_or_tied'))

    for filename, language in [('main.tex', 'en'), ('main_zh.tex', 'zh')]:
        path = PAPER / filename
        text = path.read_text()
        if language == 'en':
            replacements = [
                ('90 of 117 classification comparisons', f'{up} of 124 classification comparisons'),
                ('135 paired evaluations', '142 paired evaluations'),
                ('contains 117 classification and 18 spatial comparisons', 'contains 124 classification and 18 spatial comparisons'),
                ('Of the 135 main comparisons', 'Of the 142 main comparisons'),
                ('All remaining 120 comparisons used these fixed settings, covering eight additional backbones, AlphaRAD/CheXpert, and ten-model CheXchoNet.',
                 'All remaining 127 comparisons used these fixed settings, covering eight additional backbones, AlphaRAD/CheXpert, ten-model CheXchoNet, and seven additional AlphaRAD/RadZero evaluations completing the supported classification coverage.'),
                ('AlphaRAD gains 1.70/3.30 pp in dataset-averaged AUROC/AUPRC with Q+V, and RadZero gains 1.97/3.80 pp. BiomedCLIP and MaCo show larger gains',
                 f'Q+V changes dataset-averaged AUROC/AUPRC by {delta["AlphaRAD"]} pp for AlphaRAD and {delta["RadZero"]} pp for RadZero. BiomedCLIP and MaCo show gains'),
                ('Across all 117 classification comparisons, 90 improve both metrics, 16 decline in both, and 11 have mixed signs. Removing duplicate SIIM ranges yields 109 tasks (85 joint improvements, 14 joint declines, and 10 mixed), corresponding to 89 model--dataset combinations after grouping the cardiac tasks.',
                 f'Across all 124 classification comparisons, {up} improve both metrics, {down} decline in both, and {mixed} have mixed signs. Removing duplicate SIIM ranges yields 114 tasks ({uup} joint improvements, {udown} joint declines, and {umixed} mixed), corresponding to 94 model--dataset combinations after grouping the cardiac tasks.'),
                ("We count only the 1,704-image SIIM protocol for the eight transferred models, while retaining RadZero's 1,694-image protocol.",
                 "We retain one SIIM protocol per model: 1,694 images for RadZero and 1,704 images for every other model."),
                ('Across the 89 model--dataset combinations, Q+V adds only 0.25/0.12 pp over Q. For RadZero, AUPRC decreases from 44.64 with Q to 44.59 with Q+V.',
                 f"Across the 94 model--dataset combinations, the mean Q+V--Q change is {qv_delta} pp. RadZero's AUPRC is {rad_q:.2f} with Q and {rad_qv:.2f} with Q+V."),
            ]
            replacements.append(('The subsequent evaluations apply the fixed adaptation rule and its parameters to eight additional backbones, AlphaRAD/CheXpert, and ten-model CheXchoNet.', 'The subsequent evaluations apply the fixed adaptation rule and its parameters to eight additional backbones, AlphaRAD/CheXpert, ten-model CheXchoNet, and seven additional AlphaRAD/RadZero classification evaluations.'))
            replacements.append(('MaCo also performs better with Q alone', 'MaCo performs better with Q alone'))
            component_input = r'\input{sec/component_ablations}'
        else:
            replacements = [
                ('在 117 项分类比较中的 90 项同时提高', f'在 124 项分类比较中的 {up} 项同时提高'),
                ('和 135 项配对评价', '和 142 项配对评价'),
                ('包含 117 项分类比较和 18 项空间比较', '包含 124 项分类比较和 18 项空间比较'),
                ('在 135 项主要比较中', '在 142 项主要比较中'),
                ('其余 120 项比较均在此后采用固定方案开展，覆盖八个新增底座、AlphaRAD/CheXpert 和十模型 CheXchoNet。',
                 '其余 127 项比较均在此后采用固定方案开展，覆盖八个新增底座、AlphaRAD/CheXpert、十模型 CheXchoNet，以及用于补齐可支持分类范围的七项 AlphaRAD/RadZero 评价。'),
                ('使用 Q+V 后，AlphaRAD 按数据集平均的 AUROC/AUPRC 提高 1.70/3.30 个百分点，RadZero 提高 1.97/3.80 个百分点。BiomedCLIP 和 MaCo 的提升更大，',
                 f'使用 Q+V 后，AlphaRAD 按数据集平均的 AUROC/AUPRC 变化为 {delta["AlphaRAD"]} 个百分点，RadZero 为 {delta["RadZero"]} 个百分点。BiomedCLIP 和 MaCo 的提升'),
                ('在全部 117 项分类比较中，90 项双指标提高，16 项双指标下降，11 项一升一降。去掉 SIIM 重复范围后，得到 109 项任务：85 项双指标提高、14 项双指标下降、10 项一升一降；合并心脏任务后，对应 89 个模型--数据集组合。',
                 f'在全部 124 项分类比较中，{up} 项双指标提高，{down} 项双指标下降，{mixed} 项一升一降。去掉 SIIM 重复范围后，得到 114 项任务：{uup} 项双指标提高、{udown} 项双指标下降、{umixed} 项一升一降；合并心脏任务后，对应 94 个模型--数据集组合。'),
                ('对于八个迁移模型，SIIM 仅计入 1,704 张图像的协议，同时保留 RadZero 的 1,694 张图像协议。',
                 '每个模型仅保留一种 SIIM 协议：RadZero 使用 1,694 张图像，其余模型使用 1,704 张图像。'),
                ('在 89 个模型--数据集组合中，Q+V 相对 Q 仅增加 0.25/0.12 个百分点。对于 RadZero，AUPRC 从 Q 的 44.64 降至 Q+V 的 44.59。',
                 f'在 94 个模型--数据集组合中，Q+V 相对 Q 的平均变化为 {qv_delta} 个百分点。RadZero 的 AUPRC 在 Q 和 Q+V 下分别为 {rad_q:.2f} 和 {rad_qv:.2f}。'),
            ]
            replacements.append(('后续评价将固定的适配规则及其参数用于八个新增模型、AlphaRAD/CheXpert 和十模型 CheXchoNet。', '后续评价将固定的适配规则及其参数用于八个新增模型、AlphaRAD/CheXpert、十模型 CheXchoNet，以及补齐的七项 AlphaRAD/RadZero 分类评价。'))
            replacements.append(('MaCo 也在仅使用 Q 时表现更好', 'MaCo 在仅使用 Q 时表现更好'))
            component_input = r'\input{sec/component_ablations_zh}'
        for before, after in replacements:
            text = replace_once(text, before, after)
        label_at = text.index(r'\label{tab:components}')
        table_start = text.rfind(r'\begin{table*}', 0, label_at)
        table_end = text.index(r'\end{table*}', label_at) + len(r'\end{table*}')
        text = text[:table_start] + component_input + text[table_end:]
        updates[path] = text

    path = PAPER / 'sec/supplementary.tex'
    text = path.read_text()
    for before, after in [
        ('all 135 completed paired evaluations: 117 classification', 'all 142 completed paired evaluations: 124 classification'),
        ('all 1,008 branch/metric values', 'all 1,064 branch/metric values'),
        ("Dataset-averaged main-paper results exclude the duplicate 1,694-image range for the eight transferred models, while retaining RadZero's own protocol.",
         'Dataset-averaged main-paper results retain one SIIM protocol per model: 1,694 images for RadZero and 1,704 images for every other model.'),
        ('Thus 117 classification scopes become 109 after excluding duplicate SIIM ranges, and then 89 model--dataset combinations',
         'Thus 124 classification scopes become 114 after excluding duplicate SIIM ranges, and then 94 model--dataset combinations'),
        ('additional to the 135 main evaluations', 'additional to the 142 main evaluations'),
        ('the 135-task main-evaluation count', 'the 142-task main-evaluation count'),
        ('Of the 135 main comparisons', 'Of the 142 main comparisons'),
        ('all remaining 120 comparisons, covering eight additional backbones, AlphaRAD/CheXpert, and ten-model CheXchoNet.',
         'all remaining 127 comparisons, covering eight additional backbones, AlphaRAD/CheXpert, ten-model CheXchoNet, and seven additional AlphaRAD/RadZero evaluations completing the supported classification coverage.'),
        ("Separate training data are used for AlphaRAD's SIIM supports", "Separate training data are used for the 1,704-image SIIM protocol's supports"),
    ]:
        text = replace_once(text, before, after)
    text = replace_once(text, 'Their main SIIM evaluation uses the 1,704-image scope; the 1,694-image RadZero scope is listed separately.',
                        'Both SIIM protocols are evaluated for all ten models; each protocol uses the same support and test images across models.')
    insertion = r'\paragraph{Additional classification coverage.}'
    assert insertion not in text
    paragraph = (
        insertion + '\n'
        'Seven additional fixed-support evaluations cover AlphaRAD on OpenI and SIIM (1,694), and RadZero on CheXpert, Shenzhen, Montgomery, CovidKaggle, and SIIM (1,704). '
        'They reuse the existing support/test identities, labels, and class order without resampling or parameter selection. '
        'For all models, SIIM (1,704) uses ten supports from separate training DICOMs; SIIM (1,694) uses the ten supports held out from the original 1,704-image evaluation list. '
        'Each model retains its native image preprocessing. '
        'For the additional RadZero tasks, disease names are inserted into the fixed native template \\texttt{There is [disease]}; '
        'Consolidation, Edema, and Fracture use the phrases \\texttt{Pulmonary Consolidation}, \\texttt{Pulmonary Edema}, and \\texttt{Bone Fracture}, respectively. '
        'Original, Q, V, and Q+V share these descriptions and the frozen weights.\n\n'
    )
    marker = r'\paragraph{CheXpert expansion.}'
    assert text.count(marker) == 1
    text = text.replace(marker, paragraph + marker, 1)
    updates[path] = text

    source_table = PAPER / 'sec/support_sources.tex'
    text = source_table.read_text().replace('SIIM (RadZero)', 'SIIM (1,694)').replace(
        'SIIM (AlphaRAD and transferred models)', 'SIIM (1,704)')
    updates[source_table] = text
    data_readme = PAPER / 'supplement_data/README.txt'
    text = replace_once(data_readme.read_text(),
                        '135 completed tasks, four branches, 1,008 metric values.',
                        '142 completed tasks, four branches, 1,064 metric values.')
    text = replace_once(text, '120 subsequent comparisons', '127 subsequent comparisons')
    updates[data_readme] = text
    # Validate every replacement before touching any manuscript source.
    for path, text in updates.items():
        path.write_text(text)
    report = {'classification_changes': dict(all_counts), 'deduplicated_changes': dict(unique_counts),
              'model_delta_pp': {m: summary[m]['delta_pp'] for m in ('AlphaRAD', 'RadZero')},
              'query_plus_view_minus_query_pp': effect}
    (Path(__file__).resolve().parent / 'integration/MANUSCRIPT_UPDATE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
