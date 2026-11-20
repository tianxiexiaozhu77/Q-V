#!/usr/bin/env python3
"""Independently compare rendered Markdown table cells to verified raw JSON."""
import hashlib
import json
import math
import re
from pathlib import Path

import mistune

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'CVPR_中文初稿.md'
RAW = json.loads((HERE / 'verified_results.json').read_text())
FEWSHOT_SOURCE = (HERE.parents[1] / 'experiments/fewshot_native_query/'
                  'first_round_20261024/RESULTS.json')
FEWSHOT_METHODS = {
    '原始模型': 'original',
    '原型分类器': 'prototype_score_only',
    '原型分数混合': 'prototype_score_mix_oof',
    'Tip-Adapter': 'tip_adapter_ovr_oof',
    'GDA': 'gda_ovr_ks_oof',
    'SimNL': 'simnl_feature_ovr',
    '仅阳性原型修正查询': 'positive_query',
    '同特征正负差分修正分数': 'class_local_difference_score_oof',
    '查询修正 Q': 'query',
    '仅双视图 V': 'view',
    '查询修正＋双视图 Q+V': 'query_view',
}
MULTISEED_SOURCE = (HERE.parents[1] / 'experiments/fewshot_multiseed_20261024/'
                    'runs/full_20261024_v2/RESULTS.json')
MULTISEED_METHODS = {
    '原始模型': 'original',
    '原型分类器': 'prototype_score_only',
    '原型分数混合': 'prototype_score_mix_oof',
    'Tip-Adapter': 'tip_adapter_ovr_oof',
    'GDA': 'gda_ovr_ks_oof',
    'SimNL': 'simnl_feature_ovr',
    '仅阳性原型查询': 'positive_query',
    '同特征差分分数': 'class_local_difference_score_oof',
    '查询修正 Q': 'query',
    '仅双视图 V': 'view',
    'Q+V': 'query_view',
}
MULTISEED_COLUMNS = [
    ('BiomedCLIP RSNA', 'BiomedCLIP', 'rsna_pneumonia'),
    ('BiomedCLIP ChestXDet10', 'BiomedCLIP', 'chestxdet10'),
    ('AlphaRAD RSNA', 'AlphaRAD', 'rsna_pneumonia'),
    ('AlphaRAD ChestXDet10', 'AlphaRAD', 'chestxdet10'),
]
BRANCHES = ['original', 'query_refined', 'tta_center_original', 'tta_center_refined']
LABEL = {
    'chest14': 'ChestXray14', 'chestxdet10': 'ChestXDet10 分类',
    'chestxdet10_gt': 'ChestXDet10 定位', 'rsna_pneumonia': 'RSNA',
    'siim_pneumothorax': 'SIIM（1704）', 'siim_radzero_holdout': 'SIIM（1694）',
    'openi': 'OpenI', 'chexpert': 'CheXpert', 'shenzhenxray': 'Shenzhen',
    'montgomery': 'Montgomery', 'covidkaggle': 'CovidKaggle',
    'node21': 'Node21 定位', 'qatacovid_seg': 'QaTaCOV19 分割',
    'chexchonet_slvh': 'CheXchoNet CSLV', 'chexchonet_dlv': 'CheXchoNet CDLV',
    'chexchonet_composite_slvh_dlv': 'CheXchoNet CCOM',
}
LOOKUP = {(r['model'], LABEL[r['dataset']]): r for r in RAW['records']}
checked = 0


def text(node):
    if isinstance(node, list):
        return ''.join(text(x) for x in node)
    return node.get('text', node.get('raw', '')) or text(node.get('children', []))


def check(cell, expected):
    global checked
    decimals = len(cell.rsplit('.', 1)[-1]) if '.' in cell else 0
    assert abs(float(cell) - expected) <= 0.5 * 10 ** -decimals + 1e-9, (cell, expected)
    checked += 1


def comparison(cells, metrics):
    original, new = metrics['original'], metrics['tta_center_refined']
    check(cells[0], 100 * original['macro_auroc'])
    check(cells[1], 100 * new['macro_auroc'])
    check(cells[2], 100 * (new['macro_auroc'] - original['macro_auroc']))
    check(cells[3], 100 * original['macro_auprc'])
    check(cells[4], 100 * new['macro_auprc'])
    check(cells[5], 100 * (new['macro_auprc'] - original['macro_auprc']))


def independent_dataset_means(model):
    raw = [r for r in RAW['records'] if r['status'] == 'complete' and r['model'] == model
           and r['task'] == 'classification'
           and (r['dataset'] != 'siim_radzero_holdout' or model == 'RadZero')]
    groups = {}
    for r in raw:
        # Independent of the precomputed summary and its dataset_group field.
        key = 'chexchonet' if r['dataset'].startswith('chexchonet_') else r['dataset']
        groups.setdefault(key, []).append(r)
    means = {b: {m: sum(sum(r['branches'][b][m] for r in group) / len(group)
                        for group in groups.values()) / len(groups)
                 for m in ('macro_auroc', 'macro_auprc')} for b in BRANCHES}
    return groups, means


def check_query_score_table(header, rows, multiseed_raw):
    """Audit the derived S + unchanged Original map control without assigning S a map."""
    before = checked
    assert header == ['方法／配对比较', 'AUROC', 'AUPRC', 'PG'], header
    assert [cells[0] for cells in rows] == [
        '原始模型 Original', '同特征分数修正 S＋保留原始空间分支†',
        '原生查询修正 Q', 'Q − S'], rows
    assert all(len(cells) == 4 for cells in rows), rows
    dataset = multiseed_raw['datasets']['chestxdet10']
    model = dataset['models']['AlphaRAD']
    spatial = model['spatial']
    score_method = 'class_local_difference_score_oof'
    assert dataset['test_images'] == 492 and spatial['images'] == 409
    assert dataset['resampling_unit'] == 'image_exact_duplicate_group'
    assert spatial['draw_plan_sha256'] == dataset['draw_plan_sha256']
    assert score_method not in spatial['methods']
    assert spatial['method_availability'][score_method] == 'NA: no native spatial output'

    def seed_mean_sd(raw):
        values = raw['seed_metric_percent']
        assert set(values) == {'42', '43', '44', '45', '46'}
        mean = math.fsum(values.values()) / len(values)
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values.values()) / (len(values) - 1))
        assert abs(mean - raw['mean_percent']) < 1e-10
        assert abs(sd - raw['support_seed_sd_pp']) < 1e-10
        return mean, sd

    for row_index, method in enumerate(('original', score_method, 'query')):
        for column, metric in enumerate(('auroc', 'auprc', 'pg'), 1):
            # S has no native spatial output. This explicitly composed control
            # retains Original's maps, so its PG is Original's by definition.
            spatial_method = 'original' if method == score_method else method
            raw = (spatial['methods'][spatial_method]['macro']['pg'] if metric == 'pg'
                   else model['methods'][method]['macro'][metric])
            mean, sd = seed_mean_sd(raw)
            cell = rows[row_index][column]
            if method == 'original':
                assert sd == 0.0
                check(cell, mean)
            elif method == score_method and metric == 'pg':
                assert sd == 0.0 and cell.endswith('（同 Original）'), cell
                check(cell.removesuffix('（同 Original）').strip(), mean)
            else:
                values = cell.split('±')
                assert len(values) == 2, cell
                check(values[0].strip(), mean)
                check(values[1].strip(), sd)

    for column, metric in enumerate(('auroc', 'auprc', 'pg'), 1):
        reference = 'original' if metric == 'pg' else score_method
        branch = spatial if metric == 'pg' else model
        pair = branch['comparisons'][f'query__minus__{reference}']['macro'][metric]
        query_seeds = branch['methods']['query']['macro'][metric]['seed_metric_percent']
        reference_seeds = branch['methods'][reference]['macro'][metric]['seed_metric_percent']
        gains = [query_seeds[seed] - reference_seeds[seed] for seed in query_seeds]
        mean_gain = math.fsum(gains) / len(gains)
        assert abs(mean_gain - pair['mean_gain_pp']) < 1e-10
        for seed, gain in zip(query_seeds, gains):
            assert abs(gain - pair['seed_gain_pp'][seed]) < 1e-10
        interval = pair['mean_gain_paired_cluster_bootstrap']
        assert interval['confidence_level'] == 0.95
        assert interval['valid_replicates'] == 2000 and interval['invalid_replicates'] == 0
        assert abs(interval['point'] - mean_gain) < 1e-10
        cell = rows[3][column].replace('−', '-')
        match = re.fullmatch(r'([+-]?\d+\.\d+)\s*［\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\s*］', cell)
        assert match, cell
        for value, expected in zip(match.groups(), (mean_gain, *interval['ci'])):
            check(value, expected)
    assert checked - before == 23, checked - before
    return checked - before


def main():
    fewshot_raw = json.loads(FEWSHOT_SOURCE.read_text())
    assert fewshot_raw['status'] == 'complete', fewshot_raw['status']
    fewshot_lookup = {(r['model'], r['dataset'], r['method']): r
                      for r in fewshot_raw['records']
                      if r['model'] in ('BiomedCLIP', 'AlphaRAD')}
    assert len(fewshot_lookup) == 44, len(fewshot_lookup)
    fewshot_seen = set()
    fewshot_tables = set()
    fewshot_numeric_cells = 0
    multiseed_raw = json.loads(MULTISEED_SOURCE.read_text())
    assert multiseed_raw['status'] == 'complete', multiseed_raw['status']
    assert multiseed_raw['support_seeds'] == [42, 43, 44, 45, 46]
    assert multiseed_raw['coverage']['per_seed_rows'] == 220
    assert multiseed_raw['coverage']['means_rows'] == 44
    assert set(multiseed_raw['methods']) == set(MULTISEED_METHODS.values())
    multiseed_seen = set()
    multiseed_numeric_cells = 0
    multiseed_tables = 0
    query_score_numeric_cells = 0
    query_score_tables = 0
    ast = mistune.create_markdown(renderer='ast', plugins=['table', 'strikethrough'])(SOURCE.read_text())
    tables = []
    heading = ''
    for node in ast:
        if node['type'] == 'heading':
            heading = text(node)
        elif node['type'] == 'table':
            header = [text(c) for c in node['children'][0]['children']]
            rows = [[text(c) for c in row['children']] for row in node['children'][1]['children']]
            tables.append((heading, header, rows))
    counts = {'main_mean_rows': 0, 'main_spatial_rows': 0, 'main_ablation_rows': 0,
              'chexchonet_main_rows': 0, 'classification_appendix_rows': 0,
              'ablation_appendix_rows': 0, 'fewshot_comparison_rows': 0,
              'fewshot_method_dataset_combinations': 0,
              'multiseed_comparison_rows': 0, 'multiseed_method_dataset_combinations': 0,
              'query_score_mechanism_rows': 0}
    for heading, header, rows in tables:
        if heading.startswith('5.1'):
            for cells in rows:
                model = cells[0]
                groups, means = independent_dataset_means(model)
                check(cells[1], len(groups))
                comparison(cells[2:], means)
                counts['main_mean_rows'] += 1
        elif heading.startswith('5.2'):
            for cells in rows:
                r = LOOKUP[(cells[0], cells[1])]
                check(cells[2], r['rows'])
                metric = 'pointing_game' if r['task'] == 'grounding' else 'dice'
                check(cells[4], 100 * r['branches']['original'][metric])
                check(cells[5], 100 * r['branches']['tta_center_refined'][metric])
                check(cells[6], r['delta_pp'][metric])
                counts['main_spatial_rows'] += 1
        elif heading.startswith('5.3'):
            for cells in rows:
                groups, means = independent_dataset_means(cells[0])
                check(cells[1], len(groups))
                for b, cell in zip(BRANCHES, cells[2:]):
                    values = cell.split('／')
                    assert len(values) == 2
                    check(values[0], 100 * means[b]['macro_auroc'])
                    check(values[1], 100 * means[b]['macro_auprc'])
                counts['main_ablation_rows'] += 1
        elif heading.startswith('5.5'):
            for cells in rows:
                r = LOOKUP[(cells[0].rstrip('*†'), 'CheXchoNet ' + cells[1])]
                check(cells[2], r['rows'])
                comparison(cells[3:], r['branches'])
                counts['chexchonet_main_rows'] += 1
        elif heading.startswith('5.6.3'):
            multiseed_tables += 1
            assert header == ['方法'] + [col[0] for col in MULTISEED_COLUMNS], header
            assert [cells[0] for cells in rows] == list(MULTISEED_METHODS), rows
            for cells in rows:
                assert len(cells) == 5, cells
                method = MULTISEED_METHODS[cells[0]]
                for (_, model, dataset), cell in zip(MULTISEED_COLUMNS, cells[1:]):
                    key = (model, dataset, method)
                    assert key not in multiseed_seen, ('duplicate five-seed branch', key)
                    multiseed_seen.add(key)
                    macro = multiseed_raw['datasets'][dataset]['models'][model]['methods'][method]['macro']
                    values = cell.split(' / ')
                    assert len(values) == 2, (key, cell)
                    for value, metric in zip(values, ('auroc', 'auprc')):
                        raw_metric = macro[metric]
                        seed_metrics = raw_metric['seed_metric_percent']
                        assert set(seed_metrics) == {'42', '43', '44', '45', '46'}
                        mean = math.fsum(seed_metrics.values()) / 5
                        assert abs(mean - raw_metric['mean_percent']) < 1e-10, (key, metric)
                        check(value, mean)
                        multiseed_numeric_cells += 1
                    counts['multiseed_method_dataset_combinations'] += 1
                counts['multiseed_comparison_rows'] += 1
        elif heading.startswith('5.6.4'):
            query_score_tables += 1
            query_score_numeric_cells += check_query_score_table(header, rows, multiseed_raw)
            counts['query_score_mechanism_rows'] += len(rows)
        elif heading.startswith('5.6'):
            # This addition is restricted to the two requested models.  These
            # method branches are not new independent datasets or main tasks.
            if heading.startswith('5.6.1'):
                model = 'BiomedCLIP'
            elif heading.startswith('5.6.2'):
                model = 'AlphaRAD'
            else:
                raise AssertionError(('unexpected few-shot table', heading))
            assert model in heading and 'RadZero' not in heading, heading
            assert model not in fewshot_tables, ('duplicate few-shot table', model)
            fewshot_tables.add(model)
            assert [cells[0] for cells in rows] == list(FEWSHOT_METHODS), (model, rows)
            for cells in rows:
                assert len(cells) == 3, cells
                method = FEWSHOT_METHODS[cells[0]]
                for dataset, cell in zip(('rsna_pneumonia', 'chestxdet10'), cells[1:]):
                    key = (model, dataset, method)
                    assert key not in fewshot_seen, ('duplicate few-shot branch', key)
                    fewshot_seen.add(key)
                    record = fewshot_lookup[key]
                    values = cell.split(' / ')
                    assert len(values) == 2, (key, cell)
                    check(values[0], 100 * record['auroc'])
                    check(values[1], 100 * record['auprc'])
                    fewshot_numeric_cells += 2
                    counts['fewshot_method_dataset_combinations'] += 1
                counts['fewshot_comparison_rows'] += 1
        elif heading.startswith('A.'):
            model = heading.split(' ', 1)[1]
            for cells in rows:
                r = LOOKUP[(model, cells[0])]
                assert r['task'] == 'classification' and r['status'] == 'complete'
                check(cells[1], r['rows'])
                comparison(cells[2:], r['branches'])
                counts['classification_appendix_rows'] += 1
        elif heading.startswith('C.'):
            model = heading.split(' ', 1)[1]
            for cells in rows:
                r = LOOKUP[(model, cells[0])]
                assert r['status'] == 'complete'
                for b, cell in zip(BRANCHES, cells[1:]):
                    if r['task'] == 'classification':
                        v = cell.split('／')
                        assert len(v) == 2
                        check(v[0], 100 * r['branches'][b]['macro_auroc'])
                        check(v[1], 100 * r['branches'][b]['macro_auprc'])
                    else:
                        m = 'pointing_game' if r['task'] == 'grounding' else 'dice'
                        check(cell, 100 * r['branches'][b][m])
                counts['ablation_appendix_rows'] += 1
    assert counts == {'main_mean_rows': 10, 'main_spatial_rows': 18, 'main_ablation_rows': 10,
                      'chexchonet_main_rows': 30, 'classification_appendix_rows': 117,
                      'ablation_appendix_rows': 135, 'fewshot_comparison_rows': 22,
                      'fewshot_method_dataset_combinations': 44,
                      'multiseed_comparison_rows': 11,
                      'multiseed_method_dataset_combinations': 44,
                      'query_score_mechanism_rows': 4}, counts
    assert fewshot_tables == {'BiomedCLIP', 'AlphaRAD'}, fewshot_tables
    assert fewshot_seen == set(fewshot_lookup), fewshot_seen ^ set(fewshot_lookup)
    assert fewshot_numeric_cells == 88, fewshot_numeric_cells
    assert multiseed_tables == 1, multiseed_tables
    assert multiseed_seen == {(model, dataset, method)
                              for _, model, dataset in MULTISEED_COLUMNS
                              for method in MULTISEED_METHODS.values()}
    assert multiseed_numeric_cells == 88, multiseed_numeric_cells
    assert query_score_tables == 1, query_score_tables
    assert query_score_numeric_cells == 23, query_score_numeric_cells
    result = {'status': 'passed', 'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'numeric_cells_checked': checked, **counts,
              'fewshot_numeric_cells_checked': fewshot_numeric_cells,
              'fewshot_source': str(FEWSHOT_SOURCE),
              'fewshot_source_sha256': hashlib.sha256(FEWSHOT_SOURCE.read_bytes()).hexdigest(),
              'fewshot_models': sorted(fewshot_tables),
              'multiseed_numeric_cells_checked': multiseed_numeric_cells,
              'multiseed_source': str(MULTISEED_SOURCE),
              'multiseed_source_sha256': hashlib.sha256(MULTISEED_SOURCE.read_bytes()).hexdigest(),
              'multiseed_support_seeds': multiseed_raw['support_seeds'],
              'query_score_numeric_cells_checked': query_score_numeric_cells,
              'query_score_spatial_control': 'S + unchanged Original maps; native S remains NA; PG contrast uses Q minus Original'}
    (HERE / 'MANUSCRIPT_TABLES_INDEPENDENT_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
