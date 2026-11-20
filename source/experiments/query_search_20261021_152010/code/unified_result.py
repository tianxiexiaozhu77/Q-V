#!/usr/bin/env python3
"""Publish one complete shared candidate, never mix dataset-wise winners."""
from pathlib import Path
import json
import statistics
from datetime import datetime, timezone

from queue_append import atomic_json

ROOT = Path(__file__).resolve().parents[1]
KEYS = {'classification': ('macro_auroc', 'macro_auprc'),
        'grounding': ('pointing_game',), 'segmentation': ('dice',)}


def main():
    summary = json.loads((ROOT/'summary.json').read_text())
    eligible = []
    for phase in summary['phases']:
        if phase['complete_tasks'] != 15 or phase['failed_or_interrupted']:
            continue
        for name, record in phase['candidates'].items():
            if record['complete_tasks'] != 15 or len(record['per_pair_min_gain_pp']) != 14:
                continue
            minima = list(record['per_pair_min_gain_pp'].values())
            rank = (record['all_metric_pairs_positive'] == 14,
                    record['all_metric_pairs_positive'], record['pairs_all_metrics_ge_1pp'],
                    min(minima), statistics.median(minima))
            eligible.append((rank, phase['phase'], name, record))
    if not eligible:
        raise RuntimeError('No complete fifteen-task shared candidate')
    _, phase, name, record = max(eligible)
    rows = []
    for path in sorted((ROOT/'runs'/phase).glob('*/result.json')):
        result = json.loads(path.read_text())
        assert result['status'] == 'complete' and result['is_smoke'] is False
        assert name in result['candidates']
        baseline = result['candidates']['original']['metrics']
        previous = result['candidates']['uniform_eta03']['metrics']
        current = result['candidates'][name]['metrics']
        rows.append({'model': result['model'], 'dataset': result['dataset'], 'task': result['task'],
            'rows': result['rows'], 'support_count': result['support_count'], 'result_path': str(path),
            'metrics': {key: {'original_percent': 100*baseline[key], 'eta03_percent': 100*previous[key],
                             'candidate_percent': 100*current[key], 'delta_pp': 100*(current[key]-baseline[key])}
                        for key in KEYS[result['task']]}})
    assert len(rows) == 15
    result = {'updated_at': datetime.now(timezone.utc).isoformat(), 'source_summary_updated_at': summary['updated_at'],
        'phase': phase, 'candidate': name, 'complete_tasks': 15, 'model_dataset_pairs': 14,
        'all_metric_pairs_positive': record['all_metric_pairs_positive'],
        'pairs_all_metrics_ge_1pp': record['pairs_all_metrics_ge_1pp'], 'target_met': record['target_met'],
        'interpretation': 'Exploratory selection on repeatedly reused test sets, not independent final validation.',
        'selection_rule': 'One shared candidate; prioritize all-pair positive coverage, then >=1pp coverage, minimum pair gain, median pair gain.',
        'rows': rows}
    atomic_json(ROOT/'leading_unified.json', result)
    lines = ['当前完整评估中的统一方案（探索性结果）', f'方案：{phase} / {name}',
        f'完整任务：15；模型×数据集组合：14；全部主指标提升：{record["all_metric_pairs_positive"]}/14；'
        f'全部主指标提升至少1个百分点：{record["pairs_all_metrics_ge_1pp"]}/14。',
        '本表使用同一方案；没有按数据集拼接各自最好结果。固定测试集已反复用于方案筛选，尚非独立最终验证。',
        '指标均为百分数；Δ为百分点。AUPRC使用PR曲线梯形面积；AP不是本表AUPRC。', '',
        '| 模型 | 数据集/任务 | 指标 | 原始 | 原eta0.3 | 当前 | Δ原始 |',
        '|---|---|---|---:|---:|---:|---:|']
    labels = {'macro_auroc': 'AUROC', 'macro_auprc': 'AUPRC', 'pointing_game': 'PG', 'dice': 'Dice'}
    for row in rows:
        for key, value in row['metrics'].items():
            lines.append(f'| {row["model"]} | {row["dataset"]} | {labels[key]} | '
                f'{value["original_percent"]:.2f} | {value["eta03_percent"]:.2f} | '
                f'{value["candidate_percent"]:.2f} | {value["delta_pp"]:+.2f} |')
    lines.extend(['', 'QaTa沿用原评估的阳性样本/测试阈值扫描协议；患者分组证据限制沿用各数据清单审计。',
                  '完整来源与未四舍五入数值：leading_unified.json；全部候选及失败结果：RESULTS.txt、COMPARISONS.csv、runs/。'])
    path = ROOT/'LEADING_UNIFIED_RESULT.txt'
    temporary = path.with_suffix('.txt.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(path)
    print(json.dumps({key: result[key] for key in ('phase', 'candidate', 'all_metric_pairs_positive',
                                                   'pairs_all_metrics_ge_1pp', 'target_met')}))


if __name__ == '__main__':
    main()
