#!/usr/bin/env python3
"""Read-only current-queue comparison; never rank incomplete unified methods.

Only CURRENT_QUEUE_RESULTS.txt/json are published. Existing watcher reports,
queue/state, model code and prediction artifacts are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import statistics
import tempfile

from campaign_report import TASKS, METRICS, summarize_phase

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('phase6', 'phase12', 'phase14', 'phase17b')
FOLLOWUPS = {'phase12': 'phase12_no_deadline', 'phase14': 'phase14_no_deadline'}
REFERENCE = ('phase6', 'tta_center_refined')
TITLES = {'phase6': '固定视图融合', 'phase12': '支持集 OOF 视图权重（最多25折）',
          'phase14': '支持集 OOF 视图权重（全部有效组对）', 'phase17b': '多视图支持原型（5项先行评估）'}
NAMES = {'original': '原始模型', 'uniform_eta03': '原有 η=0.3 修正',
    'tta_center_refined': '中心裁剪＋修正query', 'tta_center_original': '中心裁剪＋原query',
    'tta_flip_refined': '翻转＋修正query', 'tta_flip_original': '翻转＋原query',
    'tta_fivecrop_refined': '五裁剪＋修正query', 'tta_fivecrop_original': '五裁剪＋原query',
    'tta_all7_refined': '七视图＋修正query',
    'view_oof_original': 'OOF权重＋原query', 'view_oof_refined': 'OOF权重＋修正query',
    'view_oof_half_original': 'OOF半幅权重＋原query', 'view_oof_half_refined': 'OOF半幅权重＋修正query',
    'prototype_flip_refined': '翻转支持原型', 'prototype_center_refined': '中心支持原型',
    'prototype_fivecrop_refined': '五裁剪支持原型', 'prototype_all7_refined': '七视图支持原型'}
DATASETS = {'chest14': 'ChestXray14', 'chestxdet10': 'ChestXDet10',
    'chestxdet10_gt': 'ChestXDet10定位', 'rsna_pneumonia': 'RSNA',
    'siim_pneumothorax': 'SIIM', 'shenzhenxray': 'Shenzhen', 'montgomery': 'Montgomery',
    'covidkaggle': 'CovidKaggle', 'node21': 'Node21定位', 'qatacovid_seg': 'QaTaCOV19分割'}
MODEL_NAMES = {'radzero': 'RadZero', 'alpharad': 'AlphaRAD'}
METRIC_NAMES = {'macro_auroc': 'AUROC', 'macro_auprc': '梯形AUPRC', 'pointing_game': 'PG', 'dice': 'Dice'}
TASK_ORDER = tuple(TASKS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_id(key):
    return '/'.join(key)


def pair_key(key):
    return (key[0], 'chestxdet10' if key == ('alpharad', 'chestxdet10_gt') else key[1])


EXPECTED_PAIRS = {}
for _key in TASK_ORDER:
    EXPECTED_PAIRS.setdefault(pair_key(_key), set()).add(_key)


def complete_pair_counts(rows):
    """A pair counts only after *all* its tasks/metrics exist, including CXD PG."""
    indexed = {(row['model'], row['dataset'], row['metric']): row for row in rows}
    completed, positives, ge1, ge2, minimums = 0, 0, 0, 0, {}
    for pair, tasks in EXPECTED_PAIRS.items():
        required = [(m, d, metric) for m, d in tasks for metric in METRICS[TASKS[m, d]]]
        if not all(key in indexed for key in required):
            continue
        gains = [indexed[key]['gain_pp'] for key in required]
        completed += 1
        positives += all(value > 1e-9 for value in gains)
        ge1 += all(value >= 1-1e-9 for value in gains)
        ge2 += all(value >= 2-1e-9 for value in gains)
        minimums[task_id(pair)] = min(gains)
    return {'complete_pairs': completed, 'all_metrics_positive_pairs': positives,
            'all_metrics_ge_1pp_pairs': ge1, 'all_metrics_ge_2pp_pairs': ge2,
            'per_complete_pair_min_gain_pp': minimums}


def full_rank_key(candidate):
    if not candidate['eligible_for_full_ranking']:
        raise ValueError('Cannot rank an incomplete unified candidate')
    # Exactly the same numerical rule as code/unified_result.py.
    minima = list(candidate['per_complete_pair_min_gain_pp'].values())
    if len(minima) != 14:
        raise ValueError('Full unified ranking requires fourteen pair minima')
    return (candidate['all_metrics_positive_pairs'] == 14,
            candidate['all_metrics_positive_pairs'], candidate['all_metrics_ge_1pp_pairs'],
            min(minima), statistics.median(minima))


def read_controller(root):
    newer = root/'controller_no_deadline/state.json'
    path = newer if newer.exists() else root/'controller_v2/state.json'
    if not path.exists():
        return {'status': 'unavailable', 'path': str(path), 'jobs': {}, 'deadline': None}
    raw = path.read_bytes()
    value = json.loads(raw)
    jobs = value.get('jobs', {})
    if isinstance(jobs, list):
        jobs = {item.get('id', str(i)): item for i, item in enumerate(jobs)}
    return {'status': value.get('status'), 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
        'deadline': value.get('deadline'), 'jobs': jobs,
        'job_status_counts': dict(Counter(item.get('status', 'unknown') for item in jobs.values())),
        'no_deadline_controller': path == newer}


def queue_counts(controller, logical_phase, physical_phase):
    selected = []
    for identifier, record in controller['jobs'].items():
        job = record.get('job', record)
        output = str(job.get('output', ''))
        if (f'/runs/{physical_phase}/' in output or f'/runs/{logical_phase}/' in output
                or identifier.startswith(physical_phase+'_') or identifier.startswith(logical_phase+'_')):
            selected.append(record)
    return dict(Counter(item.get('status', 'unknown') for item in selected))


def build_report(root):
    root = Path(root).resolve()
    controller = read_controller(root)
    phases, rows, sources = [], [], {}
    for logical in PHASES:
        resumed = root/'runs'/FOLLOWUPS.get(logical, '__unused__')
        directory = resumed if resumed.exists() else root/'runs'/logical
        summary, details = summarize_phase(directory)
        phase = {'name': logical, 'title': TITLES[logical], 'physical_directory': str(directory),
            'physical_phase': directory.name, 'deadline_followup': directory.name != logical,
            'algorithm_change_in_followup': False,
            'followup_note': ('仅取消时间上限；沿用同一算法，继承已完成项，不重复统计旧目录。'
                             if directory.name != logical else None),
            'complete_tasks': summary['complete_tasks'], 'expected_tasks': 15,
            'shared_fitter_sha256': summary['shared_fitter_sha256'],
            'queue_status_counts': queue_counts(controller, logical, directory.name),
            'failed_or_interrupted': summary['failed_or_interrupted'], 'candidates': []}
        for path in directory.glob('*/result.json'):
            sources[str(path)] = {'sha256': sha(path), 'resolved_path': str(path.resolve())}
        for name, data in summary['candidates'].items():
            selected = [dict(item, phase=logical, physical_phase=directory.name)
                        for item in details if item['candidate'] == name]
            counts = complete_pair_counts(selected)
            seen = {(item['model'], item['dataset']) for item in selected}
            eligible = (data['complete_all_tasks'] and counts['complete_pairs'] == 14
                        and not summary['failed_or_interrupted'])
            minima = list(counts['per_complete_pair_min_gain_pp'].values())
            candidate = {'phase': logical, 'physical_phase': directory.name, 'name': name, 'label': NAMES.get(name, name),
                'complete_tasks': len(seen), 'expected_tasks': 15, **counts,
                'missing_tasks': [task_id(key) for key in TASK_ORDER if key not in seen],
                'main_metric_count': len(selected), 'minimum_gain_pp': data['minimum_gain_pp'],
                'minimum_pair_gain_pp': min(minima) if minima else None,
                'median_pair_gain_pp': statistics.median(minima) if minima else None,
                'mean_gain_pp': data['mean_gain_pp'], 'eligible_for_full_ranking': eligible,
                'full_scheme_rank': None, 'target_met': eligible and counts['all_metrics_positive_pairs'] == 14
                    and counts['all_metrics_ge_1pp_pairs'] >= 12,
                'evaluation_scope': '完整统一方案' if eligible else '未完整覆盖，仅列已完成任务，不参加完整方案排名'}
            phase['candidates'].append(candidate); rows.extend(selected)
        phases.append(phase)
    reference_rows = {(r['model'], r['dataset'], r['metric']): r for r in rows
                      if (r['phase'], r['candidate']) == REFERENCE}
    reference_tasks = {(r['model'], r['dataset']) for r in reference_rows.values()}
    for row in rows:
        ref = reference_rows.get((row['model'], row['dataset'], row['metric']))
        row['phase6_reference_percent'] = ref['candidate_percent'] if ref else None
        row['difference_vs_phase6_reference_pp'] = row['candidate_percent']-ref['candidate_percent'] if ref else None
        row['original_control_difference_vs_phase6_pp'] = row['original_percent']-ref['original_percent'] if ref else None
    all_candidates = [candidate for phase in phases for candidate in phase['candidates']]
    ranked = sorted([c for c in all_candidates if c['eligible_for_full_ranking']],
                    key=lambda c: (full_rank_key(c), c['physical_phase'], c['name']), reverse=True)
    for rank, candidate in enumerate(ranked, 1):
        candidate['full_scheme_rank'] = rank
    best = ranked[0] if ranked else None
    for candidate in all_candidates:
        selected = [r for r in rows if (r['phase'], r['candidate']) == (candidate['phase'], candidate['name'])]
        differences = [r['difference_vs_phase6_reference_pp'] for r in selected
                       if r['difference_vs_phase6_reference_pp'] is not None]
        candidate['vs_phase6_reference'] = {'compared_metric_count': len(differences),
            'improved_metric_count': sum(v > 1e-9 for v in differences),
            'equal_metric_count': sum(abs(v) <= 1e-9 for v in differences),
            'lower_metric_count': sum(v < -1e-9 for v in differences),
            'minimum_difference_pp': min(differences) if differences else None,
            'maximum_difference_pp': max(differences) if differences else None}
    jobs = controller.pop('jobs')
    unfinished_statuses = ('pending', 'running', 'queued', 'not_started', 'stopped')
    unfinished = sum(record.get('status') in unfinished_statuses for record in jobs.values())
    controller['unfinished_jobs'] = unfinished
    queue_path = root/'queue_no_deadline.json'
    expected_ids = set()
    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
        expected_ids = {job['id'] for job in queue['jobs']}
        controller['finite_queue_path'] = str(queue_path)
        controller['finite_queue_sha256'] = sha(queue_path)
    accepted = bool(expected_ids) and expected_ids.issubset(jobs)
    unsettled = [identifier for identifier in expected_ids
                 if jobs.get(identifier, {}).get('status') in (None, 'pending', 'running', 'queued')]
    queue_finished = (controller.get('no_deadline_controller', False) and accepted and not unsettled
                      and controller['status'] not in ('running', 'starting', 'stopping'))
    controller['expected_queue_jobs'] = len(expected_ids) if expected_ids else None
    controller['expected_jobs_all_accepted'] = accepted
    controller['queue_transition_pending'] = not controller.get('no_deadline_controller', False)
    return {'schema_version': 1, 'updated_at': datetime.now(timezone.utc).isoformat(),
        'root': str(root), 'expected_pairs': 14, 'expected_tasks': 15,
        'required_target': '14个组合各主指标均提升；至少12个组合各主指标均提升≥1个百分点。',
        'selection_policy': '一行一个统一算法/参数方案；不拼接逐数据集优胜者；不完整方案不排名。',
        'ranking_scope': '仅本报告phase6、phase12、phase14、phase17b中的完整统一方案，不代表对所有历史实验的全局排序。',
        'ranking_rule': ['all_pairs_positive', 'positive_pairs', 'pairs_ge_1pp',
                         'minimum_pair_gain_pp', 'median_pair_gain_pp'],
        'ranking_tie_break': '与unified_result.py相同：physical_phase、candidate名称按字典序取最大值。',
        'interpretation': '重复使用测试集开发方法的探索性结果；不构成独立最终验证或统计显著性结论。',
        'reference': {'phase': REFERENCE[0], 'candidate': REFERENCE[1], 'complete_tasks': len(reference_tasks)},
        'current_best_complete_unified': best,
        'all_tracked_phases_complete': all(phase['complete_tasks'] == 15 for phase in phases),
        'current_queue_finished': queue_finished,
        'controller': controller, 'phases': phases, 'metric_rows': rows,
        'task_coverage': [{'model': key[0], 'dataset': key[1], 'task': TASKS[key],
            'pair': task_id(pair_key(key)), 'phases': {phase['name']: sorted({r['candidate'] for r in rows
                if r['phase'] == phase['name'] and (r['model'], r['dataset']) == key}) for phase in phases}}
            for key in TASK_ORDER], 'result_sources': sources,
        'source_sha256': {str(Path(__file__).resolve()): sha(__file__),
                          str(Path(__file__).with_name('campaign_report.py')): sha(Path(__file__).with_name('campaign_report.py'))}}


def fmt(value, signed=False):
    return '未完成' if value is None else (f'{value:+.2f}' if signed else f'{value:.2f}')


def task_label(key):
    return f'{MODEL_NAMES[key[0]]} / {DATASETS.get(key[1], key[1])}'


def task_values(index, phase, candidate, key, field='candidate_percent', signed=False):
    values = [index.get((phase, candidate, *key, metric), {}).get(field) for metric in METRICS[TASKS[key]]]
    if all(value is None for value in values):
        return '未完成'
    return ' / '.join(fmt(value, signed) for value in values)


def render_text(report):
    controller = report['controller']; best = report['current_best_complete_unified']
    lines = ['RadZero / AlphaRAD 当前队列结果对照', f'更新时间（UTC）：{report["updated_at"]}',
        '口径：14个模型×数据集组合、15项评估。分类同时看AUROC和梯形AUPRC；定位看PG；分割看Dice。',
        'AlphaRAD ChestXDet10 的分类、定位共同构成一个组合；必须两项齐全才计入完整组合。',
        '数值单位为%；Δ表示百分点。梯形AUPRC与average precision不是同一指标。',
        '范围仅含phase6、phase12、phase14、phase17b；排序规则与LEADING相同，但不涵盖LEADING的全部历史阶段。',
        report['selection_policy'], report['interpretation'],
        f'当前控制器：{controller["status"]}；任务状态：{json.dumps(controller.get("job_status_counts", {}), ensure_ascii=False)}；状态文件：{controller["path"]}。']
    if controller.get('no_deadline_controller'):
        lines.append('已读取取消固定时间上限后的后续队列；已完成旧任务沿用，不重复计算。')
    else:
        lines.append('当前仍读取旧控制器快照；旧控制器停止不表示本次任务完成，正在等待无截止时间后续队列。')
    if report['current_queue_finished']:
        lines.append('固定后续队列已被全部接收且已结束；失败、中断及未安排的评估仍明确保留，不能视为全部复现成功。')
    else:
        lines.append('当前队列尚未结束；以下是已有完整结果的即时汇总。')
    if best:
        lines += ['', f'本报告已完整覆盖方案中的当前最好：{best["phase"]} / {best["name"]}（{best["label"]}）。',
            f'完整15/15任务；各指标均上涨：{best["all_metrics_positive_pairs"]}/14；各指标均≥1点：{best["all_metrics_ge_1pp_pairs"]}/14；'
            f'各指标均≥2点：{best["all_metrics_ge_2pp_pairs"]}/14；最小提升：{best["minimum_gain_pp"]:+.4f}点；'
            f'预设目标：{"达到" if best["target_met"] else "尚未达到"}。',
            '排序只用于本轮完整方案的描述性比较，依次看是否14组合全上涨、上涨组合数、≥1点组合数、组合最小增益的最小值与中位数。']
    else:
        lines.append('尚无完整覆盖15项评估的统一方案，暂不排名。')
    for phase in report['phases']:
        lines += ['', f'{phase["name"]}：{phase["title"]}；完整任务 {phase["complete_tasks"]}/15',
                  f'读取目录：{phase["physical_directory"]}；队列状态：{json.dumps(phase["queue_status_counts"], ensure_ascii=False)}']
        if phase['followup_note']: lines.append(phase['followup_note'])
        if phase['complete_tasks'] < 15:
            lines.append('以下计数仅反映已完成部分；不完整候选不参加完整方案排名，也不宣称覆盖全部数据集。')
        lines += ['', '| 统一候选 | 任务/15 | 完整组合/14 | 已完成组合全上涨 | 已完成组合全≥1点 | 最小Δ原始 | 与phase6中心方案比：升/平/降指标 | 完整排名 |',
                  '|---|---:|---:|---:|---:|---:|---|---:|']
        for c in phase['candidates']:
            if c['name'] == 'original': continue
            comparison = c['vs_phase6_reference']
            lines.append(f'| {c["name"]} | {c["complete_tasks"]} | {c["complete_pairs"]} | {c["all_metrics_positive_pairs"]} | '
                f'{c["all_metrics_ge_1pp_pairs"]} | {fmt(c["minimum_gain_pp"], True)} | '
                f'{comparison["improved_metric_count"]}/{comparison["equal_metric_count"]}/{comparison["lower_metric_count"]} | '
                f'{c["full_scheme_rank"] if c["full_scheme_rank"] is not None else "不排名"} |')
    index = {(r['phase'], r['candidate'], r['model'], r['dataset'], r['metric']): r for r in report['metric_rows']}
    lines += ['', '全部15项评估对照（同一方案用于全部数据集）', '',
        '| 模型 / 数据集 | 指标顺序 | 原始模型 | phase6中心修正 | 当前完整最佳 | 当前最佳Δ原始 | 当前最佳Δphase6 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key in TASK_ORDER:
        bp, bn = (best['phase'], best['name']) if best else ('__none__', '__none__')
        lines.append(f'| {task_label(key)} | {" / ".join(METRIC_NAMES[m] for m in METRICS[TASKS[key]])} | '
            f'{task_values(index, *REFERENCE, key, "original_percent")} | {task_values(index, *REFERENCE, key)} | '
            f'{task_values(index, bp, bn, key)} | {task_values(index, bp, bn, key, "gain_pp", True)} | '
            f'{task_values(index, bp, bn, key, "difference_vs_phase6_reference_pp", True)} |')
    # Four OOF alternatives / four prototype alternatives remain individually
    # named, including partial coverage. No per-dataset winner is selected.
    for phase in report['phases']:
        if phase['name'] == 'phase6': continue
        choices = [c for c in phase['candidates'] if c['name'].startswith(('view_oof_', 'prototype_'))]
        if not choices: continue
        lines += ['', f'{phase["name"]} 候选逐任务对照：单元格为“结果（Δ原始；Δphase6中心方案）”。',
                  '分类单元格依次为 AUROC / 梯形AUPRC；定位为 PG，分割为 Dice。', '',
                  '| 模型 / 数据集 | '+' | '.join(c['label'] for c in choices)+' |',
                  '|---|'+'---:|'*len(choices)]
        for key in TASK_ORDER:
            cells = []
            for c in choices:
                value = task_values(index, phase['name'], c['name'], key)
                cells.append('未完成' if value == '未完成' else f'{value}（'
                    f'{task_values(index, phase["name"], c["name"], key, "gain_pp", True)}；'
                    f'{task_values(index, phase["name"], c["name"], key, "difference_vs_phase6_reference_pp", True)}）')
            lines.append('| '+task_label(key)+' | '+' | '.join(cells)+' |')
    lines += ['', '完整机器可读数据、覆盖情况与源文件SHA：CURRENT_QUEUE_RESULTS.json。',
              '未完成项表示尚无经控制校验通过的完整评估结果，不能用部分结果替代完整方案排名。']
    return '\n'.join(lines)+'\n'


def publish(path, content):
    path = Path(path); data = content.encode('utf-8')
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if temporary.read_bytes() != data:
            raise OSError('Summary temporary read-back mismatch')
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise OSError('Summary publication read-back mismatch')
    finally:
        if temporary.exists(): temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    report = build_report(args.root)
    prefix = args.output_prefix or args.root/'CURRENT_QUEUE_RESULTS'
    publish(prefix.with_suffix('.json'), json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    publish(prefix.with_suffix('.txt'), render_text(report))
    best = report['current_best_complete_unified']
    print(json.dumps({'status': 'complete', 'output_prefix': str(prefix.resolve()),
        'queue_finished': report['current_queue_finished'], 'coverage': {p['name']: p['complete_tasks'] for p in report['phases']},
        'best_complete': [best['phase'], best['name']] if best else None}, ensure_ascii=False))


if __name__ == '__main__':
    main()
