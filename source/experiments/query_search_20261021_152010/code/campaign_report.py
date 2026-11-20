#!/usr/bin/env python3
"""Aggregate complete matched evaluations without choosing per-dataset winners."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
MEDICAL = ROOT.parents[1]
TASKS = {
    **{('radzero', n): 'classification' for n in ('ChestXDet10', 'ChestXray14', 'OpenI', 'SIIM', 'RSNA')},
    **{('alpharad', n): 'classification' for n in ('chestxdet10', 'chest14', 'rsna_pneumonia', 'siim_pneumothorax', 'shenzhenxray', 'montgomery', 'covidkaggle')},
    ('alpharad', 'chestxdet10_gt'): 'grounding',
    ('alpharad', 'node21'): 'grounding',
    ('alpharad', 'qatacovid_seg'): 'segmentation',
}
METRICS = {'classification': ('macro_auroc', 'macro_auprc'), 'grounding': ('pointing_game',), 'segmentation': ('dice',)}


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify_protected(root=ROOT):
    protected = json.loads((root / 'protected_manifest.json').read_text())
    failures = []
    for relative, record in protected.items():
        path = MEDICAL / relative
        if not path.is_file():
            failures.append({'path': relative, 'reason': 'missing'})
        elif path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            failures.append({'path': relative, 'reason': 'changed'})
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'files_checked': len(protected), 'passed': not failures, 'failures': failures}
    write_json(root / 'audits/protected_integrity.json', result)
    return result


def summarize_phase(directory):
    loaded, failures, fitter_hashes = {}, [], set()
    for path in sorted(directory.glob('*/result.json')):
        result = json.loads(path.read_text())
        key = (result.get('model'), result.get('dataset'))
        if result.get('is_smoke') is True:
            continue
        if result.get('status') != 'complete':
            failures.append({'path': str(path), 'status': result.get('status'), 'error': result.get('error')})
            continue
        if key not in TASKS or result.get('task') != TASKS[key]:
            raise ValueError(f'Unrecognized task: {path}: {key}')
        if result.get('is_smoke') is not False:
            raise ValueError(f'Explicit full-evaluation flag required: {path}')
        if key in loaded:
            raise ValueError(f'Duplicate complete task within phase: {key}')
        if 'original' not in result['candidates']:
            raise ValueError(f'Original control is missing: {path}')
        checks = result.get('baseline_checks', {})
        if key[0] == 'radzero':
            references = checks
            count_key = 'checked_images'
            helper = checks.get('original_helper', {})
            provenance = result.get('config', {})
        else:
            references = checks.get('formal_prediction_controls', {})
            count_key = 'images_checked'
            helper = checks.get('author_helper', {})
            provenance = json.loads((path.parent / 'data_audit.json').read_text())
        if helper.get('passed') is not True:
            raise ValueError(f'Original helper verification missing: {path}')
        for control in ('original', 'uniform_eta03'):
            proof = references.get(control, {})
            if proof.get('passed') is not True or proof.get(count_key) != result['rows']:
                raise ValueError(f'Incomplete historical control verification: {path}: {control}')
        hashes = [digest for filename, digest in provenance.get('source_sha256', {}).items()
                  if Path(filename).name == 'query_candidates.py']
        if len(hashes) != 1:
            raise ValueError(f'Shared fitter source hash missing: {path}')
        fitter_hashes.add(hashes[0])
        loaded[key] = result
    if len(fitter_hashes) > 1:
        raise ValueError(f'Mixed query fitting code within one unified phase: {directory}')
    candidates = sorted(set().union(*(set(r['candidates']) for r in loaded.values()))) if loaded else []
    rows, summaries = [], {}
    for name in candidates:
        pair_gains, metric_gains, seen = {}, [], []
        for key, result in loaded.items():
            if name not in result['candidates']:
                continue
            seen.append(key)
            pair = (key[0], 'chestxdet10' if key == ('alpharad', 'chestxdet10_gt') else key[1])
            metrics = result['candidates'][name]['metrics']
            baseline = result['candidates']['original']['metrics']
            prior = result['candidates'].get('uniform_eta03', {}).get('metrics', {})
            for metric in METRICS[TASKS[key]]:
                value, base = metrics[metric], baseline[metric]
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (value, base)):
                    raise ValueError(f'Invalid full-evaluation metric: {key}/{name}/{metric}')
                gain = 100 * (value - base)
                pair_gains.setdefault('/'.join(pair), []).append(gain)
                metric_gains.append(gain)
                rows.append({'phase': directory.name, 'candidate': name, 'model': key[0], 'dataset': key[1],
                    'task': TASKS[key], 'metric': metric, 'test_images': result['rows'], 'support': result['support_count'],
                    'original_percent': 100 * base, 'previous_eta03_percent': 100 * prior[metric] if metric in prior else None,
                    'candidate_percent': 100 * value, 'gain_pp': gain})
        all_tasks = len(seen) == len(TASKS)
        positive_pairs = sum(all(g > 1e-9 for g in gains) for gains in pair_gains.values())
        ge1 = sum(all(g >= 1 - 1e-9 for g in gains) for gains in pair_gains.values())
        ge2 = sum(all(g >= 2 - 1e-9 for g in gains) for gains in pair_gains.values())
        summaries[name] = {'complete_tasks': len(seen), 'expected_tasks': len(TASKS), 'complete_all_tasks': all_tasks,
            'all_metric_pairs_positive': positive_pairs, 'pairs_all_metrics_ge_1pp': ge1, 'pairs_all_metrics_ge_2pp': ge2,
            'main_metric_count': len(metric_gains), 'minimum_gain_pp': min(metric_gains) if metric_gains else None,
            'mean_gain_pp': sum(metric_gains) / len(metric_gains) if metric_gains else None,
            'per_pair_min_gain_pp': {pair: min(gains) for pair, gains in pair_gains.items()},
            'target_met': all_tasks and len(pair_gains) == 14 and positive_pairs == 14 and ge1 >= 12}
    return {'phase': directory.name, 'shared_fitter_sha256': next(iter(fitter_hashes), None), 'complete_tasks': len(loaded), 'failed_or_interrupted': failures, 'candidates': summaries}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--verify-protected', action='store_true')
    args = parser.parse_args()
    if args.verify_protected and not verify_protected(args.root)['passed']:
        raise RuntimeError('Protected baseline changed; see audits/protected_integrity.json')
    phases, rows = [], []
    for directory in sorted((args.root / 'runs').iterdir()):
        if directory.is_dir():
            phase, details = summarize_phase(directory)
            if phase['complete_tasks'] or phase['failed_or_interrupted']:
                phases.append(phase); rows.extend(details)
    report = {'updated_at': datetime.now(timezone.utc).isoformat(), 'expected_tasks': len(TASKS), 'expected_pairs': 14,
        'selection_policy': 'One shared candidate across all tasks; no per-dataset winner selection.',
        'interpretation': 'Exploratory reuse of test sets. This is not independent final validation.', 'phases': phases}
    write_json(args.root / 'summary.json', report)
    columns = ['phase', 'candidate', 'model', 'dataset', 'task', 'metric', 'test_images', 'support', 'original_percent', 'previous_eta03_percent', 'candidate_percent', 'gain_pp']
    with (args.root / 'COMPARISONS.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    lines = ['RadZero / AlphaRAD 统一方案持续实验', f'更新：{report["updated_at"]}',
        '截止：2026-10-22 10:00 北京时间；仅使用已核查空闲的物理 GPU 3–7。',
        '14个模型×数据集组合，15项任务；分类看AUROC和梯形AUPRC，定位看PG，分割看Dice。',
        '固定支持集/测试集/作者评估口径；每行候选是覆盖全部任务的统一方法。',
        '结果反复用于方法开发，属于探索性结果，不能作为独立测试确认。', '']
    for phase in phases:
        lines.extend([f'{phase["phase"]}: {phase["complete_tasks"]}/15项完整任务',
            '统一候选 | 完整任务 | 各指标均上涨的组合/14 | 各指标均≥1点的组合/14 | 最小提升pp | 达标'])
        for name, summary in phase['candidates'].items():
            lines.append(f'{name} | {summary["complete_tasks"]}/15 | {summary["all_metric_pairs_positive"]} | {summary["pairs_all_metrics_ge_1pp"]} | {summary["minimum_gain_pp"]:.4f} | {summary["target_met"]}')
        lines.append('')
    lines += ['详细逐数据集比较：COMPARISONS.csv；任务日志与预测：runs/；队列与GPU记录：controller_v2/state.json。',
        '保护副本：protected_baseline/；原文件完整性：audits/protected_integrity.json。']
    (args.root / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'phases':len(phases),'comparison_rows':len(rows),'target_met':any(c['target_met'] for p in phases for c in p['candidates'].values())}))


if __name__ == '__main__':
    main()
