#!/usr/bin/env python3
"""Audit fixed-test predictions and report support-seed and patient uncertainty."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from cluster_bootstrap import (make_patient_draw_plan, save_patient_draw_plan,
    paired_cluster_bootstrap, save_bootstrap, interval_summary)

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
TASKS = ('slvh', 'dlv', 'composite_slvh_dlv')
DISPLAY = dict(zip(TASKS, ('CSLV', 'CDLV', 'CCOM')))
BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')
NAMES = dict(zip(BRANCHES, ('原始', 'Q', 'V', 'Q+V')))
MODELS = ('BiomedCLIP', 'AlphaRAD')
METRICS = ('auroc', 'auprc', 'average_precision')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def csv_write(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def expected_test(protocol):
    tasks = []
    for t in TASKS:
        entry = protocol['tasks'][t]
        if sha(entry['manifest']) != entry['sha256']:
            raise ValueError('Changed fixed test manifest')
        tasks.append(json.loads(Path(entry['manifest']).read_text())['rows'])
    paths = [r['image'] for r in tasks[0]]
    patients = [str(r['patient_id']) for r in tasks[0]]
    if len(paths) != 7407 or len(set(paths)) != 7407 or len(set(patients)) != 2470:
        raise ValueError('Fixed test universe differs')
    for rows in tasks:
        if paths != [r['image'] for r in rows] or patients != [str(r['patient_id']) for r in rows]:
            raise ValueError('Task identities differ')
    return dict(image_paths=np.asarray(paths), patient_ids=np.asarray(patients),
                labels=np.asarray([[rows[i]['labels'][0] for rows in tasks] for i in range(len(paths))]))


def historical(model):
    if model == 'AlphaRAD':
        folder = MEDICAL / 'experiments/chexchonet_comparison_20261023_233926/alpharad_paired'
        out = dict(labels=np.load(folder / 'labels.npy'),
                   image_paths=np.asarray(json.loads((folder / 'image_paths.json').read_text())),
                   patient_ids=np.asarray(json.loads((folder / 'patient_ids.json').read_text())))
        sources = [folder / f for f in ('labels.npy', 'image_paths.json', 'patient_ids.json', 'result.json')]
        for branch in BRANCHES:
            old = 'uniform_eta03' if branch == 'query_refined' else branch
            path = folder / f'{old}_probabilities.npy'
            out[branch] = np.load(path)
            sources.append(path)
    else:
        folder = MEDICAL / 'baselines/runs/chexchonet_20261024_000447/BiomedCLIP_full'
        sources = [folder / ('chexchonet_' + t) / 'predictions.npz' for t in TASKS]
        arrays = []
        for path in sources:
            with np.load(path, allow_pickle=False) as data:
                arrays.append({k: data[k] for k in data.files})
        out = {k: arrays[0][k] for k in ('image_paths', 'patient_ids')}
        for item in arrays:
            for key in ('image_paths', 'patient_ids'):
                if not np.array_equal(item[key], out[key]):
                    raise ValueError('Historical tasks not paired')
        for key in ('labels', *BRANCHES):
            out[key] = np.concatenate([item[key] for item in arrays], axis=1)
    return out, {str(p): sha(p) for p in sources}


def verify(data, expected):
    for key in ('labels', 'image_paths', 'patient_ids'):
        if not np.array_equal(data[key], expected[key]):
            raise ValueError('Predictions do not match fixed test: ' + key)
    for key in BRANCHES:
        if data[key].shape != (7407, 3) or not np.isfinite(data[key]).all():
            raise ValueError('Invalid scores: ' + key)


def sklearn_audit(data, points):
    differences = []
    for b, branch in enumerate(BRANCHES):
        for j in range(3):
            y, score = data['labels'][:, j], data[branch][:, j]
            precision, recall, _ = precision_recall_curve(y, score)
            expected = np.asarray([roc_auc_score(y, score), auc(recall, precision), average_precision_score(y, score)])
            differences.extend(np.abs(expected - points[b, j]).tolist())
    error = max(differences)
    if error > 1e-12:
        raise ValueError(f'Independent sklearn metric audit failed: {error}')
    return error


def process(data, expected, plan, folder, provenance):
    verify(data, expected)
    result = paired_cluster_bootstrap(data['labels'], {k: data[k] for k in BRANCHES},
                                     data['patient_ids'], task_names=TASKS, plan=plan)
    error = sklearn_audit(data, result.point_metrics)
    result.summary['provenance'] = provenance
    result.summary['independent_sklearn_point_metric_max_abs_error'] = error
    save_bootstrap(result, folder)
    return result


def interval_pp(point, replicates):
    record = interval_summary(point, replicates)
    return {**record, 'point': None if record['point'] is None else record['point'] * 100,
            'ci': None if record['ci'] is None else [100 * x for x in record['ci']], 'unit': 'percentage_points'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocol_v1/catalog.json')
    a = parser.parse_args()
    a.run = a.run.resolve()
    protocol = json.loads(a.protocol.read_text())
    seeds = protocol['protocol']['support_seeds']
    if seeds != [42, 43, 44, 45, 46]:
        raise ValueError('Support seeds changed')
    expected = expected_test(protocol)
    plan = make_patient_draw_plan(expected['patient_ids'], n_resamples=2000, seed=20261024)
    save_patient_draw_plan(plan, a.run / 'patient_draws.npz')
    records, per_seed, history_rows, metric_audits = [], [], [], []
    summary = dict(status='running', created_at=datetime.now().isoformat(),
        protocol=str(a.protocol.resolve()), protocol_sha256=sha(a.protocol),
        fixed_test_images=7407, fixed_test_patients=2470, support_seeds=seeds,
        bootstrap_replicates=2000, bootstrap_seed=20261024, draw_plan_sha256=plan.sha256,
        models={}, uncertainty_note='Five-seed SD measures support sensitivity within the reserved 30-patient pool. '
        'Patient CIs condition on these fixed support sets. The CI for the mean averages each paired bootstrap '
        'replicate across the five seeds; it is not a joint support-population-and-test-population CI. '
        'All images remain image-level observations; resampling unit is patient. No multiplicity correction.',
        historical_support_note='Historical support is separate; newly sampled seed 42 is not the historical support set.',
        source_sha256={str(p.resolve()): sha(p) for p in [Path(__file__), ROOT / 'cluster_bootstrap.py', a.protocol]})
    for model in MODELS:
        old, sources = historical(model)
        print('BOOTSTRAP historical', model, flush=True)
        old_result = process(old, expected, plan, a.run / 'bootstrap' / model / 'historical_support', sources)
        metric_audits.append(old_result.summary['independent_sklearn_point_metric_max_abs_error'])
        for b, branch in enumerate(BRANCHES):
            for j, task in enumerate(TASKS):
                row = dict(model=model, task=DISPLAY[task], branch=branch)
                for k, metric in enumerate(METRICS[:2]):
                    ci = interval_pp(old_result.point_metrics[b, j, k] - old_result.point_metrics[0, j, k],
                                     old_result.replicate_metrics[:, b, j, k] - old_result.replicate_metrics[:, 0, j, k])
                    row.update({metric + '_percent': 100 * old_result.point_metrics[b, j, k],
                                metric + '_delta_pp': ci['point'], metric + '_ci_low_pp': ci['ci'][0],
                                metric + '_ci_high_pp': ci['ci'][1]})
                history_rows.append(row)
        fitted = []
        reference = None
        for seed in seeds:
            path = a.run / model / f'seed_{seed}/predictions.npz'
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            if reference is None:
                reference = data
            for key in ('original', 'tta_center_original'):
                if not np.array_equal(data[key], reference[key]):
                    raise ValueError(f'{model}/{seed}: nonadaptive branches changed with support seed')
                # Same predeclared tolerance as the native helper and worker parity audit.
                if not np.allclose(data[key], old[key], rtol=1e-5, atol=2e-6):
                    raise ValueError(f'{model}/{seed}: historical baseline mismatch for {key}')
            print('BOOTSTRAP', model, seed, flush=True)
            provenance = {str(path): sha(path), 'support_seed': seed,
                          'support_catalog_sha256': protocol['seeds'][str(seed)][
                              'alpharad_catalog_sha256' if model == 'AlphaRAD' else 'baseline_catalog_sha256']}
            result = process(data, expected, plan, a.run / 'bootstrap' / model / f'seed_{seed}', provenance)
            fitted.append(result)
            metric_audits.append(result.summary['independent_sklearn_point_metric_max_abs_error'])
            for b, branch in enumerate(BRANCHES):
                for j, task in enumerate(TASKS):
                    row = dict(model=model, seed=seed, task=DISPLAY[task], branch=branch)
                    for k, metric in enumerate(METRICS[:2]):
                        ci = interval_pp(result.point_metrics[b, j, k] - result.point_metrics[0, j, k],
                                         result.replicate_metrics[:, b, j, k] - result.replicate_metrics[:, 0, j, k])
                        row.update({metric + '_percent': 100 * result.point_metrics[b, j, k],
                                    metric + '_delta_pp': ci['point'], metric + '_ci_low_pp': ci['ci'][0],
                                    metric + '_ci_high_pp': ci['ci'][1], metric + '_valid_fraction': ci['valid_fraction']})
                    per_seed.append(row)
        points = np.stack([result.point_metrics for result in fitted])
        draws = np.stack([result.replicate_metrics for result in fitted])
        # Average metrics, never average predictions or pretend repeated test images are new patients.
        mean_points = points.mean(axis=0)
        sd_points = points.std(axis=0, ddof=1)
        mean_draws = draws.mean(axis=0)
        summary['models'][model] = dict(test_counts=fitted[0].summary['test_counts'], branches={})
        for b, branch in enumerate(BRANCHES):
            items = {}
            for j, task in enumerate(TASKS):
                row = dict(model=model, task=DISPLAY[task], branch=branch)
                item = {}
                for k, metric in enumerate(METRICS):
                    gains = points[:, b, j, k] - points[:, 0, j, k]
                    ci = interval_pp(mean_points[b, j, k] - mean_points[0, j, k],
                                     mean_draws[:, b, j, k] - mean_draws[:, 0, j, k])
                    item[metric] = dict(mean_percent=100 * mean_points[b, j, k],
                                        support_seed_sd_pp=100 * sd_points[b, j, k],
                                        positive_gain_seed_count=int((gains > 0).sum()),
                                        minimum_gain_pp=float(100 * gains.min()),
                                        maximum_gain_pp=float(100 * gains.max()),
                                        mean_gain_paired_patient_bootstrap=ci)
                    if k < 2:
                        row.update({metric + '_mean_percent': item[metric]['mean_percent'],
                            metric + '_sd_pp': item[metric]['support_seed_sd_pp'],
                            metric + '_delta_pp': ci['point'], metric + '_ci_low_pp': ci['ci'][0],
                            metric + '_ci_high_pp': ci['ci'][1], metric + '_positive_seeds': int((gains > 0).sum()),
                            metric + '_valid_fraction': ci['valid_fraction']})
                items[task] = item
                records.append(row)
            summary['models'][model]['branches'][branch] = items
        write(a.run / 'RESULTS.json', summary)
    csv_write(a.run / 'MEAN_SD_COMPARISON.csv', records)
    csv_write(a.run / 'PER_SEED_RESULTS.csv', per_seed)
    csv_write(a.run / 'HISTORICAL_SUPPORT_CI.csv', history_rows)
    lines = ['CheXchoNet：固定测试集、五支持抽样种子与患者级配对置信区间', '',
        '完成：BiomedCLIP、AlphaRAD × CSLV、CDLV、CCOM；每种子 Original/Q/V/Q+V。',
        '固定测试 7407 张 / 2470 位患者；候选支持 87 张 / 30 位患者。',
        '每任务每种子5位阳性+5位全候选图阴性患者，各取一张；种子42—46；η=0.3、中心裁剪80%。',
        '新seed42属于严格阴性的新协议，不等于历史支持。所有种子均保留，未按结果筛选。',
        '均值±样本标准差来自五支持种子。AUROC/AUPRC单位%；Δ及CI单位百分点；AUPRC为PR曲线梯形面积。',
        '95%CI：2470患者成组、有放回、配对重抽2000次；每次纳入患者所有图，抽样计划跨模型/种子/任务共用。',
        '本表CI针对五种子指标平均增益，在给定这五组支持样本条件下反映测试患者不确定性；支持变动另由SD/逐种子结果表示。',
        '指标按图像计算，仅重采样单位为患者。未将同一测试集的五次预测拼接成五倍样本。',
        '仅30位支持候选患者，种子间可以重叠；结果不等于独立医院验证，也不能消除之前测试集选方法的偏差。', '',
        '| 模型 | 任务 | 分支 | AUROC 均值±SD | ΔAUROC [95%CI] | 正增益种子 | AUPRC 均值±SD | ΔAUPRC [95%CI] | 正增益种子 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        values = []
        for metric in METRICS[:2]:
            values.extend([f"{row[metric+'_mean_percent']:.2f}±{row[metric+'_sd_pp']:.2f}",
                f"{row[metric+'_delta_pp']:+.2f} [{row[metric+'_ci_low_pp']:+.2f}, {row[metric+'_ci_high_pp']:+.2f}]",
                f"{row[metric+'_positive_seeds']}/5"])
        lines.append('| ' + ' | '.join([row['model'], row['task'], NAMES[row['branch']], *values]) + ' |')
    lines += ['', '完整逐种子成绩及CI：PER_SEED_RESULTS.csv。',
              '历史支持集点估计与患者CI（不纳入五种子均值）：HISTORICAL_SUPPORT_CI.csv。',
              '所有分支的绝对指标CI、配对差值CI、有效抽样比例、阳性患者数：bootstrap/<模型>/<支持集>/bootstrap_summary.json。',
              '实际患者抽样倍数：patient_draws.npz；每次bootstrap指标：同目录bootstrap_metrics.npz。',
              'CSLV/CDLV/CCOM测试阳性患者数为197/90/269，阳性图像数567/315/828。',
              'CI跨0时不能声称本次观察到的提升已得到明确统计支持；未做多重检验校正。']
    (a.run / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    summary.update(status='complete', finished_at=datetime.now().isoformat(),
                   independent_metric_audit=dict(status='passed', arrays_audited=len(metric_audits),
                        point_values_audited=len(metric_audits)*4*3*3, max_abs_error=max(metric_audits)),
                   per_seed_rows=len(per_seed), aggregate_rows=len(records), historical_rows=len(history_rows))
    write(a.run / 'RESULTS.json', summary)
    print('COMPLETE', a.run / 'RESULTS.txt', flush=True)


if __name__ == '__main__':
    main()
