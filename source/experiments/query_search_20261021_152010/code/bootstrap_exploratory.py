#!/usr/bin/env python3
"""Conditional paired image bootstrap; not independent post-selection inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class BinaryCurves:
    def __init__(self, labels, scores):
        labels, scores = np.asarray(labels), np.asarray(scores)
        if labels.ndim != 1 or scores.shape != labels.shape:
            raise ValueError('Expected matching one-dimensional arrays')
        if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
            raise ValueError('Invalid binary labels or scores')
        self.order = np.argsort(scores, kind='stable')[::-1]
        self.y = labels[self.order].astype(np.float64)
        ordered = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered)-1]

    def metrics(self, weights):
        w = np.asarray(weights, dtype=np.float64)[self.order]
        if np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError('Invalid sample weights')
        tp = np.cumsum(w*self.y)[self.ends]
        fp = np.cumsum(w*(1-self.y))[self.ends]
        # sklearn removes zero-weight observations, including their thresholds.
        counts = tp+fp
        keep = np.r_[counts[0] > 0, np.diff(counts) > 0]
        tp, fp = tp[keep], fp[keep]
        if not len(tp) or tp[-1] <= 0 or fp[-1] <= 0:
            return np.full(2, np.nan)
        tpr, fpr = np.r_[0., tp/tp[-1]], np.r_[0., fp/fp[-1]]
        precision = np.r_[1., tp/(tp+fp)]
        return np.array([np.trapz(tpr, fpr), np.trapz(precision, tpr)])


def self_test():
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    rng = np.random.default_rng(29)
    checks = 0
    for n in [4, 17, 201]:
        for _ in range(30):
            y = rng.integers(0, 2, n)
            s = rng.integers(0, 7, n).astype(np.float32)
            w = rng.integers(0, 5, n)
            if np.sum(w*y) == 0 or np.sum(w*(1-y)) == 0:
                continue
            pr, re, _ = precision_recall_curve(y, s, sample_weight=w)
            expected = [roc_auc_score(y, s, sample_weight=w), auc(re, pr)]
            np.testing.assert_allclose(BinaryCurves(y, s).metrics(w), expected,
                                       atol=5e-15, rtol=0)
            checks += 1
    print(json.dumps({'status': 'passed', 'weighted_sklearn_cases': checks}))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    from queue_append import atomic_json
    root = args.root.resolve()
    outputs = []
    for directory in sorted((root/'runs'/args.phase).iterdir()):
        result_path = directory/'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result.get('status') != 'complete' or result.get('is_smoke', False):
            continue
        if args.candidate not in result['candidates']:
            continue
        expected = result['candidates'][args.candidate]['metrics']
        if 'macro_auroc' not in expected:
            continue
        label_path = directory/'labels.npy'
        y = np.load(label_path, allow_pickle=False)
        if y.ndim == 1:
            y = y[:, None]
        # Preserve exact formal protocol: AlphaRAD uses FP32 sigmoid scores.
        filename = 'probabilities.npy' if result['model'] == 'alpharad' else 'logits.npy'
        files = [directory/'candidates'/name/filename for name in ['original', args.candidate]]
        scores = [np.load(f, allow_pickle=False) for f in files]
        if any(s.shape != y.shape for s in scores):
            raise ValueError('Shape mismatch: '+directory.name)
        curves = [[BinaryCurves(y[:, c], s[:, c]) for c in range(y.shape[1])]
                  for s in scores]
        point = np.array([[curve.metrics(np.ones(len(y))) for curve in variant]
                          for variant in curves]).mean(1)
        for index, name in enumerate(['original', args.candidate]):
            recorded = result['candidates'][name]['metrics']
            np.testing.assert_allclose(point[index],
                [recorded['macro_auroc'], recorded['macro_auprc']], atol=2e-12, rtol=0)
        rng = np.random.default_rng(args.seed)
        deltas = []
        for _ in range(args.replicates):
            w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
            metrics = np.array([[curve.metrics(w) for curve in variant] for variant in curves])
            # Retain a fixed class macro-average; do not drop missing classes.
            if np.isfinite(metrics).all():
                deltas.append((metrics[1].mean(0)-metrics[0].mean(0))*100)
        values = np.asarray(deltas)
        enough = len(values) >= .9*args.replicates
        outputs.append({
            'model': result['model'], 'dataset': result['dataset'], 'images': len(y),
            'classes': y.shape[1], 'positive_counts': y.sum(0).astype(int).tolist(),
            'retained_replicates': len(values),
            'discarded_replicates_missing_class': args.replicates-len(values),
            'interval_status': ('available_conditional_on_defined_classes' if enough else
                                'unavailable_many_bootstraps_omit_a_class'),
            'metrics': {name: {'point_gain_pp': float((point[1,i]-point[0,i])*100),
                              'percentile_95_interval_pp': np.quantile(values[:, i], [.025, .975]).tolist() if enough else None,
                              'fraction_replicates_positive': float((values[:, i] > 0).mean()) if enough else None}
                        for i, name in enumerate(['AUROC', 'trapezoidal_AUPRC'])},
            'source_sha256': {str(f): sha(f) for f in [result_path, label_path, *files]}})
        print(directory.name, outputs[-1]['metrics'], flush=True)
    if len(outputs) != 12:
        raise ValueError('Expected all twelve classification tasks, got '+str(len(outputs)))
    output = {'status': 'complete', 'phase': args.phase, 'candidate': args.candidate,
              'replicates': args.replicates, 'seed': args.seed, 'unit': 'paired image bootstrap',
              'limitations': [
                  'Conditional exploratory uncertainty only; candidate selected using reused tests.',
                  'Images are resampled, not patients; repeated-patient correlation is not accounted for.',
                  'Not a correction for searching candidates or comparing many metrics.',
                  'Intervals conditional on all classes having positives and negatives; omitted if fewer than 90% of resamples qualify.',
                  'No model, query, support, test subset or hyperparameter changed.',
                  'Localization and segmentation are outside this classification-only diagnostic.'],
              'source_sha256': sha(Path(__file__)), 'tasks': outputs}
    atomic_json(args.output, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase', default='phase6')
    parser.add_argument('--candidate', default='tta_center_refined')
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.output is None or args.output.exists():
            raise ValueError('A new output path is required')
        run(args)
