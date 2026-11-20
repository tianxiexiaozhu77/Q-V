#!/usr/bin/env python3
"""Check all SIIM1694 support-independent predictions against historical SIIM1704."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NEW = Path(__file__).resolve().parent / 'siim_radzero_holdout'
OLD = ROOT / 'experiments/query_search_20261021_152010/runs/phase6/alpharad_siim_pneumothorax'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((NEW / 'result.json').read_text())
    assert result['status'] == 'complete' and result['rows'] == 1694
    new_paths = json.loads((NEW / 'image_paths.json').read_text())
    old_paths = json.loads((OLD / 'image_paths.json').read_text())
    lookup = {p: i for i, p in enumerate(old_paths)}
    assert len(lookup) == len(old_paths) == 1704
    indices = np.asarray([lookup[p] for p in new_paths])
    labels = np.load(NEW / 'labels.npy', allow_pickle=False)
    np.testing.assert_array_equal(labels, np.load(OLD / 'labels.npy', allow_pickle=False)[indices])
    errors = {}
    checked_files = [NEW / 'result.json', NEW / 'image_paths.json', NEW / 'labels.npy',
                     OLD / 'image_paths.json', OLD / 'labels.npy']
    for branch in ('original', 'tta_center_original'):
        new_file = NEW / 'branches' / branch / 'probabilities.npy'
        old_file = OLD / 'candidates' / branch / 'probabilities.npy'
        actual = np.load(new_file, allow_pickle=False)
        expected = np.load(old_file, allow_pickle=False)[indices]
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
        errors[branch] = float(np.max(np.abs(actual - expected)))
        checked_files += [new_file, old_file]
    record = {'passed': True, 'images': len(new_paths), 'branches': list(errors),
              'max_probability_absolute_error': errors,
              'source_sha256': {str(p): sha256(p) for p in checked_files},
              'validation_script_sha256': sha256(Path(__file__).resolve()),
              'note': 'Original/V are support-independent; Q/Q+V use the frozen SIIM1694 supports and are not compared with old SIIM1704 adapters.'}
    destination = NEW / 'full_historical_original_v_check.json'
    destination.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'passed': True, 'images': len(new_paths), 'errors': errors}))


if __name__ == '__main__':
    main()
