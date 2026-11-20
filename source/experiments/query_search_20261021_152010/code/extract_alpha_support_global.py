#!/usr/bin/env python3
"""Extract fresh, verified support patch means; never consume evaluation labels."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

CODE = Path(__file__).resolve().parent
MEDICAL = CODE.parents[2]
ALPHA = MEDICAL / 'AlphaRAD'
sys.path[:0] = [str(ALPHA / 'local_setup'), str(ALPHA)]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from eval_pretrained import ASSETS, sha256, write_json
from eval_query_refinement import audit_csv, CHECKPOINT_SHA


def run(args):
    import numpy as np
    import torch
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator
    torch.set_num_threads(4); seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned CUDA GPU required')
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint identity changed')
    baseline = ALPHA / 'runs/query_refinement_20261021_1441'
    for relative, expected in json.loads((baseline / 'source_sha256.json').read_text()).items():
        if sha256(ALPHA / relative) != expected:
            raise ValueError('Protected AlphaRAD source changed: ' + relative)
    catalog = json.loads((baseline / 'catalog.json').read_text())['datasets']
    entries, data = {}, {}
    for name, entry in catalog.items():
        rows, paths, labels = audit_csv(entry['support_csv_path'], entry['support_csv_sha256'], entry['class_names'])
        old = json.loads((baseline / name / 'data_audit.json').read_text())
        if labels != old['support_labels']:
            raise ValueError('Historical support labels/order changed: ' + name)
        entries[name] = (entry, rows, paths, labels)
        data[name + '_support'] = {'name': name + '_support', 'data_type': 'image_classification',
            'data_path': entry['support_csv_path'], 'n_class': len(entry['class_names']), 'normalize': None}
    config = {'base': {'seed': 1234}, 'test': {'checkpoint': [str(args.checkpoint)]}, 'data_test': data,
        'dataloader': {'test': {'batch_size': 8, 'num_workers': 2, 'pin_memory': True, 'shuffle': False, 'drop_last': False, 'prefetch_factor': 2}}}
    evaluator = Evaluator(config, [str(args.checkpoint)], 'test')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    _strict_load_published_checkpoint(evaluator.model, checkpoint['model']); del checkpoint
    evaluator.model.requires_grad_(False).eval()
    results = {}
    with torch.inference_mode():
        for name, (entry, rows, paths, labels) in entries.items():
            if time.time() >= args.deadline:
                raise TimeoutError('Campaign deadline')
            directory = args.output / name; directory.mkdir()
            query, _ = original_queries_from_evaluator(evaluator, entry['class_names'])
            old_adapter = np.load(baseline / name / 'query_adapter.npz', allow_pickle=False)
            np.testing.assert_array_equal(query, old_adapter['original_queries'])
            matcher = QueryRefinementMatcher(evaluator.model, query[None], (0.,), evaluator.use_all_image_tokens)
            globals_, locals_, observed, tokens = [], [], [], []
            for batch in evaluator.data_loader_dict[name + '_support']:
                if time.time() >= args.deadline:
                    raise TimeoutError('Campaign deadline')
                vision = matcher.encode_vision(batch['images'])
                patch = vision['image_patch_raw_features'][0]
                cls = vision['image_cls_raw_features'][0]
                if not evaluator.use_all_image_tokens:
                    raise ValueError('Expected published all-token branch')
                globals_.append(patch.mean(1).cpu().numpy())
                tokens.append(torch.cat([cls[:, None], patch], dim=1).cpu().numpy())
                locals_.append(matcher.match_prepared(vision, query, return_features=True)['class_local'].cpu().numpy())
                observed.extend(batch['image_paths'])
            if observed != paths:
                raise ValueError('Support path order changed')
            features = np.concatenate(locals_)
            old_features = np.load(baseline / name / 'support_class_local.npy', allow_pickle=False)
            np.testing.assert_allclose(features, old_features, atol=1e-6, rtol=0)
            values = np.concatenate(globals_)
            if values.shape != (len(paths), query.shape[-1]) or not np.isfinite(values).all():
                raise ValueError('Invalid global support features')
            np.save(directory / 'global_features.npy', values, allow_pickle=False)
            np.save(directory / 'vision_tokens.npy', np.concatenate(tokens), allow_pickle=False)
            np.save(directory / 'class_local.npy', features, allow_pickle=False)
            record = {'status': 'complete', 'model': 'alpharad', 'dataset': name, 'split': 'support_only',
                'checkpoint_sha256': CHECKPOINT_SHA, 'support_csv_sha256': entry['support_csv_sha256'],
                'image_paths': paths, 'image_file_sha256': [sha256(path) for path in paths],
                'labels': labels, 'num_heads': evaluator.model.num_radzero_heads,
                'shape': list(values.shape), 'dtype': str(values.dtype), 'batch_size': 8,
                'global_definition': 'Arithmetic mean of raw projected spatial patch tokens, excluding CLS; head normalization is deferred to the fitter.',
                'global_features_sha256': sha256(directory / 'global_features.npy'),
                'vision_tokens_sha256': sha256(directory / 'vision_tokens.npy'),
                'class_local_sha256': sha256(directory / 'class_local.npy'),
                'reference_class_local_sha256': sha256(baseline / name / 'support_class_local.npy'),
                'reference_check': {'passed': True, 'max_abs_error': float(np.max(np.abs(features - old_features))),
                    'bitwise_equal': bool(np.array_equal(features, old_features))},
                'source_sha256': {str(Path(__file__)): sha256(__file__)},
                'completed_at': datetime.now(timezone.utc).isoformat()}
            write_json(directory / 'manifest.json', record)
            results[name] = {'support_images': len(paths), 'reference_check': record['reference_check']}
            print(name, results[name], flush=True)
    write_json(args.output / 'result.json', {'status': 'complete', 'model': 'alpharad', 'dataset': 'all_support',
        'is_smoke': False, 'datasets': results, 'no_test_labels_consumed': True, 'deadline': args.deadline})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ASSETS / 'model/xraydinov2_mpnet/eccv_mpnet_518.tar')
    args = parser.parse_args()
    if not time.time() < args.deadline <= 1792634400:
        parser.error('Invalid campaign deadline')
    args.output.mkdir(parents=True, exist_ok=False)
    def stop(signum, _frame):
        raise InterruptedError(f'Stopped by signal {signum}')
    for signum in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    signal.setitimer(signal.ITIMER_REAL, args.deadline - time.time())
    try:
        run(args)
    except BaseException as error:
        write_json(args.output / 'result.json', {'status': 'failed', 'model': 'alpharad', 'dataset': 'all_support',
            'error': repr(error), 'traceback': traceback.format_exc(), 'is_smoke': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
