#!/usr/bin/env python3
"""Evaluate support-only query candidates with one frozen vision pass per batch.

All reference code and baseline artifacts are imported/read without alteration.
The original and uniform_eta03 controls are checked against the frozen formal
run on every evaluated image. A deadline or signal produces interrupted status,
never a complete result. Grounding/segmentation retain the author's protocols.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

MEDICAL = Path(__file__).resolve().parents[3]
ALPHA = MEDICAL / 'AlphaRAD'
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE), str(ALPHA / 'local_setup'), str(ALPHA)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'

from eval_pretrained import ASSETS, sha256, write_json

DEFAULT_BASELINE = ALPHA / 'runs/query_refinement_20261021_1441'
CHECKPOINT_SHA = '97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9'
CONTROLS = ('original', 'uniform_eta03')
STOP_REASON = None


class DeadlineReached(RuntimeError):
    pass


def handle_signal(signum, _frame):
    global STOP_REASON
    STOP_REASON = f'signal {signum}'


def check_deadline(deadline):
    if STOP_REASON is not None:
        raise DeadlineReached(STOP_REASON)
    if deadline is not None and time.time() >= deadline:
        raise DeadlineReached(f'deadline {deadline} reached')


def support_groups(rows):
    """Use available linkage; unknown IDs are explicitly per-image groups."""
    groups, kinds = [], []
    for row in rows:
        if row.get('group'):
            value, kind = row['group'], 'existing_audited_group'
        elif row.get('patient_id'):
            value, kind = 'patient:' + row['patient_id'], 'patient_id'
        elif row.get('study_id'):
            value, kind = 'study:' + row['study_id'], 'study_id'
        else:
            value, kind = 'unlinked_image:' + row['image'], 'image_only_no_patient_linkage'
        groups.append(value); kinds.append(kind)
    return groups, {'row_group_basis': kinds, 'groups': groups,
        'note': 'Image-only groups are not verified patient identities; existing groups retain source audit limitations.'}


def controls_helper_check(evaluator, image, batch, classes, output):
    """Check original helpers using the already computed image encoding."""
    import numpy as np
    import torch
    raw = image['cls2_token_raw'] if image['cls2_token_raw'] is not None else image['cls_token_raw']
    normalized = image['cls2_token'] if image['cls2_token'] is not None else image['cls_token']
    inputs = (raw, normalized, image['patch_tokens_raw'], image['patch_tokens'], batch['label_names'], classes)
    expected_prob, _ = evaluator._zeroshot_binary_dqn_helper(*inputs)
    expected_map, _ = evaluator._zeroshot_pointinggame_dqn_helper(*inputs)
    prob = torch.sigmoid(output['logits']).cpu().numpy()
    actual_map = output['grounding_map'].cpu().numpy()
    expected_prob = expected_prob.reshape(prob.shape)
    np.testing.assert_allclose(prob, expected_prob, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(actual_map, expected_map, atol=2e-5, rtol=1e-5)
    return {'passed': True, 'images': len(prob),
        'probability_max_abs_error': float(np.max(np.abs(prob - expected_prob))),
        'map_max_abs_error': float(np.max(np.abs(actual_map - expected_map))),
        'probabilities_bitwise_equal': bool(np.array_equal(prob, expected_prob)),
        'maps_bitwise_equal': bool(np.array_equal(actual_map, expected_map)),
        'vision_passes_for_this_batch': 1}


class FastPositiveDice:
    """Exact author threshold grid with cached positive GT and histogram counts.

    The supported fast case is the published QaTa square-image protocol. Map
    normalization includes CLS, CLS is then discarded, bilinear interpolation
    and sigmoid match gather_seg_statistics. Each image contributes equally.
    Histogram counts replace 101 repeated full-resolution threshold tensors;
    thresholds are cast to float32 exactly as torch float32 > Python scalar.
    """
    def __init__(self, masks, label_names, classes):
        import numpy as np
        import torch
        self.classes = classes
        self.thresholds = np.arange(0, 1.01, 0.01)
        self.thresholds32 = self.thresholds.astype(np.float32)
        self.data = []
        for name in classes:
            indices, arrays = [], []
            for i, (mask, names) in enumerate(zip(masks, label_names)):
                if name not in names:
                    continue
                if len(names) != 1 or mask.shape[0] != 1:
                    raise ValueError('Fast Dice requires one mask/class per positive image')
                value = mask[0].cpu().numpy().astype(bool)
                if value.ndim != 2 or value.shape[0] != value.shape[1] or not value.any():
                    raise ValueError('Fast Dice only supports nonempty square positive masks')
                indices.append(i); arrays.append(value)
            if not arrays or len({a.shape for a in arrays}) != 1:
                raise ValueError('Fast Dice requires equal positive image shapes')
            truth = np.stack(arrays).reshape(len(arrays), -1)
            self.data.append((np.asarray(indices), truth, truth.sum(1), arrays[0].shape))

    def evaluate(self, attention, deadline=None):
        import numpy as np
        import torch
        from torch.nn import functional as F
        from cxrclip.evaluator_utils import seg_score
        array = torch.from_numpy(np.asarray(attention, dtype=np.float32))
        minimum = array.amin(dim=-1, keepdim=True)
        array = (array - minimum) / (array.amax(dim=-1, keepdim=True) - minimum + 1e-8)
        token_count = array.shape[-1]
        if token_count in (1370, 257, 325, 197):
            array = array[..., 1:]
        side = math.isqrt(array.shape[-1])
        if side * side != array.shape[-1]:
            raise ValueError('Non-square map token grid')
        results, curves = {}, {}
        for q, name in enumerate(self.classes):
            indices, truth, truth_count, shape = self.data[q]
            scores = np.empty((len(indices), len(self.thresholds)), dtype=np.float32)
            for start in range(0, len(indices), 32):
                check_deadline(deadline)
                stop = min(start + 32, len(indices))
                heat = array[indices[start:stop], q].reshape(stop-start, 1, side, side)
                probs = torch.sigmoid(F.interpolate(heat, size=shape, mode='bilinear', align_corners=False))
                probs = probs[:, 0].numpy().reshape(stop-start, -1)
                for j, prob in enumerate(probs, start):
                    bins = np.searchsorted(self.thresholds32, prob, side='left')
                    total_hist = np.bincount(bins, minlength=102)
                    positive_hist = np.bincount(bins[truth[j]], minlength=102)
                    predicted_count = np.cumsum(total_hist[::-1])[::-1][1:]
                    intersection = np.cumsum(positive_hist[::-1])[::-1][1:]
                    scores[j] = (2 * intersection).astype(np.float32) / (predicted_count + truth_count[j]).astype(np.float32)
            # Same float32, samplewise reduction as torchmetrics DiceScore.
            means = torch.from_numpy(np.ascontiguousarray(scores.T)).nanmean(dim=1).numpy()
            best = int(np.argmax(means))
            results[name] = {'zeroshot_dice': float(means[best]), 'best_threshold': float(self.thresholds[best])}
            curves[name] = {'thresholds': self.thresholds.tolist(), 'mean_dice': means.tolist()}
        author = seg_score(results)
        return {'dice': float(author['zeroshot_segmentation(Avg)']), 'author_metrics': author,
            'threshold_curves': curves,
            'protocol': 'Author positive-only samplewise Dice; independent 0:0.01:1 threshold search per candidate on evaluation labels',
            'implementation': 'Cached GT and exact threshold histograms; float32 per-image Dice mean'}


def run(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.evaluator_utils import classification_score, gather_pointinggame_statistics, pointing_game_score, get_grounding_type
    from cxrclip.prompt import constants
    from eval_query_refinement import audit_csv, classification_metrics, prior_classification_check
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator
    from query_candidates import fit_candidates

    started = time.time()
    check_deadline(args.deadline)
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('No assigned CUDA GPU')
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Published checkpoint identity mismatch')
    reference = args.baseline_run / args.dataset
    prior_result = json.loads((reference / 'result.json').read_text())
    if prior_result['status'] != 'complete' or prior_result['is_smoke'] or prior_result['eta'] != .3:
        raise ValueError('Expected full formal reference run with eta=.3')
    entry = json.loads((args.baseline_run / 'catalog.json').read_text())['datasets'][args.dataset]
    if prior_result['entry'] != entry:
        raise ValueError('Reference dataset provenance differs')
    original_sources = json.loads((args.baseline_run / 'source_sha256.json').read_text())
    for relative, digest in original_sources.items():
        if sha256(ALPHA / relative) != digest:
            raise ValueError(f'Original AlphaRAD source changed: {relative}')
    classes = entry['class_names']
    support_rows, support_paths, support_labels = audit_csv(entry['support_csv_path'], entry['support_csv_sha256'], classes)
    test_rows, test_paths, test_labels = audit_csv(entry['csv_path'], entry['csv_sha256'], classes, entry['task'] == 'classification')
    if len(test_rows) != entry['actual_count'] or len(support_rows) != entry['support_count']:
        raise ValueError('Reference manifest counts changed')
    if set(test_paths) & set(support_paths):
        raise ValueError('Support and evaluation image overlap')
    old_audit = json.loads((reference / 'data_audit.json').read_text())
    if old_audit['support_labels'] != support_labels:
        raise ValueError('Cached feature support label order differs')
    support_labels = np.asarray(support_labels, dtype=np.uint8)
    adapter_meta = json.loads((reference / 'query_adapter.json').read_text())
    if adapter_meta['support_csv_sha256'] != entry['support_csv_sha256'] or adapter_meta['test_csv_sha256'] != entry['csv_sha256']:
        raise ValueError('Frozen adapter support/evaluation identities differ')
    if sha256(reference / 'query_adapter.npz') != adapter_meta['query_file_sha256']:
        raise ValueError('Frozen adapter changed')
    cached = np.load(reference / 'query_adapter.npz', allow_pickle=False)
    features = np.load(reference / 'support_class_local.npy', allow_pickle=False)
    original = cached['original_queries']
    if features.shape != (*support_labels.shape, original.shape[-1]):
        raise ValueError('Support feature cache shape mismatch')
    groups, group_metadata = support_groups(support_rows)
    fitted = fit_candidates(features, support_labels, original, num_heads=adapter_meta['num_heads'], groups=groups)
    names, queries, metadata = list(fitted['names']), np.asarray(fitted['queries'], dtype=np.float32), list(fitted['metadata'])
    if len(names) != len(set(names)) or len(names) > 16 or queries.shape != (len(names), *original.shape) or len(metadata) != len(names):
        raise ValueError('Candidate API shape/name contract violation')
    if any(not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name) for name in names):
        raise ValueError('Unsafe candidate identifier')
    for name, expected in zip(CONTROLS, (cached['queries'][0], cached['queries'][1])):
        if name not in names:
            raise ValueError(f'Missing required control {name}')
        np.testing.assert_array_equal(queries[names.index(name)], expected)
    if args.candidate_names:
        unknown = set(args.candidate_names) - set(names)
        if unknown:
            raise ValueError(f'Unknown candidates: {sorted(unknown)}')
        requested = set(args.candidate_names) | set(CONTROLS)
        indices = [i for i, name in enumerate(names) if name in requested]
        names, queries, metadata = [names[i] for i in indices], queries[indices], [metadata[i] for i in indices]
    if not np.isfinite(queries).all():
        raise ValueError('Nonfinite candidate query')
    if names[:2] != list(CONTROLS):
        raise ValueError('Controls must be first two candidates in declared order')
    np.savez_compressed(args.output / 'candidate_queries.npz', names=np.asarray(names), queries=queries)
    provenance = {'entry': entry, 'candidate_names': names, 'candidate_metadata': metadata,
        'query_fit_inputs': 'Frozen original support class-local features and labels only',
        'groups': group_metadata, 'support_labels': support_labels.tolist(),
        'source_sha256': {str(path): sha256(path) for path in (Path(__file__), CODE/'query_candidates.py',
            reference/'support_class_local.npy', reference/'query_adapter.npz', reference/'data_audit.json')},
        'baseline_run': str(args.baseline_run), 'original_sources_verified': original_sources,
        'support_count': len(support_rows), 'evaluation_count': len(test_rows), 'deadline_epoch': args.deadline,
        'baseline_protocol': 'Identical fixed manifests; all candidates exploratory, no held-out model selection claim'}
    write_json(args.output / 'data_audit.json', provenance)
    check_deadline(args.deadline)
    if hasattr(constants, args.dataset.upper()):
        if list(getattr(constants, args.dataset.upper())) != classes:
            raise ValueError('Author class constant order changed')
    else:
        setattr(constants, args.dataset.upper(), classes)
    loader_options = {'batch_size': args.batch_size, 'num_workers': args.workers,
        'pin_memory': True, 'shuffle': False, 'drop_last': False}
    if args.workers:
        loader_options['prefetch_factor'] = 2
    config = {'base': {'seed': 1234}, 'test': {'checkpoint': [str(args.checkpoint)]},
        'data_test': {args.dataset: {'name': args.dataset, 'data_type': 'image_classification',
            'data_path': entry['csv_path'], 'n_class': len(classes), 'normalize': None}},
        'dataloader': {'test': loader_options}}
    write_json(args.output / 'eval_config.json', config)
    evaluator = Evaluator(config, [str(args.checkpoint)], 'test')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint['model']); del checkpoint
    evaluator.model.requires_grad_(False).eval()
    live_original, prompts = original_queries_from_evaluator(evaluator, classes)
    np.testing.assert_array_equal(live_original, original)
    matcher = QueryRefinementMatcher(evaluator.model, queries, tuple(float(i) for i in range(len(names))),
        use_all_image_tokens=evaluator.use_all_image_tokens)
    loader = evaluator.data_loader_dict[args.dataset]
    if args.limit_test:
        limit = min(args.limit_test, len(test_paths))
        loader = DataLoader(Subset(loader.dataset, range(limit)), collate_fn=loader.dataset.collate_fn, **loader_options)
        test_paths, test_labels = test_paths[:limit], test_labels[:limit]
    n, c = len(test_paths), len(classes)
    cached_paths = json.loads((reference/'image_paths.json').read_text())
    if test_paths != cached_paths[:n]:
        raise ValueError('Reference prediction order changed')
    requires_maps = entry['task'] != 'classification'
    prior_probabilities = {name: np.load(reference/f'{branch}_probabilities.npy', mmap_mode='r', allow_pickle=False)
                           for name, branch in zip(CONTROLS, ('baseline', 'refined'))}
    prior_maps = ({name: np.load(reference/f'{branch}_grounding_map.npy', mmap_mode='r', allow_pickle=False)
                  for name, branch in zip(CONTROLS, ('baseline', 'refined'))} if requires_maps else {})
    old_examples = np.load(reference/'similarity_map_examples.npz', allow_pickle=False)
    logits = {name: np.empty((n,c), dtype=np.float32) for name in names}
    maps = {}
    check_records = {name: {'probability_max_abs_error':0., 'probabilities_bitwise_equal':True,
        'map_max_abs_error':0., 'maps_bitwise_equal':True, 'images_checked':0,
        'map_check_scope':'all evaluation images' if requires_maps else 'first-batch saved examples'} for name in CONTROLS}
    baseline_checks = {'formal_run':str(args.baseline_run),'source_hashes_unchanged':True,
        'query_controls_bitwise_equal':True,'formal_prediction_controls':check_records}
    image_paths, label_names, boxes, masks = [], [], [], []
    candidate_dirs = {}
    for name, query, md in zip(names, queries, metadata):
        directory = args.output/'candidates'/name;directory.mkdir(parents=True)
        candidate_dirs[name] = directory
        np.save(directory/'query.npy', query, allow_pickle=False)
        write_json(directory/'query_metadata.json', {'name':name,'metadata':md,'prompts':prompts})
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f'{args.dataset}: {len(names)} shared-vision candidates'):
            check_deadline(args.deadline)
            start = len(image_paths); end = start + len(batch['image_paths'])
            if start == 0:
                # The author encoder result is reused by the control audit and all candidates.
                first_image = evaluator.encode_image(batch['images'])
                vision = matcher.prepare_vision_results(first_image)
            else:
                vision = matcher.encode_vision(batch['images'])
            examples = {}
            for index, name in enumerate(names):
                check_deadline(args.deadline)
                output = matcher.match_prepared(vision, matcher.queries[index], return_maps=requires_maps or start==0)
                actual_logits = output['logits'].cpu().numpy()
                logits[name][start:end] = actual_logits
                if requires_maps:
                    array = output['grounding_map'].cpu().numpy()
                    if name not in maps:
                        maps[name] = np.empty((n,*array.shape[1:]), dtype=np.float32)
                    maps[name][start:end] = array
                if start == 0:
                    examples[name] = output['similarity_scores'].cpu().numpy()
                    if name == 'original':
                        baseline_checks['author_helper'] = controls_helper_check(evaluator, first_image, batch, classes, output)
                if name in CONTROLS:
                    probability = torch.sigmoid(output['logits']).cpu().numpy()
                    expected = prior_probabilities[name][start:end]
                    np.testing.assert_allclose(probability, expected, atol=2e-6, rtol=1e-5)
                    record = check_records[name]
                    record['probability_max_abs_error'] = max(record['probability_max_abs_error'],float(np.max(np.abs(probability-expected))))
                    record['probabilities_bitwise_equal'] &= bool(np.array_equal(probability,expected))
                    record['images_checked'] += len(probability)
                    if requires_maps:
                        expected_map = prior_maps[name][start:end]
                        np.testing.assert_allclose(array, expected_map, atol=2e-5, rtol=1e-5)
                        record['map_max_abs_error'] = max(record['map_max_abs_error'],float(np.max(np.abs(array-expected_map))))
                        record['maps_bitwise_equal'] &= bool(np.array_equal(array,expected_map))
                    elif start == 0:
                        branch = 'baseline' if name=='original' else 'refined'
                        overlap = min(len(examples[name]),len(old_examples[branch]))
                        current_map, saved_map = examples[name][:overlap], old_examples[branch][:overlap]
                        np.testing.assert_allclose(current_map,saved_map,atol=2e-5,rtol=1e-5)
                        record['map_max_abs_error'] = float(np.max(np.abs(current_map-saved_map)))
                        record['maps_bitwise_equal'] = bool(np.array_equal(current_map,saved_map))
            if start == 0:
                np.savez_compressed(args.output/'similarity_map_examples.npz',**examples,image_paths=np.asarray(batch['image_paths']))
                write_json(args.output/'baseline_checks.json',baseline_checks)
                del first_image
            image_paths.extend(batch['image_paths']); label_names.extend(batch['label_names']);boxes.extend(batch['boxes'])
            # Bool GT saves memory; original masks contain only 0/1.
            masks.extend([m.bool() for m in batch['masks']])
            write_json(args.output/'progress.json',{'status':'inference','images_completed':len(image_paths),
                'images_total':n,'candidate_names':names,'deadline_epoch':args.deadline})
    if image_paths != test_paths:
        raise ValueError('Evaluation image order/coverage changed')
    check_deadline(args.deadline)
    write_json(args.output/'image_paths.json',image_paths)
    if entry['task']=='classification':
        np.save(args.output/'labels.npy',np.asarray(test_labels,dtype=np.uint8),allow_pickle=False)
        baseline_probability = torch.sigmoid(torch.from_numpy(logits['original'])).numpy()
        baseline_checks['original_pretrained_cache'] = prior_classification_check(
            ALPHA/'runs/pretrained_20261021_122700',args.dataset,image_paths,classes,test_labels,baseline_probability)
    dice_evaluator = FastPositiveDice(masks,label_names,classes) if entry['task']=='segmentation' else None
    candidate_results = {}
    keys = ('macro_auroc','macro_average_precision','macro_auprc','pointing_game','dice')
    for name, md in zip(names,metadata):
        check_deadline(args.deadline)
        directory = candidate_dirs[name]
        probability = torch.sigmoid(torch.from_numpy(logits[name])).numpy()
        np.save(directory/'logits.npy',logits[name],allow_pickle=False)
        np.save(directory/'probabilities.npy',probability,allow_pickle=False)
        if requires_maps and args.save_full_maps:
            np.save(directory/'grounding_map.npy',maps[name],allow_pickle=False)
        if entry['task']=='classification':
            metrics = classification_metrics(test_labels,probability)
            metrics['per_class'] = dict(zip(classes,metrics['per_class']))
            if not args.limit_test:
                stats, counts = evaluator.gather_zeroshot_dqn_classification_statistics(classes,probability[...,None],label_names)
                author = classification_score(stats,counts)
                np.testing.assert_allclose(metrics['macro_auroc'],author['AUROC(Avg)'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(metrics['macro_average_precision'],author['PR_AUROC(Avg)'],atol=1e-10,rtol=0)
                metrics['author_metrics'] = author
        elif entry['task']=='grounding':
            author = pointing_game_score(gather_pointinggame_statistics(classes,maps[name],boxes,label_names,image_paths,
                grounding_type=get_grounding_type(args.dataset)))
            metrics = {'pointing_game':float(author['Pointing_Game(Avg)']),'author_metrics':author}
        else:
            metrics = dice_evaluator.evaluate(maps[name],args.deadline)
        check_deadline(args.deadline)
        if name in CONTROLS and not args.limit_test:
            prior = prior_result['baseline' if name=='original' else 'refined']
            for key in keys:
                if key in metrics:
                    # Float32 Dice reduction permits at most two ulps; other metrics are exact.
                    tolerance = 2e-7 if key=='dice' else 1e-10
                    np.testing.assert_allclose(metrics[key],prior[key],atol=tolerance,rtol=0)
            if entry['task']=='segmentation':
                for cl in classes:
                    if metrics['author_metrics'][cl]['best_threshold'] != prior['author_metrics'][cl]['best_threshold']:
                        raise ValueError('Fast Dice selected a different threshold from the frozen author reference')
        baseline_metrics = metrics if name=='original' else candidate_results['original']['metrics']
        delta = {key:100*(metrics[key]-baseline_metrics[key]) for key in keys if metrics.get(key) is not None}
        candidate_results[name] = {'metrics':metrics,'delta_pp':delta,'metadata':md}
        write_json(directory/'metrics.json',candidate_results[name])
        write_json(args.output/'progress.json',{'status':'metrics','images_completed':n,'images_total':n,
            'candidates_completed':list(candidate_results),'candidate_names':names,'deadline_epoch':args.deadline})
    check_deadline(args.deadline)
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError('Model unexpectedly contains trainable parameters')
    for record in check_records.values():
        record['passed']=True
    write_json(args.output/'baseline_checks.json',baseline_checks)
    result = {'status':'complete','model':'alpharad','dataset':args.dataset,'task':entry['task'],
        'rows':n,'support_count':len(support_rows),'is_smoke':bool(args.limit_test),'candidates':candidate_results,
        'baseline_checks':baseline_checks,'baseline_run':str(args.baseline_run),'entry':entry,
        'checkpoint_sha256':CHECKPOINT_SHA,'ignored_checkpoint_keys':ignored,'no_network_weight_updates':True,
        'batch_size':args.batch_size,'vision_forward_passes':len(loader),'elapsed_seconds':time.time()-started,
        'peak_gpu_memory_mib':torch.cuda.max_memory_allocated()/1024**2,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'deadline_epoch':args.deadline}
    write_json(args.output/'result.json',result)
    write_json(args.output/'progress.json',{'status':'complete','images_completed':n,'images_total':n,'candidate_names':names})
    print(json.dumps({'status':'complete','dataset':args.dataset,'candidates':{k:v['delta_pp'] for k,v in candidate_results.items()}}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--checkpoint',type=Path,default=ASSETS/'model/xraydinov2_mpnet/eccv_mpnet_518.tar')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--deadline',type=float,required=True,help='Hard cutoff in Unix epoch seconds')
    parser.add_argument('--limit-test',type=int,default=0,help='Smoke only: prefix of fixed evaluation manifest')
    parser.add_argument('--candidate-names',nargs='+')
    parser.add_argument('--save-full-maps',action='store_true',help='Persist complete grounding maps for nonclassification tasks')
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<0 or args.limit_test<0 or not math.isfinite(args.deadline):
        parser.error('Invalid batch/workers/limit/deadline')
    args.output=args.output.resolve();args.baseline_run=args.baseline_run.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if any((args.output/p).exists() for p in ('result.json','data_audit.json','candidate_queries.npz','failure.json')):
        raise FileExistsError('Use a fresh output directory; prior experiment artifacts are preserved')
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,handle_signal)
    try:
        run(args)
    except DeadlineReached as exc:
        progress_path=args.output/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'images_completed':0}
        interruption={'status':'interrupted','model':'alpharad','dataset':args.dataset,'reason':str(exc),
            'deadline_epoch':args.deadline,'is_smoke':bool(args.limit_test),'progress':progress}
        write_json(args.output/'interrupted.json',interruption)
        write_json(args.output/'result.json',interruption)
        write_json(progress_path,{**progress,'status':'interrupted','reason':str(exc)})
        print(json.dumps(interruption),flush=True)
        raise SystemExit(124)
    except BaseException as exc:
        write_json(args.output/'failure.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
