#!/usr/bin/env python3
"""Evaluate fixed spatial-view ensembles through unchanged frozen AlphaRAD.

All reference code and baseline artifacts are imported/read without alteration.
The original and uniform_eta03 controls are checked against the frozen formal
run on every evaluated image. A deadline or signal produces interrupted status,
never a complete result. Original image tokens come from certified FP32 caches,
and every real image is decoded for the extra spatial views. Each extra view is
encoded once for both unchanged support-derived queries. Grounding/segmentation
retain the author's original boxes, masks, sample sets, and scoring protocols.
"""
from __future__ import annotations

import argparse
import importlib.util
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
ACTIVE_CACHE = None


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


def load_candidate_components(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('explicit_query_candidates', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'candidate_plan', None)):
        raise ValueError('Spatial-view candidate module must provide candidate_plan(name)')
    return module.fit_candidates, module.candidate_plan


def validate_view_plans(names, queries, candidate_plan, view_specs):
    """Every ensemble reuses exactly one of the two authenticated old queries."""
    import numpy as np
    plans = {}
    for index, name in enumerate(names):
        plan = candidate_plan(name)
        if (set(plan) != {'views', 'query'} or plan['query'] not in CONTROLS
                or not isinstance(plan['views'], list) or not plan['views']
                or len(plan['views']) != len(set(plan['views']))
                or 'identity' not in plan['views']
                or any(view not in view_specs for view in plan['views'])):
            raise ValueError('Invalid fixed spatial-view plan: ' + name)
        if name in CONTROLS and plan != {'views': ['identity'], 'query': name}:
            raise ValueError('The two formal controls must remain original-view-only')
        reference = names.index(plan['query'])
        np.testing.assert_array_equal(queries[index], queries[reference])
        plans[name] = plan
    return plans


def score_view_candidates(matcher, vision, pixels, names, plans, return_maps, deadline):
    """One encoding per extra view serves both old native matching queries."""
    import torch
    from view_geometry import apply_view, aggregate_outputs

    ordered_views = list(dict.fromkeys(view for name in names for view in plans[name]['views']))
    outputs = {'identity': {}}
    for query_name in CONTROLS:
        check_deadline(deadline)
        outputs['identity'][query_name] = matcher.match_prepared(
            vision, matcher.queries[names.index(query_name)], return_maps=return_maps)
    extra = [name for name in ordered_views if name != 'identity']
    if extra:
        pixels = pixels.to(matcher.device, dtype=torch.float32, non_blocking=True)
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 518, 518):
            raise ValueError('Spatial views require the original author-preprocessed [B,3,518,518] pixels')
    for view_name in extra:
        check_deadline(deadline)
        view_pixels = apply_view(pixels, view_name)
        prepared = matcher.encode_vision(view_pixels)
        outputs[view_name] = {}
        for query_name in CONTROLS:
            check_deadline(deadline)
            outputs[view_name][query_name] = matcher.match_prepared(
                prepared, matcher.queries[names.index(query_name)], return_maps=return_maps)
        del prepared, view_pixels
    result = {}
    for name in names:
        check_deadline(deadline)
        plan = plans[name]
        result[name] = aggregate_outputs({view: outputs[view][plan['query']] for view in plan['views']},
                                         plan['views'])
        if result[name]['logits'].shape != outputs['identity'][plan['query']]['logits'].shape:
            raise ValueError('Spatial ensemble changed classification shape')
        if not torch.isfinite(result[name]['logits']).all():
            raise ValueError('Nonfinite spatial ensemble logits')
        if return_maps:
            for field in ('grounding_map', 'similarity_scores'):
                if (result[name][field].shape != outputs['identity'][plan['query']][field].shape
                        or not torch.isfinite(result[name][field]).all()):
                    raise ValueError('Invalid inverse-warped native AlphaRAD map: ' + field)
    return result, {'extra_vision_forward_passes': len(extra), 'extra_view_names': extra,
                    'native_matching_calls': 2 * (1 + len(extra)), 'real_image_rows': len(pixels)}


def load_global_support(path, dataset, entry, paths, labels, num_heads, original, reference):
    import numpy as np
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    checks = {'status': 'complete', 'model': 'alpharad', 'dataset': dataset,
        'split': 'support_only', 'checkpoint_sha256': CHECKPOINT_SHA,
        'support_csv_sha256': entry['support_csv_sha256'], 'image_paths': paths,
        'labels': labels.tolist(), 'num_heads': num_heads,
        'global_features_sha256': sha256(path),
        'reference_class_local_sha256': sha256(reference / 'support_class_local.npy')}
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise ValueError('Global support manifest mismatch: ' + key)
    if manifest.get('reference_check', {}).get('passed') is not True:
        raise ValueError('Global support extraction did not verify old local features')
    if [sha256(image) for image in paths] != manifest['image_file_sha256']:
        raise ValueError('Global support source image bytes changed')
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (len(paths), original.shape[-1]) or not np.isfinite(values).all():
        raise ValueError('Invalid global support feature array')
    return values, {'path': str(path), 'sha256': sha256(path), 'manifest_path': str(manifest_path),
                    'manifest_sha256': sha256(manifest_path), 'global_definition': manifest['global_definition']}


def pack_vision(vision):
    import torch
    if len(vision['image_cls_raw_features']) != 1 or len(vision['image_patch_raw_features']) != 1:
        raise ValueError('Visual cache supports only the published one-layer AlphaRAD checkpoint')
    return torch.cat([vision['image_cls_raw_features'][0][:, None],
                      vision['image_patch_raw_features'][0]], 1).cpu().numpy()


def unpack_vision(tokens, device):
    import numpy as np
    import torch
    value = torch.from_numpy(np.array(tokens, copy=True, order='C')).to(device)
    return {'image_cls_raw_features': [value[:, 0]], 'image_patch_raw_features': [value[:, 1:]]}


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
        'vision_passes_for_this_batch': 1,
        'vision_pass_count_scope': 'Original-view live cache/helper audit only; extra TTA views are counted separately'}


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
    global ACTIVE_CACHE
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
    from vision_cache import VisionTokenCache, image_fingerprints
    from view_geometry import VIEW_SPECS
    fit_candidates, candidate_plan = load_candidate_components(args.candidate_module)

    started = time.time()
    check_deadline(args.deadline)
    if args.cache_mode not in ('read', 'auto'):
        raise ValueError('Spatial-view ensembles require an existing certified original-image token cache')
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
    fit_options = {'num_heads': adapter_meta['num_heads'], 'groups': groups}
    global_provenance = None
    if args.global_support_features:
        raise ValueError('Spatial-view ensembles retain the original two support queries; no global support features are used')
    fitted = fit_candidates(features, support_labels, original, **fit_options)
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
    plans = validate_view_plans(names, queries, candidate_plan, VIEW_SPECS)
    np.savez_compressed(args.output / 'candidate_queries.npz', names=np.asarray(names), queries=queries)
    provenance = {'entry': entry, 'candidate_names': names, 'candidate_metadata': metadata,
        'query_fit_inputs': 'Frozen original support class-local features and labels only',
        'groups': group_metadata, 'support_labels': support_labels.tolist(),
        'source_sha256': {str(path): sha256(path) for path in (Path(__file__), args.candidate_module, CODE/'vision_cache.py', CODE/'view_geometry.py',
            reference/'support_class_local.npy', reference/'query_adapter.npz', reference/'data_audit.json')},
        'baseline_run': str(args.baseline_run), 'original_sources_verified': original_sources,
        'support_count': len(support_rows), 'evaluation_count': len(test_rows), 'deadline_epoch': args.deadline,
        'baseline_protocol': 'Identical fixed manifests; all candidates exploratory, no held-out model selection claim'}
    provenance['global_support_features'] = global_provenance
    provenance['view_plans'] = plans
    provenance['view_geometry'] = VIEW_SPECS
    provenance['view_aggregation'] = {
        'classification': 'Equal-weight arithmetic mean of native logits over each fixed view list, then sigmoid',
        'spatial_maps': 'Native raw spatial similarities inverse-warped to original 37x37 coordinates and averaged by valid coverage',
        'cls': 'CLS grounding score averaged independently across views',
        'labels_boxes_masks': 'Unchanged original author evaluation annotations and coordinate conventions',
        'support_queries_recomputed_for_views': False, 'extra_view_tokens_persisted': False,
        'all_original_images_decoded': True}
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
    vision_cache = None
    cache_record = {'requested_mode': args.cache_mode, 'active': False}
    if args.cache_mode != 'off':
        if not evaluator.use_all_image_tokens:
            raise ValueError('Cache identity currently requires the published all-token branch')
        input_fingerprints = image_fingerprints(test_paths)
        identity = {'model': 'alpharad', 'checkpoint_sha256': CHECKPOINT_SHA,
            'stage': 'prepared_projected_before_multihead_branch', 'dtype': 'float32',
            'expected_token_shape': [1370, 768], 'num_heads': adapter_meta['num_heads'],
            'use_all_image_tokens': evaluator.use_all_image_tokens,
            'original_source_sha256': original_sources,
            'preprocessing': repr(loader.dataset.image_transforms),
            'normalization': loader.dataset.normalize,
            'test_csv_sha256': entry['csv_sha256'], 'support_csv_sha256': entry['support_csv_sha256'],
            'batch_size': args.batch_size, 'seed': 1234, 'matmul_allow_tf32': False,
            'torch_version': torch.__version__, 'input_file_fingerprints': input_fingerprints}
        vision_cache = VisionTokenCache(args.vision_cache_root, identity, test_paths, mode=args.cache_mode,
            shard_rows=args.cache_shard_rows, prefetch=args.cache_prefetch, deadline=args.deadline)
        ACTIVE_CACHE = vision_cache
        if vision_cache.mode != 'read':
            raise ValueError('Original-image token cache is not complete/certified; build it with the frozen cache worker first')
        cache_record.update(active=True, key=vision_cache.key, path=str(vision_cache.path),
            initial_mode=vision_cache.mode, initial_cached_rows=vision_cache.cached_rows,
            image_identity='Ordered source paths and size/mtime/ctime fingerprints; full formal prediction controls mandatory')
        # Keep the author's genuine image decoder for every row. All extra views
        # derive from these pixels; cached tokens represent the original only.
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
        'query_controls_bitwise_equal':True,'formal_prediction_controls':check_records,
        'vision_cache': cache_record}
    image_paths, label_names, boxes, masks = [], [], [], []
    candidate_dirs = {}
    for name, query, md in zip(names, queries, metadata):
        directory = args.output/'candidates'/name;directory.mkdir(parents=True)
        candidate_dirs[name] = directory
        np.save(directory/'query.npy', query, allow_pickle=False)
        write_json(directory/'query_metadata.json', {'name':name,'metadata':md,'prompts':prompts})
    vision_forward_passes = 0
    extra_view_forward_passes = native_matching_calls = decoded_image_rows = 0
    extra_view_rows = {}
    cache_rows_read = cache_rows_encoded = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f'{args.dataset}: {len(names)} shared-vision candidates'):
            check_deadline(args.deadline)
            start = len(image_paths); end = start + len(batch['image_paths'])
            if batch['image_paths'] != test_paths[start:end]:
                raise ValueError('Loader order differs from cache manifest')
            cached_tokens = vision_cache.get(start, end) if vision_cache is not None else None
            if cached_tokens is None:
                raise ValueError('Certified original-image cache unexpectedly lacks this batch')
            if start == 0:
                # One live original encoding audits the cache and author's helpers.
                # Matching uses certified cached tokens, as on all later batches.
                first_image = evaluator.encode_image(batch['images'])
                live_vision = matcher.prepare_vision_results(first_image)
                vision_forward_passes += 1
                actual_tokens = pack_vision(live_vision)
                np.testing.assert_allclose(cached_tokens, actual_tokens, atol=1e-6, rtol=1e-6)
                cache_record['fresh_first_batch_tokens'] = {'passed': True, 'images_checked': end,
                    'max_abs_error': float(np.max(np.abs(cached_tokens - actual_tokens))),
                    'bitwise_equal': bool(np.array_equal(cached_tokens, actual_tokens))}
                del live_vision
            vision = unpack_vision(cached_tokens, matcher.device)
            cache_rows_read += end - start
            examples = {}
            view_outputs, view_audit = score_view_candidates(
                matcher, vision, batch['images'], names, plans, requires_maps or start == 0, args.deadline)
            extra_view_forward_passes += view_audit['extra_vision_forward_passes']
            vision_forward_passes += view_audit['extra_vision_forward_passes']
            native_matching_calls += view_audit['native_matching_calls']
            decoded_image_rows += view_audit['real_image_rows']
            for view_name in view_audit['extra_view_names']:
                extra_view_rows[view_name] = extra_view_rows.get(view_name, 0) + end - start
            for index, name in enumerate(names):
                check_deadline(args.deadline)
                output = view_outputs[name]
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
                'images_total':n,'candidate_names':names,'deadline_epoch':args.deadline,
                'cache_rows_read':cache_rows_read,'cache_rows_encoded':cache_rows_encoded,
                'decoded_image_rows':decoded_image_rows,'extra_view_rows':extra_view_rows,
                'extra_view_forward_passes':extra_view_forward_passes})
            del view_outputs, vision
    if image_paths != test_paths:
        raise ValueError('Evaluation image order/coverage changed')
    check_deadline(args.deadline)
    write_json(args.output/'image_paths.json',image_paths)
    if entry['task']=='classification':
        np.save(args.output/'labels.npy',np.asarray(test_labels,dtype=np.uint8),allow_pickle=False)
        baseline_probability = torch.sigmoid(torch.from_numpy(logits['original'])).numpy()
        baseline_checks['original_pretrained_cache'] = prior_classification_check(
            ALPHA/'runs/pretrained_20261021_122700',args.dataset,image_paths,classes,test_labels,baseline_probability)
    for record in check_records.values():
        if record['images_checked'] != n:
            raise ValueError('Incomplete formal control prediction verification')
        record['passed'] = True
    if vision_cache is not None:
        cache_record.update(rows_read=cache_rows_read, rows_encoded=cache_rows_encoded,
                            final_cached_rows=vision_cache.cached_rows)
        if not args.limit_test:
            if image_fingerprints(test_paths) != input_fingerprints:
                raise ValueError('Input source files changed during evaluation/cache creation')
            manifest = vision_cache.finalize(check_records, extra={
                'formal_run':str(args.baseline_run), 'worker_output':str(args.output),
                'worker_sha256':sha256(__file__), 'baseline_checks':baseline_checks})
            cache_record.update(certified=True, final_cached_rows=vision_cache.cached_rows,
                manifest_sha256=sha256(vision_cache.manifest_path))
        else:
            cache_record.update(certified=vision_cache.mode == 'read', smoke_did_not_certify=True)
        write_json(args.output/'vision_cache.json',cache_record)
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
        'batch_size':args.batch_size,'vision_forward_passes':vision_forward_passes,'elapsed_seconds':time.time()-started,
        'extra_view_forward_passes':extra_view_forward_passes,'native_matching_calls':native_matching_calls,
        'decoded_image_rows':decoded_image_rows,'extra_view_rows':extra_view_rows,'view_plans':plans,
        'vision_cache':cache_record,'global_support_features':global_provenance,
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
    parser.add_argument('--candidate-module',type=Path,default=CODE/'query_candidates.py')
    parser.add_argument('--global-support-features',type=Path)
    parser.add_argument('--vision-cache-root',type=Path,default=CODE.parent/'assets/vision_tokens_fp32')
    parser.add_argument('--cache-mode',choices=('auto','read'),default='read')
    parser.add_argument('--cache-shard-rows',type=int,default=64)
    parser.add_argument('--cache-prefetch',type=int,choices=(0,1),default=1)
    args=parser.parse_args()
    if args.batch_size<1 or args.workers<0 or args.limit_test<0 or args.cache_shard_rows<1 or not math.isfinite(args.deadline):
        parser.error('Invalid batch/workers/limit/deadline')
    args.output=args.output.resolve();args.baseline_run=args.baseline_run.resolve()
    args.candidate_module=args.candidate_module.resolve()
    if args.global_support_features:
        args.global_support_features=args.global_support_features.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if any((args.output/p).exists() for p in ('result.json','data_audit.json','candidate_queries.npz','failure.json')):
        raise FileExistsError('Use a fresh output directory; prior experiment artifacts are preserved')
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,handle_signal)
    try:
        run(args)
    except (DeadlineReached, TimeoutError) as exc:
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
    finally:
        if ACTIVE_CACHE is not None:
            ACTIVE_CACHE.abort('Worker ended; preserve completed cache or committed partial prefix')

if __name__=='__main__':main()
