#!/usr/bin/env python3
"""Four-branch MAVL evaluations with a frozen missing-description overlay."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import MethodType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = ROOT / 'baselines/MAVL'
COMMON = ROOT / 'baselines/local_setup'
sys.path.insert(0, str(COMMON))
sys.dont_write_bytecode = True
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from evaluate_baseline_buffered import write, now, classification_metrics
from evaluate_query_transfer import load_query, paired, check_native
from query_refinement_common import center_pixels
from adapter import Adapter, state_digest, DESCRIPTIONS, DESCRIPTIONS_SHA, NATIVE

BRANCHES = ('original','query_refined','tta_center_original','tta_center_refined')
VIEW_ARRAYS = ('identity_original','identity_refined','center_original','center_refined')
DATASETS = ('openi','shenzhenxray','montgomery')


def cached_query_pixels(query, paths, view):
    query.native.cache_batch(paths)
    images = [query.native._read_image(str(path)) for path in paths]
    tensors = [query.native.transform(image) for image in images]
    sizes = [(image.height, image.width) for image in images]
    return center_pixels(torch.stack(tensors).to(query.device), view), sizes


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inputs(dataset):
    support_path = ROOT / 'baselines/evaluation/query_transfer_20261022_v1' / f'{dataset}.json'
    support = json.loads(support_path.read_text())
    test_path = Path(support['test_manifest'])
    assert sha(test_path) == support['test_manifest_sha256']
    test = json.loads(test_path.read_text())
    assert test['class_names'] == support['class_names']
    assert test['count'] == len(test['rows']) == support['count']
    assert support['support_count'] == len(support['support_rows'])
    for path,digest in support['support_sources'].items():
        assert sha(path) == digest, path
    for row in support['support_rows']:
        assert sha(row['image']) == row['byte_sha256'], row['image']
    assert not ({row['image'] for row in support['support_rows']} & {row['image'] for row in test['rows']})
    labels = np.asarray([row['labels'] for row in support['support_rows']])
    assert labels.sum(0).tolist() == support['support_positive_counts']
    source = {str(support_path):sha(support_path),str(test_path):sha(test_path),**support['support_sources']}
    return test,support,source


def fit_and_capture(native, test, supports, batch_size):
    """Reuse unmodified query adapter, observing pooled support outputs only."""
    native.set_dataset(test)
    native.configure(test['class_names'])
    query = load_query('MAVL', native)
    query._pixels = MethodType(cached_query_pixels, query)
    query._activate(test['class_names'], 'classification')
    model = native.model
    old = model.image_encoder
    had_override = 'image_encoder' in model.__dict__
    captured = []
    def capture(pixels):
        output = old(pixels)
        captured.append(output[1].detach().cpu().clone())
        return output
    model.image_encoder = capture
    try:
        fit = query.fit([r['image'] for r in supports['support_rows']],
                        [r['labels'] for r in supports['support_rows']],test['class_names'],
                        batch_size=batch_size,task='classification')
    finally:
        if had_override:
            model.image_encoder = old
        else:
            del model.image_encoder
    features = torch.cat(captured).numpy()
    assert features.shape == (supports['support_count'],256)
    assert np.isfinite(features).all()
    return query,fit,features


def run_views(query, paths, classes):
    query.native.cache_batch(paths)
    identity = paired(query,'predict',paths,classes,'identity')
    center = paired(query,'predict',paths,classes,'center80')
    result = dict(identity_original=np.asarray(identity['baseline']),
                  identity_refined=np.asarray(identity['refined']),
                  center_original=np.asarray(center['baseline']),
                  center_refined=np.asarray(center['refined']))
    result.update(original=result['identity_original'],query_refined=result['identity_refined'],
                  tta_center_original=(result['identity_original']+result['center_original'])*.5,
                  tta_center_refined=(result['identity_refined']+result['center_refined'])*.5)
    return result


def regression(native, output, batch_size):
    """Check unchanged 75-disease context against historical supported results."""
    test,support,sources = inputs('chest14')
    historical = ROOT / 'baselines/runs/query_transfer_20261022_224826/MAVL_full_1/chest14'
    recorded = json.loads((historical / 'result.json').read_text())
    assert recorded['status'] == 'complete'
    query,fit,features = fit_and_capture(native,test,support,batch_size)
    assert not native.extra and native.model.n_disease == 75
    oldfit = json.loads((historical / 'query_adapter.json').read_text())
    query_check = {}
    for key in ('original_queries','refined_queries'):
        values = np.asarray(fit[key],dtype=np.float32)
        expected = np.asarray(oldfit[key],dtype=np.float32)
        np.testing.assert_allclose(values,expected,atol=2e-6,rtol=1e-5)
        query_check[key] = dict(max_abs_error=float(np.abs(values-expected).max()),
                               bitwise_equal=bool(np.array_equal(values,expected)))
    count = 8
    paths = [r['image'] for r in test['rows'][:count]]
    pixel_check = native.verify_pixel_cache(paths)
    actual = run_views(query,paths,test['class_names'])
    expected = np.load(historical/'predictions.npz',allow_pickle=False)
    assert paths == expected['image_paths'][:count].tolist()
    checks = {}
    for key in BRANCHES:
        values,baseline = actual[key],expected[key][:count]
        np.testing.assert_allclose(values,baseline,atol=2e-4,rtol=2e-4)
        checks[key] = dict(passed=True,max_abs_error=float(np.abs(values-baseline).max()),
                           bitwise_equal=bool(np.array_equal(values,baseline)))
    result = dict(passed=True,rows=count,dataset='chest14',context_diseases=75,
                  context_concepts=675,branches=checks,query_checks=query_check,
                  historical_result=str(historical/'result.json'),
                  historical_result_sha256=sha(historical/'result.json'),
                  historical_predictions_sha256=sha(historical/'predictions.npz'),
                  pixel_cache=pixel_check,
                  sources=sources)
    write(output,result)
    return result


def common_sources():
    checkpoint = MODEL/'local_assets/checkpoint_full_46.pth'
    original_manifest = json.loads((MODEL/'local_setup/manifest.json').read_text())
    expected = next(r['sha256'] for r in original_manifest['checkpoints'] if r['path'].endswith('checkpoint_full_46.pth'))
    assert sha(checkpoint) == expected
    paths = [Path(__file__).resolve(),HERE/'adapter.py',NATIVE,MODEL/'local_setup/query_refinement_adapter.py',
             MODEL/'local_setup/manifest.json',COMMON/'query_refinement_common.py',
             COMMON/'evaluate_baseline_buffered.py',COMMON/'evaluate_query_transfer.py',
             DESCRIPTIONS,MODEL/'Zero-shot_classification/observation explanation.json',
             MODEL/'Pretrain/concept_gen/gpt4_mimic.json']
    paths += sorted((MODEL/'Zero-shot_classification/models').glob('*.py'))
    bert = MODEL/'local_assets/Bio_ClinicalBERT'
    paths += [p for p in sorted(bert.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors')]
    hashes = {str(p.resolve()):sha(p) for p in paths}
    hashes[str(checkpoint.resolve())] = expected
    return hashes


def snapshot_sources(output, sources):
    snapshot = output/'source_snapshots'
    snapshot.mkdir()
    records = []
    for i,(path,digest) in enumerate(sources.items()):
        p = Path(path)
        if p.suffix not in ('.py','.json','.txt'):
            continue
        dest = snapshot/f'{i:03d}_{p.name}'
        shutil.copyfile(p,dest)
        assert sha(dest) == digest
        records.append(dict(source=path,snapshot=str(dest.relative_to(output)),sha256=digest))
    write(snapshot/'index.json',records)


def evaluate(args,dataset,native,regression_record,static_sources):
    started = time.time()
    output = args.output/dataset
    output.mkdir(parents=True,exist_ok=False)
    test,support,sources = inputs(dataset)
    sources = {**static_sources,**sources}
    classes = test['class_names']
    rows = test['rows'][:args.smoke] if args.smoke else test['rows']
    assert support['baseline_references']['MAVL']['status'] == 'unsupported'
    write(output/'status.json',dict(status='fitting_support_queries',rows=len(rows)))
    query,fit,features = fit_and_capture(native,test,support,args.batch_size)
    expected_extra = 2 if dataset=='openi' else 1
    assert len(native.extra) == expected_extra
    fit['query_context'] = f'Original75 diseases plus {expected_extra} missing disease descriptions; nine slots each; existing slots unchanged.'
    fit['frozen_descriptions_sha256'] = DESCRIPTIONS_SHA
    fit['test_labels_used_for_query_fit'] = False
    support_labels = np.asarray([r['labels'] for r in support['support_rows']],dtype=np.int8)
    np.save(output/'support_features.npy',features,allow_pickle=False)
    np.save(output/'support_labels.npy',support_labels,allow_pickle=False)
    np.savez_compressed(output/'queries.npz',original=np.asarray(fit['original_queries'],np.float32),
                        refined=np.asarray(fit['refined_queries'],np.float32))
    context = dict(native.extension_arrays)
    context.update(base_raw_queries=context['original_disease_book'],full_raw_queries=context['disease_book'],
                   base_concept_queries=context['original_concept_book'],full_concept_queries=context['concept_book'],
                   full_projected_queries=native.model.disease_embedding_layer(native.model.concept_book).reshape(native.model.n_disease,9,-1).cpu().numpy(),
                   context_class_names=np.asarray(native.classes),target_class_indices=np.asarray(fit['class_query_indices']))
    np.savez_compressed(output/'query_context.npz',**context)
    write(output/'query_adapter.json',fit)
    write(output/'extension_integrity.json',native.metadata['extension_integrity'])
    write(output/'historical_regression.json',regression_record)
    write(output/'manifest.json',support)
    input_rows=[]
    for row in rows:
        input_rows.append({**row,'byte_sha256':sha(row['image'])})
    assert not ({r['byte_sha256'] for r in support['support_rows']} & {r['byte_sha256'] for r in input_rows})
    write(output/'input_manifest.json',dict(dataset=dataset,class_names=classes,rows=input_rows,
                                          support_rows=support['support_rows'],source_sha256=sources))
    snapshot_sources(output,sources)
    write(output/'run_config.json',dict(model='MAVL',dataset=dataset,batch_size=args.batch_size,
         cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],seed=1234,eta=.3,
         native_checkpoint=str(MODEL/'local_assets/checkpoint_full_46.pth'),
         precision='float32; cuda.matmul.allow_tf32=False; cudnn.allow_tf32=True, matching historical worker default',
         frozen_descriptions_sha256=DESCRIPTIONS_SHA,source_sha256=sources,
         original_query_context_classes=75,extended_query_context_classes=native.model.n_disease,
         query_class_names=list(native.classes),score_kind='positive_class_softmax_probability',
         views=['identity','center80'],center_ratio=.8,probability_fusion='arithmetic mean',
         test_labels_used_for_query_fit=False,no_gradient_training=True,command=sys.argv))
    arrays={k:[] for k in BRANCHES+VIEW_ARRAYS}
    parity=None
    last_print=0
    for offset in range(0,len(rows),args.batch_size):
        paths=[r['image'] for r in rows[offset:offset+args.batch_size]]
        values=run_views(query,paths,classes)
        if offset==0:
            write(output/'pixel_cache_parity.json',native.verify_pixel_cache(paths))
            parity=check_native(values['original'],native.predict(paths,classes))
            write(output/'native_parity.json',parity)
            # A zero adaptation coefficient is the unchanged projected query.
            saved=query.refined_queries
            query.refined_queries=torch.as_tensor(np.asarray(fit['original_queries'],np.float32),device=native.device)
            zero=query.predict(paths,classes,refined=True,view='identity')
            query.refined_queries=saved
            write(output/'zero_eta_parity.json',check_native(zero,values['original']))
        for key,value in values.items():
            assert value.shape==(len(paths),len(classes)) and np.isfinite(value).all()
            assert np.min(value)>=0 and np.max(value)<=1
            arrays[key].append(value)
        done=offset+len(paths)
        if time.time()-last_print>15 or done==len(rows):
            write(output/'status.json',dict(status='running',done=done,total=len(rows),updated_at=now()))
            print(f'MAVL/{dataset}: {done}/{len(rows)}',flush=True)
            last_print=time.time()
    labels=np.asarray([r['labels'] for r in rows],dtype=np.int8)
    saved={key:np.concatenate(parts) for key,parts in arrays.items()}
    saved.update(labels=labels,image_paths=np.asarray([r['image'] for r in rows]),class_names=np.asarray(classes))
    np.savez_compressed(output/'predictions.npz',**saved)
    np.save(output/'labels.npy',labels,allow_pickle=False)
    write(output/'image_paths.json',[r['image'] for r in rows])
    branches={key:classification_metrics(labels,saved[key],classes) for key in BRANCHES}
    assert state_digest(native.model)==native.metadata['extension_integrity']['state_sha256_before']
    for path,digest in sources.items():
        assert sha(path)==digest, 'Source changed during evaluation: '+path
    artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='status.json'}
    result=dict(status='complete',model='MAVL',dataset=dataset,task='classification',rows=len(rows),
                class_names=classes,support_count=len(support_labels),branches=branches,
                eta=.3,is_smoke=bool(args.smoke),smoke=bool(args.smoke),native_parity=parity,
                frozen_descriptions_sha256=DESCRIPTIONS_SHA,artifact_sha256=artifacts,source_sha256=sources,
                checkpoint_sha256=sources[str((MODEL/'local_assets/checkpoint_full_46.pth').resolve())],
                extension_integrity=native.metadata['extension_integrity'],
                original_adapter_metadata=native.metadata,query_metadata=fit,
                no_gradient_training=True,test_labels_used_for_query_fit=False,
                protocol='Study-defined append-only missing disease query resources; frozen original head/weights; exact frozen support/test manifests; eta=.3; native norm preserved; four branches share context; center80 probability mean.',
                historical_unsupported_reference=support['baseline_references']['MAVL'],
                elapsed_seconds=time.time()-started,completed_at=now())
    write(output/'result.json',result)
    write(output/'status.json',dict(status='complete',done=len(rows),total=len(rows),updated_at=now()))
    print(json.dumps(dict(dataset=dataset,status='complete',branches={k:{m:v[m] for m in ('macro_auroc','macro_auprc')} for k,v in branches.items()})),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,required=True)
    parser.add_argument('--output',type=Path,default=HERE)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--smoke',type=int,default=0)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('6','7')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(1234);np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    # Match the historical worker: it disabled matmul TF32 only, leaving the
    # default cuDNN TF32 enabled. The diagnostic proves exact old-query parity.
    torch.backends.cudnn.allow_tf32=True
    assert torch.cuda.is_available()
    sources=common_sources()
    with torch.inference_mode():
        native=Adapter('cuda')
        check_path=args.output/f'historical_regression_gpu{os.environ["CUDA_VISIBLE_DEVICES"]}.json'
        historical=regression(native,check_path,args.batch_size)
        for dataset in args.datasets:
            evaluate(args,dataset,native,historical,sources)


if __name__=='__main__':
    main()
