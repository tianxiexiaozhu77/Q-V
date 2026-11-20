#!/usr/bin/env python3
"""Paired four-arm evaluation of frozen support query refinement and center TTA."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from evaluate_baseline_buffered import (sha,write,now,load_adapter,classification_metrics,
                                       spatial_batch,spatial_metrics)
from query_refinement_common import fuse_center_maps

ROOT=Path(__file__).resolve().parents[1]
BRANCHES=('original','query_refined','tta_center_original','tta_center_refined')

def load_query(name,native):
    path=ROOT/name/'local_setup/query_refinement_adapter.py'
    spec=importlib.util.spec_from_file_location('query_adapter_'+name.replace('-','_'),path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod.create_query_adapter(native)

def paired(adapter,method,paths,classes,view):
    pair=getattr(adapter,method+'_pair',None)
    if callable(pair):return pair(paths,classes,view=view)
    fn=getattr(adapter,method)
    return {k:fn(paths,classes,refined=refined,view=view)
            for k,refined in (('baseline',False),('refined',True))}

def check_native(a,b,spatial=False):
    aa=list(a) if spatial else [np.asarray(a)]
    bb=list(b) if spatial else [np.asarray(b)]
    maxerr=0.
    for x,y in zip(aa,bb):
        x,y=np.asarray(x),np.asarray(y)
        if x.shape!=y.shape or not np.allclose(x,y,rtol=1e-5,atol=2e-6,equal_nan=spatial):
            raise ValueError('Query wrapper baseline differs from original native adapter')
        diff=np.abs(x-y);valid=np.isfinite(diff)
        if valid.any():maxerr=max(maxerr,float(diff[valid].max()))
    return {'passed':True,'max_abs_error':maxerr,'rtol':1e-5,'atol':2e-6}

def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        view_weight = float(os.environ.get('QUERY_REFINEMENT_VIEW_WEIGHT', '0.5'))
    except ValueError as exc:
        raise ValueError('QUERY_REFINEMENT_VIEW_WEIGHT must be numeric') from exc
    if not np.isfinite(view_weight) or not 0.0 <= view_weight <= 1.0:
        raise ValueError('QUERY_REFINEMENT_VIEW_WEIGHT must lie in [0, 1]')
    if not torch.cuda.is_available():raise RuntimeError('Assigned CUDA GPU unavailable')
    catalog=json.loads(args.catalog.read_text())
    source_paths=[Path(__file__).resolve(),Path(__file__).with_name('query_refinement_common.py'),
      Path(__file__).with_name('evaluate_baseline_buffered.py'),
      ROOT/args.model/'local_setup/query_refinement_adapter.py',
      args.model_root/'local_setup/adapter.py',args.model_root/'local_setup/manifest.json',args.catalog]
    sources={str(p):sha(p) for p in source_paths}
    write(args.output/'source_sha256.json',sources)
    write(args.output/'status.json',dict(status='loading',model=args.model,started_at=now(),
          gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),smoke=args.smoke))
    native=load_adapter(args.model_root,'cuda')
    names=args.datasets or sorted(catalog['datasets'],key=lambda k:catalog['datasets'][k]['count'])
    outcomes={};totalstart=time.time()
    for name in names:
        entry=catalog['datasets'][name]
        manifest=Path(entry['manifest']);supportfile=Path(entry['support_manifest'])
        if sha(manifest)!=entry['sha256'] or sha(supportfile)!=entry['support_manifest_sha256']:
            raise ValueError('Fixed data manifest changed')
        data=json.loads(manifest.read_text());supports=json.loads(supportfile.read_text())
        # CheXpert's frozen support manifest predates the generic-baseline
        # campaign and has no saved prediction reference for these models.
        # In that case compute the native original arm in this run and retain
        # the explicit native-parity check; do not silently substitute a
        # different historical test result.
        reference=supports.get('baseline_references', {}).get(args.model)
        out=args.output/name;out.mkdir(exist_ok=False)
        start=time.time();task=data['task'];classes=data['class_names']
        rows=data['rows'][:args.smoke] if args.smoke else data['rows']
        try:
            if reference is not None and reference.get('status') != 'complete':
                reason=json.loads(Path(reference['result']).read_text()).get('reason','Prior protocol unsupported')
                raise NotImplementedError(reason)
            if hasattr(native,'set_dataset'):native.set_dataset(data)
            if hasattr(native,'check_classes'):native.check_classes(classes)
            if task!='classification' and not callable(getattr(native,'grounding',None)):
                raise NotImplementedError('No verified native spatial map')
            query=load_query(args.model,native)
            for path,expected in supports['support_sources'].items():
                if sha(path)!=expected:raise ValueError('Changed support source: '+path)
            for row in supports['support_rows']:
                if sha(row['image'])!=row['byte_sha256']:raise ValueError('Changed support image')
            write(out/'status.json',{'status':'fitting_support_queries','support_count':supports['support_count']})
            fit_kwargs=dict(batch_size=args.batch_size,task=task)
            if args.model == 'CARZero':
                # ``one_shot`` uses exactly the first support-induced update;
                # the adapter still performs its final native support pass so
                # the representation-gap diagnostic remains well defined.
                # ``support_gate`` runs a two-pass support-only probe first;
                # the evaluator decides whether to continue from q^(2) using
                # frozen thresholds and never inspects test outcomes.
                fit_kwargs.update(max_iter=(1 if args.adaptation_mode == 'one_shot'
                                            else 2 if args.adaptation_mode == 'support_gate'
                                            else args.fixed_point_max_iter),
                                  query_tol=args.fixed_point_query_tol,
                                  patience=args.fixed_point_patience)
            # Separate support adaptation from public-test inference.  The
            # legacy result only stored a single aggregate elapsed time,
            # which obscured the fact that test inference is paid by every
            # arm.  These outer timers are the authoritative end-to-end cost
            # fields for the CARZero audit.
            fit_start = time.time()
            fit=query.fit([r['image'] for r in supports['support_rows']],
                          [r['labels'] for r in supports['support_rows']],classes, **fit_kwargs)
            fit_elapsed_seconds = time.time() - fit_start
            if args.model == 'CARZero' and args.adaptation_mode == 'support_gate':
                # Preserve the q^(1)->q^(2) probe record before optionally
                # replacing it with the continuation fit.  Otherwise a
                # selected episode would lose the pre-gate timing and report
                # the post-decision q^(2)->q^(3) signal as if it made the
                # decision.
                probe_fit = fit
                probe_elapsed = float(probe_fit.get('fit_elapsed_seconds', 0.0))
                signals = probe_fit.get('support_gate_signals') or {}
                dm = signals.get('support_margin_delta_q2_minus_q1')
                kappa = signals.get('local_map_ratio')
                selected = (dm is not None and kappa is not None and
                            float(dm) > args.gate_gamma_m and
                            float(kappa) < args.gate_gamma_kappa)
                if selected:
                    continuation_fit = query.continue_fit(
                        [r['image'] for r in supports['support_rows']],
                        [r['labels'] for r in supports['support_rows']], classes,
                        batch_size=args.batch_size,
                        max_iter=args.fixed_point_max_iter,
                        query_tol=args.fixed_point_query_tol,
                        patience=args.fixed_point_patience)
                    fit = continuation_fit
                    fit['support_gate_signals'] = dict(signals)
                    fit['gate_probe_elapsed_seconds'] = probe_elapsed
                    fit['fit_elapsed_seconds'] = probe_elapsed + float(continuation_fit.get('fit_elapsed_seconds', 0.0))
                    fit_elapsed_seconds = time.time() - fit_start
                else:
                    # Keep q^(1) as the deployed query.  The probe's q^(2)
                    # remains recorded in query_trajectory for auditing, but
                    # is never used for test prediction when the gate rejects.
                    trajectory = fit.get('query_trajectory') or []
                    if len(trajectory) >= 2:
                        q_one = torch.as_tensor(trajectory[1], device=query.device,
                                                 dtype=torch.float32)
                        query.fits[tuple(classes)] = q_one
                        fit['refined_queries'] = trajectory[1]
                    fit['gate_probe_elapsed_seconds'] = probe_elapsed
                fit['support_gate_selected_full'] = bool(selected)
                fit['support_gate_gamma_M'] = float(args.gate_gamma_m)
                fit['support_gate_gamma_kappa'] = float(args.gate_gamma_kappa)
                fit['support_gate_test_labels_used'] = False
                fit['support_gate_selected_arm'] = 'full_fixed_point' if selected else 'one_shot'
                # Common audit schema used by AlphaRAD/RadZero collectors.
                # The decision is made after the q^(1)->q^(2) support probe
                # and before any test batch is touched.
                fit['gate_name'] = 'support_only_pre_gate_v1'
                fit['gate_selected_full'] = bool(selected)
                fit['gate_selected_full_rate'] = float(selected)
                fit['gate_thresholds'] = {'gamma_margin': float(args.gate_gamma_m),
                                          'gamma_kappa': float(args.gate_gamma_kappa)}
                fit['gate_delta_margin_mean'] = float(dm) if dm is not None else None
                fit['gate_local_map_ratio_mean'] = float(kappa) if kappa is not None else None
                fit['gate_direction_residual_l2_mean'] = None
                fit['gate_probe_native_passes'] = 3
                fit['gate_total_native_passes'] = int(3 + (1 if selected else 0))
                fit['gate_total_fit_elapsed_seconds'] = float(fit.get('fit_elapsed_seconds', probe_elapsed))
                fit['gate_test_labels_used'] = False
                fit['gate_pre_full_decision'] = True
                # Preserve the gate arm in the manifest; the selected arm is
                # explicit above and collector can report its coverage.
                fit['adaptation_mode'] = 'support_gate'
            # Keep the arm identity in both metadata files.  This lets the
            # collector pair a newly run CARZero one-shot evaluation with the
            # existing full fixed-point result using the same frozen support
            # manifest, seed, and test rows.
            fit['adaptation_mode'] = args.adaptation_mode
            fit['method'] = args.adaptation_mode
            write(out/'query_adapter.json',fit)
            write(out/'manifest.json',supports)
            if args.support_only:
                # Support-only gate calibration needs the native history but
                # does not need to rerun the large public test split.  The
                # archived full result is paired later by collect.py for
                # outcome reporting; this record deliberately contains no
                # test metrics and no test labels.
                write(out/'result.json', {
                    'status': 'complete', 'model': args.model, 'dataset': name,
                    'task': task, 'support_count': supports['support_count'],
                    'metadata': fit, 'adaptation_mode': args.adaptation_mode,
                    'method': args.adaptation_mode, 'test_labels_used_for_query_fit': False,
                    'support_only': True, 'seed': args.seed,
                    'fit_elapsed_seconds': float(fit_elapsed_seconds),
                    'inference_elapsed_seconds': 0.0,
                    'end_to_end_elapsed_seconds': float(time.time() - start),
                })
                write(out/'status.json', {'status': 'complete', 'support_only': True})
                outcomes[name] = 'complete'
                print('COMPLETE SUPPORT-ONLY', args.model, name, flush=True)
                del query, fit
                torch.cuda.empty_cache()
                continue
            arrays={k:[] for k in BRANCHES};items={k:[] for k in BRANCHES}
            inference_start = time.time()
            parity=None;last_update=0
            oldpred=None
            oldresult=None
            if reference is not None:
                if sha(reference['result'])!=reference['result_sha256']:raise ValueError('Baseline result changed')
                oldresult=json.loads(Path(reference['result']).read_text())
            if task=='classification' and reference is not None:
                oldpred=np.load(Path(reference['result']).parent/'predictions.npz',allow_pickle=False)
                if list(oldpred['class_names'])!=classes:raise ValueError('Baseline class order changed')
            for offset in range(0,len(rows),args.batch_size):
                batch=rows[offset:offset+args.batch_size];paths=[r['image'] for r in batch]
                method='predict' if task=='classification' else 'grounding'
                with torch.inference_mode():
                    identity=paired(query,method,paths,classes,'identity')
                    if offset==0:
                        expected=getattr(native,method)(paths,classes)
                        parity=check_native(identity['baseline'],expected,task!='classification')
                        write(out/'native_parity.json',parity)
                        del expected
                    center=paired(query,method,paths,classes,'center80')
                outputs={'original':identity['baseline'],'query_refined':identity['refined']}
                if task=='classification':
                    if oldpred is not None and paths!=list(oldpred['image_paths'][offset:offset+len(batch)]):raise ValueError('Baseline image order differs')
                    if oldpred is not None and not np.allclose(identity['baseline'],oldpred['scores'][offset:offset+len(batch)],atol=2e-4,rtol=2e-4):
                        if not args.allow_baseline_parity_drift:
                            raise ValueError('Saved baseline prediction parity failed')
                        # A small number of legacy Chest14 references contain
                        # a final-batch numerical drift after image decoding.
                        # Keep the run auditable when explicitly requested:
                        # report the mismatch and continue evaluating the
                        # gate, rather than silently substituting predictions.
                        parity = dict(parity or {}, saved_baseline_reference_match=False,
                                      saved_baseline_reference_drift_max=float(np.max(np.abs(
                                          np.asarray(identity['baseline']) - oldpred['scores'][offset:offset+len(batch)]))))
                    for ref,key in (('baseline','tta_center_original'),('refined','tta_center_refined')):
                        outputs[key]=(1.0-view_weight)*np.asarray(identity[ref])+view_weight*np.asarray(center[ref])
                    for key,v in outputs.items():
                        v=np.asarray(v)
                        if v.shape!=(len(batch),len(classes)) or not np.isfinite(v).all():raise ValueError('Invalid prediction')
                        arrays[key].append(v)
                else:
                    inputsize=448 if args.model=='BioViL-T' else 224
                    for ref,key in (('baseline','tta_center_original'),('refined','tta_center_refined')):
                        outputs[key]=fuse_center_maps(identity[ref],center[ref],inputsize)
                    for key,v in outputs.items():items[key].extend(spatial_batch(v,batch,classes,task))
                del outputs,center,identity
                if time.time()-last_update>15 or offset+len(batch)==len(rows):
                    progress=dict(status='running',model=args.model,dataset=name,done=offset+len(batch),
                                  total=len(rows),elapsed_seconds=time.time()-start,updated_at=now())
                    write(out/'status.json',progress);write(args.output/'status.json',progress)
                    print(f'{args.model}/{name} {progress["done"]}/{len(rows)}',flush=True);last_update=time.time()
            inference_elapsed_seconds = time.time() - inference_start
            end_to_end_elapsed_seconds = time.time() - start
            branches={}
            if task=='classification':
                labels=np.asarray([r['labels'] for r in rows],dtype=np.uint8)
                saved={'labels':labels,'image_paths':np.asarray([r['image'] for r in rows]),'class_names':np.asarray(classes)}
                for key,parts in arrays.items():
                    values=np.concatenate(parts);saved[key]=values
                    branches[key]=classification_metrics(labels,values,classes)
                np.savez_compressed(out/'predictions.npz',**saved)
                if oldpred is not None:
                    oldpred.close()
            else:
                for key,v in items.items():branches[key]=spatial_metrics(v,classes,task)
                write(out/'spatial_predictions.json',{'branches':items,'image_paths':[r['image'] for r in rows],
                                                    'class_names':classes})
                if not args.smoke and oldresult is not None:
                    metric='pointing_game' if task=='grounding' else 'dice'
                    if abs(branches['original'][metric]-oldresult['metrics'][metric])>1e-6:
                        raise ValueError('Saved original spatial metric parity failed')
            metric_names=('macro_auroc','macro_auprc') if task=='classification' else (('pointing_game',) if task=='grounding' else ('dice',))
            delta={k:(100*(branches['tta_center_refined'][k]-branches['original'][k])
                      if branches['original'].get(k) is not None else None) for k in metric_names}
            fixed_delta={k:(100*(branches['query_refined'][k]-branches['original'][k])
                            if branches['original'].get(k) is not None else None) for k in metric_names}
            result=dict(status='complete',model=args.model,dataset=name,task=task,rows=len(rows),
                smoke=bool(args.smoke),eta=float(query.metadata.get('eta', os.environ.get('QUERY_REFINEMENT_ETA', '.3'))),
                crop_ratio=float(os.environ.get('QUERY_REFINEMENT_CROP_RATIO', '.8')),
                view_weight=view_weight,support_count=supports['support_count'],class_names=classes,
                branches=branches,delta_pp=delta,fixed_point_delta_pp=fixed_delta,
                native_parity=parity,metadata=query.metadata,
                original_adapter_metadata=native.metadata,elapsed_seconds=end_to_end_elapsed_seconds,
                adaptation_metadata={k: fit.get(k) for k in (
                    'fit_elapsed_seconds', 'wall_clock_seconds',
                    'native_reaggregation_passes', 'estimated_flops',
                    'support_gate_signals', 'gate_probe_elapsed_seconds',
                    'gate_total_fit_elapsed_seconds', 'gate_total_native_passes',
                    'gate_selected_full', 'gate_pre_full_decision',
                    'gate_test_labels_used', 'adaptation_mode', 'method')
                    if isinstance(fit, dict) and k in fit},
                gate_selected_full=(fit.get('gate_selected_full') if isinstance(fit, dict) else None),
                gate_pre_full_decision=(fit.get('gate_pre_full_decision') if isinstance(fit, dict) else None),
                gate_test_labels_used=(fit.get('gate_test_labels_used') if isinstance(fit, dict) else False),
                fit_elapsed_seconds=float(fit_elapsed_seconds),
                inference_elapsed_seconds=float(inference_elapsed_seconds),
                end_to_end_elapsed_seconds=float(end_to_end_elapsed_seconds),
                test_labels_used_for_query_fit=False,no_gradient_training=True,
                protocol='Sensitivity-grid run; original-view supports; positive query direction only; native norm preserved; native score fusion weight recorded; center crop ratio and geometry recorded explicitly.',
                baseline_reference=reference,source_sha256=sources,seed=args.seed,
                fixed_point_max_iter=((1 if args.adaptation_mode == 'one_shot' else args.fixed_point_max_iter)
                                      if args.model == 'CARZero' else 1),
                adaptation_mode=args.adaptation_mode, method=args.adaptation_mode)
            write(out/'result.json',result);write(out/'status.json',{'status':'complete','elapsed_seconds':time.time()-start})
            outcomes[name]='complete'
            print('COMPLETE',args.model,name,json.dumps(delta),flush=True)
            del query,fit,arrays,items
            torch.cuda.empty_cache()
        except NotImplementedError as e:
            write(out/'result.json',dict(status='unsupported',model=args.model,dataset=name,task=task,
                                        reason=str(e),rows=len(rows)))
            outcomes[name]='unsupported';print('UNSUPPORTED',name,str(e),flush=True)
        except Exception as e:
            write(out/'result.json',dict(status='failed',model=args.model,dataset=name,task=task,
                  error=repr(e),traceback=traceback.format_exc(),rows=len(rows)))
            outcomes[name]='failed';traceback.print_exc()
            # Continue independent datasets; controller never promotes a failed smoke run.
    changed={p for p,h in sources.items() if sha(p)!=h}
    status='failed' if 'failed' in outcomes.values() or changed else 'complete'
    write(args.output/'status.json',dict(status=status,model=args.model,outcomes=outcomes,
          elapsed_seconds=time.time()-totalstart,changed_sources=sorted(changed),finished_at=now()))
    if status!='complete':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--model-root',required=True,type=Path);p.add_argument('--catalog',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--datasets',nargs='+')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--smoke',type=int,default=0)
    p.add_argument('--support-only',action='store_true',
                   help='Fit and serialize support history without public-test inference.')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--fixed-point-max-iter',type=int,default=3)
    p.add_argument('--fixed-point-query-tol',type=float,default=1e-4)
    p.add_argument('--fixed-point-patience',type=int,default=1)
    p.add_argument('--adaptation-mode',choices=('full_fixed_point','one_shot','support_gate'),
                   default='full_fixed_point',
                   help='CARZero arm identity; one_shot forces max_iter=1')
    p.add_argument('--allow-baseline-parity-drift', action='store_true',
                   help='Continue after an explicitly recorded legacy saved-baseline mismatch.')
    p.add_argument('--gate-gamma-m',type=float,default=0.0,
                   help='Frozen support-only threshold for M_S(q2)-M_S(q1).')
    p.add_argument('--gate-gamma-kappa',type=float,default=1.0,
                   help='Frozen support-only threshold for the local map ratio.')
    run(p.parse_args())
