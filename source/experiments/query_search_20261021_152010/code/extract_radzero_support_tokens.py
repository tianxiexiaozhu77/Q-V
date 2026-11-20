#!/usr/bin/env python3
"""Fresh support tokens, audited against the accepted frozen feature cache."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE.parent / 'code_phase1'))
from eval_radzero_candidates import load_inputs, sha256, write_json, now, DATASETS


def run(args):
    import numpy as np
    import torch
    from infer_batch import load_model, CachedInference
    from extract_frozen_features import image_loader, pool_original_matching
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, processor, tokenizer = load_model(torch.device('cuda'))
    model.requires_grad_(False).eval()
    summaries = {}
    with torch.inference_mode():
        for name in DATASETS:
            inputs = load_inputs(SimpleNamespace(dataset=name, deadline=args.deadline, baseline_run=None))
            cached = inputs['cached']; support = inputs['support']
            helper = CachedInference(model, tokenizer, list(inputs['adapter'].prompts))
            np.testing.assert_array_equal(helper.features.cpu().numpy(), cached['text_features'])
            directory = args.output / name; directory.mkdir()
            token_values, local_values, global_values, logits, seen = [], [], [], [], []
            loader_args = SimpleNamespace(batch_size=16, workers=2, device='cuda', dataset=name)
            for pixels, _, indices in image_loader(support, processor, loader_args):
                if time.time() >= args.deadline:
                    raise TimeoutError('Campaign deadline')
                aligned = model.forward_vision_model(pixels.to('cuda', non_blocking=True))['vision_tokens']
                local, scores, _ = pool_original_matching(model.loss_fns['KeyPhraseAlignmentLoss'], helper.features, aligned)
                token_values.append(aligned.cpu().numpy()); local_values.append(local.cpu().numpy())
                global_values.append(aligned[:, 1:].mean(1).cpu().numpy()); logits.append(scores.cpu().numpy()); seen.extend(indices.tolist())
            assert seen == list(range(len(support['image_paths'])))
            local = np.concatenate(local_values); scores = np.concatenate(logits); global_ = np.concatenate(global_values)
            np.testing.assert_allclose(local, cached['support_class_local'], atol=1e-6, rtol=0)
            old_scores = np.load(inputs['cache'] / 'support_base_logits.npy', allow_pickle=False)
            np.testing.assert_allclose(scores, old_scores, atol=1e-6, rtol=0)
            old_global = np.load(inputs['cache'] / 'support_global_aligned.npy', allow_pickle=False)[:, 1]
            np.testing.assert_allclose(global_, old_global, atol=1e-6, rtol=0)
            np.save(directory / 'vision_tokens.npy', np.concatenate(token_values), allow_pickle=False)
            np.save(directory / 'class_local.npy', local, allow_pickle=False)
            np.save(directory / 'global_features.npy', global_, allow_pickle=False)
            record = {'status':'complete','model':'radzero','dataset':name,'split':'support_only','num_heads':1,
                'image_paths':support['image_paths'],'groups':support['groups'],'labels':support['labels'].tolist(),
                'image_file_sha256':[sha256(Path(support['data_root']) / path) for path in support['image_paths']],
                'checkpoint': inputs['adapter'].metadata['checkpoint'], 'cache_manifest_sha256':sha256(inputs['cache']/'manifest.json'),
                'reference_check':{'passed':True,'class_local_max_error':float(np.max(np.abs(local-cached['support_class_local']))),
                    'logit_max_error':float(np.max(np.abs(scores-old_scores))), 'global_max_error':float(np.max(np.abs(global_-old_global)))},
                'batch_size':16,'dtype':'float32','shape':list(np.concatenate(token_values).shape),
                'vision_tokens_sha256':sha256(directory/'vision_tokens.npy'),
                'class_local_sha256':sha256(directory/'class_local.npy'),
                'global_features_sha256':sha256(directory/'global_features.npy'),
                'source_sha256':{str(Path(__file__)):sha256(__file__)},'completed_at':now()}
            write_json(directory/'manifest.json',record); summaries[name]=record['reference_check']; print(name,summaries[name],flush=True)
    write_json(args.output/'result.json',{'status':'complete','model':'radzero','dataset':'all_support','is_smoke':False,
        'datasets':summaries,'no_test_labels_consumed':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args()
    if not time.time()<args.deadline<=1792634400:
        parser.error('Invalid campaign deadline')
    args.output.mkdir(parents=True,exist_ok=False)
    def stop(signum,_frame):
        raise InterruptedError(f'Signal {signum}')
    for signum in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM):signal.signal(signum,stop)
    signal.setitimer(signal.ITIMER_REAL,args.deadline-time.time())
    try:run(args)
    except BaseException as error:
        write_json(args.output/'result.json',{'status':'failed','model':'radzero','dataset':'all_support','is_smoke':False,
            'error':repr(error),'traceback':traceback.format_exc()});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
