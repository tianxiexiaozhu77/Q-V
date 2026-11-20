#!/usr/bin/env python3
"""AlphaRAD single-view Original/R/Q on frozen patient-separated CheXchoNet."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
STABILITY = MEDICAL / "experiments/chexchonet_stability_20261024"
INTERVENTION = MEDICAL / "experiments/query_path_intervention_20261024"
sys.path[:0] = [str(STABILITY), str(INTERVENTION)]
from infer_multiseed import load_protocol, sha, write, progress, ALPHA, BASELINES, PHASE6, TASKS, SEEDS, OLD_ALPHA
BRANCHES = ("original", "readout_only", "query")

def alpha_inference(args, protocol, rows, started):
    import numpy as np
    import torch
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from eval_query_refinement import CHECKPOINT_SHA, original_helper_check
    from eval_chexchonet import native_manifest
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator, fit_query_directions
    from intervention import evaluate_interventions, audit_native_equivalence

    seed_everything(1234)
    checkpoint_path = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
    if sha(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Published AlphaRAD checkpoint changed")
    source_paths = [ALPHA / "local_setup" / name for name in
        ("query_refinement.py", "eval_query_refinement.py", "eval_chexchonet.py")]
    source_paths += [PHASE6 / "view_geometry.py", ALPHA / "cxrclip/model/cxrclip_plus_dqn.py",
                     ALPHA / "cxrclip/evaluator.py", ALPHA / "cxrclip/base_evaluator.py"]
    protocol["sources"].update({str(p): sha(p) for p in source_paths})
    native_manifest(args.output / "pool_inputs.csv", protocol["pool"], TASKS)
    native_manifest(args.output / "test_inputs.csv", rows, TASKS)
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(checkpoint_path)]},
        "data_test": {name: {"name": name, "data_type": "image_classification",
            "data_path": str(args.output / f"{name}_inputs.csv"), "n_class": 3,
            "normalize": None} for name in ("pool", "test")}, "dataloader": {"test": options}}
    write(args.output / "eval_config.json", config)
    progress(args, "loading", started)
    evaluator = Evaluator(config, [str(checkpoint_path)], "test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    classes = [name.lower() for name in protocol["classes"]]
    original, prompts = original_queries_from_evaluator(evaluator, classes)
    prompt_path = ALPHA / "local_setup/chexchonet_20261023/prompts.json"
    prompt_record = json.loads(prompt_path.read_text())
    if prompts != [prompt_record["tasks"][task]["prompt"] for task in TASKS]:
        raise ValueError("Frozen CheXchoNet prompts changed")
    protocol["sources"][str(prompt_path)] = sha(prompt_path)
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    parts, observed = [], []
    progress(args, "encoding_support_pool", started, total=len(protocol["pool"]))
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["pool"]:
            vision = matcher.encode_vision(batch["images"])
            parts.append(matcher.match_prepared(vision, original, return_features=True)["class_local"].cpu().numpy())
            observed.extend(batch["image_paths"])
    if observed != [r["image_path"] for r in protocol["pool"]]:
        raise ValueError("Candidate pool order changed")
    features = np.concatenate(parts)
    np.save(args.output / "candidate_class_local.npy", features, allow_pickle=False)
    lookup = {r["image_path"]: i for i, r in enumerate(protocol["pool"])}
    queries = {}
    for seed in SEEDS:
        fits = []
        for j, task in enumerate(TASKS):
            supports = protocol["supports"][seed][task]
            ids = [lookup[r["image_path"]] for r in supports]
            fit = fit_query_directions(features[ids, j:j + 1], [[r[task]] for r in supports],
                                       original[j:j + 1], etas=(0., .3))
            np.savez_compressed(args.output / f"seed_{seed}" / f"query_fit_{task}.npz",
                                **fit, support_indices=np.asarray(ids))
            fits.append(fit["queries"][1])
        queries[seed] = np.concatenate(fits)
        np.save(args.output / f"seed_{seed}" / "queries.npy", queries[seed], allow_pickle=False)
    arrays = {seed: {} for seed in SEEDS}
    logits_out = {seed: {key: [] for key in BRANCHES} for seed in SEEDS}
    offset, parity = 0, {}
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["test"]:
            stop = offset + len(batch["images"])
            if batch["image_paths"] != [r["image_path"] for r in rows[offset:stop]]:
                raise ValueError("Test image order changed")
            identity = matcher.encode_vision(batch["images"])
            if offset == 0:
                parity["author_helpers"] = original_helper_check(evaluator, matcher, batch, classes)
            for seed in SEEDS:
                outputs = evaluate_interventions(evaluator.model, identity, original, queries[seed])
                if offset == 0:
                    parity[f"native_seed_{seed}"] = audit_native_equivalence(
                        evaluator.model, identity, original, queries[seed])
                    expected = matcher.match_prepared(identity, queries[seed])["logits"]
                    torch.testing.assert_close(outputs["query"]["logits"], expected, atol=2e-6, rtol=2e-5)
                for branch in BRANCHES:
                    logits_out[seed][branch].append(outputs[branch]["logits"].cpu().numpy())
            if offset == 0 or offset // args.batch_size % 25 == 0 or stop == len(rows):
                progress(args, "inference", started, stop, len(rows))
            offset = stop
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("AlphaRAD parameters must remain frozen")
    for seed in SEEDS:
        joined_logits = {key: np.concatenate(value) for key, value in logits_out[seed].items()}
        # Historical eval_chexchonet applies sigmoid on the full CPU logit
        # matrix. Retain that operation order, including floating-point ties.
        arrays[seed] = {key: torch.sigmoid(torch.from_numpy(value)).numpy()
                        for key, value in joined_logits.items()}
        np.savez_compressed(args.output / f"seed_{seed}" / "logits.npz",
                            **joined_logits)
    return arrays, parity, dict(checkpoint_sha256=CHECKPOINT_SHA, ignored_checkpoint_keys=ignored,
        prompts=prompts, classification_aggregation="single native logit then sigmoid", control="readout_only", primary_contrast="query - readout_only",
        support_feature="native original-query class-local FLaS context", no_gradient_training=True)


def historical_parity(rows, arrays):
    import numpy as np
    old_paths = json.loads((OLD_ALPHA / "image_paths.json").read_text())
    lookup = {str(Path(p).resolve()): i for i, p in enumerate(old_paths)}
    ids = [lookup[r["image_path"]] for r in rows]
    expected = np.load(OLD_ALPHA / "original_probabilities.npy", allow_pickle=False)[ids]
    np.testing.assert_allclose(arrays[42]["original"], expected, atol=2e-6, rtol=1e-5)
    report = {"historical_original": dict(passed=True, max_abs_error=float(np.max(np.abs(arrays[42]["original"] - expected))))}
    prior = STABILITY / "runs/full_20261024_v1/AlphaRAD"
    for seed in SEEDS:
        path = prior / f"seed_{seed}/predictions.npz"
        with np.load(path, allow_pickle=False) as old:
            positions = {str(Path(p).resolve()): i for i,p in enumerate(old["image_paths"].tolist())}
            selected = [positions[r["image_path"]] for r in rows]
            np.testing.assert_array_equal(old["labels"][selected], [[r[t] for t in TASKS] for r in rows])
            for branch, old_branch in (("original", "original"), ("query", "query_refined")):
                expected = old[old_branch][selected]
                np.testing.assert_allclose(arrays[seed][branch], expected, atol=2e-6, rtol=1e-5)
                report[f"stability_{seed}_{branch}"] = dict(passed=True, max_abs_error=float(np.max(np.abs(arrays[seed][branch] - expected))), reference=str(path), reference_sha256=sha(path))
    return report


def run(args):
    import numpy as np
    import torch
    started=time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol=load_protocol(args.protocol)
    exp=protocol["master"]["experiment"]
    # Inherited protocol.views describe historical assets; this experiment is single view.
    if exp["views"] != ["identity"]:
        raise ValueError("This worker implements single-view inference only")
    protocol["sources"].update({str(p):sha(p) for p in (Path(__file__).resolve(), INTERVENTION/"intervention.py", STABILITY/"infer_multiseed.py")})
    rows=protocol["test"][:args.smoke] if args.smoke else protocol["test"]
    for seed in SEEDS:
        folder=args.output/f"seed_{seed}"
        folder.mkdir()
        write(folder/"support_manifest.json", dict(seed=seed, task_support_rows=protocol["supports"][seed], eta=.3, support_images_per_task=10, test_labels_used_for_fit=False))
    write(args.output/"input_audit.json", dict(protocol=str(protocol["path"]), protocol_sha256=sha(protocol["path"]), test_images=len(rows), support_test_patient_overlap=0, task_keys=list(TASKS), class_names=protocol["classes"], seeds=list(SEEDS), views=["identity"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA unavailable")
    arrays,parity,metadata=alpha_inference(args,protocol,rows,started)
    parity["historical"] = historical_parity(rows,arrays)
    labels=np.asarray([[r[t] for t in TASKS] for r in rows],dtype=np.uint8)
    for seed in SEEDS:
        for branch,values in arrays[seed].items():
            if values.shape != (len(rows),3) or not np.isfinite(values).all():
                raise ValueError("Incomplete predictions")
        np.testing.assert_array_equal(arrays[seed]["original"], arrays[42]["original"])
        np.savez_compressed(args.output/f"seed_{seed}/predictions.npz", labels=labels,
            image_paths=np.asarray([r["image_path"] for r in rows]), patient_ids=np.asarray([r["patient_id"] for r in rows]),
            task_keys=np.asarray(TASKS), class_names=np.asarray(protocol["classes"]), seed=np.asarray(seed), **arrays[seed])
    changed=[p for p,digest in protocol["sources"].items() if sha(p)!=digest]
    if changed:
        raise ValueError("Source changed during inference: "+repr(changed))
    write(args.output/"source_sha256.json",protocol["sources"])
    write(args.output/"native_parity.json",parity)
    status="smoke_complete" if args.smoke else "complete"
    write(args.output/"result.json",dict(status=status,model=args.model,seeds=list(SEEDS),rows=len(rows),native_parity=parity,metadata=metadata,elapsed_seconds=time.time()-started,peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2,gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))
    progress(args,status,started,len(rows),len(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=["AlphaRAD"],default="AlphaRAD")
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--smoke",type=int,default=0)
    args=p.parse_args()
    if args.batch_size<1 or args.workers<0 or args.smoke<0:
        p.error("Invalid inference size")
    args.output=args.output.resolve()
    try:
        run(args)
    except BaseException as exc:
        if args.output.is_dir():
            write(args.output/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=="__main__":
    main()
