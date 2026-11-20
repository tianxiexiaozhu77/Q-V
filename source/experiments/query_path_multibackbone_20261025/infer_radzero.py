#!/usr/bin/env python3
"""RadZero Original/R/Q, three fixed CheXchoNet tasks and five support seeds.

Only this worker's new output directory is written. GPU selection/locking is
the launcher's responsibility. All predictions use original-view FP32 native
logits, matching historical RadZero scoring; sigmoid values are saved apart.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "RadZero_i2cl/local_setup"
PRIOR = ROOT / "experiments/chexchonet_comparison_20261023_233926/radzero_paired"
SHARED = ROOT / "experiments/chexchonet_stability_20261024/infer_multiseed.py"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(LOCAL))


def shared_protocol_module():
    name = "radzero_multibackbone_shared_protocol"
    spec = importlib.util.spec_from_file_location(name, SHARED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared = shared_protocol_module()
TASKS, SEEDS = shared.TASKS, shared.SEEDS
write, sha = shared.write, shared.sha


def progress(args, stage, started, done=0, total=0):
    record = dict(status=stage, model="RadZero", done=done, total=total,
                  elapsed_seconds=time.time() - started, seeds=list(SEEDS), smoke=args.smoke)
    write(args.output / "status.json", record)
    print(json.dumps(record), flush=True)


def load_historical(rows, original_queries, prompts):
    """Check identity and queries before comparing any historical predictions."""
    import numpy as np
    with (PRIOR / "predictions_paired.csv").open(newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    lookup = {str(Path(r["image_path"]).resolve()): i for i, r in enumerate(old_rows)}
    if len(lookup) != len(old_rows):
        raise ValueError("Duplicate historical RadZero test images")
    ids = []
    for row in rows:
        i = lookup[row["image_path"]]
        reference = old_rows[i]
        if str(reference["patient_id"]) != row["patient_id"]:
            raise ValueError("Historical RadZero patient identity changed")
        for task in TASKS:
            if int(reference["label:" + task]) != row[task]:
                raise ValueError("Historical RadZero test label changed")
        ids.append(i)
    with np.load(PRIOR / "query_adapter.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["tasks"], TASKS)
        np.testing.assert_array_equal(saved["prompts"], prompts)
        np.testing.assert_allclose(saved["original_queries"], original_queries, atol=2e-6, rtol=1e-5)
        old_refined = saved["refined_queries"].copy()
    return {"indices": np.asarray(ids), "old_refined": old_refined,
            "original": np.load(PRIOR / "baseline_paired_logits.npy", allow_pickle=False)[ids],
            "query": np.load(PRIOR / "query_only_paired_logits.npy", allow_pickle=False)[ids]}


def run(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from infer_batch import TestImages, checkpoint_identity, load_model
    from extract_frozen_features import FrozenFeatureExtractor
    from overnight_query_refinement import QueryRefinementMatcher, fit_query_directions
    from radzero_intervention import (ARMS, VERSION, audit_native_equivalence,
        evaluate_interventions, match_intervention, prepare_vision, source_provenance)

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = shared.load_protocol(args.protocol)
    rows = protocol["test"][:args.smoke] if args.smoke else protocol["test"]
    if not rows:
        raise ValueError("Empty test evaluation")
    for seed in SEEDS:
        directory = args.output / f"seed_{seed}"
        directory.mkdir()
        write(directory / "support_manifest.json", dict(seed=seed,
            task_support_rows=protocol["supports"][seed], eta=.3,
            support_images_per_task=10, test_labels_used_for_fit=False,
            original_historical_seed42_is_not_this_seed42=True))
    source_paths = [Path(__file__).resolve(), Path(__file__).with_name("radzero_intervention.py"), SHARED]
    source_paths += [LOCAL / name for name in ("infer.py", "infer_batch.py",
                    "extract_frozen_features.py", "overnight_query_refinement.py")]
    source_paths += [PRIOR / name for name in ("predictions_paired.csv", "query_adapter.npz",
                    "baseline_paired_logits.npy", "query_only_paired_logits.npy")]
    prompt_path = ROOT / "AlphaRAD/local_setup/chexchonet_20261023/prompts.json"
    source_paths.append(prompt_path)
    protocol["sources"].update({str(p): sha(p) for p in source_paths})
    prompts = [json.loads(prompt_path.read_text())["tasks"][task]["prompt"] for task in TASKS]
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    device = torch.device("cuda")
    write(args.output / "input_audit.json", dict(protocol=str(protocol["path"]),
        protocol_sha256=sha(protocol["path"]), candidate_images=len(protocol["pool"]),
        candidate_patients=len({r["patient_id"] for r in protocol["pool"]}),
        fixed_test_images=7407, fixed_test_patients=2470, inference_images=len(rows),
        seeds=list(SEEDS), task_keys=list(TASKS), class_names=protocol["classes"],
        positive_supports=5, negative_supports=5, support_test_patient_overlap=0,
        support_negative_patient_rule="all candidate images for the task are negative"))
    progress(args, "loading", started)
    checkpoint = checkpoint_identity()
    model, processor, tokenizer = load_model(device)
    model = model.float().requires_grad_(False).eval()
    extractor = FrozenFeatureExtractor(model, tokenizer, prompts)
    original = extractor.text_features.detach().cpu().numpy().copy()
    historical = load_historical(rows, original, prompts)
    parity = {"support_native_api": None, "test_native_api": None, "paths": {}}

    def loader(selected):
        dataset = TestImages([r["image_path"] for r in selected], Path("/"), processor, False)
        options = dict(batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                       drop_last=False, pin_memory=True)
        if args.workers:
            options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        return DataLoader(dataset, **options)

    features = np.full((len(protocol["pool"]), len(TASKS), original.shape[-1]), np.nan, np.float32)
    progress(args, "encoding_support_pool", started, total=len(features))
    with torch.inference_mode():
        for pixels, _, indices in loader(protocol["pool"]):
            pixels = pixels.to(device, dtype=torch.float32, non_blocking=True)
            if parity["support_native_api"] is None:
                parity["support_native_api"] = extractor.verify_official_api(pixels)
            features[indices.numpy()] = extractor(pixels)["class_local"].cpu().numpy()
    if not np.isfinite(features).all():
        raise ValueError("Incomplete candidate features")
    np.save(args.output / "candidate_class_local.npy", features, allow_pickle=False)
    lookup = {r["image_path"]: i for i, r in enumerate(protocol["pool"])}
    queries = {}
    for seed in SEEDS:
        fits = []
        for j, task in enumerate(TASKS):
            supports = protocol["supports"][seed][task]
            ids = [lookup[r["image_path"]] for r in supports]
            fit = fit_query_directions(features[ids, j:j+1], [[r[task]] for r in supports],
                                       original[j:j+1], (0., .3))
            np.savez_compressed(args.output / f"seed_{seed}" / f"query_fit_{task}.npz",
                **fit, support_indices=np.asarray(ids),
                support_image_paths=np.asarray([r["image_path"] for r in supports]),
                support_patient_ids=np.asarray([r["patient_id"] for r in supports]))
            fits.append(fit["queries"][1])
        queries[seed] = np.concatenate(fits)
        np.save(args.output / f"seed_{seed}" / "queries.npy", queries[seed], allow_pickle=False)
    np.save(args.output / "original_queries.npy", original, allow_pickle=False)
    # All support fits are saved before inference or any classification scoring.
    arrays = {seed: {arm: np.full((len(rows), len(TASKS)), np.nan, np.float32)
                    for arm in ARMS} for seed in SEEDS}
    baseline_matcher = QueryRefinementMatcher(model, np.stack((original, queries[SEEDS[0]])), (0., .3))
    offset = 0
    with torch.inference_mode():
        for pixels, _, indices in loader(rows):
            ids = indices.numpy()
            if not np.array_equal(ids, np.arange(offset, offset + len(ids))):
                raise ValueError("Fixed test image order changed")
            pixels = pixels.to(device, dtype=torch.float32, non_blocking=True)
            aligned = model.forward_vision_model(pixels)["vision_tokens"]
            prepared = prepare_vision(model, aligned)
            if offset == 0:
                parity["test_native_api"] = extractor.verify_official_api(pixels)
                for seed in SEEDS:
                    parity["paths"][str(seed)] = {}
                    for batch_count in sorted({1, min(2, len(ids)), len(ids)}):
                        parity["paths"][str(seed)][str(batch_count)] = audit_native_equivalence(
                            model, prepared[:batch_count], original, queries[seed],
                            reference_matcher=baseline_matcher)
                old_q = match_intervention(model, prepared, historical["old_refined"],
                                           historical["old_refined"])["logits"].cpu().numpy()
                expected = historical["query"][ids]
                np.testing.assert_allclose(old_q, expected, atol=2e-5, rtol=1e-4)
                parity["historical_query_same_frozen_query"] = dict(passed=True,
                    images=len(ids), max_abs_error=float(np.max(np.abs(old_q - expected))),
                    reference=str(PRIOR / "query_only_paired_logits.npy"))
            for seed in SEEDS:
                outputs = evaluate_interventions(model, prepared, original, queries[seed])
                for arm in ARMS:
                    arrays[seed][arm][ids] = outputs[arm]["logits"].cpu().numpy()
            offset += len(ids)
            if offset == len(ids) or offset // args.batch_size % 25 == 0 or offset == len(rows):
                progress(args, "inference", started, offset, len(rows))
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("RadZero parameters must remain frozen")
    np.testing.assert_allclose(arrays[SEEDS[0]]["original"], historical["original"], atol=2e-5, rtol=1e-4)
    parity["historical_original"] = dict(passed=True, images=len(rows),
        max_abs_error=float(np.max(np.abs(arrays[SEEDS[0]]["original"] - historical["original"]))),
        reference=str(PRIOR / "baseline_paired_logits.npy"))
    write(args.output / "native_parity.json", parity)
    labels = np.asarray([[r[t] for t in TASKS] for r in rows], dtype=np.uint8)
    identity = dict(labels=labels, image_paths=np.asarray([r["image_path"] for r in rows]),
        patient_ids=np.asarray([r["patient_id"] for r in rows]), task_keys=np.asarray(TASKS),
        class_names=np.asarray(protocol["classes"]))
    metadata = dict(version=VERSION, checkpoint=checkpoint, prompts=prompts,
        score_domain="native_raw_logits", probability_export="separate float32 CPU sigmoid; not scored",
        eta=.3, views=["identity"], arms=list(ARMS), model_frozen=True,
        no_gradient_training=True, primary_comparison="query_minus_readout_only",
        support_features="normalized native original-query pooled contexts",
        native_matching=source_provenance(model), no_test_selection=True)
    status = "smoke_complete" if args.smoke else "complete"
    for seed in SEEDS:
        directory = args.output / f"seed_{seed}"
        for arm, values in arrays[seed].items():
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise ValueError("Incomplete/nonfinite predictions")
        np.testing.assert_array_equal(arrays[seed]["original"], arrays[SEEDS[0]]["original"])
        np.savez_compressed(directory / "logits.npz", **arrays[seed])
        np.savez_compressed(directory / "predictions.npz", **identity,
                            seed=np.asarray(seed), **arrays[seed])
        probabilities = {arm: torch.sigmoid(torch.from_numpy(value)).numpy()
                         for arm, value in arrays[seed].items()}
        np.savez_compressed(directory / "probabilities.npz", **identity,
                            seed=np.asarray(seed), **probabilities)
        write(directory / "result.json", dict(status=status, model="RadZero", seed=seed,
            rows=len(rows), metadata=metadata, prediction_sha256=sha(directory / "predictions.npz")))
    changed = [p for p, digest in protocol["sources"].items() if sha(p) != digest]
    if changed:
        raise ValueError("Source/protocol changed during inference: " + repr(changed))
    write(args.output / "source_sha256.json", protocol["sources"])
    write(args.output / "result.json", dict(status=status, model="RadZero", seeds=list(SEEDS),
        rows=len(rows), metadata=metadata, native_parity=parity, batch_size=args.batch_size,
        elapsed_seconds=time.time() - started,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 1024**2,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_version=torch.__version__))
    progress(args, status, started, len(rows), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("RadZero",), default="RadZero")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.smoke < 0:
        parser.error("Invalid batch size, workers or smoke")
    args.protocol, args.output = args.protocol.resolve(), args.output.resolve()
    try:
        run(args)
    except BaseException as exc:
        if args.output.is_dir():
            write(args.output / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
