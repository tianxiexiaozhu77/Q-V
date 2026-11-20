#!/usr/bin/env python3
"""Frozen AlphaRAD 2x2 attention-query/readout-query intervention.

The two certified development cohorts are mechanism diagnostics, not an
independent validation cohort. Test labels and boxes are not read by this worker.
"""
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
MULTISEED = MEDICAL / "experiments/fewshot_multiseed_20261024"
ALPHA = MEDICAL / "AlphaRAD"
PHASE6 = MEDICAL / "experiments/query_search_20261021_152010/code_phase6"
sys.path[:0] = [str(HERE), str(MULTISEED)]
from cache_multiseed import (load_job, json_spec, read, checked, sha, write_json,
    CHECKPOINT, CHECKPOINT_SHA, SEEDS, parity)

BRANCHES = ("original", "readout_only", "aggregation_only", "query")


def progress(args, stage, started, **extra):
    record = dict(status=stage, dataset=args.dataset, model="AlphaRAD",
        elapsed_seconds=time.time()-started, smoke=args.smoke, **extra)
    write_json(args.output / "status.json", record)
    print(json.dumps(record), flush=True)


def run(args):
    import numpy as np
    import torch
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from query_refinement import (QueryRefinementMatcher,
        original_queries_from_evaluator, fit_query_directions)
    from vision_cache import VisionTokenCache, image_fingerprints
    from cache_alpharad import write_images
    from intervention import evaluate_interventions, audit_native_equivalence

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    progress(args, "authenticating_inputs", started)
    job = load_job(args)
    write_json(args.output / "source_sha256.json", job["sources"])
    write_json(args.output / "support_union.json", job["union"])
    write_json(args.output / "reuse_audit.json", dict(
        reuse_manifest=str(args.reuse_manifest), reuse_sha256=sha(args.reuse_manifest),
        sources=job["sources"], test_labels_read=False, test_boxes_read=False,
        cohort_role="development_cohort_mechanism_diagnostic"))
    for seed in SEEDS:
        (args.output / f"seed_{seed}").mkdir()
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    for name, rows in (("support", job["union"]), ("live_test", job["test"][:args.batch_size])):
        write_images(args.output / f"{name}_inputs.csv", rows)
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(CHECKPOINT)]},
        "data_test": {name: dict(name=name, data_type="image_classification",
            data_path=str(args.output / f"{name}_inputs.csv"),
            n_class=len(job["classes"]), normalize=None) for name in ("support", "live_test")},
        "dataloader": {"test": options}}
    write_json(args.output / "eval_config.json", config)
    progress(args, "loading_model", started)
    evaluator = Evaluator(config, [str(CHECKPOINT)], "test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ignored_keys = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    original, prompts = original_queries_from_evaluator(evaluator, job["classes"])
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    checks = {"text": parity(original, job["text"]["positive_raw"], "native positive text")}
    support_parts, observed = [], []
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["support"]:
            prepared = matcher.encode_vision(batch["images"])
            value = matcher.match_prepared(prepared, original, return_features=True)
            support_parts.append(value["class_local"].cpu().numpy())
            observed.extend(batch["image_paths"])
    if observed != [r["image"] for r in job["union"]]:
        raise ValueError("Support loader reordered images")
    support_features = np.concatenate(support_parts)
    checks["support_features"] = parity(support_features,
        np.load(job["stable"] / "support_union_features.npy", allow_pickle=False),
        "new support encoding against authenticated cache")
    np.save(args.output / "support_union_features.npy", support_features, allow_pickle=False)
    lookup = {r["image"]: i for i, r in enumerate(job["union"])}
    queries, checks["query_fit"] = {}, {}
    for seed in SEEDS:
        rows = job["supports"][seed]
        indices = np.asarray([lookup[r["image"]] for r in rows])
        fit = fit_query_directions(support_features[indices], [r["labels"] for r in rows],
                                   original, etas=(0., .3))
        with np.load(job["stable"] / f"seed_{seed}/query_fit.npz", allow_pickle=False) as old:
            np.testing.assert_array_equal(indices, old["support_union_indices"])
            checks["query_fit"][str(seed)] = parity(fit["queries"], old["queries"], "recomputed support query")
        queries[seed] = torch.as_tensor(fit["queries"][1], device=matcher.device)
        np.savez_compressed(args.output / f"seed_{seed}/query_fit.npz", **fit,
                            support_union_indices=indices)
        write_json(args.output / f"seed_{seed}/support_manifest.json", dict(
            support_seed=seed, support_rows=rows, eta=.3, test_labels_used=False))
    progress(args, "support_fitted", started, support_union_count=len(observed))

    manifest_path = json_spec(job["reuse"]["vision_manifest"], args.reuse_manifest.parent)
    manifest = read(manifest_path)
    identity = manifest["identity"]
    paths_all = [r["image"] for r in job["test_all"]]
    if not (manifest["status"] == "complete" and manifest["image_ids"] == paths_all
            and identity["checkpoint_sha256"] == CHECKPOINT_SHA
            and identity["use_all_image_tokens"] == evaluator.use_all_image_tokens
            and identity["num_heads"] == evaluator.model.num_radzero_heads
            and identity["input_file_fingerprints"] == image_fingerprints(paths_all)):
        raise ValueError("Certified visual token identities changed")
    cache = VisionTokenCache(manifest_path.parent.parent, identity, paths_all, mode="read", prefetch=1)
    predictions = {s: {b: [] for b in BRANCHES} for s in SEEDS}
    maps = {s: {b: [] for b in BRANCHES} for s in SEEDS}
    checks["native_formula"], checks["historical_predictions"] = {}, {}
    original_tensor = torch.as_tensor(original, device=matcher.device)
    n = len(job["test"])
    includes_cls = bool(evaluator.model.radzero_head_enable_cross_attend_cls_token)
    grid = None
    try:
        with torch.inference_mode():
            for offset in range(0, n, args.batch_size):
                stop = min(offset + args.batch_size, n)
                tokens = torch.as_tensor(np.array(cache.get(offset, stop), copy=True),
                                         device=matcher.device, dtype=torch.float32)
                vision = dict(image_cls_raw_features=[tokens[:, 0]], image_patch_raw_features=[tokens[:, 1:]])
                if offset == 0:
                    fresh_batch = next(iter(evaluator.data_loader_dict["live_test"]))
                    if fresh_batch["image_paths"] != paths_all[:stop]:
                        raise ValueError("Live cache audit reordered images")
                    fresh = matcher.encode_vision(fresh_batch["images"])
                    checks["fresh_tokens"] = {}
                    for key in vision:
                        torch.testing.assert_close(vision[key][0], fresh[key][0], atol=2e-5, rtol=1e-5)
                        checks["fresh_tokens"][key] = float((vision[key][0]-fresh[key][0]).abs().max())
                for seed in SEEDS:
                    values = evaluate_interventions(evaluator.model, vision, original_tensor, queries[seed])
                    if offset == 0:
                        checks["native_formula"][str(seed)] = audit_native_equivalence(
                            evaluator.model, vision, original_tensor, queries[seed])
                    seed_checks = checks["historical_predictions"].setdefault(str(seed), {})
                    for branch in BRANCHES:
                        p = torch.sigmoid(values[branch]["logits"]).cpu().numpy()
                        predictions[seed][branch].append(p)
                        if branch in ("original", "query"):
                            check = parity(p, job["predictions"][seed][branch][offset:stop],
                                           f"all-image unchanged {branch}")
                            seed_checks.setdefault(branch, []).append(check)
                        if args.dataset == "chestxdet10":
                            m = values[branch]["grounding_map"].cpu().numpy()
                            grid = int(round((m.shape[-1] - int(includes_cls)) ** .5))
                            if grid * grid != m.shape[-1] - int(includes_cls):
                                raise ValueError("Spatial tokens do not form a square grid")
                            maps[seed][branch].append(m)
                    # Structural equalities, asserted for every image, not inferred from performance.
                    torch.testing.assert_close(values["original"]["grounding_map"],
                        values["readout_only"]["grounding_map"], atol=0, rtol=0)
                    torch.testing.assert_close(values["aggregation_only"]["grounding_map"],
                        values["query"]["grounding_map"], atol=0, rtol=0)
                if offset == 0 or stop == n or (offset // args.batch_size) % 25 == 0:
                    progress(args, "matching_interventions", started, done=stop, total=n)
    finally:
        cache.close()
    meta = dict(image_paths=np.asarray([r["image"] for r in job["test"]]),
        group_ids=np.asarray([r["group"] for r in job["test"]]), class_names=np.asarray(job["classes"]))
    result_paths = {}
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        values = {b: np.concatenate(predictions[seed][b]) for b in BRANCHES}
        if any(v.shape != (n, len(job["classes"])) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Malformed predictions")
        np.savez_compressed(folder / "predictions.npz", **meta, **values)
        artifacts = ["predictions.npz", "query_fit.npz", "support_manifest.json"]
        if args.dataset == "chestxdet10":
            np.savez_compressed(folder / "native_maps.npz", **meta,
                includes_cls=np.asarray(includes_cls), grid_size=np.asarray(grid),
                **{b: np.concatenate(maps[seed][b]) for b in BRANCHES})
            artifacts.append("native_maps.npz")
        result_paths[str(seed)] = {name: sha(folder / name) for name in artifacts}
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Frozen backbone became trainable")
    write_json(args.output / "native_parity.json", checks)
    for path, digest in job["sources"].items():
        checked(path, digest)
    status = "smoke_complete" if args.smoke else "complete"
    write_json(args.output / "result.json", dict(status=status, model="AlphaRAD", dataset=args.dataset,
        branches=list(BRANCHES), support_seeds=list(SEEDS), test_count=n,
        cohort_role="development_cohort_mechanism_diagnostic", eta=.3, views=1,
        seed_artifacts=result_paths, protocol_sha256=sha(args.protocol),
        native_parity_sha256=sha(args.output / "native_parity.json"),
        checkpoint_sha256=CHECKPOINT_SHA, ignored_checkpoint_keys=ignored_keys,
        gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"), physical_gpu=args.gpu,
        frozen_model=True, test_labels_read=False, test_boxes_read=False,
        elapsed_seconds=time.time()-started, positive_prompts=prompts,
        grounding_definition="Mean per-head scaled cosine of ATTENTION query and image keys; not a classification attribution",
        spatial_equalities="Original=readout_only; aggregation_only=query, by construction",
        source_sha256={str(HERE / name): sha(HERE / name) for name in ("infer.py", "intervention.py")}))
    progress(args, status, started, done=n, total=n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("rsna_pneumonia", "chestxdet10"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--protocol", type=Path, default=MULTISEED / "protocol_v1/catalog.json")
    p.add_argument("--reuse-manifest", type=Path, default=MULTISEED / "cache_reuse_sources_v1.json")
    args = p.parse_args()
    args.model = "AlphaRAD"
    args.output = args.output.resolve()
    if args.batch_size < 1 or args.smoke < 0 or args.workers < 0 or args.gpu not in (4, 5, 6, 7):
        p.error("Positive batch size, nonnegative counts, and physical GPU 4–7 required")
    try:
        run(args)
    except Exception as exc:
        if args.output.is_dir():
            write_json(args.output / "failure.json", dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
