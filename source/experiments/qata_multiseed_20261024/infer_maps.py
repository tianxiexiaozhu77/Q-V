#!/usr/bin/env python3
"""Fit five frozen queries and predict QaTa native maps without opening masks.

All support sets share the unchanged image encoder. Each evaluation image is
encoded once per view, and each support seed uses its own fixed query. Original
and V are computed once per image and copied to every seed, not independent runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from common import (ALPHA, BRANCHES, CHECKPOINT, CHECKPOINT_SHA, LEGACY, MEDICAL,
                    PHASE6, SEEDS, atomic_json, checked, load_rows, load_supports,
                    read_protocol, resolve, row_image, row_patient, sha256, source_snapshot)

OLD_RUN = MEDICAL / "experiments/query_search_20261021_152010/runs/phase6/alpharad_qatacovid_seg"
REFERENCE = LEGACY / "runs/validation_selected_20261024_v1/validation_maps/test_cache_manifest.json"


def preflight(protocol):
    """Read-only asset validation: no CUDA allocation, annotations or model load."""
    path, config = read_protocol(protocol)
    rows = {split: load_rows(path, config, split, identities=True) for split in ("validation", "test")}
    supports = load_supports(path, config)
    if {row_patient(r) for r in rows["validation"]} & {row_patient(r) for r in rows["test"]}:
        raise ValueError("Validation/test patient overlap")
    if {r["pixel_sha256"] for r in rows["validation"]} & {r["pixel_sha256"] for r in rows["test"]}:
        raise ValueError("Validation/test exact-pixel overlap")
    all_rows = [*rows["validation"], *rows["test"], *[r for values in supports.values() for r in values]]
    images = {row_image(r): r["image_sha256"] for r in all_rows}
    for image, digest in images.items():
        checked(image, digest)
    checked(CHECKPOINT, CHECKPOINT_SHA)
    audit = json.loads((OLD_RUN / "data_audit.json").read_text())
    native_sources = {}
    for name, digest in audit["original_sources_verified"].items():
        if name.startswith("cxrclip/") or name == "local_setup/query_refinement.py":
            native_sources[str(checked(ALPHA / name, digest))] = digest
    if json.loads((OLD_RUN / "image_paths.json").read_text()) != [row_image(r) for r in rows["test"]]:
        raise ValueError("Historical Original/V reference is a different test order")
    reference = json.loads(REFERENCE.read_text())
    for branch in ("original", "tta_center_original"):
        ref = reference["branches"][branch]
        checked(ref["path"], ref["sha256"])
    return dict(status="preflight_passed", protocol=str(path), protocol_sha256=sha256(path),
                seeds=list(SEEDS), branches=list(BRANCHES), support_images_per_seed=10,
                support_union_images=len({row_image(r) for rs in supports.values() for r in rs}),
                validation_images=473, validation_patients=200, test_images=2113, test_patients=474,
                images_authenticated=len(images), checkpoint_sha256=CHECKPOINT_SHA,
                negative_support_patient_ids_verified=False, native_source_sha256=native_sources,
                annotations_loaded=False, gpu_used=False)


def fit_queries(features, labels, original):
    """The same AlphaRAD positive-only and positive-minus-negative fits as Chest."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ALPHA / "local_setup"))
    from query_refinement import fit_query_directions
    fitted = fit_query_directions(features, labels, original, etas=(0., .3))
    positive = F.normalize(F.normalize(torch.as_tensor(original), dim=-1) + .3 *
                           F.normalize(torch.as_tensor(fitted["positive_means"]), dim=-1), dim=-1)
    # Honor the declared no-change fallback even for a non-unit original query.
    query = fitted["queries"][1].copy()
    query[fitted["zero_direction_classes"]] = original[fitted["zero_direction_classes"]]
    fitted.update(positive_query=positive.numpy(), query=query)
    if not np.isfinite(positive.numpy()).all():
        raise ValueError("Nonfinite positive-only query")
    return fitted


def run(args):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator
    from eval_alpharad_candidates import controls_helper_check
    from view_geometry import apply_view, aggregate_outputs, VIEW_SPECS

    started = time.time()
    def progress(stage, **fields):
        record = dict(status=stage, smoke=args.smoke, elapsed_seconds=time.time() - started, **fields)
        atomic_json(args.output / "status.json", record)
        print(json.dumps(record), flush=True)

    progress("preflight")
    verified = preflight(args.protocol)
    atomic_json(args.output / "preflight.json", verified)
    protocol_path, protocol = read_protocol(args.protocol)
    protocol_sha = sha256(protocol_path)
    sources = source_snapshot()
    sources.update(verified["native_source_sha256"])
    for descriptor in (protocol_path, REFERENCE, OLD_RUN / "data_audit.json"):
        sources[str(descriptor)] = sha256(descriptor)
    rows = {split: load_rows(protocol_path, protocol, split, identities=True) for split in ("validation", "test")}
    supports = load_supports(protocol_path, protocol)
    union = {row_image(r): r for seed in SEEDS for r in supports[seed]}
    support_rows = [union[k] for k in sorted(union)]
    selected = {split: values[:args.smoke] if args.smoke else values for split, values in rows.items()}
    datasets = dict(support=support_rows, **selected)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                   shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    for split, values in datasets.items():
        with (args.output / (split + "_images.csv")).open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image"])
            writer.writerows([[row_image(r)] for r in values])
    config = dict(base=dict(seed=1234), test=dict(checkpoint=[str(CHECKPOINT)]),
                  data_test={split: dict(name=split, data_type="image_classification", n_class=1,
                             data_path=str(args.output / (split + "_images.csv")), normalize=None)
                             for split in datasets}, dataloader=dict(test=options))
    atomic_json(args.output / "eval_config.json", config)
    seed_everything(1234)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Run via run_campaign.py to reserve an authorized idle GPU")
    progress("loading_model")
    evaluator = Evaluator(config, [str(CHECKPOINT)], "test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    original, prompts = original_queries_from_evaluator(evaluator, ["covid"])
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    legacy_audit = json.loads((OLD_RUN / "data_audit.json").read_text())
    if legacy_audit["view_geometry"]["center80"] != VIEW_SPECS["center80"]:
        raise ValueError("Center geometry differs from frozen phase6")
    native_query_path = OLD_RUN / "candidates/original/query.npy"
    np.testing.assert_array_equal(original, np.load(native_query_path, allow_pickle=False))
    sources[str(native_query_path)] = sha256(native_query_path)
    features = []
    offset = 0
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["support"]:
            stop = offset + len(batch["images"])
            if batch["image_paths"] != [row_image(r) for r in support_rows[offset:stop]]:
                raise ValueError("Support loader order changed")
            vision = matcher.prepare_vision_results(evaluator.encode_image(batch["images"]))
            features.append(matcher.match_prepared(vision, original, return_features=True)["class_local"].cpu().numpy())
            offset = stop
    if offset != len(support_rows):
        raise ValueError("Incomplete support feature extraction")
    all_features = np.concatenate(features)
    lookup = {row_image(r): i for i, r in enumerate(support_rows)}
    fits, query_refs = {}, {}
    for seed in SEEDS:
        seed_dir = args.output / f"seed_{seed}"
        seed_dir.mkdir()
        indices = np.asarray([lookup[row_image(r)] for r in supports[seed]])
        labels = np.asarray([r["label_vector"] for r in supports[seed]], dtype=np.int64)
        fitted = fit_queries(all_features[indices], labels, original)
        query_path = seed_dir / "query_fit.npz"
        np.savez_compressed(query_path, **fitted, support_features=all_features[indices],
                            labels=labels, image_paths=np.asarray([row_image(r) for r in supports[seed]]),
                            support_seed=seed, protocol_sha256=protocol_sha)
        query_refs[str(seed)] = dict(path=str(query_path), sha256=sha256(query_path))
        fits[seed] = {name: torch.as_tensor(fitted[name], dtype=torch.float32, device=matcher.device)
                      for name in ("query", "positive_query")}
    progress("queries_frozen", support_union_images=len(support_rows))
    reference = json.loads(REFERENCE.read_text())
    references = {name: np.load(reference["branches"][name]["path"], mmap_mode="r", allow_pickle=False)
                  for name in ("original", "tta_center_original")}
    parity, map_refs = {}, {str(seed): {} for seed in SEEDS}
    with torch.inference_mode():
        for split, split_rows in selected.items():
            arrays = {seed: {b: np.empty((len(split_rows), 1, 1370), dtype=np.float32) for b in BRANCHES}
                      for seed in SEEDS}
            offset = 0
            for batch in evaluator.data_loader_dict[split]:
                stop = offset + len(batch["images"])
                if batch["image_paths"] != [row_image(r) for r in split_rows[offset:stop]]:
                    raise ValueError("Image-only inference loader order changed")
                image_results = evaluator.encode_image(batch["images"])
                identity = matcher.prepare_vision_results(image_results)
                # Preserve phase6: transfer to GPU before crop/interpolation.
                pixels = batch["images"].to(matcher.device, dtype=torch.float32, non_blocking=True)
                center = matcher.encode_vision(apply_view(pixels, "center80"))
                native = matcher.match_prepared(identity, original, return_maps=True)
                center_native = matcher.match_prepared(center, original, return_maps=True)
                views = aggregate_outputs(dict(identity=native, center80=center_native), ["identity", "center80"])
                controls = dict(original=native, tta_center_original=views)
                if offset == 0:
                    helper_batch = dict(batch, label_names=[["covid"]] * len(batch["images"]))
                    parity[split + "_author_helper"] = controls_helper_check(
                        evaluator, image_results, helper_batch, ["covid"], native)
                    if split == "test":
                        for branch in controls:
                            actual = controls[branch]["grounding_map"].cpu().numpy()
                            expected = references[branch][:len(actual)]
                            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
                            parity[branch] = dict(passed=True, images=len(actual),
                                                  max_abs_error=float(np.max(np.abs(actual - expected))))
                for seed in SEEDS:
                    q = matcher.match_prepared(identity, fits[seed]["query"], return_maps=True)
                    cq = matcher.match_prepared(center, fits[seed]["query"], return_maps=True)
                    positive = matcher.match_prepared(identity, fits[seed]["positive_query"], return_maps=True)
                    outputs = dict(**controls, positive_query=positive, query_refined=q,
                                   tta_center_refined=aggregate_outputs(dict(identity=q, center80=cq), ["identity", "center80"]))
                    for branch in BRANCHES:
                        values = outputs[branch]["grounding_map"].cpu().numpy()
                        if values.shape != (stop - offset, 1, 1370) or not np.isfinite(values).all():
                            raise ValueError("Invalid native maps")
                        arrays[seed][branch][offset:stop] = values
                offset = stop
                if offset == len(batch["images"]) or offset % (25 * args.batch_size) == 0 or offset == len(split_rows):
                    progress(split + "_inference", done=offset, total=len(split_rows))
            if offset != len(split_rows):
                raise ValueError("Incomplete native inference")
            for seed in SEEDS:
                for branch in ("original", "tta_center_original"):
                    np.testing.assert_array_equal(arrays[seed][branch], arrays[SEEDS[0]][branch])
                target = args.output / f"seed_{seed}" / (split + ".npz")
                np.savez_compressed(target, **arrays[seed], class_names=np.asarray(["covid"]),
                                    image_paths=np.asarray([row_image(r) for r in split_rows]),
                                    patient_ids=np.asarray([row_patient(r) for r in split_rows]),
                                    support_seed=seed, protocol_sha256=protocol_sha)
                map_refs[str(seed)][split] = dict(path=str(target), sha256=sha256(target))
            del arrays
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Model must remain frozen")
    for path, digest in sources.items():
        checked(path, digest)
    # Recheck split/support manifests and image bytes at completion as well.
    preflight(args.protocol)
    manifest = dict(status="smoke_complete" if args.smoke else "complete", smoke=args.smoke,
                    protocol=dict(path=str(protocol_path), sha256=protocol_sha),
                    seeds=list(SEEDS), branches=list(BRANCHES), checkpoint_sha256=CHECKPOINT_SHA,
                    maps=map_refs, query_fits=query_refs, source_sha256=sources, parity=parity,
                    original_and_v_shared_across_seeds=True, image_encoder_passes_per_image=2,
                    support_union_encoded_once=True, support_masks_used=False, test_annotations_loaded=False,
                    thresholds_selected=False, network_weight_updates=False, prompts=prompts,
                    ignored_checkpoint_keys=ignored, elapsed_seconds=time.time() - started)
    atomic_json(args.output / "manifest.json", manifest)
    atomic_json(args.output / "parity_checks.json", parity)
    progress(manifest["status"], validation_images=len(selected["validation"]), test_images=len(selected["test"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", type=int, default=0)
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1 or args.smoke < 0:
        p.error("Invalid inference sizes")
    if args.preflight:
        print(json.dumps(preflight(args.protocol), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        p.error("--output required for inference")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        atomic_json(args.output / "failure.json", dict(status="failed", error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
