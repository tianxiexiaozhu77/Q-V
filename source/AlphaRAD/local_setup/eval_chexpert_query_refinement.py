#!/usr/bin/env python3
"""Evaluate fixed phase6 query refinement and two views on a CheXpert holdout.

This adds a dataset, not a new search: eta=.3, center80 and equal native-logit
averaging are imported from the completed phase6 experiment. All model weights
stay frozen. Only support labels enter query fitting; evaluation labels are used
for metrics after inference. Original and two ablations share the same holdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT.parent / "experiments/query_search_20261021_152010/code_phase6"
sys.path[:0] = [str(ROOT), str(PHASE6)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"

from eval_pretrained import ASSETS, sha256, write_json
from eval_query_refinement import CHECKPOINT_SHA, audit_csv, classification_metrics, original_helper_check

NAMES = ("original", "uniform_eta", "tta_center_original", "tta_center_refined")


def json_safe(value):
    """Convert NumPy scalars and non-finite smoke metrics to JSON-safe values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def center_crop_view(pixels, ratio):
    if not 0.0 < ratio <= 1.0:
        raise ValueError("crop ratio must lie in (0, 1]")
    h, w = pixels.shape[-2:]
    ch, cw = int(h * ratio), int(w * ratio)
    top, left = (h - ch) // 2, (w - cw) // 2
    return torch.nn.functional.interpolate(pixels[..., top:top + ch, left:left + cw],
        size=(h, w), mode="bilinear", align_corners=False, antialias=False)


def patient(path):
    match = re.search(r"(?:^|/)(patient\d+)(?:/|$)", str(path))
    if not match:
        raise ValueError(f"Missing CheXpert patient identity: {path}")
    return match[1]


def run(args):
    import numpy as np
    import torch
    from tqdm import tqdm
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.evaluator_utils import classification_score
    from cxrclip.prompt.constants import CHEXPERT
    from query_refinement import (VERSION, REAGGREGATION_VERSION, UNCERTAINTY_VERSION, CGT_VERSION,
                                  QueryRefinementMatcher,
                                  fit_query_directions,
                                  fit_reaggregation_query_directions,
                                  fit_contrastive_grounding_query_directions,
                                  fit_uncertainty_query_directions,
                                  original_queries_from_evaluator)
    from query_candidates import fit_candidates, candidate_plan
    from view_geometry import apply_view, aggregate_outputs, VIEW_SPECS

    started = time.time()
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("No assigned CUDA GPU")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Released checkpoint SHA256 mismatch")
    entry = json.loads(args.catalog.read_text())["datasets"]["chexpert"]
    classes = entry["class_names"]
    if classes != list(CHEXPERT) or entry["task"] != "classification":
        raise ValueError("Require author's CheXpert 12-class classification definition")
    support_rows, support_paths, support_labels = audit_csv(
        entry["support_csv_path"], entry["support_csv_sha256"], classes)
    test_rows, test_paths, test_labels = audit_csv(entry["csv_path"], entry["csv_sha256"], classes)
    if len(support_rows) != entry["support_count"] or len(test_rows) != entry["actual_count"]:
        raise ValueError("Manifest count changed")
    overlap = {patient(p) for p in support_paths} & {patient(p) for p in test_paths}
    if overlap or set(support_paths) & set(test_paths):
        raise ValueError("Support/test patient or image leakage")
    support_labels = np.asarray(support_labels, dtype=np.uint8)
    test_labels = np.asarray(test_labels, dtype=np.uint8)
    for labels in (support_labels, test_labels):
        if np.any(labels.sum(0) == 0) or np.any(labels.sum(0) == len(labels)):
            raise ValueError("Every class requires both positive and negative labels")
    if args.limit_test:
        test_rows, test_paths, test_labels = (test_rows[:args.limit_test],
                                               test_paths[:args.limit_test],
                                               test_labels[:args.limit_test])
    sources = [Path(__file__), ROOT / "local_setup/query_refinement.py",
        ROOT / "local_setup/eval_query_refinement.py", PHASE6 / "query_candidates.py",
        PHASE6 / "view_geometry.py", ROOT / "cxrclip/evaluator.py",
        ROOT / "cxrclip/base_evaluator.py", ROOT / "cxrclip/model/cxrclip_plus_dqn.py"]
    source_sha = {str(p): sha256(p) for p in sources}
    write_json(args.output / "source_sha256.json", source_sha)
    write_json(args.output / "data_audit.json", {"entry": entry,
        "support_patient_count": len({patient(p) for p in support_paths}),
        "test_patient_count": len({patient(p) for p in test_paths}),
        "patient_overlap": [], "support_count": len(support_paths), "test_count": len(test_paths),
        "support_positive_counts": support_labels.sum(0).tolist(),
        "test_positive_counts": test_labels.sum(0).tolist(),
        "catalog_sha256": sha256(args.catalog)})
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
        "pin_memory": True, "shuffle": False, "drop_last": False}
    if args.workers:
        loader_options["prefetch_factor"] = 2
    data = {name: {"name": name, "data_type": "image_classification", "data_path": str(path),
                  "n_class": len(classes), "normalize": None}
        for name, path in (("chexpert", entry["csv_path"]), ("chexpert_support", entry["support_csv_path"]))}
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(args.checkpoint)]},
        "data_test": data, "dataloader": {"test": loader_options}}
    write_json(args.output / "eval_config.json", config)
    evaluator = Evaluator(config, [str(args.checkpoint)], "test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    original, prompts = original_queries_from_evaluator(evaluator, classes)
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
        use_all_image_tokens=evaluator.use_all_image_tokens)
    features, observed, support_vision = [], [], []
    with torch.inference_mode():
        for batch in tqdm(evaluator.data_loader_dict["chexpert_support"], desc="CheXpert support"):
            # Reaggregation repeatedly calls the native query-conditioned
            # branch.  Cache only frozen projected vision tensors on CPU so
            # each fixed-point step can reuse the image encoder output.
            if args.method in ("reaggregation", "sgtt", "smgt", "cvsgtt"):
                prepared = matcher.encode_vision(batch["images"])
                values = matcher.match_prepared(
                    prepared, matcher.queries[0], return_features=True)
                support_vision.append({key: [value.cpu() for value in values]
                                       for key, values in prepared.items()})
            else:
                values = matcher(batch["images"], return_features=True)[0.]
            features.append(values["class_local"].cpu().numpy())
            observed.extend(batch["image_paths"])
    if observed != support_paths:
        raise ValueError("Support order changed")
    features = np.concatenate(features)
    fit_etas = (0.,) if args.eta == 0.0 else (0., args.eta)
    if args.method == "reaggregation":
        fitted = fit_reaggregation_query_directions(
            support_vision, support_labels, original, matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience)
    elif args.method in ("sgtt", "smgt", "cvsgtt"):
        fitted = fit_contrastive_grounding_query_directions(
            support_vision, support_labels, original, matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience,
            token_weight=args.token_weight,
            adaptive_token_gate=True, geometry="adaptive",
            support_margin_gate=args.method in ("sgttm", "smgt"),
            support_margin_grid=(args.method == "smgt"),
            support_cv_grid=(args.method == "cvsgtt"))
    elif args.method == "uncertainty":
        fitted = fit_uncertainty_query_directions(
            features, support_labels, original, etas=fit_etas,
            covariance_shrinkage=args.covariance_shrinkage,
            ridge=args.covariance_ridge,
            reliability_power=args.reliability_power)
    else:
        fitted = fit_query_directions(features, support_labels, original, etas=fit_etas)
    # The historical phase6 candidate archive is an eta=.3 parity check.  It
    # remains useful for the default run, while other predeclared sensitivity
    # values are checked against the unchanged eta=0 arm only.
    phase6_equal = False
    if args.eta == .3 and args.method == "closed_form":
        reference = fit_candidates(features, support_labels, original,
            num_heads=evaluator.model.num_radzero_heads, groups=[patient(p) for p in support_paths])
        for name in NAMES:
            index = 0 if name == "original" or name == "tta_center_original" else 1
            reference_name = "uniform_eta03" if name == "uniform_eta" else name
            np.testing.assert_array_equal(fitted["queries"][index], reference["queries"][reference["names"].index(reference_name)])
        phase6_equal = True
    np.save(args.output / "support_class_local.npy", features, allow_pickle=False)
    fitted_arrays = {key: value for key, value in fitted.items()
                     if isinstance(value, np.ndarray)}
    np.savez_compressed(args.output / "query_adapter.npz", **fitted_arrays)
    adaptation_version = (REAGGREGATION_VERSION if args.method == "reaggregation"
                          else CGT_VERSION if args.method in ("sgtt", "smgt", "cvsgtt")
                          else UNCERTAINTY_VERSION if args.method == "uncertainty"
                          else VERSION)
    write_json(args.output / "query_adapter.json", {
        "version": adaptation_version,
        "dataset": "chexpert", "eta": args.eta, "class_names": classes,
        "prompts": prompts, "query_dimension": original.shape[-1],
        "num_heads": evaluator.model.num_radzero_heads, "checkpoint_sha256": CHECKPOINT_SHA,
        "query_file_sha256": sha256(args.output / "query_adapter.npz"),
        "source_sha256": source_sha, "support_csv_path": entry["support_csv_path"],
        "support_csv_sha256": entry["support_csv_sha256"], "test_csv_sha256": entry["csv_sha256"],
        "support_count": len(support_paths), "positive_counts": fitted["positive_counts"].tolist(),
        "negative_counts": fitted["negative_counts"].tolist(), "no_network_weight_updates": True,
        "test_labels_used_for_query_fit": False, "phase6_query_bitwise_equal": phase6_equal,
        "eta_selection": "Predeclared sensitivity eta; no CheXpert metric-driven selection",
        "adaptation_mode": args.method,
        "token_weight": args.token_weight if args.method in ("sgtt", "smgt", "cvsgtt") else None,
        "reaggregation": ({key: value for key, value in fitted.items()
                           if key.startswith("reaggregation_")}
                          if args.method in ("reaggregation", "sgtt", "smgt", "cvsgtt") else None),
        "uncertainty": ({"covariance_shrinkage": args.covariance_shrinkage,
                         "ridge": args.covariance_ridge,
                         "reliability_power": args.reliability_power}
                        if args.method == "uncertainty" else None)})
    matcher = QueryRefinementMatcher(evaluator.model, fitted["queries"], fitted["etas"],
        use_all_image_tokens=evaluator.use_all_image_tokens)
    predictions = {name: [] for name in NAMES}
    image_paths, label_names = [], []
    helper_check = None
    with torch.inference_mode():
        for batch in tqdm(evaluator.data_loader_dict["chexpert"], desc="CheXpert fixed two-view comparison"):
            first_batch = not image_paths
            remaining = len(test_paths) - len(image_paths)
            if remaining <= 0:
                break
            if first_batch:
                helper_check = original_helper_check(evaluator, matcher, batch, classes)
                write_json(args.output / "zero_eta_check.json", helper_check)
            identity = matcher(batch["images"], return_maps=first_batch)
            center_pixels = center_crop_view(batch["images"].to(matcher.device, dtype=torch.float32), args.crop_ratio)
            center = matcher(center_pixels, return_maps=first_batch)
            if identity[0.]["logits"].shape[0] > remaining:
                for arm in (identity, center):
                    for eta in list(arm):
                        arm[eta] = {key: value[:remaining] if hasattr(value, "shape") and value.shape[0] > remaining else value
                                    for key, value in arm[eta].items()}
                batch = dict(batch)
                batch["image_paths"] = batch["image_paths"][:remaining]
                batch["label_names"] = batch["label_names"][:remaining]
            outputs = {"original": identity[0.], "uniform_eta": identity[args.eta]}
            for eta, name in ((0., "tta_center_original"), (args.eta, "tta_center_refined")):
                outputs[name] = dict(identity[eta])
                outputs[name]["logits"] = ((1.0 - args.view_weight) * identity[eta]["logits"]
                    + args.view_weight * center[eta]["logits"])
            for name, output in outputs.items():
                if not torch.isfinite(output["logits"]).all():
                    raise ValueError("Nonfinite logits")
                predictions[name].append(output["logits"].cpu().numpy())
            if first_batch:
                np.savez_compressed(args.output / "similarity_map_examples.npz",
                    image_paths=np.asarray(batch["image_paths"]),
                    **{name: out["similarity_scores"].cpu().numpy() for name, out in outputs.items()
                       if "similarity_scores" in out})
            image_paths.extend(batch["image_paths"])
            label_names.extend(batch["label_names"])
            write_json(args.output / "progress.json", {"status": "inference",
                "images_completed": len(image_paths), "images_total": len(test_paths)})
    if image_paths != test_paths:
        raise ValueError("Evaluation order/coverage changed")
    branches = {}
    for name in NAMES:
        logits = np.concatenate(predictions[name])
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        np.save(args.output / f"{name}_logits.npy", logits, allow_pickle=False)
        np.save(args.output / f"{name}_probabilities.npy", probabilities, allow_pickle=False)
        metrics = classification_metrics(test_labels, probabilities)
        metrics["per_class"] = dict(zip(classes, metrics["per_class"]))
        stats, counts = evaluator.gather_zeroshot_dqn_classification_statistics(
            classes, probabilities[..., None], label_names)
        author = classification_score(stats, counts)
        if args.limit_test is None:
            np.testing.assert_allclose(metrics["macro_auroc"], author["AUROC(Avg)"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(metrics["macro_average_precision"], author["PR_AUROC(Avg)"], atol=1e-10, rtol=0)
        metrics["author_metrics"] = author
        branches[name] = metrics
    write_json(args.output / "image_paths.json", image_paths)
    np.save(args.output / "labels.npy", test_labels, allow_pickle=False)
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Network unexpectedly trainable")
    if {str(p): sha256(p) for p in sources} != source_sha:
        raise ValueError("Implementation changed during evaluation")
    result = {"status": "complete", "dataset": "chexpert", "task": "classification",
        "rows": len(test_paths), "support_count": len(support_paths), "class_names": classes,
        "branches": branches, "eta": args.eta, "candidate": "tta_center_refined",
        "views": {"identity": VIEW_SPECS["identity"], "center": {"operation": "crop_resize",
            "requested_crop_fraction": args.crop_ratio}}, "view_weight": args.view_weight,
        "classification_aggregation": "mean of native logits followed by sigmoid",
        "spatial_aggregation": "phase6 inverse-warped raw-map mean over valid coverage",
        "delta_pp": {key: 100*(branches["tta_center_refined"][key]-branches["original"][key])
            for key in ("macro_auroc", "macro_auprc", "macro_average_precision")},
        "entry": entry, "checkpoint_sha256": CHECKPOINT_SHA, "ignored_checkpoint_keys": ignored,
        "zero_eta_check": helper_check, "phase6_query_bitwise_equal": phase6_equal,
        "no_network_weight_updates": True, "test_labels_used_for_query_fit": False,
        "batch_size": args.batch_size, "seed": 1234, "support_sampling_seed": 42,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "elapsed_seconds": time.time()-started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated()/1024**2}
    write_json(args.output / "result.json", json_safe(result))
    write_json(args.output / "progress.json", {"status": "complete", "images_completed": len(test_paths)})
    print(json.dumps({"status": "complete", "rows": result["rows"], "delta_pp": result["delta_pp"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eta", type=float, default=.3)
    parser.add_argument("--method", choices=("closed_form", "reaggregation", "uncertainty", "sgtt", "smgt", "cvsgtt"),
                        default="closed_form",
                        help="Support query fit; reaggregation reruns native support aggregation")
    parser.add_argument("--reaggregation-max-iter", type=int, default=3)
    parser.add_argument("--reaggregation-query-tol", type=float, default=1e-4)
    parser.add_argument("--reaggregation-patience", type=int, default=1)
    parser.add_argument("--token-weight", type=float, default=.5)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.2)
    parser.add_argument("--covariance-ridge", type=float, default=1e-3)
    parser.add_argument("--reliability-power", type=float, default=1.0)
    parser.add_argument("--crop-ratio", type=float, default=.8)
    parser.add_argument("--view-weight", type=float, default=.5)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or not math.isfinite(args.eta) or args.eta < 0
            or not math.isfinite(args.crop_ratio) or not 0 < args.crop_ratio <= 1
            or not math.isfinite(args.view_weight) or not 0 <= args.view_weight <= 1
            or not math.isfinite(args.covariance_shrinkage) or not 0 <= args.covariance_shrinkage <= 1
            or not math.isfinite(args.covariance_ridge) or args.covariance_ridge < 0
            or not math.isfinite(args.reliability_power) or args.reliability_power < 0
            or args.limit_test < 0
            or args.reaggregation_max_iter < 1 or args.reaggregation_max_iter > 32
            or not math.isfinite(args.reaggregation_query_tol) or args.reaggregation_query_tol < 0
            or args.reaggregation_patience < 1
            or args.reaggregation_patience > args.reaggregation_max_iter
            or not math.isfinite(args.token_weight) or args.token_weight < 0):
        parser.error("Invalid batch size or worker count")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("result.json", "query_adapter.npz", "failure.json")):
        raise FileExistsError("Use a fresh output directory")
    try:
        run(args)
    except BaseException as exc:
        write_json(args.output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
