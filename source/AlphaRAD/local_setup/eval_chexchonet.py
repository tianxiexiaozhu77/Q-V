#!/usr/bin/env python3
"""Frozen AlphaRAD on audited CheXchoNet manifests and fixed phase6 refinement.

All tasks share each image-encoder forward. Each task's correction uses only
its own five-positive/five-negative support images. Evaluation patients must be
disjoint from the union of support patients. No training or parameter search.

The released AlphaRAD repository names the three CheXchoNet tasks but omits
their prompt constants/data configurations. Disease names below are an explicit
local reconstruction, passed through the released checkpoint's original text
prompt template and native FLaS head. Do not call them verified author prompts.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
PHASE6 = ROOT.parent / "experiments/query_search_20261021_152010/code_phase6"
sys.path[:0] = [str(ROOT), str(ROOT / "local_setup"), str(PHASE6)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"

from eval_pretrained import ASSETS, sha256, write_json

TASKS = {
    "slvh": "severe left ventricular hypertrophy",
    "dlv": "dilated left ventricle",
    "composite_slvh_dlv": "severe left ventricular hypertrophy or dilated left ventricle",
}
BRANCHES = ("original", "uniform_eta", "tta_center_original", "tta_center_refined")


def center_crop_view(pixels, ratio):
    if not 0.0 < ratio <= 1.0:
        raise ValueError("crop ratio must lie in (0, 1]")
    h, w = pixels.shape[-2:]
    ch, cw = int(h * ratio), int(w * ratio)
    top, left = (h - ch) // 2, (w - cw) // 2
    return torch.nn.functional.interpolate(pixels[..., top:top + ch, left:left + cw],
        size=(h, w), mode="bilinear", align_corners=False, antialias=False)


def parse_memberships(value):
    if not value:
        return set()
    if value.startswith("["):
        values = ast.literal_eval(value)
    else:
        values = value.replace(";", ",").split(",")
    return {"composite_slvh_dlv" if str(x).strip() == "composite" else str(x).strip() for x in values}


def catalog_arguments(args):
    """Translate audited task-specific source manifests into one native pass."""
    if args.catalog is None:
        if args.support_csv is None or args.evaluation_csv is None:
            raise ValueError("Provide --catalog or both --support-csv and --evaluation-csv")
        return
    catalog = json.loads(args.catalog.read_text())
    tasks = list(TASKS) if args.task == "all" else [args.task]
    entries = [catalog["datasets"]["chexchonet_" + t] for t in tasks]
    support_by_path = {}
    paired_paths, full_paths = [], []
    for task, entry in zip(tasks, entries):
        if [x.lower() for x in entry["class_names"]] != [TASKS[task]]:
            raise ValueError(f"Frozen class definition differs for {task}")
        for path_key, hash_key in [("csv_path", "csv_sha256"), ("support_csv_path", "support_csv_sha256"),
                                   ("original_csv_path", "original_csv_sha256")]:
            if sha256(entry[path_key]) != entry[hash_key]:
                raise ValueError(f"Catalog manifest hash mismatch: {entry[path_key]}")
        rows = read_manifest(Path(entry["support_csv_path"]))
        for row in rows:
            path = row["image_path"]
            if path in support_by_path:
                old = support_by_path[path]
                if any(row[k] != old[k] for k in ("patient_id", *TASKS)):
                    raise ValueError("Conflicting support labels across tasks")
            else:
                support_by_path[path] = row
                support_by_path[path]["support_for"] = set()
            support_by_path[path]["support_for"].add(task)
        paired_paths.append(Path(entry["csv_path"]))
        full_paths.append(Path(entry["original_csv_path"]))
    def stable_identity(path):
        return [(r["image_path"], r["patient_id"], *[r[t] for t in tasks]) for r in read_manifest(path)]
    if any(stable_identity(p) != stable_identity(paired_paths[0]) for p in paired_paths[1:]):
        raise ValueError("Tasks must share the exact ordered paired test images/labels")
    if any(stable_identity(p) != stable_identity(full_paths[0]) for p in full_paths[1:]):
        raise ValueError("Tasks must share the exact ordered full test images/labels")
    args.support_csv = args.output / "support_union_input.csv"
    with args.support_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "patient_id", *TASKS, "support_for"])
        writer.writeheader()
        for row in support_by_path.values():
            writer.writerow({**row, "support_for": repr(sorted(row["support_for"]))})
    args.evaluation_csv = paired_paths[0]
    args.full_test_csv = full_paths[0]
    write_json(args.output / "catalog_input.json", {"path": str(args.catalog), "sha256": sha256(args.catalog),
                                                   "entries": dict(zip(tasks, entries))})


def read_manifest(path):
    with path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for item in raw:
        image = item.get("image_path") or item.get("image")
        if not image:
            raise ValueError("Manifest requires image_path or image column")
        image = Path(image)
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            raise FileNotFoundError(image)
        pid = item.get("patient_id", "").strip()
        if not pid:
            raise ValueError("Every image requires a nonempty patient_id")
        labels = {}
        for task in TASKS:
            value = item.get(task, item.get("composite") if task == "composite_slvh_dlv" else None)
            if value not in ("0", "1", "0.0", "1.0"):
                raise ValueError(f"Require explicit binary {task} label: {value}")
            labels[task] = int(float(value))
        if labels["composite_slvh_dlv"] != int(labels["slvh"] or labels["dlv"]):
            raise ValueError("Composite must equal SLVH OR DLV")
        rows.append({"image_path": str(image.resolve()), "patient_id": pid, **labels,
                     "support_for": parse_memberships(item.get("support_for", ""))})
    paths = [r["image_path"] for r in rows]
    if not rows or len(set(paths)) != len(paths):
        raise ValueError("Empty or duplicate image manifest")
    return rows


def audit(args):
    support, evaluation = read_manifest(args.support_csv), read_manifest(args.evaluation_csv)
    tasks = list(TASKS) if args.task == "all" else [args.task]
    sp = {r["patient_id"] for r in support}
    ep = {r["patient_id"] for r in evaluation}
    if sp & ep or {r["image_path"] for r in support} & {r["image_path"] for r in evaluation}:
        raise ValueError("Support/evaluation patient or image overlap")
    indices, counts = {}, {}
    has_membership = any(r["support_for"] for r in support)
    if has_membership and any(not r["support_for"] for r in support):
        raise ValueError("support_for must be populated on all support rows")
    for task in tasks:
        indices[task] = [i for i, r in enumerate(support) if not has_membership or task in r["support_for"]]
        selected = [support[i] for i in indices[task]]
        pos = sum(r[task] for r in selected)
        neg = len(selected) - pos
        if (pos, neg) != (args.shots, args.shots):
            raise ValueError(f"{task}: require exactly {args.shots} positive and negative support images, got {pos}/{neg}")
        if len({r["patient_id"] for r in selected}) != len(selected):
            raise ValueError(f"{task}: supports must use distinct patients")
        tpos = sum(r[task] for r in evaluation)
        if tpos in (0, len(evaluation)):
            raise ValueError(f"{task}: evaluation requires positive and negative patients")
        counts[task] = {"support_positive": pos, "support_negative": neg,
                        "evaluation_positive": tpos, "evaluation_negative": len(evaluation) - tpos}
    full = read_manifest(args.full_test_csv) if args.full_test_csv else evaluation
    full_by_path = {r["image_path"]: r for r in full}
    for row in evaluation:
        expected = full_by_path.get(row["image_path"])
        if expected is None or any(expected[k] != row[k] for k in ["patient_id", *tasks]):
            raise ValueError("Evaluation is not an exact labelled subset of full manifest")
    result = {"support_csv": str(args.support_csv), "support_csv_sha256": sha256(args.support_csv),
              "evaluation_csv": str(args.evaluation_csv), "evaluation_csv_sha256": sha256(args.evaluation_csv),
              "full_test_csv": str(args.full_test_csv) if args.full_test_csv else None,
              "full_test_csv_sha256": sha256(args.full_test_csv) if args.full_test_csv else None,
              "support_image_count": len(support), "support_patient_count": len(sp),
              "evaluation_image_count": len(evaluation), "evaluation_patient_count": len(ep),
              "patient_overlap": [], "tasks": tasks, "counts": counts,
              "support_indices_by_task": indices, "full_image_count": len(full),
              "task_names": {task: TASKS[task] for task in tasks},
              "prompt_provenance": "Disease-name reconstruction; official CheXchoNet constants are absent."}
    return support, evaluation, full, result


def native_manifest(path, rows, tasks):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "patient_id", "label", "class"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image": row["image_path"], "patient_id": row["patient_id"],
                             "label": repr([row[task] for task in tasks]),
                             "class": repr([TASKS[task] for task in tasks if row[task]])})


def run(args, support, evaluation, full, data_audit):
    import numpy as np
    import torch
    from tqdm import tqdm
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.evaluator_utils import classification_score
    from cxrclip.prompt import constants
    from eval_query_refinement import CHECKPOINT_SHA, classification_metrics, original_helper_check
    from query_refinement import (QueryRefinementMatcher, fit_query_directions,
                                  fit_reaggregation_query_directions,
                                  fit_contrastive_grounding_query_directions,
                                  fit_uncertainty_query_directions,
                                  original_queries_from_evaluator)
    from view_geometry import apply_view, aggregate_outputs, VIEW_SPECS

    start = time.time()
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("No assigned CUDA GPU")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Published AlphaRAD checkpoint identity mismatch")
    tasks = data_audit["tasks"]
    classes = [TASKS[t] for t in tasks]
    full_mode = args.mode == "full-original"
    inference_rows = full if full_mode else evaluation
    branches = ("original",) if full_mode else BRANCHES
    sources = [Path(__file__), ROOT / "local_setup/query_refinement.py",
               ROOT / "local_setup/eval_query_refinement.py", PHASE6 / "view_geometry.py",
               ROOT / "cxrclip/evaluator.py", ROOT / "cxrclip/base_evaluator.py",
               ROOT / "cxrclip/model/cxrclip_plus_dqn.py"]
    source_sha = {str(p): sha256(p) for p in sources}
    write_json(args.output / "source_sha256.json", source_sha)
    native_manifest(args.output / "native_support.csv", support, tasks)
    native_manifest(args.output / "native_evaluation.csv", inference_rows, tasks)
    name = "chexchonet_custom"
    setattr(constants, name.upper(), classes)
    options = {"batch_size": args.batch_size, "num_workers": args.workers,
               "pin_memory": True, "shuffle": False, "drop_last": False}
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(args.checkpoint)]},
              "data_test": {key: {"name": key, "data_type": "image_classification",
                                  "data_path": str(path), "n_class": len(classes), "normalize": None}
                            for key, path in [(name, args.output / "native_evaluation.csv"),
                                              (name + "_support", args.output / "native_support.csv")]},
              "dataloader": {"test": options}}
    write_json(args.output / "eval_config.json", config)
    evaluator = Evaluator(config, [str(args.checkpoint)], "test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    original, prompts = original_queries_from_evaluator(evaluator, classes)
    if args.catalog:
        prompt_path = args.catalog.parent / "prompts.json"
        prompt_record = json.loads(prompt_path.read_text())
        if prompts != [prompt_record["tasks"][t]["prompt"] for t in tasks]:
            raise ValueError("Native checkpoint prompt template differs from frozen prompt record")
        write_json(args.output / "prompt_provenance.json", {"source": str(prompt_path),
                   "sha256": sha256(prompt_path), "record": prompt_record,
                   "actual_prompts": prompts, "native_template_verified": True})
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    reaggregation_history = {}
    if not full_mode:
        features, observed, support_vision_batches = [], [], []

        def cpu_prepared(vision):
            """Keep one frozen support vision batch on CPU for fixed-point fitting."""
            return {key: [value.detach().cpu() for value in values]
                    for key, values in vision.items()}

        def concat_prepared(batches):
            if not batches:
                raise ValueError("No support vision batches")
            return {key: [torch.cat([batch[key][layer] for batch in batches], dim=0)
                          for layer in range(len(batches[0][key]))]
                    for key in batches[0]}

        def select_prepared(vision, indices):
            indices = torch.as_tensor(indices, dtype=torch.long)
            return {key: [value.index_select(0, indices)
                          for value in values]
                    for key, values in vision.items()}

        with torch.inference_mode():
            for batch in tqdm(evaluator.data_loader_dict[name + "_support"], desc="CheXchoNet support"):
                prepared = matcher.encode_vision(batch["images"])
                support_vision_batches.append(cpu_prepared(prepared))
                features.append(matcher.match_prepared(
                    prepared, matcher.queries[0], return_features=True)["class_local"].cpu().numpy())
                observed.extend(batch["image_paths"])
        if observed != [r["image_path"] for r in support]:
            raise ValueError("Support order changed")
        features = np.concatenate(features)
        support_vision = concat_prepared(support_vision_batches)
        fitted = []
        for j, task in enumerate(tasks):
            ids = data_audit["support_indices_by_task"][task]
            y = np.asarray([[support[i][task]] for i in ids], dtype=np.uint8)
            fit_etas = (0.,) if args.eta == 0.0 else (0., args.eta)
            if args.method == "reaggregation":
                item = fit_reaggregation_query_directions(
                    [select_prepared(support_vision, ids)], y,
                    original[j:j + 1], matcher, eta=args.eta,
                    max_iter=args.reaggregation_max_iter,
                    query_tol=args.reaggregation_query_tol,
                    patience=args.reaggregation_patience)
                reaggregation_history[task] = item["reaggregation_history"]
                # The fixed-point fitter always returns the original and one
                # adapted query, independently of the requested eta grid.
                fit_etas = (0., args.eta)
            elif args.method in ("sgtt", "cvsgtt"):
                item = fit_contrastive_grounding_query_directions(
                    [select_prepared(support_vision, ids)], y,
                    original[j:j + 1], matcher, eta=args.eta,
                    max_iter=args.reaggregation_max_iter,
                    query_tol=args.reaggregation_query_tol,
                    patience=args.reaggregation_patience,
                    token_weight=args.token_weight,
                    adaptive_token_gate=True, geometry="adaptive",
                    support_cv_grid=(args.method == "cvsgtt"))
                reaggregation_history[task] = item.get("reaggregation_history", [])
                fit_etas = (0., args.eta)
            elif args.method == "uncertainty":
                item = fit_uncertainty_query_directions(
                    features[ids, j:j + 1], y, original[j:j + 1], etas=fit_etas,
                    covariance_shrinkage=args.covariance_shrinkage,
                    ridge=args.covariance_ridge,
                    reliability_power=args.reliability_power)
            else:
                item = fit_query_directions(features[ids, j:j + 1], y, original[j:j + 1], etas=fit_etas)
            np.savez_compressed(args.output / f"query_adapter_{task}.npz",
                                **{key: value for key, value in item.items()
                                   if isinstance(value, np.ndarray)},
                                support_indices=np.asarray(ids))
            fitted.append(item)
        queries = np.concatenate([item["queries"] for item in fitted], axis=1)
        np.testing.assert_array_equal(queries[0], original)
        np.save(args.output / "support_class_local.npy", features, allow_pickle=False)
        np.savez_compressed(args.output / "query_adapter.npz", queries=queries, etas=np.asarray(fit_etas),
                            task_names=np.asarray(tasks), original_queries=original)
        matcher = QueryRefinementMatcher(evaluator.model, queries, fit_etas,
                                        use_all_image_tokens=evaluator.use_all_image_tokens)
    write_json(args.output / "query_adapter.json", {"tasks": tasks, "class_names": classes,
               "prompts": prompts, "eta": 0. if full_mode else args.eta, "prompt_template": evaluator.prompt_template,
               "prompt_provenance": data_audit["prompt_provenance"], "support_indices_by_task": data_audit["support_indices_by_task"],
               "network_parameters_frozen": True, "test_labels_used_for_query_fit": False,
               "query_correction": ("support-gated token/global contrastive grounding with adaptive linear/tangent path"
                                    if args.method in ("sgtt", "cvsgtt") else
                                    "support reaggregation-aware fixed-point direction with support-only stopping"
                                    if args.method == "reaggregation" else
                                    "support covariance-shrinkage Mahalanobis direction with reliability weighting"
                                    if args.method == "uncertainty" else
                                    "norm(norm(q)+eta*norm(mean_positive-mean_negative))"),
               "adaptation_mode": args.method,
               "token_weight": args.token_weight if args.method in ("sgtt", "cvsgtt") else None,
               "reaggregation": (reaggregation_history if args.method in ("reaggregation", "sgtt", "cvsgtt") else None),
               "reaggregation_max_iter": (args.reaggregation_max_iter if args.method in ("reaggregation", "sgtt", "cvsgtt") else None),
               "reaggregation_query_tol": (args.reaggregation_query_tol if args.method in ("reaggregation", "sgtt", "cvsgtt") else None),
               "reaggregation_patience": (args.reaggregation_patience if args.method in ("reaggregation", "sgtt", "cvsgtt") else None),
               "uncertainty": ({"covariance_shrinkage": args.covariance_shrinkage,
                                "ridge": args.covariance_ridge,
                                "reliability_power": args.reliability_power}
                               if args.method == "uncertainty" else None)})
    predictions, image_paths, label_names = {b: [] for b in branches}, [], []
    helper = None
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(evaluator.data_loader_dict[name], desc="AlphaRAD CheXchoNet " + args.mode)):
            first = not image_paths
            if first:
                helper = original_helper_check(evaluator, matcher, batch, classes)
                write_json(args.output / "zero_eta_check.json", helper)
            identity = matcher(batch["images"], return_maps=first)
            outputs = {"original": identity[0.]}
            if not full_mode:
                center = matcher(center_crop_view(batch["images"].to(matcher.device, dtype=torch.float32), args.crop_ratio),
                                 return_maps=first)
                outputs["uniform_eta"] = identity[args.eta]
                for eta, branch in [(0., "tta_center_original"), (args.eta, "tta_center_refined")]:
                    outputs[branch] = dict(identity[eta])
                    outputs[branch]["logits"] = ((1.0 - args.view_weight) * identity[eta]["logits"]
                        + args.view_weight * center[eta]["logits"])
            for branch in branches:
                values = outputs[branch]["logits"].cpu().numpy()
                if values.shape != (len(batch["images"]), len(tasks)) or not np.isfinite(values).all():
                    raise ValueError("Invalid native output shape/values")
                predictions[branch].append(values)
            if first:
                np.savez_compressed(args.output / "similarity_map_examples.npz",
                                    image_paths=np.asarray(batch["image_paths"]),
                                    **{b: outputs[b]["similarity_scores"].cpu().numpy() for b in branches})
            image_paths.extend(batch["image_paths"])
            label_names.extend(batch["label_names"])
            write_json(args.output / "progress.json", {"status": "inference", "images_completed": len(image_paths),
                       "images_total": len(inference_rows), "elapsed_seconds": time.time() - start})
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
    if image_paths != [r["image_path"] for r in inference_rows[:len(image_paths)]]:
        raise ValueError("Evaluation order changed")
    if not args.smoke_batches and len(image_paths) != len(inference_rows):
        raise ValueError("Missing evaluation images")
    inference_rows = inference_rows[:len(image_paths)]
    lookup = {path: i for i, path in enumerate(image_paths)}
    metric_rows = inference_rows
    ids = list(range(len(metric_rows)))
    labels = np.asarray([[r[t] for t in tasks] for r in metric_rows], dtype=np.uint8)
    np.save(args.output / "labels.npy", labels, allow_pickle=False)
    write_json(args.output / "image_paths.json", [r["image_path"] for r in metric_rows])
    write_json(args.output / "patient_ids.json", [r["patient_id"] for r in metric_rows])
    results = {}
    full_baseline = None
    for branch in branches:
        all_logits = np.concatenate(predictions[branch])
        all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
        probs, logits = all_probs[ids], all_logits[ids]
        np.save(args.output / f"{branch}_logits.npy", logits, allow_pickle=False)
        np.save(args.output / f"{branch}_probabilities.npy", probs, allow_pickle=False)
        metrics = classification_metrics(labels, probs)
        metrics["per_task"] = dict(zip(tasks, metrics.pop("per_class")))
        if not args.smoke_batches:
            stat, count = evaluator.gather_zeroshot_dqn_classification_statistics(
                classes, probs[..., None], [label_names[i] for i in ids])
            native = classification_score(stat, count)
            np.testing.assert_allclose(metrics["macro_auroc"], native["AUROC(Avg)"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(metrics["macro_average_precision"], native["PR_AUROC(Avg)"], atol=1e-10, rtol=0)
            metrics["author_metrics"] = native
        results[branch] = metrics
        if full_mode:
            full_baseline = metrics
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Model unexpectedly trainable")
    if {str(p): sha256(p) for p in sources} != source_sha:
        raise ValueError("Source code changed during evaluation")
    for key in ("support", "evaluation", "full_test"):
        path = data_audit.get(key + "_csv")
        if path and sha256(path) != data_audit[key + "_csv_sha256"]:
            raise ValueError("Manifest changed during evaluation")
    result = {"status": "smoke_complete" if args.smoke_batches else "complete", "mode": args.mode,
              "smoke_batches": args.smoke_batches, "model": "AlphaRAD", "dataset": "chexchonet-custom-test-7494",
              "tasks": tasks, "class_names": classes, "rows": len(metric_rows), "branches": results,
              "data_audit": data_audit, "eta": 0. if full_mode else args.eta,
              "method": args.method,
              "reaggregation": (reaggregation_history if args.method == "reaggregation" else None),
              "reaggregation_settings": (None if args.method != "reaggregation" else {
                  "max_iter": args.reaggregation_max_iter,
                  "query_tol": args.reaggregation_query_tol,
                  "patience": args.reaggregation_patience}),
              "candidate": "original" if full_mode else "tta_center_refined",
              "views": {"identity": VIEW_SPECS["identity"]} if full_mode else {
                  "identity": VIEW_SPECS["identity"], "center": {
                      "operation": "crop_resize", "requested_crop_fraction": args.crop_ratio}},
              "view_weight": args.view_weight,
              "classification_aggregation": "native sigmoid" if full_mode else "mean native logits then sigmoid",
              "full_baseline_reference": full_baseline, "checkpoint_sha256": CHECKPOINT_SHA,
              "ignored_checkpoint_keys": ignored, "zero_eta_check": helper,
              "no_network_weight_updates": True, "test_labels_used_for_query_fit": False,
              "prompt_provenance": data_audit["prompt_provenance"], "prompts": prompts,
              "delta_pp": None if full_mode else {t: {m: (100 * (results["tta_center_refined"]["per_task"][t][m] - results["original"]["per_task"][t][m])
                                                          if results["original"]["per_task"][t][m] is not None else None)
                               for m in ("auroc", "auprc", "average_precision")} for t in tasks},
              "batch_size": args.batch_size, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_version": torch.__version__, "elapsed_seconds": time.time() - start,
              "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2}
    write_json(args.output / ("smoke_result.json" if args.smoke_batches else "result.json"), result)
    write_json(args.output / "progress.json", {"status": result["status"], "images_completed": len(image_paths),
                                               "elapsed_seconds": time.time() - start})
    print(json.dumps({"status": result["status"], "rows": len(metric_rows), "delta_pp": result["delta_pp"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--support-csv", type=Path)
    parser.add_argument("--evaluation-csv", "--test-csv", dest="evaluation_csv", type=Path)
    parser.add_argument("--full-test-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--mode", choices=["full-original", "paired-four-arm"], default="paired-four-arm")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--eta", type=float, default=.3)
    parser.add_argument("--method", choices=("closed_form", "reaggregation", "uncertainty", "sgtt", "cvsgtt"), default="closed_form",
                        help="Support query fit; reaggregation reruns the native FLaS branch")
    parser.add_argument("--covariance-shrinkage", type=float, default=0.2)
    parser.add_argument("--covariance-ridge", type=float, default=1e-3)
    parser.add_argument("--reliability-power", type=float, default=1.0)
    parser.add_argument("--reaggregation-max-iter", type=int, default=3)
    parser.add_argument("--reaggregation-query-tol", type=float, default=1e-4)
    parser.add_argument("--reaggregation-patience", type=int, default=1)
    parser.add_argument("--token-weight", type=float, default=.5)
    parser.add_argument("--crop-ratio", type=float, default=.8)
    parser.add_argument("--view-weight", type=float, default=.5)
    parser.add_argument("--gpu", type=int, choices=[4, 5, 6, 7])
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or args.shots < 1 or args.smoke_batches < 0
            or not math.isfinite(args.eta) or args.eta < 0
            or not math.isfinite(args.covariance_shrinkage) or not 0 <= args.covariance_shrinkage <= 1
            or not math.isfinite(args.covariance_ridge) or args.covariance_ridge < 0
            or not math.isfinite(args.reliability_power) or args.reliability_power < 0
            or args.reaggregation_max_iter < 1 or args.reaggregation_max_iter > 32
            or not math.isfinite(args.reaggregation_query_tol) or args.reaggregation_query_tol < 0
            or args.reaggregation_patience < 1 or args.reaggregation_patience > args.reaggregation_max_iter
            or not math.isfinite(args.token_weight) or args.token_weight < 0
            or not math.isfinite(args.crop_ratio) or not 0 < args.crop_ratio <= 1
            or not math.isfinite(args.view_weight) or not 0 <= args.view_weight <= 1):
        parser.error("Invalid batch size, workers, or shots")
    if args.method == "reaggregation" and args.eta <= 0:
        parser.error("Reaggregation requires a positive --eta")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for attr in ("catalog", "support_csv", "evaluation_csv", "full_test_csv", "checkpoint", "output"):
        if getattr(args, attr) is not None:
            setattr(args, attr, getattr(args, attr).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / p).exists() for p in ("result.json", "query_adapter.npz", "failure.json")):
        raise FileExistsError("Use a fresh output directory")
    try:
        catalog_arguments(args)
        support, evaluation, full, data_audit = audit(args)
        write_json(args.output / "data_audit.json", data_audit)
        if args.audit_only:
            print(json.dumps(data_audit, indent=2))
            return
        run(args, support, evaluation, full, data_audit)
    except BaseException as exc:
        write_json(args.output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
