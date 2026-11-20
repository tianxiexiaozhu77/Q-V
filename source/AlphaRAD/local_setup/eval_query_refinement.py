#!/usr/bin/env python3
"""Fit support-only queries and compare frozen AlphaRAD on one fixed holdout."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"

from eval_pretrained import ASSETS, sha256, write_json

CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"
DEFAULT_BASELINE = ROOT / "runs/pretrained_20261021_122700"


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def audit_csv(path, expected_hash, classes, classification=True):
    from cxrclip.evaluator_utils import disease_name_mapping, disease_list_mapping
    if sha256(path) != expected_hash:
        raise ValueError(f"CSV hash changed: {path}")
    rows = read_rows(path)
    paths, labels = [], []
    for row in rows:
        image = row["image"]
        if image.startswith("["):
            image = ast.literal_eval(image)[0]
        if not Path(image).is_file():
            raise FileNotFoundError(image)
        paths.append(image)
        if classification:
            raw = row["label"]
            y = ast.literal_eval(raw) if raw.startswith("[") else [float(raw)]
            names = ast.literal_eval(row["class"]) if row["class"].startswith("[") else [row["class"]]
            mapped = [int(disease_name_mapping(c) in disease_list_mapping(names)) for c in classes]
            if y != mapped or len(y) != len(classes):
                raise ValueError(f"Label/class mismatch: {image}")
            labels.append(y)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Empty or duplicate image manifest")
    return rows, paths, labels


def classification_metrics(labels, probabilities):
    import numpy as np
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    per_class = []
    for i in range(labels.shape[1]):
        y, p = labels[:, i], probabilities[:, i]
        if len(np.unique(y)) < 2:
            per_class.append({"auroc": None, "average_precision": None, "auprc": None})
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        per_class.append({"auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "auprc": float(auc(recall, precision))})
    valid = [v for v in per_class if v["auroc"] is not None]
    return {"per_class": per_class, "valid_class_count": len(valid),
        **{"macro_" + k: float(np.mean([v[k] for v in valid])) if valid else None
           for k in ("auroc", "average_precision", "auprc")}}


def support_balanced_margin(logits, labels):
    """Return a support-only balanced positive-minus-negative margin.

    This is a diagnostic for route selection, never a test metric.  Keeping
    the computation here makes the AlphaRAD readout-only arm auditable: its
    q0-aggregated/q1-readout support margin is serialized next to the fitted
    adapter and can be consumed by the support-only selector without
    reconstructing image features or using evaluation labels.
    """
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[:, None]
    if labels.ndim == 1:
        labels = labels[:, None]
    if logits.shape != labels.shape:
        raise ValueError(f"support logits/labels shape mismatch: {logits.shape} vs {labels.shape}")
    margins = []
    for c in range(labels.shape[1]):
        valid = np.isfinite(labels[:, c]) & np.isfinite(logits[:, c])
        pos = valid & (labels[:, c] > 0.5)
        neg = valid & (labels[:, c] <= 0.5)
        if pos.any() and neg.any():
            margins.append(float(logits[pos, c].mean() - logits[neg, c].mean()))
    return float(np.mean(margins)) if margins else None


def original_helper_check(evaluator, matcher, batch, classes):
    """Compare real-image encoder/projection and both original helper outputs."""
    import numpy as np
    import torch
    image = evaluator.encode_image(batch["images"])
    prepared = matcher.prepare_vision_results(image)
    direct = matcher.encode_vision(batch["images"])
    for key in prepared:
        for actual, expected in zip(direct[key], prepared[key]):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    zero = matcher.match_prepared(direct, matcher.queries[0], return_maps=True)
    cls_raw = image["cls2_token_raw"] if image["cls2_token_raw"] is not None else image["cls_token_raw"]
    cls_normed = image["cls2_token"] if image["cls2_token"] is not None else image["cls_token"]
    inputs = (cls_raw, cls_normed, image["patch_tokens_raw"], image["patch_tokens"],
              batch["label_names"], classes)
    expected_probs, _ = evaluator._zeroshot_binary_dqn_helper(*inputs)
    expected_maps, _ = evaluator._zeroshot_pointinggame_dqn_helper(*inputs)
    probabilities = torch.sigmoid(zero["logits"]).cpu().numpy()
    maps = zero["grounding_map"].cpu().numpy()
    expected_probs = expected_probs.reshape(probabilities.shape)
    np.testing.assert_allclose(probabilities, expected_probs, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(maps, expected_maps, atol=2e-5, rtol=1e-5)
    return {"passed": True, "images": len(batch["images"]),
        "probability_max_abs_error": float(np.max(np.abs(probabilities - expected_probs))),
        "map_max_abs_error": float(np.max(np.abs(maps - expected_maps))),
        "probabilities_bitwise_equal": bool(np.array_equal(probabilities, expected_probs)),
        "maps_bitwise_equal": bool(np.array_equal(maps, expected_maps)),
        "map_token_count": maps.shape[-1]}


def prior_classification_check(root, dataset, paths, classes, labels, probabilities,
                              atol=2e-6, rtol=1e-5):
    import numpy as np
    prior = root / dataset / "predictions/eccv_mpnet_518" / dataset
    if not prior.is_dir():
        raise FileNotFoundError(f"Missing frozen baseline prediction cache: {prior}")
    original_paths = json.loads((prior / "image_paths.json").read_text())
    if json.loads((prior / "class_names.json").read_text()) != classes:
        raise ValueError("Prior baseline class ordering differs")
    lookup = {path: i for i, path in enumerate(original_paths)}
    if len(lookup) != len(original_paths):
        raise ValueError("Prior cache has duplicate paths")
    indices = [lookup[path] for path in paths]
    expected_labels = np.load(prior / "labels.npy", allow_pickle=False)[indices]
    expected = np.load(prior / "predictions.npy", allow_pickle=False)[indices].reshape(probabilities.shape)
    np.testing.assert_array_equal(expected_labels, labels)
    np.testing.assert_allclose(probabilities, expected, atol=atol, rtol=rtol)
    return {"passed": True, "matched_images": len(paths),
        "max_abs_error": float(np.max(np.abs(probabilities - expected))),
        "bitwise_equal": bool(np.array_equal(probabilities, expected)),
        "comparison": "identical paths and labels; old full-set predictions subsetted to current holdout"}


def run(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.evaluator_utils import (classification_score, gather_pointinggame_statistics,
        pointing_game_score, gather_seg_statistics, seg_score, get_grounding_type)
    from cxrclip.prompt import constants
    from query_refinement import (VERSION, REAGGREGATION_VERSION, ABLATION_VERSION, GATE_VERSION, UNCERTAINTY_VERSION,
        JACKKNIFE_VERSION, ECQA_VERSION, ECQT_VERSION, SEPCAL_VERSION, TDQA_VERSION, CGT_VERSION,
        fit_query_directions, fit_reaggregation_query_directions,
        fit_cached_multistep_query_directions, fit_reaggregation_only_query_directions,
        fit_support_only_gate,
        reaggregated_direction,
        fit_uncertainty_query_directions, fit_jackknife_stable_query_directions,
        fit_reaggregation_jackknife_query_directions,
        fit_support_selected_query_directions,
        fit_evidence_calibrated_query_directions,
        fit_reaggregation_evidence_query_directions,
        fit_separation_calibrated_query_directions,
        fit_token_discriminative_query_directions,
        fit_reaggregation_token_discriminative_query_directions,
        fit_contrastive_grounding_query_directions,
        QueryRefinementMatcher, original_queries_from_evaluator)

    started = time.time()
    torch.set_num_threads(4)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("No assigned CUDA GPU")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Official checkpoint identity mismatch")
    entry = json.loads(args.catalog.read_text())["datasets"][args.dataset]
    classes = entry["class_names"]
    if hasattr(constants, args.dataset.upper()):
        if list(getattr(constants, args.dataset.upper())) != classes:
            raise ValueError("Class order differs from author's constant")
    else:
        setattr(constants, args.dataset.upper(), classes)
    support_csv = Path(entry["support_csv_path"])
    support_rows, support_paths, support_labels = audit_csv(
        support_csv, entry["support_csv_sha256"], classes)
    test_rows, test_paths, test_labels = audit_csv(
        entry["csv_path"], entry["csv_sha256"], classes, entry["task"] == "classification")
    if len(test_rows) != entry["actual_count"] or len(support_rows) != entry["support_count"]:
        raise ValueError("Manifest count changed")
    if set(support_paths) & set(test_paths):
        raise ValueError("Support/evaluation image leakage")
    support_labels = np.asarray(support_labels, dtype=np.uint8)
    if np.any(support_labels.sum(0) == 0) or np.any(support_labels.sum(0) == len(support_labels)):
        raise ValueError("Every support class requires positive and negative examples")
    write_json(args.output / "data_audit.json", {"entry": entry,
        "support_count": len(support_rows), "evaluation_count": len(test_rows),
        "exact_image_path_overlap": 0, "support_labels": support_labels.tolist(),
        "support_csv_sha256": sha256(support_csv), "test_csv_sha256": sha256(entry["csv_path"])})
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
        "pin_memory": True, "shuffle": False, "drop_last": False}
    if args.workers:
        loader_options["prefetch_factor"] = 2
    support_key = args.dataset + "_support"
    data = {}
    for name, path in ((args.dataset, entry["csv_path"]), (support_key, support_csv)):
        data[name] = {"name": name, "data_type": "image_classification",
            "data_path": str(path), "n_class": len(classes), "normalize": None}
    config = {"base": {"seed": args.seed}, "test": {"checkpoint": [str(args.checkpoint)]},
        "data_test": data, "dataloader": {"test": loader_options}}
    write_json(args.output / "eval_config.json", config)
    evaluator = Evaluator(config, [str(args.checkpoint)], "test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    original, prompts = original_queries_from_evaluator(evaluator, classes)
    support_matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
        use_all_image_tokens=evaluator.use_all_image_tokens)
    support_features, support_head_features, support_vision, observed_support_paths = [], [], [], []
    support_evidence = []
    with torch.inference_mode():
        for batch in tqdm(evaluator.data_loader_dict[support_key], desc="Support features"):
            # Cache only frozen projected vision tokens on CPU.  The native
            # query-conditioned branch is then rerun for each fixed-point step.
            prepared = support_matcher.encode_vision(batch["images"])
            output = support_matcher.match_prepared(
                prepared, support_matcher.queries[0],
                return_maps=args.method in ("ecqa", "ecqt"), return_features=True)
            support_features.append(output["class_local"].cpu().numpy())
            support_head_features.append(output["class_local_heads"].cpu().numpy())
            if args.method in ("ecqa", "ecqt"):
                # Native FLaS grounding logits are class-specific spatial
                # evidence.  Their max-minus-mean contrast measures how
                # concentrated the frozen query is on a support image.
                grounding = output["grounding_map"]
                map_evidence = grounding.amax(dim=-1) - grounding.mean(dim=-1)
                if args.evidence_source == "logit":
                    evidence = output["logits"]
                elif args.evidence_source == "mixed":
                    evidence = map_evidence + output["logits"]
                else:
                    evidence = map_evidence
                support_evidence.append(evidence.cpu().numpy())
            support_vision.append({key: [value.cpu() for value in values]
                                   for key, values in prepared.items()})
            observed_support_paths.extend(batch["image_paths"])
    if observed_support_paths != support_paths:
        raise ValueError("Support dataloader order changed")
    features = np.concatenate(support_features)
    head_features = np.concatenate(support_head_features)
    if args.method in ("ecqa", "ecqt"):
        support_evidence = np.concatenate(support_evidence)
    else:
        support_evidence = None
    fit_started = time.perf_counter()
    if args.method == "cached_multistep":
        fit = fit_cached_multistep_query_directions(
            features, support_labels, original, eta=args.eta,
            max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience)
    elif args.method == "reaggregation_only":
        fit = fit_reaggregation_only_query_directions(
            support_vision, support_labels, original, support_matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience)
    elif args.method == "support_gate":
        fit = fit_support_only_gate(
            features, support_vision, support_labels, original, support_matcher,
            eta=args.eta, gamma_margin=args.gate_gamma_margin,
            gamma_kappa=args.gate_gamma_kappa,
            gamma_residual=args.gate_gamma_residual,
            max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience)
    elif args.method == "reaggregation":
        fit = fit_reaggregation_query_directions(
            support_vision, support_labels, original, support_matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience)
    elif args.method == "uncertainty":
        fit = fit_uncertainty_query_directions(
            features, support_labels, original, etas=(0., args.eta),
            covariance_shrinkage=args.covariance_shrinkage,
            ridge=args.covariance_ridge,
            reliability_power=args.reliability_power)
    elif args.method == "scqt":
        fit = fit_reaggregation_jackknife_query_directions(
            support_vision, support_labels, original, support_matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience,
            stability_power=args.jackknife_stability_power,
            kappa=args.jackknife_kappa, geometry=args.scqt_geometry)
    elif args.method == "ecqa":
        fit = fit_evidence_calibrated_query_directions(
            features, support_labels, original, support_evidence,
            etas=(0., args.eta), temperature=args.evidence_temperature,
            weight_floor=args.evidence_weight_floor)
    elif args.method == "sepcal":
        fit = fit_separation_calibrated_query_directions(
            features, support_labels, original, etas=(0., args.eta),
            separation_temperature=args.separation_temperature,
            reliability_floor=args.reliability_floor)
    elif args.method == "support_select":
        selection = fit_support_selected_query_directions(
            features, support_labels, original,
            eta_candidates=tuple(args.eta_candidates))
        # Use the selected per-class strengths in the native query-dependent
        # path.  The selector itself is leave-one-support-out; reaggregation
        # is refit once on all support images after selection.
        fit = fit_reaggregation_query_directions(
            support_vision, support_labels, original, support_matcher,
            eta=selection["selected_eta_per_class"],
            max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience)
        for key in ("selected_eta_per_class", "selected_eta_global", "eta_candidates",
                    "support_margin_curve", "support_margin_global"):
            fit[key] = selection[key]
        # The evaluator uses one nominal refined branch key.  The actual
        # query can carry a different support-selected eta for each class;
        # retain the requested nominal key for stable result serialization.
        fit["selected_eta_requested"] = float(args.eta)
        fit["etas"] = np.asarray((0.0, float(args.eta)), dtype=np.float64)
    elif args.method == "ecqt":
        fit = fit_reaggregation_evidence_query_directions(
            support_vision, support_labels, original, support_matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience,
            temperature=args.evidence_temperature,
            weight_floor=args.evidence_weight_floor)
    elif args.method == "tdqa_reagg":
        fit = fit_reaggregation_token_discriminative_query_directions(
            support_vision, support_labels, original, support_matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience,
            reliability_temperature=args.tdqa_temperature,
            reliability_power=args.tdqa_reliability_power)
    elif args.method == "tdqa":
        fit = fit_token_discriminative_query_directions(
            head_features, support_labels, original,
            num_heads=evaluator.model.num_radzero_heads,
            etas=(0., args.eta),
            reliability_temperature=args.tdqa_temperature,
            reliability_power=args.tdqa_reliability_power)
    elif args.method in ("cgt", "cct", "scct", "scct2", "scct3", "sgtt", "sgttm", "smgt", "cvsgtt"):
        fit = fit_contrastive_grounding_query_directions(
            support_vision, support_labels, original, support_matcher,
            eta=args.eta, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience,
            token_weight=(args.token_weight if args.method in ("cct", "scct", "scct2", "scct3", "sgtt", "sgttm", "smgt", "cvsgtt") else 1.0),
            adaptive_token_gate=(args.method in ("scct", "scct2", "scct3", "sgtt", "sgttm", "smgt", "cvsgtt")),
            geometry=("adaptive" if args.method in ("sgtt", "sgttm", "smgt", "cvsgtt") else
                      "tangent" if args.method in ("scct2", "scct3") else "linear"),
            global_jackknife=(args.method == "scct3"),
            support_margin_gate=(args.method in ("sgttm", "smgt")),
            support_margin_grid=(args.method == "smgt"),
            support_cv_grid=(args.method == "cvsgtt"))
    elif args.method in ("jackknife", "tangent_jackknife"):
        fit = fit_jackknife_stable_query_directions(
            features, support_labels, original, etas=(0., args.eta),
            stability_power=args.jackknife_stability_power,
            kappa=args.jackknife_kappa,
            geometry="tangent" if args.method == "tangent_jackknife" else "linear")
    else:
        fit = fit_query_directions(features, support_labels, original, etas=(0., args.eta))
    # One-shot and cached controls must still report the true native
    # representation gap.  Their fitting functions intentionally reuse the
    # q^(0) support cache, so evaluate d(q_hat) with a fresh native pass here.
    if args.method in ("closed_form", "cached_multistep", "reaggregation_only"):
        initial_direction = np.asarray(fit["directions"], dtype=np.float32)
        final_query = np.asarray(fit["queries"][-1], dtype=np.float32)
        final_direction = reaggregated_direction(
            support_vision, support_labels, final_query, support_matcher)
        gap = np.linalg.norm(initial_direction - final_direction, axis=-1)
        cosine = (initial_direction * final_direction).sum(-1)
        fit.update({
            "direction_initial": initial_direction,
            "direction_final": final_direction,
            "direction_gap_l2": gap,
            "direction_gap_l2_mean": float(gap.mean()),
            "direction_gap_l2_median": float(np.median(gap)),
            "direction_gap_l2_max": float(gap.max()),
            "direction_gap_cos": cosine,
            "direction_gap_cos_mean": float(cosine.mean()),
            "direction_gap_cos_min": float(cosine.min()),
            "native_reaggregation_passes": int(fit.get("native_reaggregation_passes", 0)) + 1,
        })
    fit.setdefault("fit_elapsed_seconds", float(time.perf_counter() - fit_started))
    fit.setdefault("native_reaggregation_passes", int(fit.get("reaggregation_iterations", 1 if args.method == "closed_form" else 0)))
    fit.setdefault("estimated_fit_flops", float(len(support_labels) * original.shape[0] * original.shape[1] * 2))
    np.save(args.output / "support_class_local.npy", features, allow_pickle=False)
    fit_arrays = {key: value for key, value in fit.items()
                  if isinstance(value, np.ndarray)}
    np.savez_compressed(args.output / "query_adapter.npz", **fit_arrays)
    adapter_metadata = {"version": (GATE_VERSION if args.method == "support_gate"
                                    else ABLATION_VERSION if args.method in ("cached_multistep", "reaggregation_only")
                                    else REAGGREGATION_VERSION if args.method == "reaggregation"
                                    else UNCERTAINTY_VERSION if args.method == "uncertainty"
                                    else SEPCAL_VERSION if args.method == "sepcal"
                                    else CGT_VERSION if args.method in ("cgt", "cct", "scct", "scct2", "scct3", "sgtt", "sgttm", "smgt", "cvsgtt")
                                    else ECQT_VERSION if args.method == "ecqt"
                                    else JACKKNIFE_VERSION if args.method == "support_select"
                                    else ECQA_VERSION if args.method == "ecqa"
                                    else TDQA_VERSION if args.method in ("tdqa", "tdqa_reagg")
                                    else JACKKNIFE_VERSION if args.method in ("jackknife", "tangent_jackknife", "scqt")
                                    else VERSION), "dataset": args.dataset, "eta": args.eta,
        "class_names": classes, "prompts": prompts, "checkpoint_sha256": CHECKPOINT_SHA,
        "query_file_sha256": sha256(args.output / "query_adapter.npz"),
        "support_csv_path": str(support_csv), "support_csv_sha256": sha256(support_csv),
        "test_csv_sha256": sha256(entry["csv_path"]), "support_count": len(support_rows),
        "positive_counts": fit["positive_counts"].tolist(), "negative_counts": fit["negative_counts"].tolist(),
        "query_dimension": original.shape[-1], "num_heads": evaluator.model.num_radzero_heads,
        "no_network_weight_updates": True, "test_labels_used_for_query_fit": False,
        "eta_selection": f"Fixed {args.eta:g} before AlphaRAD evaluation; default 0.3 transferred from prior exploratory RadZero result; no AlphaRAD test selection",
        "method": ("cached multi-step query adaptation with fixed support features"
                   if args.method == "cached_multistep" else
                   "native reaggregation-only support diagnostic with fixed original query"
                   if args.method == "reaggregation_only" else
                   "support-only pre-gate selecting one-shot or native fixed-point adaptation"
                   if args.method == "support_gate" else
                   "reaggregation-aware fixed-point query adaptation"
                   if args.method == "reaggregation" else
                   "support covariance-shrinkage Mahalanobis query adaptation"
                   if args.method == "uncertainty" else
                   "jackknife-stable support consensus query adaptation"
                   if args.method in ("jackknife", "tangent_jackknife", "scqt") else
                   "support-separation calibrated query transport"
                   if args.method == "sepcal" else
                   "contrastive grounding token transport"
                   if args.method in ("cgt", "cct", "scct", "scct2", "scct3", "sgtt", "sgttm", "smgt", "cvsgtt") else
                   "native-grounding evidence-calibrated support query adaptation"
                   if args.method in ("ecqa", "ecqt") else
                   "support-only leave-one-out eta selection with per-class query transport"
                   if args.method == "support_select" else
                   "native-head token-discriminative query adaptation"
                   if args.method in ("tdqa", "tdqa_reagg") else
                   "global unit context; positive-negative mean direction; global unit query residual; original FLaS"),
        "adaptation_mode": "one_shot" if args.method == "closed_form" else args.method,
        "ablation_mode": fit.get("ablation_mode") if args.method in ("cached_multistep", "reaggregation_only") else None,
        "uncertainty": ({key: value for key, value in fit.items()
                         if key in ("covariance_shrinkage", "ridge", "reliability_power")}
                        if args.method == "uncertainty" else None),
        "evidence": ({**{key: fit[key] for key in ("evidence_temperature", "evidence_weight_floor")},
                      "evidence_source": args.evidence_source}
                     if args.method in ("ecqa", "ecqt") else None),
        "separation": ({key: fit[key] for key in ("separation_temperature", "reliability_floor")}
                        if args.method == "sepcal" else None),
        "support_selection": ({key: (fit[key].tolist() if isinstance(fit[key], np.ndarray) else fit[key])
                               for key in ("selected_eta_per_class", "selected_eta_global", "eta_candidates",
                                            "support_margin_global", "selected_eta_requested") if key in fit}
                              if args.method == "support_select" else None),
        "tdqa": ({key: (fit[key].tolist() if isinstance(fit[key], np.ndarray) else fit[key])
                  for key in ("num_heads", "head_dim", "reliability_temperature",
                              "reliability_power", "head_gates", "head_utility")}
                 if args.method in ("tdqa", "tdqa_reagg") else None),
        "jackknife": ({"support_stability": fit["support_stability"].tolist(),
                       "stability_power": float(fit["stability_power"]),
                       "jackknife_kappa": float(fit["jackknife_kappa"]),
                       "geometry": fit["geometry"]}
                      if args.method in ("jackknife", "tangent_jackknife", "scqt") else None),
        "reaggregation": ({key: value for key, value in fit.items()
                           if key.startswith("reaggregation_")}
                          if args.method in ("reaggregation", "cached_multistep", "reaggregation_only", "support_gate", "tdqa_reagg", "ecqt") else None),
        "support_only_gate": ({key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                for key, value in fit.items()
                                if key.startswith("gate_")}
                               if args.method == "support_gate" else None),
        "source_sha256": {str(Path(__file__).resolve()): sha256(__file__),
            str(Path(__file__).with_name('query_refinement.py')): sha256(Path(__file__).with_name('query_refinement.py'))}}
    adapter_metadata.update({
        "seed": int(args.seed),
        "fit_elapsed_seconds": float(fit.get("fit_elapsed_seconds", 0.0)),
        "native_reaggregation_passes": int(fit.get("native_reaggregation_passes", 0)),
        "estimated_fit_flops": float(fit.get("estimated_fit_flops", 0.0)),
        "direction_gap_l2_mean": float(fit.get("direction_gap_l2_mean", 0.0)),
        "direction_gap_l2_median": float(fit.get("direction_gap_l2_median", 0.0)),
        "direction_gap_l2_max": float(fit.get("direction_gap_l2_max", 0.0)),
        "direction_gap_cos_mean": float(fit.get("direction_gap_cos_mean", 1.0)),
        "direction_gap_cos_min": float(fit.get("direction_gap_cos_min", 1.0)),
        "gate_name": fit.get("gate_name"),
        "gate_selected_full": fit.get("gate_selected_full"),
        "gate_selected_full_rate": fit.get("gate_selected_full_rate"),
        "gate_thresholds": fit.get("gate_thresholds"),
        "gate_delta_margin_mean": (float(np.mean(fit["gate_delta_margin"]))
                                    if "gate_delta_margin" in fit else None),
        "gate_local_map_ratio_mean": (float(np.mean(fit["gate_local_map_ratio"]))
                                       if "gate_local_map_ratio" in fit else None),
        "gate_direction_residual_l2_mean": (float(np.mean(fit["gate_direction_residual_l2"]))
                                             if "gate_direction_residual_l2" in fit else None),
        "gate_test_labels_used": fit.get("gate_test_labels_used"),
        "gate_pre_full_decision": fit.get("gate_pre_full_decision"),
        "gate_probe_native_passes": fit.get("gate_probe_native_passes"),
        "gate_total_native_passes": fit.get("gate_total_native_passes"),
        "gate_probe_elapsed_seconds": fit.get("gate_probe_elapsed_seconds"),
    })
    write_json(args.output / "query_adapter.json", adapter_metadata)
    matcher = QueryRefinementMatcher(evaluator.model, fit["queries"], fit["etas"],
        use_all_image_tokens=evaluator.use_all_image_tokens)

    # Support-only audit for the readout-only branch.  This margin is computed
    # from q^(0)-aggregated support contexts and q_final readout scores; it is
    # serialized for later support-only routing/calibration and never uses
    # held-out labels.  We evaluate the support contexts already cached above
    # rather than running another image encoder pass.
    def _to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(matcher.device, dtype=torch.float32, non_blocking=True)
        if isinstance(value, dict):
            return {key: _to_device(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(_to_device(item) for item in value)
        return value

    support_readout_logits, support_one_logits, support_full_logits = [], [], []
    with torch.inference_mode():
        for prepared in support_vision:
            prepared = _to_device(prepared)
            q0, qfinal = matcher.queries[0], matcher.queries[-1]
            # Reaggregation stores intermediate queries in the trajectory,
            # while ``fit["queries"]`` intentionally keeps only q0 and the
            # final q^(T).  Use q^(1) for the One-step support control when
            # that trajectory is available; closed_form has q1==qfinal.
            qone = qfinal
            trajectory = fit.get("query_trajectory")
            if isinstance(trajectory, np.ndarray) and trajectory.ndim >= 3 and trajectory.shape[0] >= 2:
                qone = torch.as_tensor(trajectory[1], dtype=torch.float32,
                                        device=matcher.device)
            support_readout_logits.append(matcher.match_prepared_readout_only(
                prepared, q0, qfinal)["logits"].cpu().numpy())
            support_one_logits.append(matcher.match_prepared(
                prepared, qone)["logits"].cpu().numpy())
            support_full_logits.append(matcher.match_prepared(
                prepared, qfinal)["logits"].cpu().numpy())
    support_readout_logits = np.concatenate(support_readout_logits, axis=0)
    support_one_logits = np.concatenate(support_one_logits, axis=0)
    support_full_logits = np.concatenate(support_full_logits, axis=0)
    def _support_margin(scores):
        margins = []
        for class_index in range(scores.shape[1]):
            positive = scores[support_labels[:, class_index] == 1, class_index]
            negative = scores[support_labels[:, class_index] == 0, class_index]
            margins.append(float(np.mean(positive) - np.mean(negative)))
        return np.asarray(margins, dtype=np.float64)
    support_readout_margin = _support_margin(support_readout_logits)
    support_one_margin = _support_margin(support_one_logits)
    support_full_margin = _support_margin(support_full_logits)
    support_readout_delta_vs_one = support_readout_margin - support_one_margin
    adapter_metadata.update({
        "support_readout_only_margin": support_readout_margin.tolist(),
        "support_readout_only_margin_mean": float(np.mean(support_readout_margin)),
        "support_one_step_margin": support_one_margin.tolist(),
        "support_full_margin": support_full_margin.tolist(),
        "support_readout_only_delta_vs_one": support_readout_delta_vs_one.tolist(),
        "support_readout_only_delta_vs_one_mean": float(np.mean(support_readout_delta_vs_one)),
        "support_readout_only_aggregation_query": "q0",
        "support_readout_only_readout_query": "q_final",
        "support_readout_only_test_labels_used": False,
    })
    write_json(args.output / "query_adapter.json", adapter_metadata)
    loader = evaluator.data_loader_dict[args.dataset]
    if args.limit_test:
        count = min(args.limit_test, len(test_paths))
        loader = DataLoader(Subset(loader.dataset, range(count)),
            collate_fn=loader.dataset.collate_fn, **loader_options)
        test_paths = test_paths[:count]
        test_labels = test_labels[:count]
    expected_paths = test_paths
    predictions = {0.: [], args.eta: []}
    # Diagnostic branch: q^(0) aggregates image tokens, while the final
    # support-adapted query is used only for readout.  The query is fit from
    # support labels before this loop; test labels never affect the branch.
    readout_predictions = []
    maps = {0.: [], args.eta: []}
    readout_maps = []

    def branch_output(outputs, target):
        """Resolve eta keys after query tensors have passed through float32.

        The matcher stores eta values originating from a torch tensor, so a
        requested Python ``0.3`` can be represented as ``0.3000000119``.
        Matching by a small numerical tolerance keeps the evaluation labels
        stable without changing the actual query or logits.
        """
        if target in outputs:
            return outputs[target]
        if not outputs:
            raise KeyError(f"No matcher output for eta={target}")
        key = min(outputs, key=lambda value: abs(float(value) - float(target)))
        if abs(float(key) - float(target)) > 1e-5:
            raise KeyError(f"No matcher output for eta={target}; available={list(outputs)}")
        return outputs[key]

    image_paths, label_names, boxes, masks = [], [], [], []
    first_batch_check = None
    require_maps = entry["task"] != "classification"
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{args.dataset}: baseline + eta={args.eta}"):
            if first_batch_check is None:
                first_batch_check = original_helper_check(evaluator, matcher, batch, classes)
                write_json(args.output / "zero_eta_check.json", first_batch_check)
            outputs = matcher(batch["images"], return_maps=require_maps or not image_paths)
            for eta in predictions:
                value = branch_output(outputs, eta)
                predictions[eta].append(value["logits"].cpu().numpy())
                if require_maps:
                    maps[eta].append(value["grounding_map"].cpu().numpy())
            if len(matcher.queries) >= 2:
                # Re-encode only for the diagnostic branch.  The existing
                # native/refined path remains untouched for exact audits.
                vision = matcher.encode_vision(batch["images"])
                readout_value = matcher.match_prepared_readout_only(
                    vision, matcher.queries[0], matcher.queries[-1],
                    return_maps=require_maps)
                readout_predictions.append(readout_value["logits"].cpu().numpy())
                if require_maps:
                    readout_maps.append(readout_value["grounding_map"].cpu().numpy())
            if not image_paths:
                np.savez_compressed(args.output / "similarity_map_examples.npz",
                    baseline=branch_output(outputs, 0.)["similarity_scores"].cpu().numpy(),
                    refined=branch_output(outputs, args.eta)["similarity_scores"].cpu().numpy(),
                    image_paths=np.asarray(batch["image_paths"]))
            image_paths.extend(batch["image_paths"])
            label_names.extend(batch["label_names"])
            boxes.extend(batch["boxes"])
            masks.extend(batch["masks"])
            write_json(args.output / "progress.json", {"status": "inference",
                "images_completed": len(image_paths), "images_total": len(expected_paths)})
    if image_paths != expected_paths:
        raise ValueError("Evaluation order/coverage changed")
    branch_results = {}
    for eta, label in ((0., "baseline"), (args.eta, "refined")):
        logits = np.concatenate(predictions[eta])
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite output")
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        np.save(args.output / f"{label}_logits.npy", logits, allow_pickle=False)
        np.save(args.output / f"{label}_probabilities.npy", probabilities, allow_pickle=False)
        if entry["task"] == "classification":
            metrics = classification_metrics(test_labels, probabilities)
            metrics["per_class"] = dict(zip(classes, metrics["per_class"]))
            if not args.limit_test:
                stats, counts = evaluator.gather_zeroshot_dqn_classification_statistics(
                    classes, probabilities[..., None], label_names)
                author = classification_score(stats, counts)
                np.testing.assert_allclose(metrics["macro_auroc"], author["AUROC(Avg)"], atol=1e-10, rtol=0)
                np.testing.assert_allclose(metrics["macro_average_precision"], author["PR_AUROC(Avg)"], atol=1e-10, rtol=0)
                metrics["author_metrics"] = author
            if eta == 0.:
                prior_check = prior_classification_check(args.baseline_root, args.dataset,
                    image_paths, classes, test_labels, probabilities,
                    atol=args.prior_atol, rtol=args.prior_rtol)
                write_json(args.output / "prior_baseline_check.json", prior_check)
        else:
            attention = np.concatenate(maps[eta])
            np.save(args.output / f"{label}_grounding_map.npy", attention, allow_pickle=False)
            if entry["task"] == "grounding":
                author = pointing_game_score(gather_pointinggame_statistics(
                    classes, attention, boxes, label_names, image_paths,
                    grounding_type=get_grounding_type(args.dataset)))
                metrics = {"pointing_game": float(author["Pointing_Game(Avg)"]), "author_metrics": author}
            else:
                author = seg_score(gather_seg_statistics(attention, image_paths,
                    masks, label_names, classes))
                metrics = {"dice": float(author["zeroshot_segmentation(Avg)"]), "author_metrics": author,
                    "protocol": "author positive-only best Dice; separately search 0:0.01:1 on evaluation labels"}
        branch_results[label] = metrics
        write_json(args.output / f"{label}_metrics.json", metrics)
    if readout_predictions:
        readout_logits = np.concatenate(readout_predictions)
        if not np.isfinite(readout_logits).all():
            raise ValueError("Nonfinite readout-only output")
        readout_probabilities = torch.sigmoid(torch.from_numpy(readout_logits)).numpy()
        np.save(args.output / "readout_only_logits.npy", readout_logits, allow_pickle=False)
        np.save(args.output / "readout_only_probabilities.npy", readout_probabilities, allow_pickle=False)
        if entry["task"] == "classification":
            readout_metrics = classification_metrics(test_labels, readout_probabilities)
            readout_metrics["per_class"] = dict(zip(classes, readout_metrics["per_class"]))
            if not args.limit_test:
                stats, counts = evaluator.gather_zeroshot_dqn_classification_statistics(
                    classes, readout_probabilities[..., None], label_names)
                author = classification_score(stats, counts)
                np.testing.assert_allclose(readout_metrics["macro_auroc"],
                    author["AUROC(Avg)"], atol=1e-10, rtol=0)
                np.testing.assert_allclose(readout_metrics["macro_average_precision"],
                    author["PR_AUROC(Avg)"], atol=1e-10, rtol=0)
                readout_metrics["author_metrics"] = author
        else:
            readout_attention = np.concatenate(readout_maps)
            np.save(args.output / "readout_only_grounding_map.npy", readout_attention, allow_pickle=False)
            if entry["task"] == "grounding":
                author = pointing_game_score(gather_pointinggame_statistics(
                    classes, readout_attention, boxes, label_names, image_paths,
                    grounding_type=get_grounding_type(args.dataset)))
                readout_metrics = {"pointing_game": float(author["Pointing_Game(Avg)"]),
                                   "author_metrics": author}
            else:
                author = seg_score(gather_seg_statistics(readout_attention, image_paths,
                    masks, label_names, classes))
                readout_metrics = {"dice": float(author["zeroshot_segmentation(Avg)"]),
                                   "author_metrics": author}
        branch_results["readout_only"] = readout_metrics
        write_json(args.output / "readout_only_metrics.json", readout_metrics)
    write_json(args.output / "image_paths.json", image_paths)
    if entry["task"] == "classification":
        np.save(args.output / "labels.npy", np.asarray(test_labels, dtype=np.uint8), allow_pickle=False)
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("A model parameter was unexpectedly trainable")
    keys = ("macro_auroc", "macro_average_precision", "macro_auprc", "pointing_game", "dice")
    delta = {key: 100 * (branch_results['refined'][key] - branch_results['baseline'][key])
             for key in keys if branch_results['baseline'].get(key) is not None}
    readout_delta = ({key: 100 * (branch_results['readout_only'][key] - branch_results['baseline'][key])
                     for key in keys if branch_results['readout_only'].get(key) is not None}
                     if "readout_only" in branch_results else {})
    result = {"status": "complete", "dataset": args.dataset, "task": entry["task"],
        "rows": len(image_paths), "support_count": len(support_paths), "eta": args.eta,
        "baseline": branch_results["baseline"], "refined": branch_results["refined"], "delta_pp": delta,
        "entry": entry, "checkpoint_sha256": CHECKPOINT_SHA, "ignored_checkpoint_keys": ignored,
        "zero_eta_check": first_batch_check, "is_smoke": bool(args.limit_test),
        "batch_size": args.batch_size, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "no_network_weight_updates": True, "elapsed_seconds": time.time() - started,
        "inference_elapsed_seconds": float(time.perf_counter() - inference_started),
        "seed": int(args.seed),
        "fit_elapsed_seconds": float(fit.get("fit_elapsed_seconds", 0.0)),
        "native_reaggregation_passes": int(fit.get("native_reaggregation_passes", 0)),
        "estimated_fit_flops": float(fit.get("estimated_fit_flops", 0.0)),
        "direction_gap_l2_mean": float(fit.get("direction_gap_l2_mean", 0.0)),
        "direction_gap_l2_median": float(fit.get("direction_gap_l2_median", 0.0)),
        "direction_gap_l2_max": float(fit.get("direction_gap_l2_max", 0.0)),
        "direction_gap_cos_mean": float(fit.get("direction_gap_cos_mean", 1.0)),
        "direction_gap_cos_min": float(fit.get("direction_gap_cos_min", 1.0)),
        "gate_name": fit.get("gate_name"),
        "gate_selected_full": fit.get("gate_selected_full"),
        "gate_selected_full_rate": fit.get("gate_selected_full_rate"),
        "gate_thresholds": fit.get("gate_thresholds"),
        "gate_delta_margin_mean": (float(np.mean(fit["gate_delta_margin"]))
                                    if "gate_delta_margin" in fit else None),
        "gate_local_map_ratio_mean": (float(np.mean(fit["gate_local_map_ratio"]))
                                       if "gate_local_map_ratio" in fit else None),
        "gate_direction_residual_l2_mean": (float(np.mean(fit["gate_direction_residual_l2"]))
                                             if "gate_direction_residual_l2" in fit else None),
        "gate_test_labels_used": fit.get("gate_test_labels_used"),
        "gate_pre_full_decision": fit.get("gate_pre_full_decision"),
        "gate_probe_native_passes": fit.get("gate_probe_native_passes"),
        "gate_total_native_passes": fit.get("gate_total_native_passes"),
        "gate_probe_elapsed_seconds": fit.get("gate_probe_elapsed_seconds"),
        "support_readout_only_margin": support_readout_margin.tolist(),
        "support_readout_only_margin_mean": float(np.mean(support_readout_margin)),
        "support_one_step_margin": support_one_margin.tolist(),
        "support_full_margin": support_full_margin.tolist(),
        "support_readout_only_delta_vs_one": support_readout_delta_vs_one.tolist(),
        "support_readout_only_delta_vs_one_mean": float(np.mean(support_readout_delta_vs_one)),
        "support_readout_only_test_labels_used": False,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2}
    if "readout_only" in branch_results:
        result["readout_only"] = branch_results["readout_only"]
        result["readout_only_delta_pp"] = readout_delta
        result["readout_only_definition"] = {
            "aggregation_query": "q0",
            "readout_query": "q_final",
            "query_fit_support_only": True,
            "test_labels_used_for_query_fit": False,
        }
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", {"status": "complete", "images_completed": len(image_paths)})
    print(json.dumps({"dataset": args.dataset, "baseline": result["baseline"],
        "refined": result["refined"], "readout_only": result.get("readout_only"),
        "delta_pp": delta, "readout_only_delta_pp": readout_delta},
        ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar")
    p.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--eta", type=float, default=.3)
    p.add_argument("--seed", type=int, default=1234,
                   help="Deterministic evaluator seed; vary for repeated support episodes.")
    p.add_argument("--method", choices=("closed_form", "reaggregation", "cached_multistep", "reaggregation_only", "support_gate", "uncertainty", "jackknife", "tangent_jackknife", "scqt", "ecqa", "ecqt", "sepcal", "tdqa", "tdqa_reagg", "support_select", "cgt", "cct", "scct", "scct2", "scct3", "sgtt", "sgttm", "smgt", "cvsgtt"), default="closed_form",
                   help="Support query fitting rule; uncertainty whitens with a shrinkage covariance")
    p.add_argument("--reaggregation-max-iter", type=int, default=3)
    p.add_argument("--reaggregation-query-tol", type=float, default=1e-4)
    p.add_argument("--reaggregation-patience", type=int, default=1)
    p.add_argument("--gate-gamma-margin", type=float, default=0.0,
                   help="Frozen support-only probe margin threshold for support_gate")
    p.add_argument("--gate-gamma-kappa", type=float, default=1.0,
                   help="Frozen support-only local map ratio threshold for support_gate")
    p.add_argument("--gate-gamma-residual", type=float, default=0.0,
                   help="Frozen support-only direction residual threshold for support_gate")
    p.add_argument("--covariance-shrinkage", type=float, default=0.2,
                   help="Shrinkage weight toward the isotropic support covariance target")
    p.add_argument("--covariance-ridge", type=float, default=1e-3,
                   help="Ridge multiplier on the covariance trace target")
    p.add_argument("--reliability-power", type=float, default=1.0,
                   help="Support reliability exponent; 1 is the uncertainty-aware rule")
    p.add_argument("--jackknife-stability-power", type=float, default=1.0,
                   help="Exponent for the leave-one-support-out directional stability gate")
    p.add_argument("--jackknife-kappa", type=float, default=8.0,
                   help="Consensus concentration for jackknife directions")
    p.add_argument("--scqt-geometry", choices=("linear", "tangent"), default="tangent")
    p.add_argument("--evidence-temperature", type=float, default=1.0,
                   help="Sigmoid temperature for native grounding evidence agreement")
    p.add_argument("--evidence-weight-floor", type=float, default=0.25,
                   help="Minimum fraction of uniform support weight for ECQA")
    p.add_argument("--evidence-source", choices=("grounding", "logit", "mixed"), default="grounding",
                   help="Native support evidence used by ECQA/ECQT")
    p.add_argument("--separation-temperature", type=float, default=1.0)
    p.add_argument("--reliability-floor", type=float, default=0.25)
    p.add_argument("--token-weight", type=float, default=0.2,
                   help="Token-contrast weight for cross-scale CCT")
    p.add_argument("--eta-candidates", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.3),
                   help="Support-only leave-one-out eta candidates for support_select")
    p.add_argument("--tdqa-temperature", type=float, default=1.0,
                   help="Softmax temperature for native-head TDQA utility gates")
    p.add_argument("--tdqa-reliability-power", type=float, default=1.0,
                   help="Power applied to TDQA head gates before mean-one renormalization")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-test", type=int, default=0, help="Smoke test only; never a full result")
    p.add_argument("--prior-atol", type=float, default=2e-6,
                   help="Tolerance for comparing eta=0 logits with the archived baseline")
    p.add_argument("--prior-rtol", type=float, default=1e-5,
                   help="Relative tolerance for the archived eta=0 baseline check")
    args = p.parse_args()
    if (not math.isfinite(args.eta) or args.eta <= 0 or args.batch_size < 1
            or args.workers < 0 or args.limit_test < 0
            or args.reaggregation_max_iter < 1 or args.reaggregation_max_iter > 32
            or not math.isfinite(args.reaggregation_query_tol) or args.reaggregation_query_tol < 0
            or args.reaggregation_patience < 1
            or args.reaggregation_patience > args.reaggregation_max_iter
            or not math.isfinite(args.gate_gamma_margin)
            or not math.isfinite(args.gate_gamma_kappa) or args.gate_gamma_kappa < 0
            or not math.isfinite(args.gate_gamma_residual) or args.gate_gamma_residual < 0
            or not math.isfinite(args.prior_atol) or args.prior_atol < 0
            or not math.isfinite(args.prior_rtol) or args.prior_rtol < 0
            or not math.isfinite(args.token_weight) or args.token_weight < 0
            or not math.isfinite(args.covariance_shrinkage) or not 0 <= args.covariance_shrinkage <= 1
            or not math.isfinite(args.covariance_ridge) or args.covariance_ridge < 0
            or not math.isfinite(args.reliability_power) or args.reliability_power < 0
            or not math.isfinite(args.jackknife_stability_power) or args.jackknife_stability_power < 0
            or not math.isfinite(args.jackknife_kappa) or args.jackknife_kappa < 0):
        p.error("Invalid eta, batch size, workers, smoke limit, or reaggregation settings")
    if (not math.isfinite(args.evidence_temperature) or args.evidence_temperature <= 0
            or not math.isfinite(args.evidence_weight_floor)
            or not 0 < args.evidence_weight_floor <= 1
            or not args.eta_candidates or any(not math.isfinite(x) or x < 0 for x in args.eta_candidates)
            or len(set(args.eta_candidates)) != len(args.eta_candidates) or 0.0 not in args.eta_candidates):
        p.error("Invalid evidence calibration settings")
    if (not math.isfinite(args.tdqa_temperature) or args.tdqa_temperature <= 0
            or not math.isfinite(args.tdqa_reliability_power)
            or args.tdqa_reliability_power < 0):
        p.error("Invalid TDQA reliability settings")
    if (not math.isfinite(args.separation_temperature) or args.separation_temperature <= 0
            or not math.isfinite(args.reliability_floor)
            or not 0 < args.reliability_floor <= 1):
        p.error("Invalid support separation calibration settings")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists() or (args.output / "query_adapter.npz").exists():
        raise FileExistsError("Use a fresh worker output; existing results/queries are preserved")
    try:
        run(args)
    except BaseException as exc:
        write_json(args.output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
