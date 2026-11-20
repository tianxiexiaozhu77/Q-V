#!/usr/bin/env python3
"""Weighted-BCE linear score residuals over frozen RadZero features.

No RadZero weights change. This is a class-specific linear score adaptation,
not the original I2CL mechanism, and original similarity maps do not explain
its new scores. All choices use grouped support OOF scores before test labels
are loaded; overnight comparison between families remains exploratory.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from batch_metrics import compute_metrics
from overnight_cache_methods import (
    DeadlineReached, array_digest, cache_view, check_deadline, class_feature,
    evaluate_test_only, fingerprint, metric_delta, metric_score,
    normalized_features, parse_deadline, sha256, write_json,
)

VERSION = "radzero-frozen-linear-score-residual-group5fold-v1"
FEATURES = ("class_local", "dino_cls", "aligned_cls")


def make_group_folds(groups, count=5, seed=42):
    """Deterministic row-balanced folds; related images always stay together."""
    groups = np.asarray(groups, dtype=str)
    if groups.ndim != 1 or not len(groups) or np.any(groups == ""):
        raise ValueError("Expected nonempty support groups")
    unique, sizes = np.unique(groups, return_counts=True)
    if len(unique) < 2:
        raise ValueError("At least two support groups are needed")
    count = min(count, len(unique))
    if count < 2:
        raise ValueError("At least two folds are needed")
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(size=len(unique))
    order = sorted(range(len(unique)), key=lambda i: (-sizes[i], jitter[i]))
    load = np.zeros(count, dtype=int)
    assignments = {}
    for i in order:
        eligible = np.flatnonzero(load == load.min())
        fold = int(rng.choice(eligible))
        assignments[unique[i]] = fold
        load[fold] += sizes[i]
    folds = []
    for fold in range(count):
        held = np.asarray([assignments[group] == fold for group in groups])
        if not held.any() or held.all():
            raise RuntimeError("Empty grouped train/validation fold")
        folds.append({"fold": fold, "fit_indices": np.flatnonzero(~held).tolist(),
                      "held_indices": np.flatnonzero(held).tolist(),
                      "held_group_sha256": [hashlib.sha256(group.encode()).hexdigest()
                                            for group in sorted(set(groups[held]))]})
    return folds


def feature_means(features, classes):
    """Only fit-subset features may be passed by the caller."""
    if features.ndim == 2:
        return np.repeat(features.mean(axis=0, keepdims=True), classes, axis=0)
    return features.mean(axis=0)


def centered_class_features(features, means):
    if features.ndim == 2:
        return features[:, None, :] - means[None, :, :]
    return features - means[None, :, :]


def weighted_ranking_objective(logits, labels, weights, valid, coefficient_weights,
                               l2, ranking):
    """Averages each valid class equally; fold-missing classes contribute zero."""
    count = valid.sum().clamp(min=1)
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=weights, reduction="none")
    classification = (bce * valid.unsqueeze(0)).sum() / (len(labels) * count)
    penalty = (coefficient_weights.square().sum(dim=1) * valid).sum() / count
    rank = logits.sum() * 0.0
    if ranking:
        terms = []
        for c in range(labels.shape[1]):
            if bool(valid[c]):
                positives = logits[labels[:, c] == 1, c]
                negatives = logits[labels[:, c] == 0, c]
                terms.append(F.softplus(-(positives[:, None] - negatives[None, :])).mean())
        if terms:
            rank = torch.stack(terms).mean()
    total = classification + l2 * penalty + ranking * rank
    return total, {"weighted_bce": classification, "mean_class_weight_l2": penalty,
                   "pairwise_ranking": rank}


def fit_linear(features, labels, baseline_logits, *, l2=.1, ranking=0.,
               epochs=100, learning_rate=.01, use_bias=True, deadline=None):
    """Fit from support only. All statistics are recomputed for this subset."""
    labels = np.asarray(labels)
    baseline_logits = np.asarray(baseline_logits)
    if labels.ndim != 2 or baseline_logits.shape != labels.shape:
        raise ValueError("Labels and baseline must have equal [N,C] shape")
    if len(features) != len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Invalid training labels/features")
    if not np.isfinite(features).all() or not np.isfinite(baseline_logits).all():
        raise ValueError("Nonfinite fit inputs")
    if l2 < 0 or ranking < 0 or not math.isfinite(l2 + ranking):
        raise ValueError("Regularization strengths must be finite/nonnegative")
    if epochs < 0 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Invalid training schedule")
    classes, dimension = labels.shape[1], features.shape[-1]
    means = feature_means(features, classes)
    centered = centered_class_features(features, means)
    positive = labels.sum(axis=0).astype(int)
    negative = len(labels) - positive
    valid = (positive > 0) & (negative > 0)
    pos_weight = np.divide(negative, positive, out=np.ones(classes), where=positive > 0)
    x = torch.tensor(centered, dtype=torch.float32, device="cpu")
    y = torch.tensor(labels, dtype=torch.float32, device="cpu")
    base = torch.tensor(baseline_logits, dtype=torch.float32, device="cpu")
    mask = torch.tensor(valid, dtype=torch.float32, device="cpu")
    weight = torch.tensor(pos_weight, dtype=torch.float32, device="cpu")
    parameters = torch.nn.Parameter(torch.zeros(classes, dimension, device="cpu"))
    bias = torch.nn.Parameter(torch.zeros(classes, device="cpu"), requires_grad=use_bias)
    optimizer = torch.optim.Adam([parameters, bias] if use_bias else [parameters],
                                 lr=learning_rate)
    history = []
    for epoch in range(epochs + 1):
        check_deadline(deadline)
        logits = base + ((x * parameters.unsqueeze(0)).sum(dim=-1) + bias) * mask
        loss, components = weighted_ranking_objective(
            logits, y, weight, mask, parameters, l2, ranking)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite linear-adaptation objective")
        if epoch == 0 or epoch == epochs or epoch % 10 == 0:
            history.append({"completed_optimizer_steps": epoch, "total_loss": float(loss.detach()),
                            **{key: float(value.detach()) for key, value in components.items()}})
        if epoch == epochs:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():
            raise RuntimeError("Invalid linear-coefficient gradients")
        optimizer.step()
    result = {
        "weight": parameters.detach().numpy().astype(np.float64),
        "bias": bias.detach().numpy().astype(np.float64),
        "means": means, "valid": valid, "positive_counts": positive,
        "negative_counts": negative, "pos_weight": pos_weight,
        "history": history, "support_count": len(labels), "feature_ndim": features.ndim,
        "epochs": epochs, "learning_rate": learning_rate, "l2": l2, "ranking": ranking,
        "use_bias": use_bias,
    }
    predicted = baseline_logits + predict_linear_residual(result, features)
    np.testing.assert_allclose(predicted, logits.detach().numpy(), atol=3e-5, rtol=1e-5)
    if not np.isfinite(result["weight"]).all() or not np.isfinite(result["bias"]).all():
        raise RuntimeError("Nonfinite linear-adaptation state")
    return result


def predict_linear_residual(state, features):
    centered = centered_class_features(features, state["means"])
    return ((centered * state["weight"][None]).sum(axis=-1) + state["bias"]) * state["valid"]


def grouped_oof(features, labels, baseline, folds, **fit_kwargs):
    residual = np.zeros_like(baseline, dtype=float)
    audits = []
    for fold in folds:
        held, fit = fold["held_indices"], fold["fit_indices"]
        state = fit_linear(features[fit], labels[fit], baseline[fit], **fit_kwargs)
        residual[held] = predict_linear_residual(state, features[held])
        audits.append({
            "fold": fold["fold"], "fit_positive_counts": state["positive_counts"].tolist(),
            "fit_negative_counts": state["negative_counts"].tolist(),
            "pos_weight": state["pos_weight"].tolist(),
            "zero_residual_classes": np.flatnonzero(~state["valid"]).tolist(),
            "fit_feature_means_sha256": array_digest(state["means"]),
            "fit_labels_sha256": array_digest(labels[fit]), "history": state["history"],
        })
    return residual, audits


def select_by_feature(records, top_k=3):
    best = {}
    for record in records:
        feature = record["config"]["feature"]
        # Ties favor the smaller adaptation magnitude, then stronger L2.
        key = (record["selection_score"], -record["config"]["strength"], record["config"]["l2"])
        if feature not in best or key > best[feature][0]:
            best[feature] = (key, record)
    return sorted((entry[1] for entry in best.values()),
                  key=lambda row: (-row["selection_score"], row["candidate_id"]))[:top_k]


class FrozenCache:
    def __init__(self, path):
        self.path, self.manifest = cache_view(path)
        self.root = self.path.parent
        self.dataset = self.manifest["dataset"]
        self.split_dir = Path(self.manifest["split_dir"])
        self.metadata = {}
        for split in ("support", "test"):
            metadata_path = self.root / f"{split}_metadata.json"
            if sha256(metadata_path) != self.manifest["auxiliary_files"][metadata_path.name]["sha256"]:
                raise ValueError("Cache metadata hash mismatch")
            metadata = json.loads(metadata_path.read_text())
            if "labels" in metadata or fingerprint(metadata) != self.manifest[f"{split}_identity"]:
                raise ValueError("Cache metadata identity mismatch")
            metadata["csv_sha256"] = self.manifest["split_sha256"][
                str(self.split_dir / self.dataset / f"{split}.csv")]
            self.metadata[split] = metadata

    def array(self, split, feature):
        names = {"class_local": ("class_local", None),
                 "dino_cls": ("global_dino", 0),
                 "aligned_cls": ("global_aligned", 0),
                 "baseline_logits": ("base_logits", None)}
        key, index = names[feature]
        record = self.manifest["splits"][split]["files"][key]
        path = self.root / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Frozen feature array hash mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
            raise ValueError("Frozen feature shape/dtype mismatch")
        if index is not None:
            if values.ndim != 3 or values.shape[1] != 2:
                raise ValueError("Invalid [CLS,patchmean] global feature axis")
            return values[:, index]
        return values

    def support(self):
        from i2cl_data import load_spec
        support = load_spec(self.split_dir, self.dataset, "support")
        if sha256(self.split_dir / self.dataset / "support.csv") != self.metadata["support"]["csv_sha256"]:
            raise ValueError("Support split hash differs from cached features")
        for key in ("image_paths", "groups", "class_names", "prompts"):
            if support[key] != self.metadata["support"][key]:
                raise ValueError(f"Support {key} mismatch")
        for key in ("image_paths", "groups"):
            if set(support[key]) & set(self.metadata["test"][key]):
                raise ValueError(f"Support/test overlap in {key}")
        label_path = self.root / "support_labels.npy"
        if sha256(label_path) != self.manifest["auxiliary_files"][label_path.name]["sha256"]:
            raise ValueError("Support-label cache hash mismatch")
        if not np.array_equal(support["labels"], np.load(label_path, allow_pickle=False)):
            raise ValueError("Cached and verified support labels differ")
        return support


def save_state(path, state, support_sha):
    np.savez(path, weight=state["weight"], bias=state["bias"], means=state["means"],
             valid=state["valid"], positive_counts=state["positive_counts"],
             negative_counts=state["negative_counts"], pos_weight=state["pos_weight"],
             support_sha256=np.asarray(support_sha))


def run(args):
    cache = FrozenCache(args.cache)
    if args.dataset and args.dataset != cache.dataset:
        raise ValueError("Dataset does not match frozen cache")
    support = cache.support()
    labels = np.asarray(support["labels"], dtype=np.int8)
    baseline = np.asarray(cache.array("support", "baseline_logits"), dtype=np.float64)
    classes = labels.shape[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    folds = make_group_folds(support["groups"], args.folds, args.seed)
    write_json(output / "folds.json", folds)
    write_json(output / "run_config.json", {
        "version": VERSION, "arguments": vars(args), "dataset": cache.dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(__file__),
        "cache_helper_source_sha256": sha256(Path(__file__).with_name("overnight_cache_methods.py")),
        "cache_manifest": str(cache.path), "cache_manifest_sha256": sha256(cache.path),
        "support_sha256": cache.metadata["support"]["csv_sha256"],
        "test_sha256": cache.metadata["test"]["csv_sha256"],
        "class_names": support["class_names"],
        "objective": "per-class support-weighted BCE + L2*mean_c(||W_c||²) + ranking*mean_c(pairwise softplus)",
        "normalization": "Per-image L2 normalization, then means from fold fit support only",
        "initialization": "W=0, bias=0; baseline logits are fixed offsets",
        "bias_caveat": "A constant bias alone cannot change within-class AUROC/AUPRC rankings",
        "test_labels_used_in_fitting_or_selection": False,
        "exploratory": True, "adapted_maps_available": False,
    })
    write_json(output / "status.json", {"status": "support_group_cv"})
    records, audits, features = [], {}, {}
    support_spec = {"name": cache.dataset, "labels": labels,
                    "class_names": support["class_names"]}
    baseline_metrics = compute_metrics(support_spec, baseline)
    fit_common = {"epochs": args.epochs, "learning_rate": args.learning_rate,
                  "use_bias": args.use_bias, "deadline": args.deadline}
    try:
        for feature in args.features:
            check_deadline(args.deadline)
            x = normalized_features(cache.array("support", feature), classes)
            features[feature] = x
            for l2 in args.l2s:
                for ranking in args.rankings:
                    check_deadline(args.deadline)
                    key = f"{feature}_l2{l2:g}_rank{ranking:g}"
                    residual, audit = grouped_oof(
                        x, labels, baseline, folds, l2=l2, ranking=ranking, **fit_common)
                    audits[key] = audit
                    np.save(output / f"oof_residual_{key}.npy", residual, allow_pickle=False)
                    for strength in args.strengths:
                        metrics = compute_metrics(support_spec, baseline + strength * residual)
                        records.append({
                            "candidate_id": f"{key}_strength{strength:g}",
                            "config": {"feature": feature, "l2": l2, "ranking": ranking,
                                       "strength": strength, "method": "linear_bce_score_residual"},
                            "support_oof_metrics": metrics,
                            "support_oof_delta": metric_delta(baseline_metrics, metrics),
                            "selection_score": metric_score(metrics),
                            "fold_audit_key": key,
                        })
                    write_json(output / "candidates.json", records)
                    write_json(output / "fold_audits.json", audits)
                    print(f"{cache.dataset}: linear OOF {key} complete", flush=True)
        selected = select_by_feature(records, args.top_k)
        selection = {
            "selected_before_test_labels_loaded": True,
            "criterion": "mean support grouped-OOF macro AUROC and AUPRC",
            "fold_count": len(folds), "seed": args.seed,
            "baseline_support_metrics": baseline_metrics,
            "support_labels_sha256": array_digest(labels), "selected": selected,
        }
        write_json(output / "selection.json", selection)
        selection_hash = sha256(output / "selection.json")
        write_json(output / "status.json", {"status": "selected_test_prediction",
                                           "selection_sha256": selection_hash})
        base_test = np.asarray(cache.array("test", "baseline_logits"), dtype=np.float64)
        results = []
        for record in selected:
            check_deadline(args.deadline)
            config = record["config"]
            state = fit_linear(features[config["feature"]], labels, baseline,
                               l2=config["l2"], ranking=config["ranking"], **fit_common)
            destination = output / record["candidate_id"]
            destination.mkdir()
            save_state(destination / "linear_residual.npz", state,
                       cache.metadata["support"]["csv_sha256"])
            write_json(destination / "training.json", {
                key: state[key] if not isinstance(state[key], np.ndarray) else state[key].tolist()
                for key in ("history", "positive_counts", "negative_counts", "pos_weight",
                            "valid", "epochs", "learning_rate", "l2", "ranking", "use_bias")})
            query = cache.array("test", config["feature"])
            logits = np.empty_like(base_test)
            for first in range(0, len(query), args.chunk_size):
                check_deadline(args.deadline)
                last = min(len(query), first + args.chunk_size)
                x = normalized_features(query[first:last], classes)
                logits[first:last] = base_test[first:last] + config["strength"] * predict_linear_residual(state, x)
            np.save(destination / "predictions.npy", logits, allow_pickle=False)
            if sha256(output / "selection.json") != selection_hash:
                raise RuntimeError("Selection changed after held-out feature reading")
            check_deadline(args.deadline)
            result = evaluate_test_only(cache.split_dir, cache.dataset, logits, base_test,
                                        cache.metadata["test"], destination)
            result.update(record)
            result.update(
                selection_sha256=selection_hash,
                prediction_sha256=sha256(destination / "predictions.npy"),
                model_sha256=sha256(destination / "linear_residual.npz"))
            write_json(destination / "result.json", result)
            results.append(result)
            write_json(output / "summary.json", {
                "dataset": cache.dataset, "version": VERSION, "exploratory": True,
                "selection": selection, "results": results,
                "test_labels_used_in_fitting_or_selection": False,
            })
            print(f"{cache.dataset}: {record['candidate_id']} "
                  f"AUROC={result['metrics']['macro_auroc']:.6f} "
                  f"AUPRC={result['metrics']['macro_auprc']:.6f}", flush=True)
        write_json(output / "status.json", {"status": "complete", "selected_count": len(results),
                                           "completed_at": datetime.now(timezone.utc).isoformat()})
    except DeadlineReached as error:
        write_json(output / "status.json", {"status": "deadline", "reason": str(error)})
        raise
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "reason": repr(error)})
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--l2s", nargs="+", type=float, default=[.01, .1, 1.])
    parser.add_argument("--rankings", nargs="+", type=float, default=[0., .1])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0., .1, .3, 1.])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--no-bias", dest="use_bias", action="store_false", default=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deadline", type=parse_deadline)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or args.epochs < 1 or args.folds < 2:
        parser.error("threads must be 1..4, epochs positive, folds >=2")
    if args.top_k < 1 or args.chunk_size < 1 or args.learning_rate <= 0:
        parser.error("top-k, chunk-size and learning-rate must be positive")
    if any(not math.isfinite(value) or value < 0
           for value in args.l2s + args.rankings + args.strengths):
        parser.error("All regularization and score strengths must be finite/nonnegative")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    with threadpool_limits(limits=args.threads):
        try:
            check_deadline(args.deadline)
            run(args)
        except DeadlineReached as error:
            print(str(error), flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
