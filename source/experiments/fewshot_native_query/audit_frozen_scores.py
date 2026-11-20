#!/usr/bin/env python3
"""Independent CPU verification of frozen scores and classification metrics.

Does not import fitting methods or the production metric function. Uses dense
KS pseudoinverses instead of the fitter's Woodbury solve, rank-sum AUROC, and
explicit unique-threshold PR curves. Never changes fitted parameters/results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(value):
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return np.divide(value, norm, out=np.zeros_like(value), where=norm > 1e-12)


def independent_metrics(labels, scores):
    y = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if set(np.unique(y)) != {0, 1}:
        return {key: None for key in ("auroc", "auprc", "average_precision")}
    positive, negative = y.sum(), len(y) - y.sum()
    auc = (rankdata(scores)[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    order = np.argsort(-scores, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
    true_positive = y[order].cumsum()[end]
    recall = np.r_[0.0, true_positive / positive]
    precision = np.r_[1.0, true_positive / (end + 1)]
    return {"auroc": float(auc), "auprc": float(np.trapz(precision, recall)),
            "average_precision": float(np.sum(np.diff(recall) * precision[1:]))}


def independent_score(name, support, test, fit):
    x = support["features"].astype(np.float64)
    tx = test["features"].astype(np.float64)
    y = support["labels"].astype(np.float64)
    z = test["logits"].astype(np.float64)
    params = fit.get("selected_parameters", {})
    alpha = float(params.get("alpha", 0))
    if name != "prototype_score_only" and alpha == 0:
        return z.copy()
    if name in ("prototype_score_only", "prototype_score_mix_oof"):
        pm = y.T @ x / y.sum(0)[:, None]
        nm = (1 - y).T @ x / (1 - y).sum(0)[:, None]
        residual = tx @ (unit(pm) - unit(nm)).T
    elif name == "tip_adapter_ovr_oof":
        affinity = np.clip(tx @ x.T, -1., 1.)
        kernel = np.exp(-params["beta"] * (1. - affinity))
        # Deliberately use explicit two-column one-hot caches per class.
        residual = np.stack([(kernel @ np.eye(2)[y[:, c].astype(int)]) @ np.asarray([-1., 1.])
                             for c in range(y.shape[1])], axis=1)
    elif name == "class_local_difference_score_oof":
        local = support["local"].astype(np.float64)
        pm = (local * y[..., None]).sum(0) / y.sum(0)[:, None]
        nm = (local * (1 - y)[..., None]).sum(0) / (1 - y).sum(0)[:, None]
        residual = np.einsum("ncd,cd->nc", test["local"].astype(np.float64), unit(pm - nm))
    elif name == "gda_ovr_ks_oof":
        parts = []
        for c in range(y.shape[1]):
            means = np.stack([x[y[:, c] == label].mean(0) for label in (0, 1)])
            centered = np.concatenate([x[y[:, c] == label] - means[label] for label in (0, 1)])
            cov = np.cov(centered.T)
            precision = x.shape[1] * np.linalg.pinv((len(x) - 1) * cov + cov.trace() * np.eye(x.shape[1]))
            weights = precision @ means.T
            bias = -np.einsum("nd,dc,nc->n", means, precision, means) / 2
            two_class = tx @ weights + bias
            parts.append(two_class[:, 1] - two_class[:, 0])
        residual = np.stack(parts, axis=1)
    else:
        raise ValueError(name)
    return residual if name == "prototype_score_only" else z + alpha * residual


def audit_unit(unit_dir, protocol_dir):
    evaluation = unit_dir / "evaluation"
    result = json.loads((evaluation / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Only completed full evaluations are audited")
    protocol_path = protocol_dir / (result["dataset"] + ".json")
    if sha(protocol_path) != result["protocol_sha256"]:
        raise ValueError("Protocol hash changed")
    protocol = json.loads(protocol_path.read_text())
    for filename, key in (("fit_info.json", "fit_info_sha256"), ("predictions.npz", "predictions_sha256")):
        if sha(evaluation / filename) != result[key]:
            raise ValueError(f"Evaluation artifact hash changed: {filename}")
    if sha(protocol["test_manifest"]["path"]) != protocol["test_manifest"]["sha256"]:
        raise ValueError("Frozen test labels changed")
    rows = json.loads(Path(protocol["test_manifest"]["path"]).read_text())["rows"]
    predictions = dict(np.load(evaluation / "predictions.npz", allow_pickle=False))
    if predictions["image_paths"].tolist() != [r["image"] for r in rows]:
        raise ValueError("Prediction/test-label order differs")
    if predictions["class_names"].tolist() != protocol["class_names"]:
        raise ValueError("Class order changed")
    labels = np.asarray([r["labels"] for r in rows])
    fit = json.loads((evaluation / "fit_info.json").read_text())
    cache = unit_dir / "cache"
    if sha(cache / "result.json") != result["cache_result_sha256"]:
        raise ValueError("Cache result changed")
    support = dict(np.load(cache / "support.npz", allow_pickle=False))
    test = dict(np.load(cache / "test.npz", allow_pickle=False))
    native = dict(np.load(cache / "native_predictions.npz", allow_pickle=False))
    if "labels" in test:
        raise ValueError("Test labels leaked into the frozen feature cache")
    np.testing.assert_array_equal(support["labels"], [r["labels"] for r in protocol["support_rows"]])
    coverage = np.zeros(len(support["labels"]), dtype=int)
    groups = np.asarray(protocol["support_groups"])
    for fold in protocol["folds"]:
        train, valid = fold["train"], fold["validation"]
        if set(groups[train]) & set(groups[valid]):
            raise ValueError("Support cross-validation group leakage")
        if set(train) | set(valid) != set(range(len(groups))) or set(train) & set(valid):
            raise ValueError("Invalid fold partition")
        coverage[valid] += 1
    np.testing.assert_array_equal(coverage, np.ones(len(groups)))
    methods, max_metric_error = {}, 0.0
    for name, recorded in result["methods"].items():
        values = predictions[name]
        if values.shape != labels.shape or not np.isfinite(values).all():
            raise ValueError("Nonfinite/misaligned score matrix")
        per_class = [independent_metrics(labels[:, c], values[:, c]) for c in range(labels.shape[1])]
        macro = {"macro_" + metric: float(np.mean([v[metric] for v in per_class if v[metric] is not None]))
                 for metric in ("auroc", "auprc", "average_precision")}
        differences = []
        for key, val in macro.items():
            differences.append(abs(val - recorded["metrics"][key]))
        for c, label in enumerate(protocol["class_names"]):
            for key, val in per_class[c].items():
                if val is not None:
                    differences.append(abs(val - recorded["metrics"]["per_class"][label][key]))
        max_error = max(differences, default=0.0)
        if max_error > 2e-12:
            raise ValueError(f"Independent metric mismatch for {name}: {max_error}")
        max_metric_error = max(max_metric_error, max_error)
        details = {"metrics": macro, "max_metric_absolute_error": max_error, "metrics_passed": True}
        if name in ("original", "query", "view", "query_view", "positive_query"):
            np.testing.assert_array_equal(values, native[name])
            details["score_check"] = "exactly equal to authenticated native control export"
        elif name != "simnl_feature_ovr":
            rebuilt = independent_score(name, support, test, fit[name])
            np.testing.assert_allclose(values, rebuilt, atol=1e-8, rtol=1e-9)
            details["score_check"] = "recomputed independently from full support and locked selected parameters"
            details["score_max_absolute_error"] = float(np.max(np.abs(values - rebuilt)))
        else:
            details["score_check"] = "saved prediction metrics verified; trained SimNL residual state not reconstructed"
        if name in fit and "candidates" in fit[name] and fit[name]["candidates"]:
            best = None
            for candidate in fit[name]["candidates"]:
                value = candidate["oof_macro_auprc"]
                if value is not None and (best is None or value > best["oof_macro_auprc"] + 1e-12):
                    best = candidate
            if best["parameters"] != fit[name]["selected_parameters"]:
                raise ValueError("Chosen parameters differ from locked OOF table")
            details["selection_record_passed"] = True
        methods[name] = details
    for name, details in methods.items():
        for metric in ("macro_auroc", "macro_auprc"):
            delta = 100 * (details["metrics"][metric] - methods["original"]["metrics"][metric])
            if abs(delta - result["methods"][name]["delta_pp"][metric]) > 2e-10:
                raise ValueError("Stored delta differs from independent paired metric delta")
    return {"status": "passed", "model": result["model"], "dataset": result["dataset"],
            "test_count": len(labels), "method_count": len(methods), "methods": methods,
            "max_metric_absolute_error": max_metric_error, "group_folds_passed": True,
            "result_path": str(evaluation / "result.json"), "result_sha256": sha(evaluation / "result.json"),
            "protocol_sha256": sha(protocol_path), "predictions_sha256": result["predictions_sha256"],
            "fit_info_sha256": result["fit_info_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path(__file__).with_name("protocol_v1"))
    args = parser.parse_args()
    units = sorted(p.parent.parent for p in args.run.glob("*/evaluation/result.json"))
    if not units:
        raise ValueError("No completed evaluation units")
    report = {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "metric_implementation": "rank-sum tie-aware AUROC; explicit unique-threshold trapezoidal PR area and step-weighted AP",
              "score_implementation": "independent NumPy prototype/Tip/local formulas and dense KS pseudoinverse",
              "limitations": ["SimNL residual training was not rerun; its saved scores and all metrics were audited."],
              "source_sha256": sha(__file__),
              "units": [audit_unit(unit, args.protocol_dir) for unit in units]}
    destination = args.run / "independent_metric_audit.json"
    if destination.exists():
        raise FileExistsError("Do not overwrite an existing independent audit")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "units": len(report["units"]), "output": str(destination)}))
