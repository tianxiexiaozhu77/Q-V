"""Support-only Project and Mix / L2 logistic comparators on frozen features.

Project and Mix is independently implemented from arXiv:2603.24528v2,
equations (5), (8), (9), (10), with the 99% SVD energy rule in section 6.
The author's pinned TAMP repository has no implementation as of this audit.
This is an explicit binary-task adaptation, not an official-code reproduction.
No function accepts test labels. Fit once on identity support images, then
predict concatenated identity and center-view features with the same fit.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass

import numpy as np

VERSION = "paper-equations-binary-support-fivefold-v1"
PROJECT_MIX_LAMBDAS = tuple(float(x) for x in np.linspace(0.0, 1.0, 10))
PROJECT_MIX_GAMMAS = tuple(float(x) for x in np.linspace(0.0, 1.0, 10))
TEXT_ENERGY_THRESHOLD = 0.99
LINEAR_PROBE_CS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LINEAR_PROBE_MAX_ITER = 2000
LINEAR_PROBE_TOL = 1e-8
DEFAULT_METHODS = ("project_mix", "linear_probe")


def algorithm_plan():
    """JSON-safe, result-independent settings to pin before evaluation."""
    return {
        "version": VERSION,
        "selection": {
            "folds": 5,
            "unit": "support patient/group; exact held-out complement",
            "objective": "pooled out-of-fold macro trapezoid AUPRC",
            "tie_break": "first grid candidate with exactly maximal metric",
            "support_view": "identity only",
            "test_view_rule": "same fitted model; identity/center margins averaged equally by caller",
            "test_labels_used": False,
        },
        "project_mix": {
            "display_name": "Project and Mix (paper-equation binary adaptation)",
            "paper": "https://arxiv.org/abs/2603.24528v2",
            "official_repository": "https://github.com/dipamgoswami/TAMP",
            "official_commit": "ad90ff202d493e979c05469aeeec0f255db2c895",
            "official_code_available": False,
            "equations": [5, 8, 9, 10],
            "lambda_grid": list(PROJECT_MIX_LAMBDAS),
            "gamma_grid": list(PROJECT_MIX_GAMMAS),
            "candidate_order": "lambda ascending outer loop; gamma ascending inner loop",
            "text_svd_energy_threshold": TEXT_ENERGY_THRESHOLD,
            "text_svd_centering": False,
            "text_rank_rule": "smallest k with cumulative squared singular values >= 99%",
            "text_rank_tuned": False,
            "binary_class_order": ["negative", "positive"],
            "text_prototypes": "fixed task-specific negative and positive prompts, unit normalized",
            "image_prototypes": "arithmetic class means of unit features; not normalized before mixing",
            "covariance": "class-centered residual sample covariance with denominator N-1",
            "mahalanobis_matrix": "A=(N-1)*Sigma+trace(Sigma)*I; cosine after A^(-1/2)",
            "output": "positive CMP class score minus negative CMP class score",
            "score_domain": "raw cosine-combination margin (positive minus negative)",
            "gradient_training": False,
            "trainable_parameters": 0,
            "zero_projection_convention": "cosine of zero projected feature or prototype is zero",
            "zero_covariance_convention": "fail explicitly; no undocumented fallback",
            "adaptations": [
                "independent two-class present/absent problems; not a mutually-exclusive disease softmax",
                "support-only five-fold AUPRC selection instead of a separate validation accuracy set",
                "no ten-augmentation support expansion; same identity supports and labels as Q",
                "equal two-view test margin averaging is an experimental wrapper",
                "BiomedCLIP uses shared projected global image/text features",
                "AlphaRAD uses frozen projected patch mean and normalized task texts as a representation transfer",
            ],
        },
        "linear_probe": {
            "display_name": "L2 logistic linear probe",
            "C_grid": list(LINEAR_PROBE_CS),
            "penalty": "l2", "solver": "lbfgs", "fit_intercept": True,
            "class_weight": None, "max_iter": LINEAR_PROBE_MAX_ITER,
            "tol": LINEAR_PROBE_TOL,
            "feature_scaling": "unit image features only; no fitted standardization",
            "output": "binary logistic decision_function (positive minus negative log-odds)",
            "score_domain": "binary logistic log-odds margin",
            "gradient_training": True,
            "trainable_parameter_count_rule": "C_tasks * (feature_dimension + 1)",
            "zero_shot_mixing": False,
            "convergence_warning": "error; no silently accepted nonconverged candidate",
        },
    }


def _unit_array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    if not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError(f"{name} must contain unit features from the frozen extractor")
    return array


def _normalize(array):
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 1e-12)


def _digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_folds(groups, folds, labels):
    if folds is None or len(folds) != 5:
        raise ValueError("Exactly five explicitly shared support folds are required")
    if any(g is None or not str(g).strip() for g in groups):
        raise ValueError("Every support sample requires a nonempty patient/group ID")
    groups = np.asarray([str(g) for g in groups])
    n = len(labels)
    if groups.shape != (n,):
        raise ValueError("Support groups and labels differ in length")
    coverage = np.zeros(n, dtype=int)
    checked, audit = [], []
    for i, fold in enumerate(folds):
        parts = []
        for key in ("train", "validation"):
            raw = np.asarray(fold[key])
            if raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw):
                raise ValueError(f"Fold {i} {key} requires nonempty integer indices")
            idx = raw.astype(np.int64)
            if idx.min() < 0 or idx.max() >= n or len(np.unique(idx)) != len(idx):
                raise ValueError(f"Fold {i} {key} contains repeated/out-of-range indices")
            parts.append(idx)
        train, valid = parts
        if set(train) & set(valid) or set(train) | set(valid) != set(range(n)):
            raise ValueError(f"Fold {i} train must be the exact validation complement")
        if set(groups[train]) & set(groups[valid]):
            raise ValueError(f"Fold {i} patient/group crosses train and validation")
        for part, name in ((train, "train"), (valid, "validation")):
            sums = labels[part].sum(0)
            if np.any((sums == 0) | (sums == len(part))):
                raise ValueError(f"Fold {i} {name} is missing a binary class")
        coverage[valid] += 1
        checked.append({"train": train, "validation": valid})
        audit.append({"fold": i, "train_indices": train.tolist(),
                      "validation_indices": valid.tolist(),
                      "train_positive_counts": labels[train].sum(0).astype(int).tolist(),
                      "validation_positive_counts": labels[valid].sum(0).astype(int).tolist()})
    if not np.all(coverage == 1):
        raise ValueError("Every support sample must receive exactly one OOF prediction")
    return checked, audit


def _auprc(labels, scores):
    from sklearn.metrics import auc, precision_recall_curve
    metrics = []
    for c in range(labels.shape[1]):
        precision, recall, _ = precision_recall_curve(labels[:, c], scores[:, c])
        metrics.append(float(auc(recall, precision)))
    return float(np.mean(metrics))


@dataclass
class ProjectMixBinary:
    """Complete two-class CMP fit; class order is absent, present."""
    means: np.ndarray
    texts: np.ndarray
    basis: np.ndarray
    covariance_basis: np.ndarray
    whitening_correction: np.ndarray
    audit: dict

    def _whiten_direction(self, features):
        # A^(-1/2) multiplied by sqrt(trace(Sigma)); the common positive
        # scalar cancels exactly in both sides of the cosine in equation 9.
        transformed = features + ((features @ self.covariance_basis.T)
                                  * self.whitening_correction) @ self.covariance_basis
        return _normalize(transformed)

    def semantic_scores(self, features, lam):
        projected_features = features @ self.basis
        projected_means = (self.means @ self.basis) @ self.basis.T
        # Keep the full text prototype in equation 8 even when truncation
        # discards some text energy: projecting text again changes its norm.
        mixed = float(lam) * projected_means + (1.0 - float(lam)) * self.texts
        return _normalize(projected_features) @ (_normalize(mixed) @ self.basis).T

    def image_scores(self, features):
        return self._whiten_direction(features) @ self._whiten_direction(self.means).T

    def predict_margin(self, features, lam, gamma):
        scores = self.semantic_scores(features, lam) + float(gamma) * self.image_scores(features)
        return scores[:, 1] - scores[:, 0]


def fit_project_mix_binary(features, labels, positive_text, negative_text):
    """Fit the paper's full CMP ingredients using support statistics only."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Project and Mix needs matrix features and both binary support classes")
    texts = np.stack((negative_text, positive_text)).astype(np.float64)
    if texts.shape != (2, x.shape[1]):
        raise ValueError("Text and image dimensions must match")
    means = np.stack((x[y == 0].mean(0), x[y == 1].mean(0)))
    u, text_singular, _ = np.linalg.svd(texts.T, full_matrices=False)
    energy = text_singular ** 2
    cumulative = np.cumsum(energy) / energy.sum()
    rank = int(np.searchsorted(cumulative, TEXT_ENERGY_THRESHOLD, side="left") + 1)
    basis = u[:, :rank]
    residual = x - means[y]
    # The exact class-centered global mean is zero. This removes only roundoff
    # and matches sample covariance implementations that center their inputs.
    residual -= residual.mean(0, keepdims=True)
    _, residual_singular, residual_vt = np.linalg.svd(residual, full_matrices=False)
    covariance_trace = float(np.sum(residual * residual) / (len(x) - 1))
    if covariance_trace <= 0:
        raise ValueError("Zero support covariance makes the paper's inverse square root undefined")
    correction = np.sqrt(covariance_trace / (covariance_trace + residual_singular ** 2)) - 1.0
    audit = {
        "negative_support_count": int(np.sum(y == 0)),
        "positive_support_count": int(np.sum(y == 1)),
        "text_singular_values": text_singular.tolist(),
        "text_cumulative_squared_energy": cumulative.tolist(),
        "text_rank": rank,
        "text_energy_retained": float(cumulative[rank - 1]),
        "rank_one_binary_projection": rank == 1,
        "covariance_trace": covariance_trace,
        "covariance_denominator": len(x) - 1,
        "solver": "thin residual SVD; scaled inverse square root, cosine-equivalent",
    }
    return ProjectMixBinary(means, texts, basis, residual_vt, correction, audit)


def _fit_logistic(features, labels, c):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    estimator = LogisticRegression(C=float(c), penalty="l2", solver="lbfgs",
                                   fit_intercept=True, class_weight=None,
                                   max_iter=LINEAR_PROBE_MAX_ITER, tol=LINEAR_PROBE_TOL,
                                   random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(features, labels)
    return estimator


def run_neighbor_methods(*, support_features, labels, groups, folds,
                         queries_positive, queries_negative, test_features,
                         methods=None):
    """Return margin matrices plus complete support-only selection metadata.

    Arrays: support_features [N,D], labels [N,C], text queries [C,D],
    test_features [M,D]. Each disease is an independent binary task, so callers
    with distinct support sets per disease invoke this with C=1. Test identity
    and center views may be concatenated along M and split after this call.
    """
    x = _unit_array(support_features, "support_features")
    xt = _unit_array(test_features, "test_features")
    pos = _unit_array(queries_positive, "queries_positive")
    neg = _unit_array(queries_negative, "queries_negative")
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != len(x) or not y.shape[1] or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be a binary [N,C] matrix")
    if xt.shape[1] != x.shape[1] or pos.shape != (y.shape[1], x.shape[1]) or neg.shape != pos.shape:
        raise ValueError("Feature/text dimensions or number of binary tasks do not match")
    selected_methods = DEFAULT_METHODS if methods is None else tuple(methods)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("Choose one or both unique neighbor methods")
    if set(selected_methods) - set(DEFAULT_METHODS):
        raise ValueError(f"Unknown methods: {set(selected_methods) - set(DEFAULT_METHODS)}")
    checked, fold_audit = _validate_folds(groups, folds, y)
    plan = algorithm_plan()
    result = {"scores": {}, "fit_info": {
        "version": VERSION, "algorithm_plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "selection": plan["selection"], "folds": fold_audit,
        "support_features_sha256": _digest(x), "support_labels_sha256": _digest(y),
        "positive_queries_sha256": _digest(pos), "negative_queries_sha256": _digest(neg),
        "methods": {},
    }}
    if "project_mix" in selected_methods:
        start = time.monotonic()
        semantic_oof = np.zeros((len(PROJECT_MIX_LAMBDAS), *y.shape))
        image_oof = np.zeros_like(y)
        fold_fits = []
        for fold in checked:
            train, valid = fold["train"], fold["validation"]
            audits = []
            for c in range(y.shape[1]):
                model = fit_project_mix_binary(x[train], y[train, c], pos[c], neg[c])
                image_scores = model.image_scores(x[valid])
                image_oof[valid, c] = image_scores[:, 1] - image_scores[:, 0]
                for j, lam in enumerate(PROJECT_MIX_LAMBDAS):
                    scores = model.semantic_scores(x[valid], lam)
                    semantic_oof[j, valid, c] = scores[:, 1] - scores[:, 0]
                audits.append(model.audit)
            fold_fits.append(audits)
        candidates = []
        for j, lam in enumerate(PROJECT_MIX_LAMBDAS):
            for gamma in PROJECT_MIX_GAMMAS:
                metric = _auprc(y, semantic_oof[j] + gamma * image_oof)
                candidates.append({"lambda": lam, "gamma": gamma, "oof_macro_auprc": metric})
        best = max(candidates, key=lambda item: item["oof_macro_auprc"])
        # Refit only on the complete support set. No test statistic influences
        # projection, covariance, hyperparameters, or either prototype.
        fitted = [fit_project_mix_binary(x, y[:, c], pos[c], neg[c]) for c in range(y.shape[1])]
        fit_seconds = time.monotonic() - start
        predict_start = time.monotonic()
        result["scores"]["project_mix"] = np.column_stack([
            model.predict_margin(xt, best["lambda"], best["gamma"]) for model in fitted])
        result["fit_info"]["methods"]["project_mix"] = {
            "selected": dict(best), "candidates": candidates,
            "fold_fits": fold_fits, "full_support_fits": [model.audit for model in fitted],
            "gradient_training": False, "trainable_parameters": 0,
            "score_domain": plan["project_mix"]["score_domain"],
            "fit_seconds": fit_seconds, "prediction_seconds": time.monotonic() - predict_start,
            "implementation": plan["project_mix"],
        }
    if "linear_probe" in selected_methods:
        start = time.monotonic()
        candidates = []
        for c_value in LINEAR_PROBE_CS:
            oof = np.zeros_like(y)
            iterations = []
            for fold in checked:
                train, valid = fold["train"], fold["validation"]
                fold_iterations = []
                for c in range(y.shape[1]):
                    estimator = _fit_logistic(x[train], y[train, c], c_value)
                    oof[valid, c] = estimator.decision_function(x[valid])
                    fold_iterations.append(int(estimator.n_iter_[0]))
                iterations.append(fold_iterations)
            candidates.append({"C": c_value, "oof_macro_auprc": _auprc(y, oof),
                               "fold_iterations": iterations})
        best = max(candidates, key=lambda item: item["oof_macro_auprc"])
        fitted = [_fit_logistic(x, y[:, c], best["C"]) for c in range(y.shape[1])]
        fit_seconds = time.monotonic() - start
        predict_start = time.monotonic()
        result["scores"]["linear_probe"] = np.column_stack([
            model.decision_function(xt) for model in fitted])
        result["fit_info"]["methods"]["linear_probe"] = {
            "selected": {"C": best["C"], "oof_macro_auprc": best["oof_macro_auprc"]},
            "candidates": candidates,
            "coefficients": [model.coef_[0].tolist() for model in fitted],
            "intercepts": [float(model.intercept_[0]) for model in fitted],
            "full_support_iterations": [int(model.n_iter_[0]) for model in fitted],
            "gradient_training": True, "trainable_parameters": int(y.shape[1] * (x.shape[1] + 1)),
            "score_domain": plan["linear_probe"]["score_domain"],
            "fit_seconds": fit_seconds, "prediction_seconds": time.monotonic() - predict_start,
            "implementation": plan["linear_probe"],
        }
    for name, scores in result["scores"].items():
        if scores.shape != (len(xt), y.shape[1]) or not np.isfinite(scores).all():
            raise RuntimeError(f"{name} produced invalid margin predictions")
    return result
