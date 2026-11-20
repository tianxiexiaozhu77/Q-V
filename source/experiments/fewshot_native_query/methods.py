"""Frozen-feature few-shot score controls with group-isolated support CV.

This module never accepts test labels. These are explicitly named binary OvR
ports, not exact reproductions of the original single-label CLIP experiments.
Native binary margins are retained; there is no hidden CLIP x100 multiplier.
Spatial output is unsupported: an added classification score does not define a
new explanation map. See SOURCES_TF.json for pinned upstream formula sources.
"""
from __future__ import annotations

import hashlib
import json
from itertools import product

import numpy as np

VERSION = "native-margin-ovr-support-group-cv-v1"
ALPHAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
BETAS = (1.0, 2.0, 5.0, 10.0)
GDA_ALPHAS = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0)
DEFAULT_METHODS = (
    "native_original", "prototype_score_only", "prototype_score_mix_oof",
    "tip_adapter_ovr_oof", "gda_ovr_ks_oof", "class_local_difference_score_oof",
)


def _array(value, name, ndim):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite rank-{ndim} array")
    return result


def _unit_array(value, name, ndim):
    result = _array(value, name, ndim)
    if not np.allclose(np.linalg.norm(result, axis=-1), 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError(f"{name}: expected unit vectors; normalize in the feature extractor")
    return result


def _normalise(value):
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    return np.divide(value, norms, out=np.zeros_like(value), where=norms > 1e-12)


def _indices(values, n, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw):
        raise ValueError(f"{name}: nonempty integer index vector required")
    result = raw.astype(np.int64)
    if result.min() < 0 or result.max() >= n or len(set(result.tolist())) != len(result):
        raise ValueError(f"{name}: repeated or out-of-range indices")
    return result


def validate_folds(groups, folds=None):
    """Check a full support-only, group-disjoint out-of-fold partition.

    Explicit folds are shared with other algorithms by the caller. Without
    explicit folds, leave-one-group-out is used only as a documented fallback.
    Each train set must be the exact complement of its validation set.
    """
    groups = list(groups)
    if not groups or any(g is None or str(g).strip() == "" for g in groups):
        raise ValueError("Every support sample needs a nonempty group")
    groups = np.asarray([str(g) for g in groups])
    unique = list(dict.fromkeys(groups.tolist()))
    if len(unique) < 2:
        raise ValueError("At least two support groups are needed")
    if folds is None:
        folds = [{"train": np.flatnonzero(groups != g),
                  "validation": np.flatnonzero(groups == g)} for g in unique]
    if len(folds) < 2:
        raise ValueError("At least two folds required")
    n, checked = len(groups), []
    coverage = np.zeros(n, dtype=np.int64)
    for i, fold in enumerate(folds):
        train = _indices(fold["train"], n, f"fold {i} train")
        valid = _indices(fold["validation"], n, f"fold {i} validation")
        if set(train) & set(valid) or set(train) | set(valid) != set(range(n)):
            raise ValueError(f"fold {i}: train must be the exact validation complement")
        if set(groups[train]) & set(groups[valid]):
            raise ValueError(f"fold {i}: support group crosses train/validation")
        coverage[valid] += 1
        checked.append({"train": train, "validation": valid})
    if not np.all(coverage == 1):
        raise ValueError("Each support sample must receive exactly one OOF prediction")
    return checked


def prototype_direction(features, labels, *, local=False):
    """Means of unit image features, separately normalized before subtraction.

    Global prototype control uses cosine(x, mean_positive)-cosine(x, mean_negative).
    Class-local score control instead normalizes the positive-minus-negative mean
    direction, matching the support direction of our native-query intervention.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positive = labels.sum(0)
    negative = len(labels) - positive
    if (positive <= 0).any() or (negative <= 0).any():
        raise ValueError("Every fitted class needs both positive and negative samples")
    if features.ndim == 2:
        pm = (labels.T @ features) / positive[:, None]
        nm = ((1 - labels).T @ features) / negative[:, None]
    else:
        pm = (features * labels[..., None]).sum(0) / positive[:, None]
        nm = (features * (1 - labels)[..., None]).sum(0) / negative[:, None]
    return _normalise(pm - nm) if local else _normalise(pm) - _normalise(nm)


def tip_margin(test_features, support_features, labels, beta):
    """Binary OvR Tip-Adapter cache margin, with exact positive/negative sums.

    Exp[-beta(1-cos)] @ one_hot(y) for absent/present, followed by present-absent.
    No division by class counts: balancing would be a different cache variant.
    """
    affinity = np.clip(test_features @ support_features.T, -1.0, 1.0)
    cache = np.exp(float(beta) * (affinity - 1.0))
    return cache @ (2.0 * labels - 1.0)


def fit_gda_ks(features, labels):
    """Author KS inverse covariance, fitted independently per binary task.

    Original formula: D pinv((N-1) S + trace(S) I), S=cov(centered_features.T).
    Class centering makes the global residual mean zero; explicitly subtracting
    it again matches torch.cov in the source. A Woodbury solve avoids D x D
    inversion for few-shot N << D. Uniform two-class priors cancel in the margin.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n, dim = x.shape
    if n < 2:
        raise ValueError("GDA requires at least two support samples")
    weights, biases, details = [], [], []
    for c in range(y.shape[1]):
        positive = y[:, c] == 1
        if not positive.any() or positive.all():
            raise ValueError("GDA fitted class missing positive or negative supports")
        means = np.stack((x[~positive].mean(0), x[positive].mean(0)))
        centered = x - means[positive.astype(np.int64)]
        centered -= centered.mean(0, keepdims=True)
        lam = float(np.sum(centered * centered) / (n - 1))
        vectors = means.T
        if lam == 0.0:
            # Author pinv(0) = 0, rather than inventing an extra regularizer.
            solved = np.zeros_like(vectors)
            solver = "zero_covariance_pseudoinverse"
        elif n < dim:
            small = centered @ centered.T + lam * np.eye(n)
            solved = (vectors - centered.T @ np.linalg.solve(small, centered @ vectors)) / lam
            solver = "woodbury"
        else:
            matrix = centered.T @ centered + lam * np.eye(dim)
            solved = np.linalg.solve(matrix, vectors)
            solver = "direct_small_dimension"
        transformed = dim * solved
        weights.append(transformed[:, 1] - transformed[:, 0])
        biases.append(-0.5 * float(np.sum(means[1] * transformed[:, 1])
                                  - np.sum(means[0] * transformed[:, 0])))
        details.append({"class_index": c, "positive_count": int(positive.sum()),
                        "negative_count": int((~positive).sum()),
                        "covariance_trace": lam, "solver": solver})
    return np.stack(weights, axis=1), np.asarray(biases), details


def macro_trapezoid_auprc(labels, scores):
    from sklearn.metrics import auc, precision_recall_curve
    values = []
    for c in range(labels.shape[1]):
        if len(np.unique(labels[:, c])) != 2:
            raise ValueError("OOF metric requires both labels in every declared class")
        precision, recall, _ = precision_recall_curve(labels[:, c], scores[:, c])
        values.append(float(auc(recall, precision)))
    return float(np.mean(values))


def _fold_audit(folds, labels, groups):
    audits, valid = [], True
    for i, fold in enumerate(folds):
        train, held = fold["train"], fold["validation"]
        p = labels[train].sum(0).astype(int)
        missing = np.flatnonzero((p == 0) | (p == len(train))).tolist()
        valid &= not missing
        audits.append({"fold": i, "train_indices": train.tolist(),
                       "validation_indices": held.tolist(),
                       "train_group_count": len(set(groups[train])),
                       "validation_group_count": len(set(groups[held])),
                       "train_positive_counts": p.tolist(),
                       "train_negative_counts": (len(train) - p).tolist(),
                       "missing_binary_class_indices": missing,
                       "fit_eligible": not bool(missing)})
    return bool(valid), audits


def _param_grid(name):
    if name == "tip_adapter_ovr_oof":
        return [{"alpha": a, "beta": b} for a, b in product(ALPHAS, BETAS)]
    grid = GDA_ALPHAS if name == "gda_ovr_ks_oof" else ALPHAS
    return [{"alpha": a} for a in grid]


def _evaluate_candidates(labels, original_oof, residuals, grid, eligible):
    baseline_metric = macro_trapezoid_auprc(labels, original_oof)
    candidates = []
    for params in grid:
        alpha = params["alpha"]
        if alpha == 0:
            metric, status = baseline_metric, "valid_baseline_no_fitting"
        elif eligible:
            residual = residuals[params.get("beta", "default")]
            metric = macro_trapezoid_auprc(labels, original_oof + alpha * residual)
            status = "valid"
        else:
            metric, status = None, "invalid_fold_missing_binary_class"
        candidates.append({"parameters": params, "oof_macro_auprc": metric,
                           "status": status})
    valid = [c for c in candidates if c["oof_macro_auprc"] is not None]
    # Smallest alpha wins ties; deterministic first beta wins remaining ties.
    best = valid[0]
    for candidate in valid[1:]:
        if candidate["oof_macro_auprc"] > best["oof_macro_auprc"] + 1e-12:
            best = candidate
    return best["parameters"].copy(), candidates


def run_score_methods(*, support_features, support_local, labels, support_logits,
                      groups, test_features, test_local, test_logits,
                      queries_positive=None, queries_negative=None, methods=None,
                      folds=None):
    """Fit/choose only with support CV, then predict a previously unseen array.

    Required arrays are support/test global unit vectors [N,D], class-local unit
    vectors [N,C,Dlocal], and native binary margins [N,C]. Labels exist only for
    support. Optional text queries are accepted for a common extraction schema;
    these score-only controls deliberately do not alter or consume text queries.

    Returns {scores: {name: test_margin[Ntest,C]}, fit_info: JSON-safe metadata}.
    No output is a probability until a caller applies its specified sigmoid.
    """
    requested = list(DEFAULT_METHODS if methods is None else methods)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(DEFAULT_METHODS):
        raise ValueError(f"Unknown or duplicate score methods: {requested}")
    x = _unit_array(support_features, "support_features", 2)
    local = _unit_array(support_local, "support_local", 3)
    y = _array(labels, "labels", 2)
    z = _array(support_logits, "support_logits", 2)
    tx = _unit_array(test_features, "test_features", 2)
    tl = _unit_array(test_local, "test_local", 3)
    tz = _array(test_logits, "test_logits", 2)
    if not len(y) or not len(tx) or y.shape != z.shape or len(x) != len(y):
        raise ValueError("Support array sizes disagree or an input split is empty")
    if local.shape[:2] != y.shape or local.shape[-1] != tl.shape[-1]:
        raise ValueError("Support/test local feature dimensions disagree")
    if tz.shape != (len(tx), y.shape[1]) or tl.shape[:2] != tz.shape or tx.shape[1] != x.shape[1]:
        raise ValueError("Test dimensions/class order disagree with support schema")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Only explicit binary support labels are supported")
    if ((y.sum(0) == 0) | (y.sum(0) == len(y))).any():
        raise ValueError("Every declared class needs positive and negative supports")
    if len(groups) != len(y):
        raise ValueError("One group is required per support sample")
    for name, queries in (("queries_positive", queries_positive), ("queries_negative", queries_negative)):
        if queries is not None and _array(queries, name, 2).shape[0] != y.shape[1]:
            raise ValueError(f"{name}: class count differs")
    checked = validate_folds(groups, folds)
    group_array = np.asarray([str(g) for g in groups])
    eligible, audit = _fold_audit(checked, y, group_array)
    fold_json = [{k: v.tolist() for k, v in fold.items()} for fold in checked]
    fold_hash = hashlib.sha256(json.dumps(fold_json, sort_keys=True).encode()).hexdigest()
    common = {
        "version": VERSION, "trainable_parameters": 0, "gradient_training": False,
        "selection_data": "support-only group-isolated out-of-fold predictions",
        "selection_metric": "macro trapezoidal PR area; not average precision",
        "fold_count": len(checked), "folds_sha256": fold_hash, "fold_audit": audit,
        "fold_type": "provided_shared_group_partition" if folds is not None else "leave_one_group_out",
        "all_folds_fit_eligible": eligible, "test_labels_accepted": False,
        "test_features_used_for_selection": False, "text_queries_used": False,
        "support_count": len(y), "support_group_count": len(set(group_array)),
        "support_positive_counts": y.sum(0).astype(int).tolist(),
        "support_negative_counts": (len(y) - y.sum(0)).astype(int).tolist(),
        "test_count": len(tx), "class_count": y.shape[1],
        "score_domain": "native per-disease binary margin; no x100 CLIP scaling",
        "spatial_output": "not applicable to adapted score head; no fabricated map",
        "tie_break": "prefer smallest alpha within 1e-12; then first fixed beta",
    }
    # Fit all OOF statistics before any use of test features for predictions.
    residuals = {}
    selected = {}
    candidate_info = {}
    optimised = [name for name in requested if name.endswith("_oof")]
    if eligible:
        for name in optimised:
            keys = BETAS if name == "tip_adapter_ovr_oof" else ("default",)
            residuals[name] = {key: np.full_like(y, np.nan) for key in keys}
        for fold in checked:
            tr, va = fold["train"], fold["validation"]
            for name in optimised:
                dest = residuals[name]
                if name == "prototype_score_mix_oof":
                    direction = prototype_direction(x[tr], y[tr])
                    dest["default"][va] = x[va] @ direction.T
                elif name == "class_local_difference_score_oof":
                    direction = prototype_direction(local[tr], y[tr], local=True)
                    dest["default"][va] = np.einsum("ncd,cd->nc", local[va], direction)
                elif name == "tip_adapter_ovr_oof":
                    for beta in BETAS:
                        dest[beta][va] = tip_margin(x[va], x[tr], y[tr], beta)
                elif name == "gda_ovr_ks_oof":
                    weight, bias, _ = fit_gda_ks(x[tr], y[tr])
                    dest["default"][va] = x[va] @ weight + bias
        for values in residuals.values():
            if any(not np.isfinite(value).all() for value in values.values()):
                raise ValueError("Incomplete or nonfinite OOF residual")
    for name in optimised:
        selected[name], candidate_info[name] = _evaluate_candidates(
            y, z, residuals.get(name, {}), _param_grid(name), eligible)

    scores, fits = {}, {}
    for name in requested:
        meta = dict(common, method=name)
        if name == "native_original":
            values = tz.copy()
            meta.update(selection_status="no_selection_original_model", selected_parameters={}, candidates=[])
        elif name == "prototype_score_only":
            direction = prototype_direction(x, y)
            values = tx @ direction.T
            meta.update(selection_status="fixed_mechanism_control_no_hyperparameters",
                        selected_parameters={}, candidates=[], formula="cos(x,positive_mean)-cos(x,negative_mean)")
        else:
            params = selected[name]
            meta.update(selected_parameters=params, candidates=candidate_info[name],
                        selection_status="selected_on_support_oof" if eligible else "no_selection_invalid_folds_baseline_fallback")
            alpha = params["alpha"]
            if alpha == 0:
                values = tz.copy()
                meta["final_statistics_fitted"] = False
            else:
                meta["final_statistics_fitted"] = True
                if name == "prototype_score_mix_oof":
                    direction = prototype_direction(x, y)
                    residual = tx @ direction.T
                    meta["formula"] = "native_margin + alpha*(cos(x,positive_mean)-cos(x,negative_mean))"
                elif name == "class_local_difference_score_oof":
                    direction = prototype_direction(local, y, local=True)
                    residual = np.einsum("ncd,cd->nc", tl, direction)
                    meta["formula"] = "native_margin + alpha*dot(original_class_local,unit(positive_mean-negative_mean))"
                elif name == "tip_adapter_ovr_oof":
                    residual = tip_margin(tx, x, y, params["beta"])
                    meta["formula"] = "native_margin + alpha*exp[-beta*(1-cos(test,support))]@(2*labels-1)"
                    meta["class_count_balancing"] = False
                elif name == "gda_ovr_ks_oof":
                    weight, bias, details = fit_gda_ks(x, y)
                    residual = tx @ weight + bias
                    meta["covariance_fit"] = details
                    meta["class_prior"] = "uniform positive/negative; cancels in binary margin"
                    meta["formula"] = "native_margin + alpha*(GDA_present_logit-GDA_absent_logit)"
                values = tz + alpha * residual
        if values.shape != tz.shape or not np.isfinite(values).all():
            raise ValueError(f"{name}: invalid final score matrix")
        scores[name], fits[name] = values, meta
    # Assert the metadata is safe to write without accidentally serializing test arrays.
    json.dumps(fits, allow_nan=False)
    return {"scores": scores, "fit_info": fits}
