"""Regular TaskRes as an explicit frozen-feature binary/OvR port.

Author implementation: baselines/TaskRes/trainers/taskres.py. Each disease has
two fixed native text bases (absent/present) and two zero-initialized
learnable residuals. The only fitted parameters are these residuals:
    logits[n,c,k] = scale * unit(x[n]) @ unit(q[c,k] + alpha*r[c,k]).
Training uses a separate binary cross-entropy problem per disease, implemented
as the sum of class-wise mean two-class cross-entropies. The exported score is
the present-minus-absent logit. No test arrays are accepted by fit_taskres.

This is not the original single-label benchmark pipeline: native disease queries
replace CLIP class-name embeddings, deterministic cached native image
features replace randomly augmented training images, and multilabel labels are
handled by independent OvR tasks. The regular base is used, never the enhanced
base. No image encoder, text encoder, native image-aggregation path, or logit
scale is trained. See TASKRES_NOTES.txt for source/config provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch.nn import functional as F

VERSION = "taskres-regular-native-base-binary-ovr-oof-v1"
DEFAULT_EPOCH_CANDIDATES = (0, 20, 50, 100, 200)
DEFAULT_LR_CANDIDATES = (0.0002, 0.002)


def _matrix(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name}: expected a nonempty rank-2 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: all values must be finite")
    if (np.linalg.norm(array, axis=-1) <= 1e-12).any():
        raise ValueError(f"{name}: zero-length vectors are not valid inputs")
    return np.array(array, copy=True, order="C")


def _unit_array(value, name):
    array = _matrix(value, name)
    if not np.allclose(np.linalg.norm(array, axis=-1), 1., atol=2e-4, rtol=2e-4):
        raise ValueError(f"{name}: expected unit vectors from the frozen extractor")
    return array


def _validate_folds(groups, folds, labels):
    """Validate supplied support-only folds without generating a new partition."""
    if groups is None or len(groups) != len(labels):
        raise ValueError("groups must contain one identifier per support image")
    if any(g is None or not str(g).strip() for g in groups):
        raise ValueError("groups must contain nonempty identifiers")
    groups = np.asarray([str(g) for g in groups])
    if folds is None or len(folds) < 2:
        raise ValueError("at least two explicit support-only folds are required")
    coverage = np.zeros(len(labels), dtype=np.int64)
    checked = []
    audit = []
    for number, fold in enumerate(folds):
        parts = {}
        for key in ("train", "validation"):
            values = np.asarray(fold[key])
            if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
                raise ValueError(f"fold {number}: {key} must contain integer indices")
            indices = values.astype(np.int64)
            if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(labels):
                raise ValueError(f"fold {number}: {key} contains repeated/out-of-range indices")
            parts[key] = indices
        train, valid = parts["train"], parts["validation"]
        if set(train) & set(valid) or set(train) | set(valid) != set(range(len(labels))):
            raise ValueError(f"fold {number}: train must be the exact validation complement")
        if set(groups[train]) & set(groups[valid]):
            raise ValueError(f"fold {number}: a support group crosses train/validation")
        positive = labels[train].sum(0)
        missing = np.flatnonzero((positive == 0) | (positive == len(train))).tolist()
        if missing:
            raise ValueError(f"fold {number}: training classes {missing} lack one binary label; supports are not altered")
        coverage[valid] += 1
        checked.append(parts)
        audit.append({"fold": number, "train_indices": train.tolist(),
                      "validation_indices": valid.tolist(),
                      "train_group_count": len(set(groups[train])),
                      "validation_group_count": len(set(groups[valid])),
                      "train_positive_counts": positive.astype(int).tolist(),
                      "train_negative_counts": (len(train) - positive).astype(int).tolist()})
    if not np.all(coverage == 1):
        raise ValueError("each support image must receive exactly one OOF prediction")
    return checked, audit


def _macro_trapezoid_auprc(labels, scores):
    result = []
    for c in range(labels.shape[1]):
        precision, recall, _ = precision_recall_curve(labels[:, c], scores[:, c])
        result.append(float(auc(recall, precision)))
    return float(np.mean(result))


def learning_rate_at_epoch(epoch, learning_rate, schedule_epochs=200, warmup_lr=1e-5):
    """Dassl constant 1-epoch warmup + cosine with WARMUP_RECOUNT=True.

    Epochs are zero-based. Epoch 0 uses warmup_lr, epoch 1 the base learning
    rate, and epoch 2 cosine step 1. Checkpoint selection never changes T_max.
    """
    if epoch < 0 or schedule_epochs < 1:
        raise ValueError("epoch must be nonnegative and schedule_epochs positive")
    if epoch == 0:
        return float(warmup_lr)
    return float(learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / schedule_epochs)))


class TaskResModel(nn.Module):
    """Residual-only binary/OvR classifier; class axis precedes absent/present."""

    def __init__(self, queries_positive, queries_negative, *, logit_scale, residual_scale=0.5):
        super().__init__()
        # Preserve raw native base norms, as in the regular author method.
        # Normalize only q + alpha*r during scoring, never q before addition.
        positive = _matrix(queries_positive, "queries_positive")
        negative = _matrix(queries_negative, "queries_negative")
        if positive.shape != negative.shape:
            raise ValueError("positive/negative query shapes must agree")
        if not math.isfinite(float(logit_scale)) or logit_scale <= 0:
            raise ValueError("logit_scale must be the positive frozen native multiplier")
        if not math.isfinite(float(residual_scale)) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        base = torch.from_numpy(np.stack((negative, positive), axis=1))
        self.register_buffer("base_text_features", base)
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale), dtype=torch.float32))
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale), dtype=torch.float32))
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base))

    def adapted_text(self):
        return F.normalize(self.base_text_features + self.residual_scale * self.text_feature_residuals, dim=-1)

    def forward(self, features):
        return self.logit_scale * torch.einsum("nd,ckd->nck", F.normalize(features, dim=-1), self.adapted_text())

    @torch.no_grad()
    def predict(self, features, *, batch_size=4096):
        """Return scaled present-minus-absent logits; accepts no labels."""
        features = _unit_array(features, "prediction_features")
        if features.shape[1] != self.base_text_features.shape[-1]:
            raise ValueError("prediction feature and query dimensions disagree")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        result = []
        for start in range(0, len(features), batch_size):
            x = torch.as_tensor(features[start:start + batch_size], device=self.base_text_features.device)
            logits = self(x)
            result.append((logits[..., 1] - logits[..., 0]).cpu().numpy())
        return np.concatenate(result, axis=0)

    def export(self):
        """JSON-compatible inference state; state_dict() is also supported."""
        return {"version": VERSION,
                "queries_positive": self.base_text_features[:, 1].detach().cpu().tolist(),
                "queries_negative": self.base_text_features[:, 0].detach().cpu().tolist(),
                "residuals": self.text_feature_residuals.detach().cpu().tolist(),
                "logit_scale": float(self.logit_scale.detach().cpu()),
                "residual_scale": float(self.residual_scale.detach().cpu())}

    @classmethod
    def from_export(cls, state, *, device="cpu"):
        if state["version"] != VERSION:
            raise ValueError("unsupported TaskRes model state version")
        model = cls(state["queries_positive"], state["queries_negative"],
                    logit_scale=state["logit_scale"], residual_scale=state["residual_scale"]).to(device)
        residual = torch.as_tensor(state["residuals"], dtype=torch.float32, device=device)
        if residual.shape != model.text_feature_residuals.shape or not torch.isfinite(residual).all():
            raise ValueError("invalid saved TaskRes residual array")
        with torch.no_grad():
            model.text_feature_residuals.copy_(residual)
        return model.eval()


def _fit_path(features, labels, qpos, qneg, *, checkpoints, learning_rate, device,
              seed, residual_scale, logit_scale, schedule_epochs, batch_size,
              weight_decay, warmup_lr):
    """One optimization trajectory, snapshots shared by all epoch candidates.

    Inputs contain training samples only. Held-out features/labels never enter
    optimization, even through batch composition, normalization, or stopping.
    """
    model = TaskResModel(qpos, qneg, logit_scale=logit_scale, residual_scale=residual_scale).to(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam([model.text_feature_residuals], lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    snapshots = {}
    trace = []
    start = time.perf_counter()
    if 0 in checkpoints:
        snapshots[0] = model.text_feature_residuals.detach().cpu().clone()
    for epoch in range(max(checkpoints)):
        lr = learning_rate_at_epoch(epoch, learning_rate, schedule_epochs, warmup_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = torch.randperm(len(x), generator=generator).to(device)
        loss_sum, count = 0., 0
        for indices in order.split(batch_size):
            output = model(x[indices])
            loss = F.cross_entropy(output.reshape(-1, 2), y[indices].reshape(-1), reduction="none")
            # Sum independent class-wise CE means. Averaging across C would
            # unintentionally change regularization relative to binary fits.
            loss = loss.reshape(len(indices), y.shape[1]).mean(0).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError("TaskRes support training produced a nonfinite loss")
            loss_sum += float(loss.detach().cpu()) * len(indices)
            count += len(indices)
        trace.append({"epoch": epoch + 1, "learning_rate": lr,
                      "sum_binary_cross_entropy": loss_sum / count})
        if epoch + 1 in checkpoints:
            snapshots[epoch + 1] = model.text_feature_residuals.detach().cpu().clone()
    elapsed = time.perf_counter() - start
    return model.eval(), snapshots, {"fit_seconds": elapsed, "seed": int(seed), "epoch_trace": trace}


def fit_taskres(support_features, labels, queries_positive, queries_negative, groups, folds, *,
                logit_scale, device="cpu", seed=42,
                epoch_candidates=DEFAULT_EPOCH_CANDIDATES,
                learning_rate_candidates=DEFAULT_LR_CANDIDATES,
                residual_scale=0.5, schedule_epochs=200, batch_size=256,
                weight_decay=0.0005, warmup_lr=0.00001):
    """Choose one LR/checkpoint on pooled support-only OOF macro trapezoidal PR.

    The selected LR/epoch is shared by every disease, then refitted from zero
    on all supports. Scores tied within 1e-12 prefer fewer epochs then lower LR.
    Each LR has one full fixed-schedule path; selecting an early checkpoint
    does not restart or shorten its cosine schedule. The zero-epoch candidate
    is the unadapted cosine classifier and appears only once in the audit.
    """
    start = time.perf_counter()
    x = _unit_array(support_features, "support_features")
    qpos = _matrix(queries_positive, "queries_positive")
    qneg = _matrix(queries_negative, "queries_negative")
    y = np.asarray(labels)
    if y.ndim != 2 or y.shape != (len(x), len(qpos)) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must have shape N x C and contain only 0/1")
    y = y.astype(np.int64)
    if qpos.shape != qneg.shape or x.shape[1] != qpos.shape[1]:
        raise ValueError("support and query dimensions disagree")
    if ((y.sum(0) == 0) | (y.sum(0) == len(y))).any():
        raise ValueError("each declared disease needs both support labels")
    checked_folds, fold_audit = _validate_folds(groups, folds, y)
    epochs_raw = list(epoch_candidates)
    if not epochs_raw or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in epochs_raw):
        raise ValueError("epoch_candidates must contain nonnegative integers")
    epochs = tuple(sorted(set(int(v) for v in epochs_raw)))
    rates = tuple(sorted(set(float(v) for v in learning_rate_candidates)))
    if not rates or any(not math.isfinite(v) or v <= 0 for v in rates):
        raise ValueError("learning_rate_candidates must be finite and positive")
    if not isinstance(schedule_epochs, int) or schedule_epochs < 1 or max(epochs) > schedule_epochs:
        raise ValueError("schedule_epochs must be a positive integer >= every epoch candidate")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(weight_decay) or weight_decay < 0 or not math.isfinite(warmup_lr) or warmup_lr <= 0:
        raise ValueError("invalid weight decay or warmup learning rate")
    device = torch.device(device)
    params = dict(residual_scale=float(residual_scale), logit_scale=float(logit_scale),
                  schedule_epochs=schedule_epochs, batch_size=batch_size,
                  weight_decay=float(weight_decay), warmup_lr=float(warmup_lr))
    base_model = TaskResModel(qpos, qneg, logit_scale=logit_scale, residual_scale=residual_scale).to(device)
    candidates = []
    path_audit = []
    if 0 in epochs:
        scores = base_model.predict(x)
        candidates.append({"epoch": 0, "learning_rate": 0.0,
                           "oof_macro_trapezoid_auprc": _macro_trapezoid_auprc(y, scores),
                           "oof_score_sha256": hashlib.sha256(scores.tobytes()).hexdigest()})
    positive_epochs = tuple(v for v in epochs if v > 0)
    for rate in rates if positive_epochs else ():
        pooled = {epoch: np.empty(y.shape, dtype=np.float32) for epoch in positive_epochs}
        for fold_index, fold in enumerate(checked_folds):
            train, valid = fold["train"], fold["validation"]
            model, snapshots, path_info = _fit_path(
                x[train], y[train], qpos, qneg, checkpoints=positive_epochs,
                learning_rate=rate, device=device, seed=int(seed) + fold_index,
                **params)
            for epoch, residual in snapshots.items():
                with torch.no_grad():
                    model.text_feature_residuals.copy_(residual.to(device))
                pooled[epoch][valid] = model.predict(x[valid])
            path_audit.append({"fold": fold_index, "learning_rate": rate, **path_info})
        for epoch in positive_epochs:
            candidates.append({"epoch": epoch, "learning_rate": rate,
                               "oof_macro_trapezoid_auprc": _macro_trapezoid_auprc(y, pooled[epoch]),
                               "oof_score_sha256": hashlib.sha256(pooled[epoch].tobytes()).hexdigest()})
    best_score = max(item["oof_macro_trapezoid_auprc"] for item in candidates)
    selected = min((item for item in candidates if item["oof_macro_trapezoid_auprc"] >= best_score - 1e-12),
                   key=lambda item: (item["epoch"], item["learning_rate"]))
    selected_epoch = selected["epoch"]
    if selected_epoch:
        model, _, final_fit_info = _fit_path(
            x, y, qpos, qneg, checkpoints=(selected_epoch,),
            learning_rate=selected["learning_rate"], device=device, seed=int(seed), **params)
    else:
        model, final_fit_info = base_model.eval(), {"fit_seconds": 0.0, "seed": int(seed), "epoch_trace": []}
    fit_info = {
        "version": VERSION, "method": "TaskRes (regular-base binary/OvR port)",
        "input_feature_shape": list(x.shape), "class_count": y.shape[1],
        "support_positive_counts": y.sum(0).tolist(),
        "support_negative_counts": (len(y) - y.sum(0)).tolist(),
        "selected": dict(selected), "candidates": candidates,
        "configuration": {**params, "learning_rate_candidates": list(rates),
                          "epoch_candidates": list(epochs), "optimizer": "Adam",
                          "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-8,
                          "warmup_epochs": 1, "warmup_recount": True,
                          "seed": int(seed), "device": str(device),
                          "loss": "sum of class-wise mean two-class cross-entropies",
                          "base_normalization": "preserve supplied native norms; normalize only after residual addition",
                          "positive_base_norms": np.linalg.norm(qpos, axis=-1).tolist(),
                          "negative_base_norms": np.linalg.norm(qneg, axis=-1).tolist(),
                          "augmentation": "none; frozen deterministic single-view cache"},
        "selection_metric": "pooled support-only OOF macro trapezoidal AUPRC",
        "selection_tie_break": "within 1e-12: smallest epoch, then smallest LR",
        "folds": fold_audit, "optimization_paths": path_audit,
        "final_fit": final_fit_info,
        "trainable_parameter_count": model.text_feature_residuals.numel(),
        "total_fit_seconds": time.perf_counter() - start,
        "support_input_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
        "torch_version": torch.__version__,
    }
    # Fail at construction rather than discovering non-JSON metadata after a run.
    json.dumps(fit_info, allow_nan=False)
    return model, fit_info
