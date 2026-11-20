"""SimNL feature-only one-vs-rest transfer, NOT the authors' exact protocol.

Source: zhangce01/SimNL @ a0871bcf31a607a7ec8acc99aff82f41ed7eaff3.
Retains positive/negative trainable cache/text residuals, pseudo-negative cache,
confidence-weighted positive cache, CE + adaptive reranking + text consistency,
AdamW (negative LR x5), and cosine decay. See SOURCES_SIMNL.json for departures.

Only support labels reach fit. Epoch selection uses caller-owned group folds;
test features and native logits enter predict only after selection. The native
logits are checked for shape but are NOT mixed with SimNL's scale-100 scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import auc, precision_recall_curve


SOURCE_COMMIT = "a0871bcf31a607a7ec8acc99aff82f41ed7eaff3"
METHOD_NAME = "simnl_feature_ovr"


@dataclass(frozen=True)
class SimNLConfig:
    alpha: float = 5.0
    beta: float = 1.0
    positive_weight: float = 0.75
    learning_rate: float = 1e-4
    adam_eps: float = 1e-3
    weight_decay: float = 0.1
    negative_lr_multiplier: float = 5.0
    text_scale: float = 100.0
    negative_text_scale: float = 0.15
    consistency_weight: float = 8.0
    template_weight: float = 0.45
    reranking_scale: float = 4.0
    reranking_knn: int = 3


def _array(value: Any, name: str, ndim: int = 2) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite {ndim}-dimensional array")
    return a


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError(f"{name}: zero feature/query vector")
    return value / norms


def validate_folds(folds: Sequence, groups: np.ndarray, labels: np.ndarray):
    """Validate fixed out-of-fold split; never repair it using test data."""
    n = len(groups)
    coverage = np.zeros(n, dtype=np.int64)
    clean = []
    universe = set(range(n))
    for k, fold in enumerate(folds):
        if isinstance(fold, dict):
            train, validation = fold["train"], fold["validation"]
        else:
            train, validation = fold
        tr, va = np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)
        if tr.ndim != 1 or va.ndim != 1 or not len(tr) or not len(va):
            raise ValueError(f"fold {k}: empty or malformed indices")
        if len(set(tr.tolist())) != len(tr) or len(set(va.tolist())) != len(va):
            raise ValueError(f"fold {k}: duplicate row indices")
        if set(tr.tolist()).intersection(va.tolist()):
            raise ValueError(f"fold {k}: train/validation row overlap")
        if set(tr.tolist()).union(va.tolist()) != universe:
            raise ValueError(f"fold {k}: train/validation must partition support rows")
        if set(groups[tr].tolist()).intersection(groups[va].tolist()):
            raise ValueError(f"fold {k}: patient/group leakage")
        counts = labels[tr].sum(axis=0)
        missing = np.where((counts == 0) | (counts == len(tr)))[0]
        if len(missing):
            raise ValueError(f"fold {k}: training has one class for disease(s) {missing.tolist()}")
        coverage[va] += 1
        clean.append((tr, va))
    if len(clean) < 2 or not np.all(coverage == 1):
        raise ValueError("fold validation rows must cover every support row exactly once")
    return clean


def confidence_cache_values(keys: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Official within-class confidence, generalized to unequal class counts.

    Each class has total cache weight N/2. This equals the source's K-weight
    sum when both classes contain K examples. Singleton confidence is one.
    """
    values = torch.zeros((len(keys), 2), device=keys.device, dtype=keys.dtype)
    for cls in (0, 1):
        idx = targets == cls
        n = int(idx.sum())
        if n == 0:
            raise ValueError("both OvR support classes are required")
        class_keys = keys[idx]
        if n == 1:
            confidence = torch.ones(1, device=keys.device, dtype=keys.dtype)
        else:
            similarity = class_keys @ class_keys.T
            confidence = torch.softmax((similarity.sum(1) - 1) / (n - 1), dim=0)
        values[idx, cls] = confidence * (len(keys) / 2)
    return values


def negative_cache_values(targets: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Binary complement cache: every support item is negative for its other class.

    A balanced 2-class cache is precisely the official other-class cache up to
    row permutation. With unequal classes we keep every item, with N/2 total
    weight per target class, avoiding arbitrary duplication or dropping data.
    """
    destination = 1 - targets
    values = F.one_hot(destination, num_classes=2).to(dtype=dtype)
    for cls in (0, 1):
        idx = destination == cls
        values[idx] *= len(targets) / (2 * int(idx.sum()))
    return values


def adaptive_reranking_loss(features, prototypes, targets, scale=4.0, knn=3):
    distances = torch.cdist(features.float(), prototypes.float(), p=2)
    near_distance, near_idx = torch.sort(distances, dim=1)
    knn = min(knn, len(prototypes))
    positive = prototypes[targets]
    anchor = torch.linalg.vector_norm(features - positive, dim=1, keepdim=True)
    margins = (1 - (prototypes[near_idx[:, :knn]] * positive[:, None]).sum(-1)) / scale
    return (anchor + margins - near_distance[:, :knn]).clamp_min(0).mean()


class PositiveAdapter(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.res_template = nn.Parameter(torch.zeros(2, dimensions))
        self.res_cupl = nn.Parameter(torch.zeros(2, dimensions))
        self.res_keys = nn.Parameter(torch.zeros(2, dimensions))


class NegativeAdapter(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.res_negative = nn.Parameter(torch.zeros(2, dimensions))
        self.res_keys = nn.Parameter(torch.zeros(2, dimensions))


class BinarySimNL(nn.Module):
    """Trainable residuals only; all supplied representations are detached."""
    def __init__(self, support_features, labels, positive_query, negative_query,
                 config: SimNLConfig):
        super().__init__()
        features = torch.as_tensor(support_features, dtype=torch.float32).detach().clone()
        targets = torch.as_tensor(labels, dtype=torch.long).detach().clone()
        if set(targets.tolist()) != {0, 1}:
            raise ValueError("BinarySimNL needs both support classes")
        query = torch.stack((torch.as_tensor(negative_query), torch.as_tensor(positive_query))).float()
        query = F.normalize(query.detach().clone(), dim=1)
        self.config = config
        self.register_buffer("support_features", features)
        self.register_buffer("targets", targets)
        self.register_buffer("templates", query)
        self.register_buffer("negative_templates", query.flip(0))
        self.register_buffer("cache_values", confidence_cache_values(features, targets))
        self.register_buffer("negative_values", negative_cache_values(targets, features.dtype))
        self.positive_adapter = PositiveAdapter(features.shape[1])
        self.negative_adapter = NegativeAdapter(features.shape[1])

    def components(self):
        p, n = self.positive_adapter, self.negative_adapter
        positive_keys = F.normalize(self.support_features + p.res_keys[self.targets], dim=1)
        negative_keys = F.normalize(self.support_features + n.res_keys[1 - self.targets], dim=1)
        template = F.normalize(self.templates + p.res_template, dim=1)
        # No extra CuPL/GPT resource: a separately adaptable copy of the same
        # original text vector replaces the official second description bank.
        cupl = F.normalize(self.templates + p.res_cupl, dim=1)
        text = self.config.template_weight * template + (1 - self.config.template_weight) * cupl
        negative_text = F.normalize(self.negative_templates + n.res_negative, dim=1)
        return positive_keys, negative_keys, text, negative_text

    def forward(self, features):
        positive_keys, negative_keys, text, negative_text = self.components()
        cfg = self.config
        positive_affinity = torch.exp(-cfg.beta * (1 - features @ positive_keys.T))
        # This is exp(-beta * similarity), not the positive affinity again:
        # source R_fF2 = 1 - cosine, exp(-beta * (1 - R_fF2)).
        negative_affinity = torch.exp(-cfg.beta * (features @ negative_keys.T))
        positive = cfg.text_scale * (features @ text.T) + cfg.alpha * positive_affinity @ self.cache_values
        negative = (cfg.text_scale * cfg.negative_text_scale * (1 - features @ negative_text.T)
                    + cfg.alpha * negative_affinity @ self.negative_values)
        return cfg.positive_weight * positive + (1 - cfg.positive_weight) * negative

    def loss(self):
        logits = self(self.support_features)
        _, _, text, _ = self.components()
        ce = F.cross_entropy(logits, self.targets)
        rerank = adaptive_reranking_loss(self.support_features, text, self.targets,
                                       self.config.reranking_scale, self.config.reranking_knn)
        # Preserve official axis: source has [D,C] text and CosineSimilarity(dim=1).
        # With identical frozen template/CuPL banks the .45/.55 targets coincide.
        consistency = (1 - F.cosine_similarity(text.T, self.templates.T, dim=1, eps=1e-7)).mean()
        total = ce + rerank + self.config.consistency_weight * consistency
        return total, {"cross_entropy": float(ce.detach()), "reranking": float(rerank.detach()),
                       "text_consistency": float(consistency.detach())}


def _train_snapshots(features, labels, positive_query, negative_query, *, epochs,
                     horizon, config, device, validation_features=None):
    model = BinarySimNL(features, labels, positive_query, negative_query, config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.positive_adapter.parameters(), "lr": config.learning_rate},
        {"params": model.negative_adapter.parameters(),
         "lr": config.learning_rate * config.negative_lr_multiplier},
    ], eps=config.adam_eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(horizon, 1))
    requested = set(epochs)
    predictions, losses = {}, []
    validation = None if validation_features is None else torch.as_tensor(
        validation_features, dtype=torch.float32, device=device)
    for epoch in range(max(requested) + 1):
        if epoch in requested and validation is not None:
            with torch.no_grad():
                logits = model(validation)
                predictions[epoch] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        if epoch == max(requested):
            break
        loss, parts = model.loss()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite SimNL loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"nonfinite SimNL gradient at epoch {epoch}")
        optimizer.step()
        scheduler.step()
        losses.append({"epoch": epoch + 1, "total": float(loss.detach()), **parts})
    return model.eval(), predictions, losses


def _macro_auprc(labels, scores):
    per_class = []
    for col in range(labels.shape[1]):
        precision, recall, _ = precision_recall_curve(labels[:, col], scores[:, col])
        per_class.append(float(auc(recall, precision)))
    return float(np.mean(per_class)), per_class


class FittedSimNLOvR:
    def __init__(self, models, fit_info, dimensions, device):
        self.models = models
        self.fit_info = fit_info
        self.dimensions = dimensions
        self.device = device

    def predict(self, test_features, test_logits, *, batch_size=1024):
        features = _unit(_array(test_features, "test_features"), "test_features")
        logits = _array(test_logits, "test_logits")
        if features.shape[1] != self.dimensions or logits.shape != (len(features), len(self.models)):
            raise ValueError("test shape does not match fitted feature dimensions / classes")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        result = np.empty_like(logits)
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                x = torch.as_tensor(features[start:start + batch_size], device=self.device)
                for col, model in enumerate(self.models):
                    score = model(x)
                    result[start:start + batch_size, col] = (score[:, 1] - score[:, 0]).cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite SimNL prediction")
        return result

    def to_json(self):
        return self.fit_info


def fit_simnl_ovr(support_features, labels, support_logits, positive_queries,
                  negative_queries, groups, *, folds, epoch_candidates=(0, 10, 20, 40),
                  device="cpu", seed=42, config=None):
    """Choose one shared epoch by pooled support OOF macro trapezoidal AUPRC.

    Every fold recomputes visual/cache statistics and fits new adapters using
    only its training rows. Selection has no test-label or test-feature input.
    The selected checkpoint epoch is refit on all support rows with the SAME
    cosine horizon used during CV, not a shortened/restarted cosine schedule.
    """
    features = _unit(_array(support_features, "support_features"), "support_features")
    labels = _array(labels, "labels")
    logits = _array(support_logits, "support_logits")
    positive = _unit(_array(positive_queries, "positive_queries"), "positive_queries")
    negative = _unit(_array(negative_queries, "negative_queries"), "negative_queries")
    groups = np.asarray(groups)
    n, d = features.shape
    if labels.shape[0] != n or logits.shape != labels.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("support labels/logits must align and labels must be binary")
    c = labels.shape[1]
    if c == 0 or groups.shape != (n,) or positive.shape != (c, d) or negative.shape != (c, d):
        raise ValueError("support group/query shape mismatch")
    if not np.all((labels.sum(0) > 0) & (labels.sum(0) < n)):
        raise ValueError("all diseases need both positive and negative support")
    epochs = sorted(set(int(x) for x in epoch_candidates))
    if not epochs or epochs[0] < 0 or any(float(x) != int(x) for x in epoch_candidates):
        raise ValueError("epoch_candidates must be nonnegative integers")
    cfg = config or SimNLConfig()
    clean_folds = validate_folds(folds, groups, labels)
    oof = {epoch: np.full((n, c), np.nan, dtype=np.float32) for epoch in epochs}
    # No random mini-batch ordering or random adapter initialization: zero
    # residuals + one full support batch per epoch. Seed remains in provenance.
    for fold_id, (tr, va) in enumerate(clean_folds):
        for col in range(c):
            _, scores, _ = _train_snapshots(features[tr], labels[tr, col], positive[col], negative[col],
                                            epochs=epochs, horizon=max(epochs), config=cfg, device=device,
                                            validation_features=features[va])
            for epoch in epochs:
                oof[epoch][va, col] = scores[epoch]
    table = []
    for epoch in epochs:
        if not np.isfinite(oof[epoch]).all():
            raise FloatingPointError("OOF predictions incomplete or nonfinite")
        macro, class_scores = _macro_auprc(labels, oof[epoch])
        table.append({"epoch": epoch, "macro_auprc_trapezoidal": macro,
                      "per_class_auprc_trapezoidal": class_scores,
                      "oof_scores_sha256": hashlib.sha256(oof[epoch].tobytes()).hexdigest()})
    chosen = max(table, key=lambda row: (row["macro_auprc_trapezoidal"], -row["epoch"]))["epoch"]
    models, training = [], []
    for col in range(c):
        model, _, losses = _train_snapshots(features, labels[:, col], positive[col], negative[col],
                                            epochs=[chosen], horizon=max(epochs), config=cfg, device=device)
        models.append(model)
        training.append({"class_index": col, "support_positive": int(labels[:, col].sum()),
                         "support_negative": int(n - labels[:, col].sum()),
                         "losses": losses})
    info = {
        "method": METHOD_NAME, "source_commit": SOURCE_COMMIT,
        "protocol": "feature-only binary one-vs-rest transfer; not exact author reproduction",
        "config": asdict(cfg), "seed": int(seed), "deterministic_full_batch": True,
        "chosen_epoch": chosen, "epoch_candidates": epochs, "cosine_horizon": max(epochs),
        "selection_metric": "pooled support OOF macro trapezoidal AUPRC",
        "selection_scope": "one epoch shared by every disease", "oof_table": table,
        "folds": [{"train": tr.tolist(), "validation": va.tolist()} for tr, va in clean_folds],
        "support_count": n, "group_count": len(set(groups.tolist())), "class_count": c,
        "trainable_parameters": sum(p.numel() for model in models for p in model.parameters()),
        "native_logits_used_for_fitting_or_scores": False,
        "score_domain": "SimNL two-class logit difference, scale-100 text plus positive/negative cache",
        "test_labels_received": False, "spatial_output": "not implemented / not applicable",
        "training": training,
        "departures": ["no image augmentation or image encoder updates",
                       "same native text bank in template and CuPL slots; no external GPT descriptions",
                       "each disease is a separate two-class problem; absence/presence templates are complements",
                       "unequal class cache totals normalized to N/2; all support examples retained",
                       "separate positive confidence and negative complement values; no upstream in-place cache_values alias",
                       "full batch FP32; official inactive res_keys2 parameter omitted",
                       "shared support-group OOF AUPRC epoch choice instead of external validation accuracy"],
    }
    return FittedSimNLOvR(models, info, d, device)


def run_simnl(*, support_features, labels, support_logits, positive_queries=None,
              negative_queries=None, queries_positive=None, queries_negative=None,
              groups, test_features, test_logits, folds, device="cpu", seed=42,
              epoch_candidates=(0, 10, 20, 40), config=None):
    """Convenience API matching run_score_methods' scores/fit_info result shape."""
    positive = positive_queries if positive_queries is not None else queries_positive
    negative = negative_queries if negative_queries is not None else queries_negative
    if positive is None or negative is None:
        raise ValueError("real positive and negative native text features are required")
    model = fit_simnl_ovr(support_features, labels, support_logits, positive, negative,
                         groups, folds=folds, device=device, seed=seed,
                         epoch_candidates=epoch_candidates, config=config)
    return {"scores": {METHOD_NAME: model.predict(test_features, test_logits)},
            "fit_info": {METHOD_NAME: model.fit_info}}
