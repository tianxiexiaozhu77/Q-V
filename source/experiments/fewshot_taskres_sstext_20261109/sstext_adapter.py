"""Support-only feature adapter for the official SS-Text+ closed-form solver.

Source: https://github.com/jusiro/SS-Text, commit
4bc0c9e3b7b9e0f090079bcc51d7aa852806e91f. See ``SSTEXT_NOTES.txt``.
``logit_scale`` is the pretrained *logarithmic* scale, not its exponential.
No inputs are silently unit-normalized before fitting: the upstream solver
uses the cached feature/prototype values directly. Prediction normalizes each
image and each fitted prototype, after prototype repulsion, as upstream does.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Mapping

import numpy as np
import torch


SOURCE_URL = "https://github.com/jusiro/SS-Text"
SOURCE_COMMIT = "4bc0c9e3b7b9e0f090079bcc51d7aa852806e91f"
SOURCE_SHA256 = {
    "solvers/sstext.py": "1ccd485b9f3d5982badd5526b8785af5d0f1ec35e0a57ea2f13ae807a1fc7602",
    "solvers/runner.py": "122fd8bb47d56b15118f06fd73e8ae4cc2bfb8f72a6e3c68f2dd867cd7f8ee5f",
    "modeling/adapters/models.py": "d6c30b47b1ce6feccfdb260f5e12aeb412874447d2031b4128c603b13272b481",
}
PORT_DIFFERENCES = [
    "One independent binary [negative, positive] classifier per disease, retaining all support labels.",
    "Class identities are fixed to [0, 1], never discovered from test labels.",
    "Predict returns positive-minus-negative logits rather than multiclass softmax probabilities.",
    "Explicit device and finite/nonzero-input checks replace upstream implicit globals and NaNs.",
    "No support resampling, validation set, CV tuning, or test-distribution statistics.",
]


def _float_tensor(value, name: str, *, device: torch.device) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _nonzero_rows(value: torch.Tensor, name: str) -> None:
    norms = value.norm(dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError(f"{name} must have finite, nonzero row norms")


def _scale_tensor(logit_scale, device: torch.device) -> torch.Tensor:
    value = _float_tensor(logit_scale, "logit_scale", device=device)
    if value.numel() != 1:
        raise ValueError("logit_scale must be a scalar natural-log pretrained scale")
    value = value.reshape(())
    scale = value.exp().item()
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("exp(logit_scale) must be positive and finite in float32")
    return value


class SStextModel:
    """Frozen binary prototypes of shape C x 2 x D, in [negative, positive] order."""

    def __init__(self, prototypes, logit_scale, *, device="cpu"):
        self.device = torch.device(device)
        self._prototypes = _float_tensor(prototypes, "prototypes", device=self.device)
        if self._prototypes.ndim != 3 or self._prototypes.shape[1] != 2:
            raise ValueError("prototypes must have shape C x 2 x D")
        if self._prototypes.shape[0] < 1 or self._prototypes.shape[2] < 1:
            raise ValueError("prototypes must have at least one disease and feature")
        _nonzero_rows(self._prototypes, "prototypes")
        self._logit_scale = _scale_tensor(logit_scale, self.device)

    @torch.no_grad()
    def predict(self, test_features) -> np.ndarray:
        """Return N x C positive-minus-negative logits using no fitted test statistics."""
        features = _float_tensor(test_features, "test_features", device=self.device)
        if features.ndim != 2 or features.shape[1] != self._prototypes.shape[2]:
            raise ValueError("test_features must have shape N x D with the fitted D")
        _nonzero_rows(features, "test_features")
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        margins = []
        # Preserve upstream LinearProbeHead's normalization and operation order.
        for prototypes in self._prototypes:
            prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)
            logits = image_features_norm @ prototypes_norm.t() * self._logit_scale.exp()
            margins.append(logits[:, 1] - logits[:, 0])
        return torch.stack(margins, dim=1).cpu().numpy()

    def export(self) -> dict[str, np.ndarray]:
        """Array-only state suitable for np.savez and np.load(allow_pickle=False)."""
        return {
            "format_version": np.array(1, dtype=np.int64),
            "prototypes": self._prototypes.detach().cpu().numpy().copy(),
            "logit_scale": self._logit_scale.detach().cpu().numpy().copy(),
            "class_order": np.array([0, 1], dtype=np.int64),
            "source_commit": np.array(SOURCE_COMMIT),
        }

    @classmethod
    def from_export(cls, state: Mapping, *, device="cpu") -> "SStextModel":
        if int(np.asarray(state["format_version"]).item()) != 1:
            raise ValueError("Unsupported SStextModel export format")
        if not np.array_equal(np.asarray(state["class_order"]), [0, 1]):
            raise ValueError("Exported class order must be [negative=0, positive=1]")
        if str(np.asarray(state["source_commit"]).item()) != SOURCE_COMMIT:
            raise ValueError("Exported SS-Text source commit differs from this implementation")
        return cls(state["prototypes"], state["logit_scale"], device=device)


@torch.no_grad()
def fit_sstext(
    support_features,
    labels,
    queries_positive,
    queries_negative,
    *,
    logit_scale,
    device="cpu",
    text_lambda="adaptive_shots",
    balanced_importance=False,
    repel=True,
) -> tuple[SStextModel, dict]:
    """Fit independent binary/OvR SS-Text+ classifiers from support data only.

    Inputs: features N x D; binary labels N x C; positive/negative queries C x D.
    Defaults are exactly the official ``SStext+`` runner configuration. Its
    per-class lambda is max(1 / support_count, 1e-3). A missing support class has
    infinite lambda and zero visual contribution, as in the official solver;
    repulsion is still applied to both prototypes. Non-default booleans expose
    official solver options but are explicitly recorded as a solver variant.
    ``adaptive_perf``/arbitrary lambda modes are intentionally not supported.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    device = torch.device(device)
    if text_lambda != "adaptive_shots":
        raise ValueError("This SS-Text+ port fixes text_lambda='adaptive_shots'")
    if not isinstance(balanced_importance, bool) or not isinstance(repel, bool):
        raise ValueError("balanced_importance and repel must be booleans")
    features = _float_tensor(support_features, "support_features", device=device)
    y = _float_tensor(labels, "labels", device=device)
    q_positive = _float_tensor(queries_positive, "queries_positive", device=device)
    q_negative = _float_tensor(queries_negative, "queries_negative", device=device)
    scale = _scale_tensor(logit_scale, device)
    if features.ndim != 2 or min(features.shape) < 1:
        raise ValueError("support_features must have nonempty shape N x D")
    if y.ndim != 2 or y.shape[0] != features.shape[0] or y.shape[1] < 1:
        raise ValueError("labels must have shape N x C, with C >= 1")
    if not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("labels must be binary 0/1; unknown labels are not silently discarded")
    expected_shape = (y.shape[1], features.shape[1])
    if tuple(q_positive.shape) != expected_shape or tuple(q_negative.shape) != expected_shape:
        raise ValueError("queries_positive and queries_negative must have shape C x D")
    _nonzero_rows(features, "support_features")
    _nonzero_rows(q_positive, "queries_positive")
    _nonzero_rows(q_negative, "queries_negative")
    y = y.to(torch.long)
    n = features.shape[0]
    tau = 1 / scale.exp().item()
    fitted, counts_all, lambdas_all = [], [], []
    for disease in range(y.shape[1]):
        # Exact ss_text_solver formula and ordering, applied to one binary task.
        binary_labels = y[:, disease]
        affinity_labeled = torch.nn.functional.one_hot(binary_labels, num_classes=2).float()
        vision_mu = torch.einsum("ij,ik->jk", affinity_labeled, features) / tau
        text_mu = torch.stack((q_negative[disease], q_positive[disease]), dim=0)
        counts = affinity_labeled.sum(dim=0).to(torch.long)
        if balanced_importance:
            lambda_text = torch.tensor(1 / n, dtype=torch.float32, device=device)
        else:
            # Upstream computes reciprocal counts in NumPy float64, then casts
            # to float32. Preserve that order, including inf for absent classes.
            with np.errstate(divide="ignore"):
                reciprocal = 1 / counts.cpu().numpy()
            lambda_text = torch.as_tensor(reciprocal, dtype=torch.float32, device=device)
        lambda_text = lambda_text.clamp(min=1e-3)
        new_mu = (1 / n) * (1 / lambda_text).unsqueeze(-1) * vision_mu + text_mu
        if repel:
            delta = 2 * new_mu - new_mu.mean(0)
            _nonzero_rows(delta, "prototype repulsion directions")
            delta /= delta.norm(dim=-1, keepdim=True)
            new_mu = new_mu - delta
        fitted.append(new_mu)
        counts_all.append(counts.cpu().tolist())
        # JSON-safe metadata: null denotes +inf for a missing support class.
        weights = lambda_text.reshape(-1).cpu().tolist()
        lambdas_all.append([value if math.isfinite(value) else None for value in weights])
    model = SStextModel(torch.stack(fitted), scale, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    info = {
        "method": "SStext+" if not balanced_importance and repel else "ss_text_solver_variant",
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": dict(SOURCE_SHA256),
        "started_at_utc": started_at,
        "fit_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "config": {
            "text_lambda": text_lambda,
            "balanced_importance": balanced_importance,
            "repel": repel,
            "lambda_min": 1e-3,
            "logit_scale": float(scale.item()),
            "logit_scale_parameterization": "natural_log",
            "dtype": "float32",
        },
        "num_support": n,
        "num_diseases": y.shape[1],
        "feature_dimension": features.shape[1],
        "class_order": ["negative", "positive"],
        "support_counts": counts_all,
        "lambda_text": lambdas_all,
        "lambda_null_means": "+infinity for a missing support class",
        "validation_used": False,
        "test_information_used": False,
        "port_differences": list(PORT_DIFFERENCES),
    }
    return model, info
