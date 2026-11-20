"""Image-level metrics for the released RadZero evaluation splits.

PR AUC follows external/CARZero/utils.py: precision_recall_curve followed by
trapezoidal auc(recall, precision), rather than average_precision_score.  OpenI
is evaluated per image, with no study aggregation, as in the released evaluator.
"""

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def _binary_metrics(labels, scores):
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    result = {"positives": positives, "negatives": negatives}
    if positives and negatives:
        result["auroc"] = float(roc_auc_score(labels, scores))
    else:
        result["auroc"] = None
    if positives:
        precision, recall, _ = precision_recall_curve(labels, scores)
        result["auprc"] = float(auc(recall, precision))
    else:
        # Without any positive example, PR AUC is not a meaningful metric.
        # This also makes limited smoke runs JSON-safe without warning floods.
        result["auprc"] = None
    return result


def _mean_defined(values):
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def compute_metrics(spec, logits):
    """Return JSON-compatible classification results in the supplied row order.

    ``spec`` is returned by batch_data.load_dataset.  For a subset, callers must
    slice spec['labels'] and spec['image_paths'] to match logits first.  Undefined
    AUROC (one target value) and PR AUC (no positives) are reported as null, with
    macro coverage counts.  The full released splits have valid binary targets.
    """
    labels = np.asarray(spec["labels"])
    scores = np.asarray(logits)
    class_names = list(spec["class_names"])
    if labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError(
            f"Expected matching [images, classes] labels/logits, got "
            f"{labels.shape} and {scores.shape}"
        )
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("Cannot compute metrics for an empty dataset or class list")
    if labels.shape[1] != len(class_names) or len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique and match the labels/logits columns")
    if "image_paths" in spec and len(spec["image_paths"]) != labels.shape[0]:
        raise ValueError("Image paths and labels/logits must have the same row count")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Metrics require binary labels encoded as 0/1")
    if not np.isfinite(scores).all():
        raise ValueError("Cannot compute metrics from non-finite logits")

    labels = labels.astype(np.int8, copy=False)
    name = spec.get("name", spec.get("metadata", {}).get("dataset", "unknown"))
    per_class = {
        class_name: _binary_metrics(labels[:, index], scores[:, index])
        for index, class_name in enumerate(class_names)
    }
    per_auroc = [entry["auroc"] for entry in per_class.values()]
    per_auprc = [entry["auprc"] for entry in per_class.values()]
    micro = _binary_metrics(labels.ravel(), scores.ravel())
    result = {
        "dataset": name,
        "evaluation_unit": "image",
        "sample_count": int(labels.shape[0]),
        "class_count": int(labels.shape[1]),
        "score_type": "raw image-text logits (before sigmoid)",
        "auprc_method": "trapezoidal auc(recall, precision) from precision_recall_curve",
        "per_class": per_class,
        "macro_auroc": _mean_defined(per_auroc),
        "macro_auprc": _mean_defined(per_auprc),
        "micro_auroc": micro["auroc"],
        "micro_auprc": micro["auprc"],
        "macro_auroc_defined_classes": sum(value is not None for value in per_auroc),
        "macro_auprc_defined_classes": sum(value is not None for value in per_auprc),
        "undefined_metric_policy": "null per class; macro averages defined classes only",
    }

    if name.casefold() == "openi":
        frequency = labels.sum(axis=0)
        head_threshold = float(np.percentile(frequency, 65))
        group_masks = {
            "head": frequency > head_threshold,
            "medium": (frequency <= head_threshold) & (frequency > 10),
            "tail": frequency <= 10,
        }
        result["study_aggregation"] = "none; released evaluation scores each image independently"
        result["openi_groups"] = {
            "frequency_unit": "positive images per class in the evaluated split",
            "head_threshold_percentile": 65,
            "head_threshold_count": head_threshold,
            "medium_threshold_count": 10,
        }
        for group_name, mask in group_masks.items():
            indices = np.flatnonzero(mask).tolist()
            values = [per_auroc[index] for index in indices]
            result["openi_groups"][group_name] = {
                "class_names": [class_names[index] for index in indices],
                "class_count": len(indices),
                "defined_auroc_class_count": sum(value is not None for value in values),
                "macro_auroc": _mean_defined(values),
            }
    return result
