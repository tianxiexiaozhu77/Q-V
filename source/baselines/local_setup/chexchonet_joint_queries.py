"""Share frozen image forwards across independently fitted CheXchoNet tasks.

This is an execution optimization only: each binary task still fits its query
using its own fixed ten support images.  No support pools or class statistics
are combined.  The historical model adapters and their score scales stay
unchanged; numerical equivalence is checked before full evaluation.
"""
from __future__ import annotations

import numpy as np
import torch

from evaluate_query_transfer import check_native, load_query


SUPPORTED_MODELS = ("BiomedCLIP", "BioViL-T", "CheXzero", "MaCo")


def build_joint_query(model, native, task_inputs, batch_size=8):
    """Return ``(joint_query, individual_queries, fit_records)``.

    ``task_inputs`` is an ordered list of dictionaries containing ``classes``
    (one disease name) and ``support_rows`` (``image`` and binary ``labels``).
    The fit record list has the same order as the input tasks.  Callers must
    retain the pre-existing patient-disjoint evaluation manifest separately.
    """
    if model not in SUPPORTED_MODELS:
        raise NotImplementedError(f"Joint query inference unavailable for {model}")
    if not task_inputs or batch_size < 1:
        raise ValueError("Nonempty tasks and a positive batch size are required")

    all_classes, individual, records = [], [], []
    for task in task_inputs:
        classes = list(task["classes"])
        rows = task["support_rows"]
        if len(classes) != 1 or not isinstance(classes[0], str) or not classes[0].strip():
            raise ValueError("Each CheXchoNet task must contain one named binary class")
        if classes[0] in all_classes:
            raise ValueError("Repeated task class names cannot be concatenated")
        labels = np.asarray([row["labels"] for row in rows])
        if labels.shape != (10, 1) or not np.isin(labels, [0, 1]).all() or labels.sum() != 5:
            raise ValueError("Each task must retain exactly its fixed 5 positive and 5 negative supports")
        images = [str(row["image"]) for row in rows]
        if len(set(images)) != 10:
            raise ValueError("Each task support set must contain ten distinct images")
        query = load_query(model, native)
        record = query.fit(images, labels.tolist(), classes,
                           batch_size=batch_size, task="classification")
        all_classes.extend(classes)
        individual.append(query)
        records.append(record)

    joint = load_query(model, native)
    class_key = tuple(all_classes)
    if model in ("BiomedCLIP", "BioViL-T"):
        joint._positive = torch.cat([query._positive for query in individual], dim=0)
        joint._fit_key = class_key
        if model == "BioViL-T":
            joint._spatial_positive = torch.cat(
                [query._spatial_positive for query in individual], dim=0)
    elif model == "CheXzero":
        member_count = len(native.models)
        if member_count != 10 or any(len(query._positive) != member_count for query in individual):
            raise ValueError("CheXzero must preserve its complete ten-checkpoint ensemble")
        joint._positive = [torch.cat([query._positive[index] for query in individual], dim=0)
                           for index in range(member_count)]
        joint._fit_key = class_key
    elif model == "MaCo":
        joint.fits[(class_key, "classification")] = torch.cat(
            [query.fits[((name,), "classification")]
             for name, query in zip(all_classes, individual)], dim=0)

    joint.metadata.update(
        joint_inference=True,
        independent_task_support_count=10,
        joint_class_order=all_classes,
        support_pool_merged=False,
        image_forward_sharing="One image forward per view and native ensemble member for all task queries",
    )
    return joint, individual, records


@torch.inference_mode()
def check_joint(joint, individual_queries, paths, classes):
    """Verify shared and per-task execution for both views and both branches.

    Original-view output is additionally compared to the unmodified native
    adapter on the same image batch and complete ordered class list.  Uses the
    historical evaluator's tolerance: absolute 2e-6 and relative 1e-5.
    """
    classes = list(classes)
    if len(classes) != len(individual_queries) or not paths:
        raise ValueError("Nonempty images and one class per individual query are required")
    report = {"passed": True, "image_count": len(paths), "class_names": classes,
              "rtol": 1e-5, "atol": 2e-6, "views": {}}
    for view in ("identity", "center80"):
        combined = joint.predict_pair(paths, classes, view=view)
        separate = [query.predict_pair(paths, [name], view=view)
                    for name, query in zip(classes, individual_queries)]
        report["views"][view] = {
            branch: check_native(combined[branch], np.concatenate(
                [prediction[branch] for prediction in separate], axis=1))
            for branch in ("baseline", "refined")
        }
        if view == "identity":
            report["native_baseline"] = check_native(
                combined["baseline"], joint.native.predict(paths, classes))
    return report
