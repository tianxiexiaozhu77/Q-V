#!/usr/bin/env python3
"""Re-score existing fixed-support RSNA predictions after patient exclusion.

Only this script's output directory is written. Historical predictions, the
verified-results catalog, and all corrected repeated-support runs are read-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")
METRICS = ("macro_auroc", "macro_auprc", "macro_average_precision")
EXPECTED_MODELS = {"AlphaRAD", "RadZero", "BiomedCLIP", "BioViL-T", "CheXzero", "MedKLIP", "MAVL", "KAD", "MaCo", "CARZero"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def metrics(labels, scores):
    y, s = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    assert set(np.unique(y)) == {0, 1} and np.isfinite(s).all()
    precision, recall, _ = precision_recall_curve(y, s)
    roc = float(roc_auc_score(y, s))
    pr = float(auc(recall, precision))
    ap = float(average_precision_score(y, s))
    return {
        "per_class": {"Pneumonia": {"auroc": roc, "auprc": pr, "average_precision": ap}},
        "valid_class_count": 1, "class_count": 1, "sample_count": len(y),
        "macro_auroc": roc, "macro_auprc": pr, "macro_average_precision": ap,
        "auprc_method": "Trapezoidal area under sklearn precision_recall_curve; distinct from average precision.",
        "average_precision_method": "sklearn.metrics.average_precision_score",
    }


def change_class(branches):
    differences = [branches["tta_center_refined"][m] - branches["original"][m]
                   for m in ("macro_auroc", "macro_auprc")]
    return ("all_primary_up" if all(v > 0 for v in differences) else
            "all_primary_down" if all(v < 0 for v in differences) else "mixed_or_tied")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def register(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected:
            assert digest == expected, f"Source hash mismatch: {path}"
        if str(path) in inputs:
            assert inputs[str(path)] == digest
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(register(path, expected).read_text())

    catalog_path = root / "baselines/paper_draft_zh/verified_results.json"
    snapshot_path = out / "original_records.json"
    if not snapshot_path.exists():
        catalog = read(catalog_path)
        old_records = [r for r in catalog["records"] if r["dataset"] == "rsna_pneumonia" and r["status"] == "complete"]
        assert all(r["rows"] == 5327 and "postprocessing" not in r for r in old_records), "Cannot freeze already corrected rows"
        dump(snapshot_path, {"historical_catalog_path": str(catalog_path),
                             "historical_catalog_sha256": inputs[str(catalog_path)], "records": old_records})
    snapshot = read(snapshot_path)
    old_records = snapshot["records"]
    catalog_hash = snapshot["historical_catalog_sha256"]
    assert len(old_records) == 10 and {r["model"] for r in old_records} == EXPECTED_MODELS
    mapping_path = root / "experiments/rsna_chest_stability_20261024/source_data/pneumonia-challenge-dataset-mappings_2018.json"
    mapping_audit = read(root / "experiments/rsna_chest_stability_20261024/source_data/RSNA_NIH_PATIENT_ID_AUDIT.json")
    mapping = read(mapping_path, mapping_audit["official_mapping"]["sha256"])
    identity_map = {}
    for row in mapping:
        for key in (row["subset_img_id"], row["SOPInstanceUID"], Path(row["img_id"]).stem):
            assert key not in identity_map
            identity_map[key] = row

    def identify(path):
        return identity_map[Path(str(path)).stem]

    def exam(path):
        return identify(path)["subset_img_id"]

    def patient(path):
        return "NIH:" + identify(path)["img_id"].split("_")[0]

    original_manifest_path = root / "baselines/evaluation/matched_phase6_20261022/rsna_pneumonia.json"
    original_manifest = read(original_manifest_path)
    original_rows = original_manifest["rows"]
    canonical_exams = [exam(r["image"]) for r in original_rows]
    canonical_labels = np.asarray([r["labels"] for r in original_rows], dtype=np.uint8)
    assert len(original_rows) == 5327 and len(set(canonical_exams)) == 5327
    historical_protocol_path = root / "experiments/fewshot_native_query/protocol_v1/rsna_pneumonia.json"
    historical_protocol = read(historical_protocol_path)
    support_rows = historical_protocol["support_rows"]
    support_exam_labels = {exam(r["image"]): tuple(r["labels"]) for r in support_rows}
    support_patients = {patient(r["image"]) for r in support_rows}
    assert len(support_exam_labels) == 10 and len(support_patients) == 10
    assert not (set(support_exam_labels) & set(canonical_exams))

    # Derive exclusions from the official mapping, not from the older audit's
    # stored list/count. Those counts are only checked after the derivation.
    patients = [patient(r["image"]) for r in original_rows]
    keep = np.asarray([p not in support_patients for p in patients], dtype=bool)
    dropped_indices = np.flatnonzero(~keep).tolist()
    kept_indices = np.flatnonzero(keep).tolist()
    overlap_patients = sorted(set(patients) & support_patients)
    assert len(dropped_indices) == 30 and len(overlap_patients) == 7
    assert not ({patients[i] for i in kept_indices} & support_patients)
    corrected_rows, excluded_rows = [], []
    for i, source_row in enumerate(original_rows):
        row = copy.deepcopy(source_row)
        row.update(original_index=i, exam_id=canonical_exams[i], patient_id=patients[i],
                   nih_image_id=identify(source_row["image"])["img_id"])
        if keep[i]:
            row["index"] = len(corrected_rows)
            corrected_rows.append(row)
        else:
            row["exclusion_reason"] = "NIH patient also appears in the unchanged fixed support set"
            excluded_rows.append(row)

    separation = {
        "original_test_images": len(original_rows), "original_test_patients": len(set(patients)),
        "support_images": len(support_exam_labels), "support_patients": len(support_patients),
        "excluded_test_images": len(dropped_indices), "excluded_test_patients": len(overlap_patients),
        "corrected_test_images": len(corrected_rows),
        "corrected_test_patients": len({patients[i] for i in kept_indices}),
        "corrected_test_positive_images": int(canonical_labels[keep].sum()),
        "corrected_test_negative_images": int(len(corrected_rows) - canonical_labels[keep].sum()),
        "corrected_support_test_patient_overlap": 0,
    }
    manifest = {
        "version": "fixed-rsna-support-patient-exclusion-v1", "dataset": "rsna_pneumonia",
        "task": "classification", "class_names": ["Pneumonia"], "count": len(corrected_rows),
        "source_manifest": str(original_manifest_path), "source_manifest_sha256": inputs[str(original_manifest_path)],
        "official_mapping": {"path": str(mapping_path), "sha256": inputs[str(mapping_path)],
                             "url": mapping_audit["official_mapping"]["url"]},
        "selection_rule": "Keep each original fixed-test image if and only if its officially mapped NIH patient is absent from all ten unchanged fixed-support images.",
        "support_patient_ids": sorted(support_patients), "separation": separation,
        "rows": corrected_rows, "excluded_rows": excluded_rows,
        "original_indices_retained": kept_indices, "original_indices_excluded": dropped_indices,
        "scope": "Only fixed-support RSNA evaluation; corrected repeated-support experiments are unchanged.",
    }
    manifest_path = out / "corrected_test_manifest.json"
    dump(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)
    corrected_records, model_audits, changes = [], [], []
    for old in old_records:
        model = old["model"]
        source_path = Path(old["source_path"])
        source_result = read(source_path, old["source_sha256"])
        source_dir = source_path.parent
        prediction_sources = []
        if model in {"AlphaRAD", "RadZero"}:
            path_file, label_file = source_dir / "image_paths.json", source_dir / "labels.npy"
            prediction_paths = read(path_file)
            labels = np.load(register(label_file), allow_pickle=False)
            prediction_sources += [str(path_file), str(label_file)]
            predictions = {}
            for branch in BRANCHES:
                source_branch = old["source_query_branch_name"] if branch == "query_refined" else branch
                # RadZero's historical rank metrics use logits. A float32
                # sigmoid can introduce extra ties (notably Q+V), so preserve
                # the original metric input instead of substituting scores.
                score_filename = "logits.npy" if model == "RadZero" else "probabilities.npy"
                path = source_dir / "candidates" / source_branch / score_filename
                predictions[branch] = np.load(register(path), allow_pickle=False)
                prediction_sources.append(str(path))
            if model == "AlphaRAD":
                entry = source_result["entry"]
                support_csv = register(entry["support_csv_path"], entry["support_csv_sha256"])
                with support_csv.open() as f:
                    sr = list(csv.DictReader(f))
                model_supports = {exam(r["image"]): tuple(json.loads(r["label"])) for r in sr}
            else:
                cfg = source_result["config"]
                support_csv = Path(cfg["split_dir"]) / "RSNA/support.csv"
                register(support_csv, cfg["split_sha256"][str(support_csv)])
                with support_csv.open() as f:
                    sr = list(csv.DictReader(f))
                model_supports = {exam(r["image_path"]): tuple(json.loads(r["labels_json"])) for r in sr}
        else:
            prediction_file = register(source_dir / "predictions.npz")
            prediction_sources.append(str(prediction_file))
            with np.load(prediction_file, allow_pickle=False) as z:
                labels, prediction_paths = z["labels"], z["image_paths"].tolist()
                predictions = {branch: z[branch] for branch in BRANCHES}
            model_manifest = read(source_dir / "manifest.json")
            model_supports = {exam(r["image"]): tuple(r["labels"]) for r in model_manifest["support_rows"]}
        assert model_supports == support_exam_labels, f"Fixed-support mismatch: {model}"
        model_exams = [exam(p) for p in prediction_paths]
        assert len(model_exams) == len(set(model_exams)) == 5327
        assert set(model_exams) == set(canonical_exams)
        model_index = {e: i for i, e in enumerate(model_exams)}
        reorder = np.asarray([model_index[e] for e in canonical_exams])
        aligned_labels = labels[reorder].reshape(-1, 1)
        assert np.array_equal(aligned_labels, canonical_labels), f"Label mismatch: {model}"
        aligned = {b: np.asarray(p)[reorder].reshape(-1, 1) for b, p in predictions.items()}
        old_recomputed = {b: metrics(aligned_labels, p) for b, p in aligned.items()}
        discrepancies = {b: {m: old_recomputed[b][m] - old["branches"][b][m] for m in METRICS} for b in BRANCHES}
        max_error = max(abs(v) for values in discrepancies.values() for v in values.values())
        assert max_error < 1e-10, f"Historical metric reproduction failed: {model}: {max_error}"
        branches = {b: metrics(aligned_labels[keep], p[keep]) for b, p in aligned.items()}
        delta = {m: 100 * (branches["tta_center_refined"][m] - branches["original"][m]) for m in METRICS}
        model_dir = out / "models" / model
        model_dir.mkdir(parents=True, exist_ok=True)
        filtered_prediction_path = model_dir / "predictions.npz"
        np.savez_compressed(filtered_prediction_path, labels=aligned_labels[keep],
                            image_paths=np.asarray([r["image"] for r in corrected_rows]),
                            exam_ids=np.asarray([canonical_exams[i] for i in kept_indices]),
                            patient_ids=np.asarray([patients[i] for i in kept_indices]),
                            class_names=np.asarray(["Pneumonia"]),
                            **{b: p[keep] for b, p in aligned.items()})
        provenance = {
            "historical_result": {"path": str(source_path), "sha256": old["source_sha256"]},
            "historical_verified_catalog": {"path": snapshot["historical_catalog_path"], "sha256": catalog_hash},
            "frozen_historical_records": {"path": str(snapshot_path), "sha256": inputs[str(snapshot_path)]},
            "historical_prediction_inputs": [{"path": p, "sha256": inputs[p]} for p in prediction_sources],
            "operation": "Post-hoc patient exclusion and metric recomputation only; unchanged supports, fitted queries, model parameters, predictions, and branch definitions.",
            "score_representation": "historical logits" if model == "RadZero" else "historical prediction scores",
            "corrected_test_manifest": str(manifest_path), "corrected_test_manifest_sha256": manifest_hash,
            "filtered_predictions": str(filtered_prediction_path), "filtered_predictions_sha256": sha256(filtered_prediction_path),
            "historical_metrics_max_absolute_reproduction_error": max_error,
        }
        result = {
            "model": model, "dataset": "rsna_pneumonia", "task": "classification", "status": "complete",
            "rows": len(corrected_rows), "support_count": 10, "class_names": ["Pneumonia"],
            "branches": branches, "delta_pp": delta, "change": change_class(branches),
            "postprocessing": provenance,
        }
        result_path = model_dir / "result.json"
        dump(result_path, result)
        record = copy.deepcopy(old)
        record.update(rows=len(corrected_rows), source_path=str(result_path), source_sha256=sha256(result_path),
                      evaluation_manifest=str(manifest_path), evaluation_manifest_sha256=manifest_hash,
                      branches=branches, delta_pp=delta, change=change_class(branches), postprocessing=provenance)
        corrected_records.append(record)
        model_audits.append({"model": model, "same_fixed_support_images_and_labels": True,
                             "test_exam_sets_match": True, "path_order_reordered": not np.array_equal(reorder, np.arange(5327)),
                             "all_labels_match": True, "historical_metric_max_absolute_error": max_error,
                             "historical_metric_errors": discrepancies, "prediction_inputs": provenance["historical_prediction_inputs"]})
        for branch in BRANCHES:
            for metric in METRICS:
                old_value, new_value = old["branches"][branch][metric], branches[branch][metric]
                changes.append({"model": model, "branch": branch, "metric": metric,
                                "old_percent": 100 * old_value, "corrected_percent": 100 * new_value,
                                "change_percentage_points": 100 * (new_value - old_value)})

    summaries = {state: {branch: {m: float(np.mean([r["branches"][branch][m] for r in records]))
                                 for m in METRICS} for branch in BRANCHES}
                 for state, records in [("historical", old_records), ("corrected", corrected_records)]}
    summary = {
        "separation": separation, "model_count": 10, "branch_count_per_model": 4,
        "cross_model_metric_means_fraction": summaries,
        "cross_model_qv_minus_original_mean_pp": {
            state: {m: 100 * (means["tta_center_refined"][m] - means["original"][m]) for m in METRICS}
            for state, means in summaries.items()},
        "historical_change_counts": dict(Counter(r["change"] for r in old_records)),
        "corrected_change_counts": dict(Counter(r["change"] for r in corrected_records)),
    }
    artifact = {
        "status": "verified", "version": "fixed-rsna-support-patient-exclusion-v1",
        "metric_unit": "fraction (0-1); delta_pp is percentage points",
        "auprc_definition": "trapezoidal PR-curve area; average precision is stored separately",
        "primary_method": "tta_center_refined", "eta": 0.3, "branches": list(BRANCHES),
        "included_models": [r["model"] for r in corrected_records], "included_datasets": ["rsna_pneumonia"],
        "summary": summary, "records": corrected_records,
        "scope": "Replacement records for the ten fixed-support RSNA rows only; no repeated-support results changed.",
    }
    dump(out / "corrected_results.json", artifact)
    dump(out / "summary.json", summary)
    with (out / "metric_changes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(changes[0])); writer.writeheader(); writer.writerows(changes)
    dump(out / "metric_changes.json", changes)
    # All registered inputs must remain identical; the code writes only to out.
    for path, digest in inputs.items():
        assert sha256(path) == digest, f"Historical input changed: {path}"
    output_hashes = {str(p): sha256(p) for p in out.rglob("*")
                     if p.is_file() and p.name not in {"AUDIT.json"} and "__pycache__" not in p.parts}
    audit = {
        "status": "passed", "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
        "support_patient_exclusion_derived_from_official_mapping": True,
        "separation": separation, "model_audits": model_audits,
        "historical_input_files_unchanged": True, "input_sha256": inputs, "output_sha256": output_hashes,
        "historical_results_preserved": True, "corrected_repeated_support_results_untouched": True,
    }
    dump(out / "AUDIT.json", audit)
    print(json.dumps({"status": "passed", "output": str(out), **separation,
                      "max_historical_metric_error": max(x["historical_metric_max_absolute_error"] for x in model_audits)}, indent=2))


if __name__ == "__main__":
    main()
