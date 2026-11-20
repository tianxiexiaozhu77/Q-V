#!/usr/bin/env python3
"""Freeze independent supports against the unchanged RSNA / ChestXDet10 tests.

Data acquisition is handled by prepare_rsna_support.py / prepare_chest_support.py.
This script never reads predictions or chooses supports using test outcomes.
All methods and both models consume exactly the same support and test rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
SOURCE = HERE / "source_data"
HISTORICAL = MEDICAL / "experiments/fewshot_native_query/protocol_v1"
SEEDS = (42, 43, 44, 45, 46)
DATASETS = ("rsna_pneumonia", "chestxdet10")
VERSION = "rsna-chest-independent-support-five-seeds-fixed-test-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != value:
        raise FileExistsError(f"Refusing to alter frozen protocol: {path}")
    if not path.exists():
        path.write_bytes(value)


def write_json(path, obj):
    frozen(path, (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def write_csv(path, rows, classes):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["image", "class", "label", "patient_id", "group_id", "anchor_class"])
    writer.writeheader()
    for r in rows:
        writer.writerow({"image": r["image"], "class": repr([cls for cls, value in zip(classes, r["labels"]) if value]),
                         "label": repr(r["labels"]), "patient_id": r.get("patient_id", ""),
                         "group_id": r["group_id"], "anchor_class": r.get("anchor_class", "")})
    frozen(path, stream.getvalue().encode())


def check_ref(ref):
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError(f"Frozen source changed: {ref['path']}")


def checked_sources(pool):
    for path, expected in pool["sources"].items():
        if sha(path) != expected:
            raise ValueError(f"Support source changed: {path}")


def prepare_dataset(dataset, output):
    pool_path = SOURCE / ("rsna_pool.json" if dataset == "rsna_pneumonia" else "chestxdet10_pool.json")
    pool = json.loads(pool_path.read_text())
    checked_sources(pool)
    history_path = HISTORICAL / (dataset + ".json")
    old = json.loads(history_path.read_text())
    check_ref(old["test_manifest"])
    if old["spatial_manifest"]:
        check_ref(old["spatial_manifest"])
    old_test = json.loads(Path(old["test_manifest"]["path"]).read_text())
    classes = old["class_names"]
    if pool["class_names"] != classes:
        raise ValueError("New training labels and historical class order differ")
    if set(pool["seeds"]) != {str(x) for x in SEEDS}:
        raise ValueError("Expected the five predetermined seeds")
    test = [dict(r) for r in old_test["rows"]]
    expected_count = 5327 if dataset == "rsna_pneumonia" else 492
    if len(test) != expected_count or len({r["image"] for r in test}) != expected_count:
        raise ValueError("Historical fixed-test image count changed")
    test_inventory = {r["image"]: r for r in pool.get("test_image_inventory", [])}
    mapping_path = SOURCE / "pneumonia-challenge-dataset-mappings_2018.json"
    mapping = None
    if dataset == "rsna_pneumonia":
        mapping = {r["subset_img_id"]: r for r in json.loads(mapping_path.read_text())}
        if len(mapping) != 30000:
            raise ValueError("RSNA official mapping is incomplete or nonunique")
    for row in test:
        if row["image"] in test_inventory:
            identity = test_inventory[row["image"]]
            row.update({k: identity[k] for k in ("byte_sha256", "pixel_sha256") if k in identity})
        if mapping is not None:
            image_id = Path(row["image"]).stem
            source = mapping[image_id]
            row["dicom_patient_id"] = image_id
            row["nih_image_id"] = source["img_id"]
            row["patient_id"] = "NIH:" + Path(source["img_id"]).stem.split("_")[0]
            row["group_id"] = row["patient_id"]
            row["study_id"] = source["StudyInstanceUID"]
        else:
            if "pixel_sha256" not in row:
                raise ValueError("ChestXDet10 test requires decoded-pixel identity audit")
            row["group_id"] = "ChestXDet10:pixel:" + row["pixel_sha256"]
    test_groups = {r["group_id"] for r in test}
    test_images = {r["image"] for r in test}
    test_pixels = {r["pixel_sha256"] for r in test if "pixel_sha256" in r}
    test_bytes = {r["byte_sha256"] for r in test if "byte_sha256" in r}
    destination = output / dataset
    manifest_path = destination / "test_manifest.json"
    frozen(manifest_path, Path(old["test_manifest"]["path"]).read_bytes())
    test_ref = {"path": str(manifest_path), "sha256": sha(manifest_path)}
    write_json(destination / "test_identities.json", test)
    write_csv(destination / "evaluation.csv", test, classes)
    union, seed_entries, distinct_draws = {}, {}, set()
    for seed in SEEDS:
        entry = pool["seeds"][str(seed)]
        support = [dict(row) for row in entry["support_rows"]]
        budget = 10 if dataset == "rsna_pneumonia" else 50
        if len(support) != budget or len({r["image"] for r in support}) != budget:
            raise ValueError("Support images are not distinct or budget changed")
        for i, row in enumerate(support):
            row["index"] = i
            row["labels"] = [int(x) for x in row["labels"]]
            if len(row["labels"]) != len(classes) or any(x not in (0, 1) for x in row["labels"]):
                raise ValueError("Invalid support label vector")
            if sha(row["image"]) != row["byte_sha256"]:
                raise ValueError("Selected support image bytes changed")
            if mapping is not None:
                if not str(row.get("patient_id", "")).startswith("NIH:"):
                    raise ValueError("RSNA patient group must identify original NIH subject, not per-image UUID")
                row["group_id"] = row["patient_id"]
            else:
                row["group_id"] = "ChestXDet10:pixel:" + row["pixel_sha256"]
            row["class_names_present"] = [c for c, y in zip(classes, row["labels"]) if y]
            if row["image"] in test_images or row["group_id"] in test_groups or row["pixel_sha256"] in test_pixels or row["byte_sha256"] in test_bytes:
                raise ValueError("New support overlaps the fixed test")
            union[row["image"]] = {k: v for k, v in row.items() if k not in ("index", "anchor_class")}
        if len({r["group_id"] for r in support}) != budget:
            raise ValueError("Repeated support sampling group")
        positive = [sum(r["labels"][c] for r in support) for c in range(len(classes))]
        if any(x <= 0 or x >= budget for x in positive):
            raise ValueError("Both support outcomes are required for every class")
        if dataset == "rsna_pneumonia" and positive != [5]:
            raise ValueError("RSNA positive/negative budget changed")
        if dataset == "chestxdet10" and Counter(r["anchor_class"] for r in support) != {c: 5 for c in classes}:
            raise ValueError("ChestXDet10 positive anchor quota changed")
        identity = tuple(sorted(r["image"] for r in support))
        if identity in distinct_draws:
            raise ValueError("Seeds merely reshuffle the same support set")
        distinct_draws.add(identity)
        seed_dir = destination / f"seed_{seed}"
        support_path = seed_dir / "support.json"
        csv_path = seed_dir / "support.csv"
        support_manifest = {"dataset": dataset, "class_names": classes, "task": "classification",
                            "support_seed": seed, "support_rows": support, "support_count": budget,
                            "support_positive_counts": positive, "support_negative_counts": [budget - x for x in positive],
                            "test_manifest": str(manifest_path), "test_manifest_sha256": test_ref["sha256"],
                            "protocol_version": VERSION, "sampling": entry["sampling"],
                            "support_source_pool": {"path": str(pool_path), "sha256": sha(pool_path)},
                            "support_test_group_overlap": 0, "seed42_is_historical_support": False}
        write_json(support_path, support_manifest)
        write_csv(csv_path, support, classes)
        compatibility_path = seed_dir / "native_protocol.json"
        compatibility = {"version": VERSION, "status": "frozen", "dataset": dataset, "class_names": classes,
                         "eta": .3, "crop_fraction": .8, "support_rows": support,
                         "support_groups": [r["group_id"] for r in support],
                         "support_manifest": {"path": str(support_path), "sha256": sha(support_path)},
                         "test_manifest": test_ref, "test_images": [r["image"] for r in test],
                         "test_count": len(test), "support_count": budget, "spatial_manifest": old["spatial_manifest"],
                         "references": old["references"], "historical_reference_adapted_branches_comparable": False}
        write_json(compatibility_path, compatibility)
        seed_entries[str(seed)] = {"seed": seed, "support_rows": support, "support_count": budget,
                                  "support_groups": [r["group_id"] for r in support],
                                  "positive_counts": positive, "negative_counts": [budget - x for x in positive],
                                  "manifest": {"path": str(support_path), "sha256": sha(support_path)},
                                  "csv_path": str(csv_path), "csv_sha256": sha(csv_path),
                                  "native_protocol": {"path": str(compatibility_path), "sha256": sha(compatibility_path)},
                                  "sampling": entry["sampling"]}
    historical_overlap = None
    if mapping is not None:
        historical_support_patients = {"NIH:" + Path(mapping[Path(r["image"]).stem]["img_id"]).stem.split("_")[0] for r in old["support_rows"]}
        overlap = sorted(historical_support_patients & test_groups)
        historical_overlap = {"historical_support_patient_count": len(historical_support_patients),
                              "historical_support_test_patient_overlap_count": len(overlap),
                              "overlapping_patient_ids": overlap,
                              "note": "Old DICOM PatientID was a per-image recoded UUID. New protocol uses original NIH subject IDs from official RSNA mapping; historical results are preserved but are not verified patient-disjoint."}
    groups_ordered = [r["group_id"] for r in test]
    label_counts = [sum(r["labels"][c] for r in test) for c in range(len(classes))]
    record = {"version": VERSION, "status": "frozen", "dataset": dataset, "class_names": classes,
              "models": ["BiomedCLIP", "AlphaRAD"], "branches": ["original", "query_only", "view_only", "query_plus_view"],
              "eta": .3, "crop_fraction": .8, "support_seeds": list(SEEDS), "test_fixed": True,
              "test_manifest": test_ref, "test_rows": test, "test_groups": groups_ordered,
              "test_count": len(test), "test_group_count": len(test_groups), "test_positive_counts": label_counts,
              "test_negative_counts": [len(test) - x for x in label_counts],
              "resampling_unit": "patient" if mapping is not None else "image_exact_duplicate_group",
              "group_id_source": "Original NIH image filename subject prefix via official RSNA mapping; DICOM UUID is not a patient identity." if mapping is not None else "Exact decoded RGB pixel SHA256 groups; verified patient identity unavailable.",
              "spatial_manifest": old["spatial_manifest"], "seeds": seed_entries,
              "support_union_rows": [union[key] for key in sorted(union)], "support_union_count": len(union),
              "candidate_count": len(pool["candidate_rows"]), "candidate_metadata_only": True,
              "candidate_pool": {"path": str(pool_path), "sha256": sha(pool_path)},
              "historical_protocol": {"path": str(history_path), "sha256": sha(history_path)},
              "historical_reference_adapted_branches_comparable": False, "references": old["references"],
              "seed42_is_historical_support": False, "historical_patient_audit": historical_overlap,
              "same_support_across_models_and_branches": True, "all_support_images_enter_each_class_fit": True,
              "sampling_uses_model_predictions": False, "report_all_seeds": True, "source_audit": pool["audit"],
              "bootstrap": {"replicates": 2000, "seed": 20261024, "confidence_level": .95,
                            "paired": True, "shared_draws_within_dataset_across_models_methods_seeds": True,
                            "interval": "percentile", "metrics": ["macro_AUROC", "macro_trapezoidal_PR_AUPRC"],
                            "report_support_seed_std_ddof": 1,
                            "invalid_single_class_draws": "exclude any macro replicate lacking either outcome in any target class; report valid count"},
              "limitations": ["Previously exposed datasets; resampling cannot remove method-selection bias.",
                              "New supports come from a separate source training/adaptation pool, while historical supports came from test holdout; this is a new protocol, not historical-seed reproduction.",
                              "Patient isolation is verified for RSNA via original NIH subject identity; ChestXDet10 has only exact-image separation.",
                              "Only five prespecified support draws; no claim that five draws exhaust support uncertainty.",
                              "No near-duplicate or pretraining-overlap search is implied."]}
    if mapping is not None:
        record["patient_mapping"] = {"path": str(mapping_path), "sha256": sha(mapping_path)}
    path = output / (dataset + ".json")
    write_json(path, record)
    return {"path": str(path), "sha256": sha(path), "test_count": len(test), "test_group_count": len(test_groups),
            "support_count_per_seed": len(seed_entries["42"]["support_rows"]), "support_union_count": len(union),
            "resampling_unit": record["resampling_unit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "protocol_v1")
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    output = args.output.resolve()
    files = {}
    for dataset in args.datasets:
        files[dataset] = prepare_dataset(dataset, output)
        print(json.dumps({dataset: files[dataset]}, ensure_ascii=False), flush=True)
    for dataset in DATASETS:
        path = output / (dataset + ".json")
        if dataset not in files and path.exists():
            d = json.loads(path.read_text())
            files[dataset] = {"path": str(path), "sha256": sha(path), "test_count": d["test_count"],
                              "test_group_count": d["test_group_count"], "support_count_per_seed": d["seeds"]["42"]["support_count"],
                              "support_union_count": d["support_union_count"], "resampling_unit": d["resampling_unit"]}
    catalog = {"version": VERSION, "status": "frozen" if set(files) == set(DATASETS) else "partially_prepared",
               "support_seeds": list(SEEDS), "models": ["BiomedCLIP", "AlphaRAD"], "datasets": files,
               "eta": .3, "crop_fraction": .8, "bootstrap_replicates": 2000, "bootstrap_seed": 20261024,
               "protocol_preparation_script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}}
    if catalog["status"] == "frozen":
        write_json(output / "catalog.json", catalog)
    else:
        write_json(output / "catalog_partial_chest.json", catalog)


if __name__ == "__main__":
    main()
