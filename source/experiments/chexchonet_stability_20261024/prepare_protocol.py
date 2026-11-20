#!/usr/bin/env python3
"""Freeze five patient-sampled supports against the existing CheXchoNet test.

The historical 7,407 images / 2,470 patients are never resampled or reused as
support candidates. All new supports come from the already excluded 87 images
/ 30 patients. This is a new, stricter support protocol, not historical seed 42.
No model prediction or result is read when choosing supports.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
ALPHA_SOURCE = MEDICAL / "AlphaRAD/local_setup/chexchonet_20261023"
BASELINE_SOURCE = MEDICAL / "baselines/evaluation/chexchonet_20261023"
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SEEDS = (42, 43, 44, 45, 46)
VERSION = "chexchonet-fixed-test-five-support-seeds-v1"
BASE_COLUMNS = ("image", "image_path", "patient_id", *TASKS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != value:
        raise FileExistsError(f"Refusing to replace a changed frozen file: {path}")
    if not path.exists():
        path.write_bytes(value)


def frozen_json(path, value):
    frozen_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def frozen_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    frozen_bytes(path, stream.getvalue().encode())


def snapshot(source, dest):
    frozen_bytes(dest, Path(source).read_bytes())
    if sha(source) != sha(dest):
        raise ValueError(f"Snapshot differs: {source}")


def identity(rows):
    return [(r["image"], r["patient_id"], *(int(r[t]) for t in TASKS)) for r in rows]


def validate_binary(rows):
    for row in rows:
        if row["image"] != row["image_path"] or not row["patient_id"]:
            raise ValueError("Missing patient ID or inconsistent image paths")
        if any(row[task] not in ("0", "1") for task in TASKS):
            raise ValueError("Nonbinary source label")
        if int(row[TASKS[2]]) != max(int(row[t]) for t in TASKS[:2]):
            raise ValueError("Source composite labels unexpectedly differ from OR")
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image path in manifest")


def sample_task(pool, task, seed):
    """Uniform patients, then one uniform eligible image; no image-count weighting."""
    by_patient = defaultdict(list)
    for row in pool:
        by_patient[row["patient_id"]].append(row)
    positive = sorted(p for p, rows in by_patient.items() if any(int(r[task]) == 1 for r in rows))
    negative = sorted(p for p, rows in by_patient.items() if all(int(r[task]) == 0 for r in rows))
    if set(positive) & set(negative) or min(len(positive), len(negative)) < 5:
        raise ValueError("Insufficient disjoint patient pools")
    rng_seed = seed + 1000003 * (TASKS.index(task) + 1)
    rng = random.Random(rng_seed)
    selected = []
    for label, patients in ((1, positive), (0, negative)):
        for patient in rng.sample(patients, 5):
            eligible = sorted((r for r in by_patient[patient] if int(r[task]) == label), key=lambda r: r["image"])
            selected.append(dict(rng.choice(eligible)))
    if len({r["patient_id"] for r in selected}) != 10:
        raise ValueError("A task selected the same patient more than once")
    if Counter(int(r[task]) for r in selected) != {0: 5, 1: 5}:
        raise ValueError("Support label counts differ from 5 positive + 5 negative")
    return selected, {
        "random_seed": seed, "task_rng_seed": rng_seed,
        "positive_patient_count": len(positive), "all_negative_patient_count": len(negative),
        "positive_candidate_patient_ids": positive, "all_negative_candidate_patient_ids": negative,
        "selected_positive_patient_ids": [r["patient_id"] for r in selected[:5]],
        "selected_negative_patient_ids": [r["patient_id"] for r in selected[5:]],
    }


def task_csv_rows(rows, task, cls):
    return [{**{field: row[field] for field in BASE_COLUMNS},
             "label": repr([int(row[task])]),
             "class": repr([cls] if int(row[task]) else [])} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "protocol_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ALPHA_SOURCE or output == BASELINE_SOURCE:
        raise ValueError("Cannot replace historical protocol")
    source_catalog = json.loads((ALPHA_SOURCE / "catalog.json").read_text())
    baseline_catalog = json.loads((BASELINE_SOURCE / "catalog.json").read_text())
    split_audit = json.loads((ALPHA_SOURCE / "split_audit.json").read_text())
    integrity = json.loads((ALPHA_SOURCE / "image_integrity_audit.json").read_text())
    prompts = json.loads((ALPHA_SOURCE / "prompts.json").read_text())
    full = read_csv(ALPHA_SOURCE / "full_evaluation.csv")
    test = read_csv(ALPHA_SOURCE / "evaluation.csv")
    pool = read_csv(ALPHA_SOURCE / "excluded_patients_images.csv")
    for rows in (full, test, pool):
        validate_binary(rows)
    if (len(full), len(test), len(pool)) != (7494, 7407, 87):
        raise ValueError("Historical image counts changed")
    patient_sets = [{r["patient_id"] for r in rows} for rows in (full, test, pool)]
    if tuple(map(len, patient_sets)) != (2500, 2470, 30) or patient_sets[1] & patient_sets[2]:
        raise ValueError("Historical patient counts or separation changed")
    if identity([r for r in full if r["patient_id"] not in patient_sets[2]]) != identity(test):
        raise ValueError("Fixed test is not the historical patient-excluded subset")
    if identity([r for r in full if r["patient_id"] in patient_sets[2]]) != identity(pool):
        raise ValueError("Candidate pool is not all images of the 30 historical support patients")
    for name, digest in split_audit["manifest_sha256"].items():
        if sha(ALPHA_SOURCE / name) != digest:
            raise ValueError(f"Historical manifest changed: {name}")
    if integrity["status"] != "passed":
        raise ValueError("Existing image integrity audit did not pass")
    image_inventory = {r["image"]: r for r in integrity["inventory"]}
    if set(image_inventory) != {r["image"] for r in full}:
        raise ValueError("Image integrity inventory does not cover exactly the full subset")
    test_pixels = {image_inventory[r["image"]]["pixel_sha256"] for r in test}
    pool_pixels = {image_inventory[r["image"]]["pixel_sha256"] for r in pool}
    if test_pixels & pool_pixels:
        raise ValueError("Exact decoded-image duplicate across candidate/test patients")
    def check_image(row):
        inv = image_inventory[row["image"]]
        if inv["patient_id"] != row["patient_id"] or sha(row["image"]) != inv["sha256"]:
            raise ValueError(f"Image bytes or identity changed: {row['image']}")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(check_image, full))

    sources = [ALPHA_SOURCE / name for name in ("catalog.json", "split_audit.json", "image_integrity_audit.json",
               "prompts.json", "full_evaluation.csv", "evaluation.csv", "excluded_patients_images.csv")]
    sources += [BASELINE_SOURCE / "catalog.json", Path(__file__).resolve()]
    source_shas = {str(p): sha(p) for p in sources}
    shared = output / "shared_test"
    snapshot(ALPHA_SOURCE / "evaluation.csv", shared / "evaluation.csv")
    snapshot(ALPHA_SOURCE / "prompts.json", output / "prompts.json")
    snapshot(ALPHA_SOURCE / "excluded_patients_images.csv", output / "candidate_pool.csv")
    frozen_json(shared / "image_paths.json", [r["image"] for r in test])
    frozen_json(shared / "patient_ids.json", [r["patient_id"] for r in test])
    frozen_json(shared / "labels.json", [[int(r[t]) for t in TASKS] for r in test])
    frozen_json(output / "candidate_pool.json", [
        {**{key: row[key] for key in BASE_COLUMNS},
         "labels": {task: int(row[task]) for task in TASKS},
         "byte_sha256": image_inventory[row["image"]]["sha256"],
         "pixel_sha256": image_inventory[row["image"]]["pixel_sha256"]} for row in pool])
    frozen_json(output / "image_integrity_inventory.json", {
        "source": str(ALPHA_SOURCE / "image_integrity_audit.json"),
        "source_sha256": sha(ALPHA_SOURCE / "image_integrity_audit.json"),
        "all_7494_image_bytes_reverified": True,
        "candidate_test_duplicate_pixel_groups": 0,
        "inventory": integrity["inventory"],
    })
    fixed_tasks = {}
    for task in TASKS:
        key = "chexchonet_" + task
        old = source_catalog["datasets"][key]
        old_base = baseline_catalog["datasets"][key]
        task_test = read_csv(old["csv_path"])
        if sha(old["csv_path"]) != old["csv_sha256"] or identity(task_test) != identity(test):
            raise ValueError("Task-specific test differs from the common fixed test")
        original_manifest = json.loads(Path(old_base["manifest"]).read_text())
        if sha(old_base["manifest"]) != old_base["sha256"]:
            raise ValueError("Historical baseline test manifest changed")
        if [(r["image"], r["patient_id"], r["labels"][0]) for r in original_manifest["rows"]] != [
            (r["image"], r["patient_id"], int(r[task])) for r in test]:
            raise ValueError("Historical baseline test order or labels differ")
        csv_dest = shared / task / "evaluation.csv"
        json_dest = shared / (key + ".json")
        snapshot(old["csv_path"], csv_dest)
        snapshot(old_base["manifest"], json_dest)
        fixed_tasks[task] = {
            "dataset": key, "source_label_column": task, "class_names": old["class_names"],
            "fixed_semantic_prompt": prompts["tasks"][task],
            "csv_path": str(csv_dest), "csv_sha256": sha(csv_dest),
            "manifest": str(json_dest), "sha256": sha(json_dest),
            "image_count": len(test), "patient_count": len(patient_sets[1]),
            "positive_image_count": sum(int(r[task]) for r in test),
            "positive_patient_count": len({r["patient_id"] for r in test if int(r[task])}),
            "historical_test_csv_path": old["csv_path"], "historical_test_csv_sha256": old["csv_sha256"],
        }

    protocol = {
        "version": VERSION, "support_seeds": list(SEEDS), "eta": .3,
        "center_crop_ratio": .8, "views": ["identity", "center80"], "no_gradient_training": True,
        "negative_query_policy": "unchanged", "query_norm_policy": "preserve original norm after direction update",
        "fixed_model_specific_scoring_and_view_fusion": True,
        "support_per_task": {"positive_patients": 5, "all_negative_patients": 5, "images_per_patient": 1},
        "test_fixed": True, "test_image_count": 7407, "test_patient_count": 2470,
        "candidate_image_count": 87, "candidate_patient_count": 30,
        "positive_rule": "Patient has at least one positive candidate image for this task; choose one positive image.",
        "negative_rule": "All candidate images of the patient are negative for this task; choose one image.",
        "sampling_algorithm": "For each task (slvh, dlv, composite_slvh_dlv in this order), random.Random(seed + 1000003 * (task_index + 1)); sorted eligible patient IDs; rng.sample(..., 5) positives then negatives; per selected patient rng.choice on sorted eligible image paths.",
        "cross_task_support_patient_overlap_allowed": True,
        "task_input_rule": "Each task receives only its own 10 support images; physical feature caching across tasks grants no additional support examples to a task.",
        "selection_uses_predictions": False, "seed42_is_historical_support": False,
        "seed42_note": "New seed 42 is sampled under this strict-negative protocol and is not the historical 20261023 support.",
        "report_all_seeds": True, "support_seed_std_ddof": 1,
        "bootstrap": {"unit": "patient", "replicates": 2000, "seed": 20261024,
                      "confidence_level": .95, "interval": "paired percentile",
                      "shared_patient_draws_across_models_methods_and_tasks": True,
                      "metric": ["AUROC", "trapezoidal_PR_AUPRC"],
                      "all_images_from_sampled_patients_included": True,
                      "invalid_single_class_draws": "exclude and report valid count",
                      "fixed_support_ci_and_support_seed_variation_reported_separately": True},
        "limitations": [
            "Only 30 pre-reserved patients are eligible; five support samples can overlap and do not represent a new independent clinical cohort.",
            "Patient-level test bootstrap does not remove earlier test-driven method-selection bias.",
            "The stricter negative-patient rule differs from the historical support selection.",
            "No new patient is removed from or added to the fixed 7407-image test.",
            "The data are the custom 7494-image subset, not a verified AlphaRAD-paper split.",
            "No check of pretraining-source overlap or near-duplicate images is implied.",
        ],
    }
    seed_tasks, all_support_images = {}, set()
    for seed in SEEDS:
        seed_dir = output / f"seed_{seed}"
        snapshot(ALPHA_SOURCE / "prompts.json", seed_dir / "prompts.json")
        seed_tasks[str(seed)] = {}
        for task in TASKS:
            key = "chexchonet_" + task
            cls = fixed_tasks[task]["class_names"][0]
            support, selection = sample_task(pool, task, seed)
            if {r["patient_id"] for r in support} & patient_sets[1]:
                raise ValueError("Support/test patient leakage")
            old_support = read_csv(source_catalog["datasets"][key]["support_csv_path"])
            csv_dest = seed_dir / task / "support.csv"
            json_dest = seed_dir / task / "support.json"
            frozen_csv(csv_dest, task_csv_rows(support, task, cls), (*BASE_COLUMNS, "label", "class"))
            support_rows = [{"index": i, "image": row["image"], "labels": [int(row[task])],
                             "class_names_present": [cls] if int(row[task]) else [],
                             "patient_id": row["patient_id"],
                             "byte_sha256": image_inventory[row["image"]]["sha256"]}
                            for i, row in enumerate(support)]
            support_sources = {str(ALPHA_SOURCE / "excluded_patients_images.csv"): sha(ALPHA_SOURCE / "excluded_patients_images.csv"),
                               str(csv_dest): sha(csv_dest), str(ALPHA_SOURCE / "prompts.json"): sha(ALPHA_SOURCE / "prompts.json")}
            frozen_json(json_dest, {
                "dataset": key, "task": "classification", "class_names": [cls],
                "source_label_column": task, "count": len(test), "support_rows": support_rows,
                "support_count": 10, "support_positive_counts": [5], "support_negative_counts": [5],
                "test_manifest": fixed_tasks[task]["manifest"], "test_manifest_sha256": fixed_tasks[task]["sha256"],
                "fixed_semantic_prompt": prompts["tasks"][task], "support_sources": support_sources,
                "sampling": selection, "protocol_version": VERSION, "support_seed": seed,
                "negative_patient_rule": protocol["negative_rule"],
                "identity_audit": {"patient_overlap_count": 0, "image_overlap_count": 0,
                                   "candidate_patient_membership_checked": True,
                                   "all_negative_patient_rule_checked": True},
                "baseline_references": {},
                "baseline_reference_policy": "Compare original and adapted predictions on the same fixed test; historical original predictions may be reused after image/label alignment verification.",
            })
            entry = {**selection, "support_csv_path": str(csv_dest), "support_csv_sha256": sha(csv_dest),
                     "support_manifest": str(json_dest), "support_manifest_sha256": sha(json_dest),
                     "support_image_count": 10, "support_patient_count": 10,
                     "support_image_paths": [r["image"] for r in support],
                     "support_patient_ids": [r["patient_id"] for r in support],
                     "support_positive_count": 5, "support_negative_count": 5,
                     "historical_support_csv": source_catalog["datasets"][key]["support_csv_path"],
                     "matches_historical_support_image_set": {r["image"] for r in support} == {r["image"] for r in old_support}}
            seed_tasks[str(seed)][task] = entry
            all_support_images.update(entry["support_image_paths"])

    audit = {
        "status": "passed", "protocol": protocol, "source_files_sha256": source_shas,
        "candidate_pool_patient_ids": sorted(patient_sets[2]),
        "test_patient_ids": sorted(patient_sets[1]), "test_order_preserved": True,
        "patient_overlap_count": 0, "candidate_test_exact_pixel_duplicate_groups": 0,
        "all_7494_image_bytes_reverified": True,
        "fixed_tasks": fixed_tasks, "seed_tasks": seed_tasks,
        "total_unique_support_images_across_seeds_and_tasks": len(all_support_images),
        "seed_union_patient_counts": {seed: len({p for t in tasks.values() for p in t["support_patient_ids"]}) for seed, tasks in seed_tasks.items()},
        "task_cross_seed_patient_intersections": {
            task: {f"{a}_{b}": len(set(seed_tasks[str(a)][task]["support_patient_ids"]) & set(seed_tasks[str(b)][task]["support_patient_ids"]))
                   for ai, a in enumerate(SEEDS) for b in SEEDS[ai+1:]} for task in TASKS},
    }
    frozen_json(output / "split_audit.json", audit)
    catalog = {"status": "complete", "version": VERSION, "protocol": protocol,
               "source_files_sha256": source_shas, "tasks": fixed_tasks, "seeds": {},
               "candidate_pool_csv": str(output / "candidate_pool.csv"),
               "candidate_pool_sha256": sha(output / "candidate_pool.csv"),
               "common_test_csv": str(shared / "evaluation.csv"), "common_test_csv_sha256": sha(shared / "evaluation.csv"),
               "split_audit": str(output / "split_audit.json"), "split_audit_sha256": sha(output / "split_audit.json")}
    for seed in SEEDS:
        seed_dir = output / f"seed_{seed}"
        alpha = {"status": "complete", "protocol": {**protocol, "seed": seed}, "datasets": {}}
        baseline = {k: v for k, v in baseline_catalog.items() if k not in ("datasets", "source_catalogs", "baseline_reference_policy")}
        baseline.update({"version": VERSION, "protocol": protocol, "support_seed": seed,
                         "source_catalogs": source_shas, "datasets": {}})
        for task in TASKS:
            key = "chexchonet_" + task
            spec = seed_tasks[str(seed)][task]
            fixed = fixed_tasks[task]
            old = source_catalog["datasets"][key]
            alpha["datasets"][key] = {**old,
                "csv_path": fixed["csv_path"], "csv_sha256": fixed["csv_sha256"],
                "support_csv_path": spec["support_csv_path"], "support_csv_sha256": spec["support_csv_sha256"],
                "support_seed": seed, "support_sampling_protocol": VERSION,
                "scope": "Fixed historical 7407-image patient holdout; new 5+5 support sampled only from pre-reserved 30 patients.",
                "split_audit_path": str(output / "split_audit.json"), "split_audit_sha256": sha(output / "split_audit.json")}
            baseline["datasets"][key] = {
                "dataset": key, "task": "classification", "manifest": fixed["manifest"], "sha256": fixed["sha256"],
                "count": 7407, "class_names": fixed["class_names"],
                "support_manifest": spec["support_manifest"], "support_manifest_sha256": spec["support_manifest_sha256"],
                "support_count": 10, "support_seed": seed, "comparison_references": []}
        alpha_path, baseline_path = seed_dir / "alpharad_catalog.json", seed_dir / "baseline_catalog.json"
        frozen_json(alpha_path, alpha)
        frozen_json(baseline_path, baseline)
        catalog["seeds"][str(seed)] = {
            "alpharad_catalog": str(alpha_path), "alpharad_catalog_sha256": sha(alpha_path),
            "baseline_catalog": str(baseline_path), "baseline_catalog_sha256": sha(baseline_path),
            "tasks": seed_tasks[str(seed)], "union_support_patient_count": audit["seed_union_patient_counts"][str(seed)]}
    frozen_json(output / "catalog.json", catalog)
    frozen_json(output / "ARTIFACT_MANIFEST.json", {
        "status": "complete", "files_sha256": {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}})
    print(json.dumps({"status": "complete", "catalog": str(output / "catalog.json"),
                      "test_images": 7407, "test_patients": 2470, "candidate_images": 87,
                      "candidate_patients": 30, "seeds": list(SEEDS),
                      "seed_union_patient_counts": audit["seed_union_patient_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
