#!/usr/bin/env python3
"""Independently audit completed CheXchoNet stability files without inference.

Reads frozen inputs and inference outputs only. It does not import inference
code, modify a model/result, recompute statistical intervals, or select a seed.
Exit status 2 means inference files are not complete yet; 1 means audit failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("BiomedCLIP", "AlphaRAD")
BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")
OLD_ALPHA = MEDICAL / "experiments/chexchonet_comparison_20261023_233926/alpharad_paired"
OLD_BIO = MEDICAL / "baselines/runs/chexchonet_20261024_000447/BiomedCLIP_full"


class NotReady(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(rows):
    return [{"image_path": str(Path(r.get("image_path", r.get("image"))).resolve()),
             "patient_id": str(r["patient_id"]), **{t: int(r[t]) for t in TASKS}}
            for r in rows]


def hash_check(mapping, checked):
    for path, expected in mapping.items():
        actual = sha(path)
        require(actual == expected, "File drift: " + path)
        if path in checked:
            require(checked[path] == expected, "Inconsistent source hashes across snapshots: " + path)
        checked[path] = actual


def ensure_ready(run):
    pending = []
    for model in MODELS:
        folder = run / model
        result = folder / "result.json"
        if not result.exists() or read_json(result).get("status") != "complete":
            state = read_json(folder / "status.json") if (folder / "status.json").exists() else {}
            pending.append({"model": model, "status": state.get("status", "missing"),
                            "done": state.get("done"), "total": state.get("total")})
        elif not (folder / "source_sha256.json").exists():
            pending.append({"model": model, "status": "missing source_sha256.json"})
    if pending:
        raise NotReady(json.dumps(pending, ensure_ascii=False))


def check_recorded_passes(obj, context):
    if isinstance(obj, dict):
        if "passed" in obj:
            require(obj["passed"] is True, "Recorded native parity failed: " + context)
        for key, value in obj.items():
            check_recorded_passes(value, context + "/" + key)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            check_recorded_passes(value, context + "/" + str(index))


def historical_check(model, arrays, paths, labels, patients, parity, checked):
    result = {}
    recorded = parity["historical_unchanged_arms"]
    require(recorded["images"] == 7407 and recorded["passed"] is True, "Historical parity scope differs")
    require(recorded["atol"] == 2e-6 and recorded["rtol"] == 1e-5, "Historical tolerances changed")
    if model == "AlphaRAD":
        old_paths = read_json(OLD_ALPHA / "image_paths.json")
        lookup = {str(Path(path).resolve()): i for i, path in enumerate(old_paths)}
        ids = np.asarray([lookup[path] for path in paths])
        np.testing.assert_array_equal(np.load(OLD_ALPHA / "labels.npy", allow_pickle=False)[ids], labels)
        np.testing.assert_array_equal(np.asarray(read_json(OLD_ALPHA / "patient_ids.json"))[ids], patients)
        for branch in ("original", "tta_center_original"):
            ref = OLD_ALPHA / f"{branch}_probabilities.npy"
            old = np.load(ref, allow_pickle=False)[ids]
            info = recorded["checks"][branch]
            require(info["reference"] == str(ref), "Unexpected AlphaRAD historical reference")
            hash_check({str(ref): info["reference_sha256"]}, checked)
            np.testing.assert_allclose(arrays[branch], old, atol=2e-6, rtol=1e-5)
            error = float(np.max(np.abs(arrays[branch] - old)))
            equal = bool(np.array_equal(arrays[branch], old))
            require(info["max_abs_error"] == error and info["bitwise_equal"] == equal, "Recorded historical parity differs from arrays")
            result[branch] = {"max_abs_error": error, "bitwise_equal": equal}
    else:
        for j, task in enumerate(TASKS):
            dataset = "chexchonet_" + task
            ref = OLD_BIO / dataset / "predictions.npz"
            with np.load(ref, allow_pickle=False) as old:
                lookup = {str(Path(path).resolve()): i for i, path in enumerate(old["image_paths"].tolist())}
                ids = np.asarray([lookup[path] for path in paths])
                np.testing.assert_array_equal(old["labels"][ids, 0], labels[:, j])
                if "patient_ids" in old.files:
                    np.testing.assert_array_equal(old["patient_ids"][ids], patients)
                for branch in ("original", "tta_center_original"):
                    info = recorded["checks"][dataset + "/" + branch]
                    require(info["reference"] == str(ref), "Unexpected BiomedCLIP historical reference")
                    hash_check({str(ref): info["reference_sha256"]}, checked)
                    expected = old[branch][ids, 0]
                    np.testing.assert_allclose(arrays[branch][:, j], expected, atol=2e-6, rtol=1e-5)
                    error = float(np.max(np.abs(arrays[branch][:, j] - expected)))
                    equal = bool(np.array_equal(arrays[branch][:, j], expected))
                    require(info["max_abs_error"] == error and info["bitwise_equal"] == equal, "Recorded historical parity differs from arrays")
                    result[task + "/" + branch] = {"max_abs_error": error, "bitwise_equal": equal}
    return result


def audit(run, protocol_path):
    ensure_ready(run)
    catalog = read_json(protocol_path)
    require(catalog["protocol"]["support_seeds"] == list(SEEDS), "Unexpected support seed order")
    require(catalog["protocol"]["eta"] == .3 and catalog["protocol"]["center_crop_ratio"] == .8, "Method changed")
    checked = {}
    # This chain includes the original candidate CSV and prepare_protocol.py;
    # a worker merely recording the current pool hash would not be sufficient.
    hash_check(catalog["source_files_sha256"], checked)
    hash_check({catalog["common_test_csv"]: catalog["common_test_csv_sha256"],
                catalog["candidate_pool_csv"]: catalog["candidate_pool_sha256"],
                catalog["split_audit"]: catalog["split_audit_sha256"]}, checked)
    artifact = read_json(protocol_path.parent / "ARTIFACT_MANIFEST.json")
    hash_check({str(protocol_path.parent / p): h for p, h in artifact["files_sha256"].items()}, checked)
    test = normalized(read_csv(catalog["common_test_csv"]))
    pool = normalized(read_csv(catalog["candidate_pool_csv"]))
    paths = np.asarray([r["image_path"] for r in test])
    patients = np.asarray([r["patient_id"] for r in test])
    labels = np.asarray([[r[t] for t in TASKS] for r in test])
    classes = [catalog["tasks"][t]["class_names"][0] for t in TASKS]
    require(len(test) == 7407 and len(set(patients)) == 2470, "Wrong fixed test scope")
    require(not set(patients) & {r["patient_id"] for r in pool}, "Candidate/test patient leakage")
    alpha_source = MEDICAL / "AlphaRAD/local_setup/chexchonet_20261023"
    baseline_source = MEDICAL / "baselines/evaluation/chexchonet_20261023"
    historical_alpha_catalog = read_json(alpha_source / "catalog.json")
    historical_baseline_catalog = read_json(baseline_source / "catalog.json")
    require(sha(catalog["candidate_pool_csv"]) == sha(alpha_source / "excluded_patients_images.csv"),
            "Candidate snapshot differs from the original reserved pool")
    require(sha(catalog["common_test_csv"]) == sha(alpha_source / "evaluation.csv"),
            "Common test CSV differs from the historical test")
    for task in TASKS:
        fixed = catalog["tasks"][task]
        original_alpha = historical_alpha_catalog["datasets"]["chexchonet_" + task]
        original_baseline = historical_baseline_catalog["datasets"]["chexchonet_" + task]
        hash_check({fixed["historical_test_csv_path"]: fixed["historical_test_csv_sha256"],
                    original_alpha["csv_path"]: original_alpha["csv_sha256"],
                    original_baseline["manifest"]: original_baseline["sha256"]}, checked)
        require(sha(fixed["csv_path"]) == original_alpha["csv_sha256"], "Task test CSV differs from historical bytes")
        require(sha(fixed["manifest"]) == original_baseline["sha256"], "Task test JSON differs from historical bytes")
    pool_lookup = {r["image_path"]: i for i, r in enumerate(pool)}
    campaign = read_json(run / "status.json")
    require(campaign["protocol_sha256"] == sha(protocol_path), "Campaign protocol hash mismatch")
    hash_check(campaign["sources"], checked)
    expected_supports = {}
    for seed in SEEDS:
        item = catalog["seeds"][str(seed)]
        hash_check({item["alpharad_catalog"]: item["alpharad_catalog_sha256"],
                    item["baseline_catalog"]: item["baseline_catalog_sha256"]}, checked)
        alpha = read_json(item["alpharad_catalog"])
        bio = read_json(item["baseline_catalog"])
        expected_supports[seed] = {}
        for task in TASKS:
            key = "chexchonet_" + task
            ae, be = alpha["datasets"][key], bio["datasets"][key]
            hash_check({ae["support_csv_path"]: ae["support_csv_sha256"],
                        be["support_manifest"]: be["support_manifest_sha256"]}, checked)
            selected = normalized(read_csv(ae["support_csv_path"]))
            support_json = read_json(be["support_manifest"])["support_rows"]
            require([(r["image_path"], r["patient_id"], r[task]) for r in selected] ==
                    [(r["image"], r["patient_id"], r["labels"][0]) for r in support_json], "Cross-model support mismatch")
            require(len(selected) == len({r["patient_id"] for r in selected}) == 10, "Support patients are not distinct")
            require(sum(r[task] for r in selected) == 5, "Support labels differ from 5+5")
            for row in selected:
                require(row == pool[pool_lookup[row["image_path"]]], "Support outside fixed pool")
                if not row[task]:
                    require(all(r[task] == 0 for r in pool if r["patient_id"] == row["patient_id"]), "Negative patient has positive candidate image")
            expected_supports[seed][task] = selected

    reports = {}
    for model in MODELS:
        folder = run / model
        model_result = read_json(folder / "result.json")
        require(model_result["model"] == model and model_result["seeds"] == list(SEEDS) and model_result["rows"] == 7407, "Model result identity differs")
        inputs = read_json(folder / "input_audit.json")
        require(inputs["protocol_sha256"] == sha(protocol_path) and Path(inputs["protocol"]).resolve() == protocol_path, "Model input protocol hash differs")
        require(inputs["task_keys"] == list(TASKS) and inputs["class_names"] == classes and inputs["seeds"] == list(SEEDS), "Input metadata order differs")
        require(inputs["fixed_test_images"] == inputs["inference_images"] == 7407 and inputs["fixed_test_patients"] == 2470, "Input test cohort differs")
        hash_check(read_json(folder / "source_sha256.json"), checked)
        parity = read_json(folder / "native_parity.json")
        check_recorded_passes(parity, model)
        require(model_result["native_parity"] == parity, "Model result/native parity record differs")
        require(parity["unchanged_arms_bitwise_equal_across_seeds"] is True, "Unchanged arms not asserted identical")
        unchanged, query_arrays, prediction_hashes = None, {}, {}
        for seed in SEEDS:
            seed_dir = folder / f"seed_{seed}"
            result = read_json(seed_dir / "result.json")
            require(result == model_result["seed_results"][str(seed)], "Seed result differs from aggregate copy")
            require(result["model"] == model and result["seed"] == seed and result["status"] == "complete", "Wrong seed result identity")
            require(result["rows"] == 7407 and result["patients"] == 2470, "Wrong seed result cohort")
            prediction = seed_dir / "predictions.npz"
            hash_check({str(prediction): result["prediction_sha256"]}, checked)
            prediction_hashes[str(seed)] = result["prediction_sha256"]
            with np.load(prediction, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["seed"], seed)
                np.testing.assert_array_equal(data["task_keys"], TASKS)
                np.testing.assert_array_equal(data["class_names"], classes)
                np.testing.assert_array_equal(data["image_paths"], paths)
                np.testing.assert_array_equal(data["patient_ids"], patients)
                np.testing.assert_array_equal(data["labels"], labels)
                for branch in BRANCHES:
                    scores = data[branch]
                    require(scores.shape == (7407, 3) and np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all(), "Invalid probability array")
                if unchanged is None:
                    unchanged = {branch: data[branch].copy() for branch in ("original", "tta_center_original")}
                else:
                    for branch in unchanged:
                        np.testing.assert_array_equal(data[branch], unchanged[branch])
            support_record = read_json(seed_dir / "support_manifest.json")
            require(support_record["seed"] == seed and support_record["eta"] == .3 and
                    support_record["test_labels_used_for_fit"] is False, "Support record protocol differs")
            require(support_record["task_support_rows"] == expected_supports[seed], "Run supports differ from frozen CSV manifests")
            queries = np.load(seed_dir / "queries.npy", allow_pickle=False)
            require(queries.shape == (3, 768 if model == "AlphaRAD" else 512) and np.isfinite(queries).all(), "Malformed queries")
            query_arrays[seed] = queries
            for j, task in enumerate(TASKS):
                selected = expected_supports[seed][task]
                expected_ids = [pool_lookup[r["image_path"]] for r in selected]
                with np.load(seed_dir / f"query_fit_{task}.npz", allow_pickle=False) as fit:
                    np.testing.assert_array_equal(fit["positive_counts"], [5])
                    np.testing.assert_array_equal(fit["negative_counts"], [5])
                    np.testing.assert_array_equal(fit["support_indices"], expected_ids)
                    fitted = fit["queries"][1, 0] if model == "AlphaRAD" else fit["queries"][0]
                    np.testing.assert_array_equal(fitted, queries[j])
                    require(np.isfinite(fit["directions"]).all(), "Nonfinite query direction")
        task_query_differences = {}
        for j, task in enumerate(TASKS):
            differences = {f"{a}_{b}": float(np.max(np.abs(query_arrays[a][j] - query_arrays[b][j])))
                           for ai, a in enumerate(SEEDS) for b in SEEDS[ai+1:]}
            require(all(error > 0 for error in differences.values()), "Two support seeds produced identical task queries")
            task_query_differences[task] = differences
        historic = historical_check(model, unchanged, paths, labels, patients, parity, checked)
        reports[model] = {"status": "passed", "prediction_hashes": prediction_hashes,
                          "seed_count": 5, "task_support_fits_checked": 15,
                          "historical_unchanged_predictions": historic,
                          "queries_pairwise_max_abs_difference": task_query_differences,
                          "unchanged_arms_bitwise_equal_across_seeds": True}
    return {"status": "passed", "run": str(run), "protocol": str(protocol_path),
            "protocol_sha256": sha(protocol_path), "audit_script_sha256": sha(__file__),
            "models": reports, "fixed_test_images": 7407, "fixed_test_patients": 2470,
            "seed_prediction_files_checked": 10, "cross_model_support_groups_checked": 15,
            "query_fit_files_checked": 30, "source_and_artifact_paths_checked": len(checked),
            "scope": "Input/output identity, provenance, native parity and support-fit bookkeeping only; no statistical significance claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/full_20261024_v1")
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol_v1/catalog.json")
    parser.add_argument("--output", type=Path, help="Optional new JSON report; refuses to replace an existing report")
    args = parser.parse_args()
    try:
        result = audit(args.run.resolve(), args.protocol.resolve())
    except NotReady as exc:
        print(json.dumps({"status": "not_ready", "reason": str(exc)}, ensure_ascii=False))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "exception": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 1
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(content)
    print(content, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
