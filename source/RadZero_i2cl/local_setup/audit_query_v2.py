#!/usr/bin/env python3
"""Independent support-vector, prediction, identity, and metric audit of query runs."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from audit_i2cl import independent_metrics, verify_metrics
from extract_frozen_features import fingerprint, public_spec, sha256
from i2cl_data import load_spec

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")
DEADLINE = datetime.fromisoformat("2026-10-21T10:00:00+08:00").timestamp()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def normalized(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def audit(name, raw):
    if time.time() >= DEADLINE:
        raise RuntimeError("Authorized deadline reached")
    directory = raw / "query_refinement" / name
    manifest = read(directory / "manifest.json")
    assert manifest["status"] == "complete"
    cache = Path(manifest["cache"])
    original = read(cache / "manifest.json")
    assert original["status"] == "complete"
    assert sha256(cache / "manifest.json") == manifest["cache_manifest_sha256"]
    for key in ("support_identity", "test_identity", "checkpoint", "class_names", "prompts"):
        assert manifest[key] == original[key], key
    for path, digest in manifest["source_sha256"].items():
        assert sha256(path) == digest, path
    for path, digest in manifest["split_sha256"].items():
        assert sha256(path) == digest, path
    support = load_spec(Path(manifest["split_dir"]), name, "support")
    test = load_spec(Path(manifest["split_dir"]), name, "test")
    for part, spec in (("support", support), ("test", test)):
        metadata = read(cache / f"{part}_metadata.json")
        assert public_spec(spec) == metadata
        assert fingerprint(metadata) == manifest[f"{part}_identity"]
    for key in ("image_paths", "ids", "groups"):
        assert not set(support[key]) & set(test[key]), key
    assert sha256(directory / "fit_vectors.npz") == manifest["fit_vectors_sha256"]
    fit = np.load(directory / "fit_vectors.npz", allow_pickle=False)
    labels = np.asarray(support["labels"])
    np.testing.assert_array_equal(labels, np.load(cache / "support_labels.npy"))
    features = np.asarray(np.load(cache / "support_class_local.npy"), dtype=np.float64)
    queries = np.asarray(np.load(cache / "text_features.npy"), dtype=np.float64)
    positives = labels.sum(0)
    negatives = len(labels) - positives
    np.testing.assert_array_equal(positives, fit["positive_counts"])
    np.testing.assert_array_equal(negatives, fit["negative_counts"])
    means_pos = np.stack([features[labels[:, c] == 1, c].mean(0)
                          for c in range(labels.shape[1])])
    means_neg = np.stack([features[labels[:, c] == 0, c].mean(0)
                          for c in range(labels.shape[1])])
    directions = normalized(means_pos - means_neg)
    np.testing.assert_allclose(means_pos, fit["positive_means"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(means_neg, fit["negative_means"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(directions, fit["directions"], atol=3e-7, rtol=0)
    np.testing.assert_array_equal(queries, fit["original_queries"])
    query_errors = {}
    for index, eta in enumerate(fit["etas"]):
        expected = queries if eta == 0 else normalized(normalized(queries) + eta * directions)
        np.testing.assert_allclose(expected, fit["queries"][index], atol=3e-7, rtol=0)
        query_errors[str(eta)] = float(np.max(np.abs(expected - fit["queries"][index])))
    baseline = np.load(cache / "test_base_logits.npy")
    np.testing.assert_array_equal(baseline, np.load(directory / "eta_0/logits.npy"))
    base_metrics = independent_metrics(test["labels"], baseline, test["class_names"])
    verify_metrics(manifest["baseline_metrics"], base_metrics, "baseline")
    results = []
    for stem, record in manifest["predictions"].items():
        if time.time() >= DEADLINE:
            raise RuntimeError("Authorized deadline reached")
        child = directory / stem
        assert sha256(child / "logits.npy") == record["logits_sha256"]
        assert sha256(child / "predictions.csv") == record["predictions_sha256"]
        logits = np.load(child / "logits.npy")
        with (child / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(test["image_paths"])
        csv_scores = []
        for index, row in enumerate(rows):
            assert int(row["index"]) == index
            assert row["image_path"] == test["image_paths"][index]
            np.testing.assert_array_equal(
                [float(row[f"label:{c}"]) for c in test["class_names"]], test["labels"][index])
            csv_scores.append([float(row[f"logit:{c}"]) for c in test["class_names"]])
        np.testing.assert_array_equal(np.asarray(csv_scores), logits)
        metrics = independent_metrics(test["labels"], logits, test["class_names"])
        verify_metrics(record["metrics"], metrics, stem)
        for metric, delta in record["delta_pp"].items():
            np.testing.assert_allclose(delta, 100*(metrics[metric] - base_metrics[metric]),
                                       atol=1e-10, rtol=0)
        results.append({"eta": record["eta"], "images": len(logits),
                        "predictions_sha256": sha256(child / "logits.npy"),
                        "metrics_independently_verified": True})
    return {
        "dataset": name, "status": "passed", "baseline_bitwise_equal": True,
        "support_only_vectors_independently_recomputed": True,
        "max_query_error_by_eta": query_errors, "trials": results,
        "audit_source_sha256": sha256(__file__),
        "limitation": "CPU artifact audit; does not independently rerun image encoder or prove generalization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=PROJECT / "local_assets/outputs/overnight_v2_20261020_2250")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "runs/overnight_v2_20261020_2250/experiments/query_audit")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.datasets:
        path = args.output / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        try:
            report = audit(name, args.raw)
        except Exception as error:
            report = {"dataset": name, "status": "failed", "error": repr(error)}
            write(path, report)
            raise
        write(path, report)
        reports.append(report)
        print(f"{name}: audit passed", flush=True)
    write(args.output / "summary.json", {"reports": reports})


if __name__ == "__main__":
    main()
