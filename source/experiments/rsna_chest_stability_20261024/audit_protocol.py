#!/usr/bin/env python3
"""Independent frozen-protocol and acquisition provenance audit; no predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(ref):
    assert sha(ref["path"]) == ref["sha256"], ref["path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol_v1/catalog.json")
    args = parser.parse_args()
    catalog = json.loads(args.protocol.read_text())
    assert catalog["status"] == "frozen"
    audit = {"status": "passed", "catalog": {"path": str(args.protocol.resolve()), "sha256": sha(args.protocol)},
             "datasets": {}, "reads_predictions": False}
    mappings_path = HERE / "source_data/pneumonia-challenge-dataset-mappings_2018.json"
    mapping = {r["subset_img_id"]: r for r in json.loads(mappings_path.read_text())}
    def patient(image_uuid):
        return "NIH:" + mapping[image_uuid]["img_id"].split("_")[0]
    for dataset, ref in catalog["datasets"].items():
        check(ref)
        d = json.loads(Path(ref["path"]).read_text())
        check(d["historical_protocol"])
        old = json.loads(Path(d["historical_protocol"]["path"]).read_text())
        check(old["test_manifest"])
        original = json.loads(Path(old["test_manifest"]["path"]).read_text())["rows"]
        test = d["test_rows"]
        assert [(r["image"], r["labels"]) for r in test] == [(r["image"], r["labels"]) for r in original]
        assert d["test_manifest"]["sha256"] == old["test_manifest"]["sha256"]
        check(d["candidate_pool"])
        pool = json.loads(Path(d["candidate_pool"]["path"]).read_text())
        candidate = {r["image_id"]: r for r in pool["candidate_rows"]}
        test_groups = {r["group_id"] for r in test}
        test_pixels = {r["pixel_sha256"] for r in test}
        assert len(test_groups) == d["test_group_count"]
        assert d["test_groups"] == [r["group_id"] for r in test]
        if dataset == "rsna_pneumonia":
            assert all(r["group_id"] == r["patient_id"] == patient(Path(r["image"]).stem) for r in test)
            assert len(test_groups) == 3720
        else:
            assert all(r["group_id"] == "ChestXDet10:pixel:" + r["pixel_sha256"] for r in test)
            assert len(test_groups) == 492
        sets, checks = {}, []
        for seed, entry in d["seeds"].items():
            support = entry["support_rows"]
            assert {r["image_id"] for r in support} <= set(candidate)
            assert not {r["group_id"] for r in support} & test_groups
            assert not {r["pixel_sha256"] for r in support} & test_pixels
            assert len({r["group_id"] for r in support}) == len(support)
            assert all(r["labels"] == candidate[r["image_id"]]["labels"] for r in support)
            assert all(sha(r["image"]) == r["byte_sha256"] for r in support)
            check(entry["manifest"])
            check(entry["native_protocol"])
            assert sha(entry["csv_path"]) == entry["csv_sha256"]
            if dataset == "rsna_pneumonia":
                assert all(r["patient_id"] == patient(r["image_id"]) for r in support)
                assert sum(r["labels"][0] for r in support) == 5
            else:
                assert Counter(r["anchor_class"] for r in support) == {c: 5 for c in d["class_names"]}
            sets[seed] = frozenset(r["image"] for r in support)
            checks.append({"seed": int(seed), "support_count": len(support), "support_groups": len({r["group_id"] for r in support}),
                           "support_test_group_overlap": 0, "support_test_decoded_pixel_overlap": 0,
                           "positive_counts": entry["positive_counts"]})
        assert len(set(sets.values())) == 5
        result = {"test_image_count": len(test), "test_group_count": len(test_groups),
                  "original_test_paths_labels_order_unchanged": True,
                  "test_manifest_sha256": d["test_manifest"]["sha256"], "five_distinct_support_draws": True,
                  "candidate_count": len(candidate), "candidate_metadata_only": True,
                  "support_union_count": len(set.union(*(set(s) for s in sets.values()))),
                  "seeds": checks,
                  "resampling_unit": d["resampling_unit"], "dataset_protocol": ref}
        if dataset == "rsna_pneumonia":
            history_patients = {patient(Path(r["image"]).stem) for r in old["support_rows"]}
            overlap = history_patients & test_groups
            result["historical_patient_audit"] = {"historical_support_patients": len(history_patients),
                "historical_support_test_overlapping_patients": len(overlap),
                "historical_test_images_belonging_to_overlapping_patients": sum(r["patient_id"] in overlap for r in test),
                "historical_results_changed": False,
                "note": "The historical DICOM PatientID UUID identified each image; original NIH subjects expose patient overlap. Newly sampled support patients exclude the entire historical cohort."}
            acquisition = [HERE / "source_data/rsna_sampling_freeze.json", HERE / "source_data/rsna_zip_index.json",
                           HERE / "prepare_rsna_support.py", HERE / "prepare_chest_support.py",
                           MEDICAL / "AlphaRAD/local_setup/prepare_storage_data.py",
                           MEDICAL / "AlphaRAD/cxrclip/data/dataformat_converter.py"]
            result["source_urls"] = [
                "https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018",
                "https://atlas.rsna.org/cards/74777c92-9b0b-4f06-a788-71a8a12c4e93",
                "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"]
        else:
            acquisition = [HERE / "source_data/chestxdet10_sampling_freeze.json", HERE / "source_data/chestxdet10_train_zip_index.json",
                           HERE / "prepare_chest_support.py"]
            result["source_urls"] = pool["source_urls"]
        paths = set(acquisition) | {Path(p) for p in pool["sources"]}
        result["acquisition_source_sha256"] = {str(p): sha(p) for p in sorted(paths)}
        result["source_pool_sha256"] = sha(d["candidate_pool"]["path"])
        audit["datasets"][dataset] = result
    audit["audit_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    destination = args.protocol.parent / "PROTOCOL_AUDIT.json"
    data = (json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if destination.exists() and destination.read_bytes() != data:
        raise FileExistsError("Refusing to replace a changed protocol audit")
    destination.write_bytes(data)
    print(json.dumps({"status": "passed", "audit": str(destination),
                      "datasets": {key: {k: v for k, v in d.items() if k in ("test_image_count", "test_group_count", "support_union_count", "historical_patient_audit")}
                                   for key, d in audit["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
