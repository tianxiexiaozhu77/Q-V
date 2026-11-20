#!/usr/bin/env python3
"""Freeze a new same-source SIIM holdout without loading a model.

The release's UUID PatientID has no verified longitudinal patient mapping.
All grouping claims therefore concern DICOM identifiers, not real patients.
Run with AlphaRAD/local_setup/python_local.sh. Existing output is never replaced.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
UID_RE = re.compile(r"1\.2\.276\.0\.7230010\.3\.1\.4\.8323329(?:\.\d+)+")
SEEDS = (42, 43, 44, 45, 46)
SPLIT_SEED = 20261024
MASK_FORMAT = "siim_relative_gap_zero_based_column_major_union"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return {"path": str(path.resolve()), "sha256": sha(path)}


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def source_functions():
    converter_source = ROOT / "AlphaRAD/cxrclip/data/dataformat_converter.py"
    converter = load_function(converter_source, "resize_dicom_and_save", {
        "cv2": cv2, "np": np, "pydicom": pydicom, "os": os,
        "apply_modality_lut": apply_modality_lut, "apply_voi_lut": apply_voi_lut,
    })
    decoder_source = ROOT / "AlphaRAD/cxrclip/evaluator_utils.py"
    decoder = load_function(decoder_source, "rle2mask", {"np": np})
    # A non-square, multiple-run case distinguishes relative gaps from absolute
    # starts and checks the column-major geometry without any clinical image.
    toy = decoder("1 2 2 1", width=3, height=2)
    expected = np.array([0, 1, 1, 0, 0, 1]).reshape(3, 2).T
    if not np.array_equal(toy, expected):
        raise ValueError("Author RLE decoder does not match the frozen convention")
    other = load_function(ROOT / "RadZero/exp/cxr_pt/inference/segmentation_utils.py",
                          "rle2mask", {"np": np})
    if not np.array_equal(other("1 2 2 1", 3, 2), expected):
        raise ValueError("The two author RLE implementations disagree")
    return converter, decoder


def historical_sources():
    prior = HERE / "SIIM_CANDIDATE_AUDIT.json"
    audit = json.loads(prior.read_text())
    paths = {ROOT / x["path"] for x in audit["historical_reference_sources"]}
    paths.update((ROOT / "data/radzero/i2cl_support/SIIM").glob("*.json"))
    paths.update((ROOT / "RadZero/runs").glob("*/inputs/siim.json"))
    return sorted(paths, key=str)


def read_header(path):
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=[
        "PatientID", "StudyInstanceUID", "SOPInstanceUID", "Rows", "Columns",
        "IssuerOfPatientID", "PatientIdentityRemoved", "DeidentificationMethod",
    ])
    patient = str(ds.get("PatientID", "")).strip()
    study = str(ds.get("StudyInstanceUID", "")).strip()
    sop = str(ds.get("SOPInstanceUID", "")).strip()
    if not patient or not study or sop != path.stem:
        raise ValueError("Missing DICOM grouping identifiers or filename/SOP mismatch")
    return {"uid": sop, "source_image": str(path),
            "patient_key": digest("SIIM_DICOM_PatientID:" + patient),
            "study_key": digest("SIIM_DICOM_StudyInstanceUID:" + study),
            "group_id": "siim_dicom_" + digest(patient + "|" + study),
            "original_shape": [int(ds.Rows), int(ds.Columns)],
            "has_issuer": bool(str(ds.get("IssuerOfPatientID", "")).strip()),
            "has_identity_removed": bool(str(ds.get("PatientIdentityRemoved", "")).strip()),
            "has_deidentification_method": bool(str(ds.get("DeidentificationMethod", "")).strip())}


def pixel_signature(array):
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def convert_one(row, folder, converter, retain):
    target = folder / (row["group_id"] + ".png")
    converter(row["source_image"], str(target), do_resize=True)
    canonical = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    if canonical is None or canonical.shape != (512, 512):
        raise ValueError("Expected author-converted square 512 PNG")
    raw = pydicom.dcmread(row["source_image"]).pixel_array
    if list(raw.shape) != row["original_shape"]:
        raise ValueError("Pixel array and DICOM header geometry differ")
    output = {"image": str(target), "group_id": row["group_id"],
              "byte_sha256": sha(target), "pixel_sha256": pixel_signature(canonical),
              "raw_pixel_sha256": hashlib.sha256(
                  str(raw.shape).encode() + str(raw.dtype).encode() + raw.tobytes()).hexdigest(),
              "source_sha256": sha(row["source_image"]),
              "original_shape": row["original_shape"], "canonical_shape": [512, 512]}
    if not retain:
        target.unlink()
        output.pop("image")
        output.pop("byte_sha256")
    return output


def validate_rles(rles, shape, decoder):
    height, width = shape
    union = np.zeros((height, width), dtype=bool)
    for value in rles:
        if value.strip() == "-1":
            continue
        tokens = [int(x) for x in value.split()]
        if not tokens or len(tokens) % 2 or min(tokens) < 0:
            raise ValueError("Invalid SIIM relative-gap RLE")
        if sum(tokens) > height * width or any(x == 0 for x in tokens[1::2]):
            raise ValueError("SIIM RLE exceeds geometry or has an empty foreground run")
        union |= decoder(value, width, height).astype(bool)
    return int(union.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "protocol_siim_v1")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    converter, decoder = source_functions()
    source_root = ROOT / "data/alpharad/SIIM"
    rle_path = source_root / "train-rle.csv"
    with rle_path.open(newline="") as handle:
        source_rles = defaultdict(list)
        for raw in csv.DictReader(handle):
            row = {k.strip(): v.strip() for k, v in raw.items()}
            source_rles[row["ImageId"]].append(row["EncodedPixels"])

    snapshots = output / "source_snapshots"
    snapshots.mkdir()
    reference_records, historical_uids = [], set()
    for number, path in enumerate(historical_sources()):
        content = path.read_bytes()
        uids = set(UID_RE.findall(content.decode()))
        historical_uids.update(uids)
        target = snapshots / f"{number:03d}_{path.name}"
        target.write_bytes(content)
        reference_records.append({"path": str(path), "sha256": sha(path),
                                  "snapshot": str(target), "snapshot_sha256": sha(target),
                                  "referenced_sop_uid_count": len(uids)})
    reference_manifest = write_json(output / "historical_source_references.json", {
        "rows": reference_records,
        "scope": "Recorded local evaluation, support, and audited candidate lists; source-machine undocumented use cannot be ruled out.",
    })
    paths = sorted((source_root / "dicom-images-train").rglob("*.dcm"))
    print(f"Reading {len(paths)} DICOM headers; no model is loaded.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        headers = list(pool.map(read_header, paths))
    # This local release has one image for every header PatientID and StudyUID.
    # If that changes, fail rather than silently treating related images as
    # independent groups; the grouping policy would need a new protocol version.
    for key in ("uid", "patient_key", "study_key", "group_id"):
        if len({r[key] for r in headers}) != len(headers):
            raise ValueError("Release no longer has one image per header group: " + key)
    by_uid = {r["uid"]: r for r in headers}
    historical = [r for r in headers if r["uid"] in historical_uids]
    absent_historical = historical_uids - set(by_uid)
    if absent_historical:
        raise ValueError("Some historical referenced DICOM images cannot be audited")
    blocked_patient = {r["patient_key"] for r in historical}
    blocked_study = {r["study_key"] for r in historical}
    eligible = []
    for row in headers:
        if row["uid"] not in source_rles:
            continue
        if row["uid"] in historical_uids or row["patient_key"] in blocked_patient or row["study_key"] in blocked_study:
            continue
        if row["original_shape"] != [1024, 1024]:
            raise ValueError("Unanticipated original SIIM mask geometry")
        row["label"] = int(any(rle != "-1" for rle in source_rles[row["uid"]]))
        eligible.append(row)
    eligible.sort(key=lambda row: row["group_id"])
    print(f"Historical {len(historical)} images; eligible {len(eligible)} header groups.", flush=True)
    historical_tmp = output / "historical_conversion_temporary"
    historical_tmp.mkdir()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        historical_pixels = list(pool.map(
            lambda row: convert_one(row, historical_tmp, converter, False), historical))
    historical_tmp.rmdir()
    historical_pixel_manifest = write_json(output / "historical_pixel_audit.json", {
        "rows": historical_pixels,
        "canonical_pixel_hash": "sha256(str(gray512.shape).encode()+gray512.tobytes())",
        "raw_pixel_hash": "sha256(str(raw.shape).encode()+str(raw.dtype).encode()+raw.tobytes())",
    })
    historical_canonical = {r["pixel_sha256"] for r in historical_pixels}
    historical_raw = {r["raw_pixel_sha256"] for r in historical_pixels}
    assets = output / "images"
    assets.mkdir()
    cache, invalid = {}, set()
    pixel_owner, raw_owner = {}, {}
    rejection_counts = Counter()

    def materialize(row):
        group = row["group_id"]
        if group in invalid:
            return None
        if group in cache:
            return cache[group]
        result = convert_one(row, assets, converter, True)
        reason = None
        if result["pixel_sha256"] in historical_canonical:
            reason = "historical_canonical_pixel_duplicate"
        elif result["raw_pixel_sha256"] in historical_raw:
            reason = "historical_raw_pixel_duplicate"
        elif result["pixel_sha256"] in pixel_owner:
            reason = "selected_canonical_pixel_duplicate"
        elif result["raw_pixel_sha256"] in raw_owner:
            reason = "selected_raw_pixel_duplicate"
        if reason:
            rejection_counts[reason] += 1
            invalid.add(group)
            Path(result["image"]).unlink()
            return None
        pixel_owner[result["pixel_sha256"]] = group
        raw_owner[result["raw_pixel_sha256"]] = group
        cache[group] = result
        return result

    # Random group order is independent of labels and model predictions. Invalid
    # or exact-duplicate candidates are skipped using the same frozen rule.
    rng = np.random.default_rng(SPLIT_SEED)
    test, test_headers = [], []
    for idx in rng.permutation(len(eligible)):
        row = eligible[int(idx)]
        image = materialize(row)
        if image is None:
            continue
        test.append(image.copy())
        test_headers.append(row)
        if len(test) % 100 == 0:
            print(f"Converted and checked {len(test)}/1000 held-out images.", flush=True)
        if len(test) == 1000:
            break
    if len(test) != 1000:
        raise ValueError("Insufficient nonduplicate groups for the fixed holdout")
    test_groups = {r["group_id"] for r in test}
    support_candidates = [r for r in eligible if r["group_id"] not in test_groups]
    seed_records = {}
    for seed in SEEDS:
        selected = []
        seed_rng = np.random.default_rng(seed)
        for label in (1, 0):
            candidates = [r for r in support_candidates if r["label"] == label]
            count = 0
            for idx in seed_rng.permutation(len(candidates)):
                row = candidates[int(idx)]
                image = materialize(row)
                if image is None:
                    continue
                selected.append({**image, "labels": [label], "label": label,
                                 "source_image": row["source_image"]})
                count += 1
                if count == 5:
                    break
            if count != 5:
                raise ValueError("Insufficient support groups after duplicate exclusions")
        assert len({r["group_id"] for r in selected}) == 10
        assert not test_groups.intersection(r["group_id"] for r in selected)
        seed_records[str(seed)] = {"seed": seed, "support_rows": selected,
                                  "support_count": 10, "positive_count": 5, "negative_count": 5}

    # Evaluation annotations are only placed in the separate evaluation file.
    evaluation_rows = []
    for identity, row in zip(test, test_headers):
        rles = source_rles[row["uid"]]
        foreground = validate_rles(rles, row["original_shape"], decoder)
        if bool(foreground) != bool(row["label"]):
            raise ValueError("RLE foreground and binary label disagree")
        if foreground == 1024 * 1024:
            raise ValueError("All-foreground image cannot support pixel AUROC")
        evaluation_rows.append({"image": identity["image"], "group_id": identity["group_id"],
                                "labels": [row["label"]], "rle": rles,
                                "original_shape": row["original_shape"],
                                "canonical_shape": [512, 512],
                                "source_image": row["source_image"],
                                "source_sha256": identity["source_sha256"],
                                "foreground_pixels": foreground})
    assert all("labels" not in row and "label" not in row and "rle" not in row for row in test)
    test_manifest = write_json(output / "test_identity.json", {"rows": test})
    evaluation_manifest = write_json(output / "evaluation.json", {
        "mask_format": MASK_FORMAT, "rle_source": str(rle_path), "rle_source_sha256": sha(rle_path),
        "rows": evaluation_rows,
    })
    source_files = {
        "preparer": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "rle": {"path": str(rle_path), "sha256": sha(rle_path)},
        "author_converter": {"path": str(ROOT / "AlphaRAD/cxrclip/data/dataformat_converter.py"),
                             "sha256": sha(ROOT / "AlphaRAD/cxrclip/data/dataformat_converter.py")},
        "author_rle_decoder": {"path": str(ROOT / "AlphaRAD/cxrclip/evaluator_utils.py"),
                               "sha256": sha(ROOT / "AlphaRAD/cxrclip/evaluator_utils.py")},
        "historical_references": reference_manifest,
        "historical_pixel_audit": historical_pixel_manifest,
    }
    geometry = {
        "source_shape": [1024, 1024], "canonical_shape": [512, 512], "model_input_shape": [518, 518],
        "conversion": "Author resize_dicom_and_save verbatim with do_resize=True: modality LUT, VOI LUT, photometric inversion if needed, resize short edge to 512, per-image min-max uint8.",
        "interpolation_note": "Preserves author cv2.resize positional call; effective interpolation is OpenCV default linear, not its positional LANCZOS constant.",
        "crop": False,
        "spatial_evaluation": "Resize native 37x37 patch response bilinearly to original 1024x1024; masks remain original geometry.",
    }
    audit = {
        "status": "frozen_before_model_inference", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dicom_files": len(headers), "rle_unique_images": len(source_rles),
        "historical_uid_count": len(historical), "eligible_header_groups": len(eligible),
        "eligible_label_counts": dict(Counter(str(r["label"]) for r in eligible)),
        "test_images": len(test), "test_header_groups": len(test_groups),
        "test_positive_images": sum(r["labels"][0] for r in evaluation_rows),
        "support_union_groups": len({r["group_id"] for s in seed_records.values() for r in s["support_rows"]}),
        "support_seed_count": len(SEEDS), "duplicate_rejections": dict(rejection_counts),
        "all_selected_canonical_pixel_unique": len(pixel_owner) == len(cache),
        "all_selected_raw_pixel_unique": len(raw_owner) == len(cache),
        "historical_overlap_uids": 0, "historical_overlap_header_patient_ids": 0,
        "historical_overlap_header_study_uids": 0, "historical_overlap_exact_pixels": 0,
        "support_test_overlap_header_groups": 0, "support_test_overlap_exact_pixels": 0,
        "longitudinal_patient_identity_verified": False,
        "grouping_unit": "dicom_identifier_group",
        "identity_evidence": "Every image in this release has a unique UUID-shaped PatientID and unique StudyInstanceUID. No issuer or deidentification provenance or authoritative longitudinal identity mapping is available.",
        "real_patient_overlap_ruled_out": False, "near_duplicate_overlap_ruled_out": False,
        "pretraining_overlap_ruled_out": False, "external_hospital_validation": False,
        "source_scope_limit": "Local recorded historical references are excluded. Undocumented source-machine use cannot be ruled out.",
        "selection": "Sort eligible hashed header-group IDs; RNG20261024 permutation; retain first 1000 passing deterministic exact-pixel exclusions. Labels and model outputs do not determine test selection. Support RNG42-46 draws five positive and five negative groups each from remainder.",
        "models_loaded": False, "model_inference_run": False,
        "rle_convention_checked_against_two_author_sources": True,
    }
    audit_ref = write_json(output / "SPLIT_AUDIT.json", audit)
    catalog = {
        "schema_version": 1, "dataset": "siim_new_holdout", "class_names": ["Pneumothorax"],
        "eta": 0.3, "split_seed": SPLIT_SEED, "support_seeds": list(SEEDS),
        "grouping_unit": "dicom_identifier_group", "patient_identity_verified": False,
        "test_identity_manifest": test_manifest, "evaluation_manifest": evaluation_manifest,
        "seeds": seed_records, "source_files": source_files, "split_audit": audit_ref,
        "geometry": geometry, "test_count": len(test),
        "description": "New same-source holdout from previously unreferenced SIIM images with audited DICOM-header-group and exact-pixel exclusion; not independent external-hospital validation.",
        "test_annotation_access": "GPU worker reads catalog and test_identity only. Evaluation labels and RLE are read by the statistics stage only after all prediction files are frozen.",
    }
    catalog_ref = write_json(output / "catalog.json", catalog)
    # Verify the published boundary and all stored image checksums, independently
    # of construction lists, before signalling the protocol is ready.
    reread = json.loads((output / "catalog.json").read_text())
    for name in ("test_identity_manifest", "evaluation_manifest", "split_audit"):
        item = reread[name]
        assert sha(item["path"]) == item["sha256"]
    for item in cache.values():
        assert sha(item["image"]) == item["byte_sha256"]
    print(json.dumps({"catalog": catalog_ref, "test_images": audit["test_images"],
                      "test_positive_images": audit["test_positive_images"],
                      "support_union_groups": audit["support_union_groups"],
                      "duplicate_rejections": dict(rejection_counts),
                      "grouping_unit": audit["grouping_unit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
