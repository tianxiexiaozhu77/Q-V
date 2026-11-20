"""Identity checks shared by the QaTa five-support-seed experiment.

This module never decodes a mask or loads a model. Annotation manifests are
opened only when explicitly requested by the validation/test scoring stages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
ALPHA = MEDICAL / "AlphaRAD"
LEGACY = MEDICAL / "experiments/qata_validation_threshold_20261024"
PHASE6 = MEDICAL / "experiments/query_search_20261021_152010/code_phase6"
SEEDS = (42, 43, 44, 45, 46)
BRANCHES = ("original", "positive_query", "query_refined", "tta_center_original", "tta_center_refined")
DISPLAY = dict(zip(BRANCHES, ("Original", "Positive query", "Q", "V", "Q+V")))
CHECKPOINT = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve(base, path):
    value = Path(path)
    return (value if value.is_absolute() else Path(base).parent / value).resolve()


def checked(path, expected):
    path = Path(path).resolve()
    if not expected or sha256(path) != expected:
        raise ValueError("Missing/mismatching frozen SHA256: " + str(path))
    return path


def read_protocol(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "catalog.json"
    config = json.loads(path.read_text())
    if "datasets" in config:
        spec = config["datasets"]["qatacovid_seg"]
        path = checked(resolve(path, spec["path"]), spec["sha256"])
        config = json.loads(path.read_text())
    if (config.get("status") != "frozen" or config.get("dataset") != "qatacovid_seg"
            or config.get("model") != "AlphaRAD" or config.get("class_names") != ["covid"]
            or config.get("eta") != .3 or config.get("crop_fraction") != .8
            or not config.get("evaluation_positive_only")
            or config.get("seeds") != list(SEEDS) or config.get("branches") != list(BRANCHES)
            or set(config.get("support_sets", {})) != {str(s) for s in SEEDS}):
        raise ValueError("Expected the frozen AlphaRAD QaTa five-seed/five-branch protocol")
    if not np.array_equal(np.asarray(config["threshold_protocol"]["grid"]), np.arange(0, 1.01, .01)):
        raise ValueError("Threshold grid must remain 0:0.01:1")
    for split, counts in (("validation", (473, 200)), ("test", (2113, 474))):
        if (config[split]["rows_count"], config[split]["patients_count"]) != counts:
            raise ValueError("Fixed validation/test cohort changed: " + split)
    return path, config


def split_descriptor(protocol_path, protocol, split):
    spec = protocol[split]
    return resolve(protocol_path, spec["rows_path"]), spec["rows_sha256"]


def row_image(row):
    return str(Path(row.get("image", row.get("image_path"))).resolve())


def row_patient(row):
    patient = row.get("patient_id")
    if not isinstance(patient, str) or not patient.startswith("sub-S"):
        raise ValueError("Explicit BIMCV patient IDs required for validation/test")
    return patient


def load_rows(protocol_path, protocol, split, identities=False):
    if split not in ("validation", "test"):
        raise ValueError("Expected validation or test split")
    if identities:
        spec = protocol[split + "_identities"]
        path, digest = resolve(protocol_path, spec["path"]), spec["sha256"]
    else:
        path, digest = split_descriptor(protocol_path, protocol, split)
    rows = json.loads(checked(path, digest).read_text())
    if len(rows) != protocol[split]["rows_count"]:
        raise ValueError("Incomplete frozen split: " + split)
    if len({row_image(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation images")
    if len({row_patient(r) for r in rows}) != protocol[split]["patients_count"]:
        raise ValueError("Patient counts changed")
    forbidden = {"class", "label", "labels", "label_vector", "mask", "mask_path", "EncodedPixels"}
    if identities and any(set(row) & forbidden for row in rows):
        raise ValueError("Identity-only files must contain no annotations")
    return rows


def load_supports(protocol_path, protocol):
    supports = {}
    held_out = [*load_rows(protocol_path, protocol, "validation", identities=True),
                *load_rows(protocol_path, protocol, "test", identities=True)]
    held_images = {row_image(r) for r in held_out}
    held_patients = {row_patient(r) for r in held_out}
    held_pixels = {r["pixel_sha256"] for r in held_out}
    for seed in SEEDS:
        spec = protocol["support_sets"][str(seed)]
        rows = json.loads(checked(resolve(protocol_path, spec["rows_path"]), spec["rows_sha256"]).read_text())
        if len(rows) != 10 or spec["rows_count"] != 10:
            raise ValueError("Each seed must have exactly ten support images")
        labels = [r["label_vector"] for r in rows]
        if sorted(labels) != [[0]] * 5 + [[1]] * 5:
            raise ValueError("Each support set requires five positive/five negative labels")
        positives = [r for r in rows if r["label_vector"] == [1]]
        if len({row_patient(r) for r in positives}) != 5:
            raise ValueError("Positive support patients must be distinct within each seed")
        if len({row_image(r) for r in rows}) != 10 or len({r["pixel_sha256"] for r in rows}) != 10:
            raise ValueError("Duplicate support image or pixels within a seed")
        for row in rows:
            if (row_image(row) in held_images or row.get("patient_id") in held_patients
                    or row["pixel_sha256"] in held_pixels):
                raise ValueError("Support overlaps fixed validation/test data")
            if set(row) & {"mask", "mask_path", "EncodedPixels"}:
                raise ValueError("Support fitting must receive only image-level labels")
        supports[seed] = rows
    return supports


def source_snapshot():
    files = list(ROOT.glob("*.py")) + [
        LEGACY / "dice_metrics.py", ALPHA / "local_setup/query_refinement.py",
        ALPHA / "cxrclip/evaluator.py", ALPHA / "cxrclip/evaluator_utils.py",
        ALPHA / "cxrclip/model/cxrclip_plus_dqn.py", PHASE6 / "view_geometry.py",
        PHASE6 / "eval_alpharad_candidates.py",
    ]
    return {str(p.resolve()): sha256(p) for p in sorted(files) if p.is_file()}


def load_maps(run_maps_dir, protocol_path, protocol, split, seed):
    """Load only authenticated complete maps; never accept smoke predictions."""
    root = Path(run_maps_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("smoke") != 0
            or manifest.get("seeds") != list(SEEDS) or manifest.get("branches") != list(BRANCHES)
            or manifest["protocol"]["sha256"] != sha256(protocol_path)
            or manifest.get("checkpoint_sha256") != CHECKPOINT_SHA):
        raise ValueError("Incomplete, smoke, or mismatching native-map provenance")
    if seed not in SEEDS:
        raise ValueError("Unknown support seed")
    source_proof = manifest.get("source_sha256", {})
    required_sources = (ROOT / "infer_maps.py", ROOT / "common.py", ALPHA / "local_setup/query_refinement.py",
                        PHASE6 / "view_geometry.py")
    if any(str(p.resolve()) not in source_proof for p in required_sources):
        raise ValueError("Missing native inference source provenance")
    for source, digest in source_proof.items():
        checked(source, digest)
    rows = load_rows(protocol_path, protocol, split, identities=True)
    spec = manifest["maps"][str(seed)][split]
    path = checked(resolve(manifest_path, spec["path"]), spec["sha256"])
    query_ref = manifest["query_fits"][str(seed)]
    query_path = checked(resolve(manifest_path, query_ref["path"]), query_ref["sha256"])
    support_spec = protocol["support_sets"][str(seed)]
    support_rows = json.loads(checked(resolve(protocol_path, support_spec["rows_path"]),
                                      support_spec["rows_sha256"]).read_text())
    with np.load(query_path, allow_pickle=False) as query:
        if (int(query["support_seed"]) != seed
                or str(query["protocol_sha256"].item()) != sha256(protocol_path)
                or query["image_paths"].tolist() != [row_image(r) for r in support_rows]
                or query["labels"].tolist() != [r["label_vector"] for r in support_rows]):
            raise ValueError("Query fit belongs to another support seed or protocol")
    with np.load(path, allow_pickle=False) as data:
        if (data["image_paths"].tolist() != [row_image(r) for r in rows]
                or data["patient_ids"].tolist() != [row_patient(r) for r in rows]
                or data["class_names"].tolist() != ["covid"]
                or int(data["support_seed"]) != seed
                or str(data["protocol_sha256"].item()) != sha256(protocol_path)):
            raise ValueError("Native-map identities/order/seed differ from frozen protocol")
        maps = {branch: data[branch].copy() for branch in BRANCHES}
    for branch, values in maps.items():
        if values.shape != (len(rows), 1, 1370) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Invalid native FP32 maps: " + branch)
    proof = dict(path=str(path), sha256=spec["sha256"],
                 manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)),
                 query_fit=dict(path=str(query_path), sha256=query_ref["sha256"]))
    return maps, proof
