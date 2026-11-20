#!/usr/bin/env python3
"""Authenticated, label-isolated access to the frozen five-draw comparison data.

Only ``load_test_labels`` parses a test-label manifest. ``load_unit`` and the
preflight CLI hash that manifest without deserializing it. The experiment runner
must commit every requested fit/prediction before calling ``load_test_labels``.
This module needs NumPy, not model packages, Torch, CUDA, or raw-image decoding.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CATALOG = ROOT / "experiments/fewshot_multiseed_20261024/protocol_v1/catalog.json"
DEFAULT_SOURCE_RUN = ROOT / "experiments/fewshot_multiseed_20261024/runs/full_20261024_v2"
MODELS = ("BiomedCLIP", "AlphaRAD")
DATASETS = ("rsna_pneumonia", "chestxdet10")
SEEDS = (42, 43, 44, 45, 46)
NATIVE_METHODS = ("original", "query", "view", "query_view")
REQUIRED_ARTIFACTS = ("support.npz", "test.npz", "queries.npz", "native_predictions.npz")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _checked(path: str | Path, expected: str, sources: dict[str, str]) -> Path:
    path = Path(path).absolute()
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    sources[str(path)] = expected
    return path


def _checked_spec(spec: dict, root: Path, sources: dict[str, str]) -> Path:
    path = Path(spec["path"])
    return _checked(path if path.is_absolute() else root / path, spec["sha256"], sources)


def _npz(path: Path, *, label_free: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if label_free and any("label" in key.lower() or "target" in key.lower()
                              or "box" in key.lower() for key in data.files):
            raise ValueError(f"Fit-visible cache contains test labels/targets/boxes: {path}")
        return {key: data[key] for key in data.files}


def _same(actual: Any, expected: Any, description: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"Identity/order mismatch: {description}")


def _finite(array: np.ndarray, shape: tuple[int, ...], description: str, *, unit=False) -> None:
    if array.shape != shape or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"Nonfinite or invalid shape/dtype: {description}")
    if unit and not np.allclose(np.linalg.norm(array, axis=-1), 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError(f"Feature/text vectors are not unit-normalized: {description}")


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DataUnit:
    model: str
    dataset: str
    seed: int
    support_features: np.ndarray
    labels: np.ndarray
    positive_text: np.ndarray
    negative_text: np.ndarray
    positive_text_raw: np.ndarray
    negative_text_raw: np.ndarray
    support_paths: tuple[str, ...]
    support_groups: tuple[str, ...]
    folds: tuple[dict[str, np.ndarray], ...]
    test_features: np.ndarray
    test_paths: tuple[str, ...]
    classes: tuple[str, ...]
    test_groups: tuple[str, ...]
    support_logits: np.ndarray
    test_logits: np.ndarray
    native_predictions: dict[str, np.ndarray]
    protocol_path: str
    protocol_sha256: str
    source_artifact_sha256: dict[str, str]
    provenance: dict[str, Any]
    _test_manifest: dict[str, str]

    @property
    def groups(self):
        return self.support_groups

    @property
    def queries_positive(self):
        return self.positive_text

    @property
    def queries_negative(self):
        return self.negative_text

    @property
    def unit_id(self):
        return f"{self.model}_{self.dataset}/seed_{self.seed}"

    def manifest(self) -> dict:
        """JSON-safe provenance, without any test-label values or test metrics."""
        return dict(self.provenance, unit_id=self.unit_id,
                    source_artifact_sha256=dict(self.source_artifact_sha256))


def load_unit(model: str, dataset: str, seed: int, catalog_path=DEFAULT_CATALOG,
              source_run=DEFAULT_SOURCE_RUN) -> DataUnit:
    """Read complete caches, keeping test annotations outside the fitting API."""
    catalog_path, source_run = Path(catalog_path).absolute(), Path(source_run).absolute()
    sources = {str(catalog_path): sha256(catalog_path)}
    catalog = _read_json(catalog_path)
    if catalog.get("status") != "frozen" or model not in catalog["models"] or seed not in catalog["support_seeds"]:
        raise ValueError("Unknown model/seed or unfrozen catalog")
    entry = catalog["datasets"][dataset]
    # Dataset-level and test manifests may contain labels: authenticate bytes
    # only. The seed-specific native protocol is deliberately label-free on test.
    _checked_spec(entry, catalog_path.parent, sources)
    ppath = _checked_spec(entry["seeds"][str(seed)], catalog_path.parent, sources)
    p = _read_json(ppath)
    if (p.get("status") != "frozen" or p["dataset"] != dataset or p["support_seed"] != seed
            or any(key in p for key in ("test_rows", "test_labels", "test_targets", "test_boxes"))):
        raise ValueError("Seed protocol identity/status or test-label isolation violated")
    for key in ("support_manifest", "test_manifest", "source_protocol", "source_catalog", "source_native_protocol"):
        if p.get(key):
            _checked_spec(p[key], ppath.parent, sources)
    if catalog.get("audit"):
        _checked_spec(catalog["audit"], catalog_path.parent, sources)
    cache = source_run / f"{model}_{dataset}" / "cache" / f"seed_{seed}"
    result_path = cache / "result.json"
    result = _read_json(result_path)
    sources[str(result_path)] = sha256(result_path)
    if (result.get("status") != "complete" or result.get("model") != model
            or result.get("dataset") != dataset or result.get("seed") != seed
            or result.get("protocol_sha256") != sha256(ppath)
            or result.get("catalog_sha256") != sources[str(catalog_path)]
            or result.get("test_labels_used") is not False
            or result.get("test_boxes_used") is not False):
        raise ValueError("Cache identity/provenance mismatch, noncomplete run, or test-label use")
    artifacts = result.get("artifact_sha256", {})
    if not set(REQUIRED_ARTIFACTS).issubset(artifacts):
        raise ValueError("Missing required artifact SHA256 values")
    for name, digest in artifacts.items():
        if Path(name).name != name:
            raise ValueError("Unsafe cache artifact path")
        _checked(cache / name, digest, sources)
    s, t, q, native = (_npz(cache / name, label_free=name != "support.npz") for name in REQUIRED_ARTIFACTS)
    classes = tuple(p["class_names"])
    _same(classes, entry["class_names"], "catalog class names")
    _same(classes, result["class_names"], "result class names")
    if len(set(classes)) != len(classes) or not classes:
        raise ValueError("Duplicate/empty classes")
    support_paths = tuple(row["image"] for row in p["support_rows"])
    test_paths, support_groups, test_groups = (tuple(p[key]) for key in
                                             ("test_images", "support_groups", "test_groups"))
    ns, nt, nc = len(support_paths), len(test_paths), len(classes)
    if ns != p["support_count"] or ns != result["support_count"] or ns != entry["support_count_per_seed"]:
        raise ValueError("Support count differs from frozen full protocol")
    if (nt != p["test_count"] or nt != result["test_count"] or nt != result["full_test_count"]
            or nt != entry["test_count"] or len(set(test_groups)) != entry["test_group_count"]):
        raise ValueError("Incomplete or altered test cohort")
    if (len(support_groups) != ns or len(test_groups) != nt or len(set(support_paths)) != ns
            or len(set(test_paths)) != nt or not all(support_groups) or not all(test_groups)
            or set(support_paths) & set(test_paths) or set(support_groups) & set(test_groups)):
        raise ValueError("Invalid identities/groups or support/test overlap")
    _same(support_groups, [r.get("group_id", r.get("group", "")) for r in p["support_rows"]], "support protocol groups")
    for arrays, paths, groups, patients, description in (
            (s, support_paths, support_groups, [r.get("patient_id", "") for r in p["support_rows"]], "support"),
            (t, test_paths, test_groups, p["test_patient_ids"], "test"),
            (native, test_paths, test_groups, p["test_patient_ids"], "native")):
        _same(arrays["image_paths"], paths, description + " paths")
        _same(arrays["groups"], groups, description + " groups")
        _same(arrays["patient_ids"], patients, description + " patient identities")
    if dataset == "rsna_pneumonia":
        if entry["resampling_unit"] != "patient" or any(not g.startswith("NIH:") for g in support_groups + test_groups):
            raise ValueError("RSNA requires mapped NIH patient groups")
        _same(s["patient_ids"], support_groups, "RSNA support patients")
        _same(t["patient_ids"], test_groups, "RSNA test patients")
    elif dataset == "chestxdet10" and entry["resampling_unit"] != "image_exact_duplicate_group":
        raise ValueError("ChestXDet10 grouping must remain image/exact-duplicate, not claimed patient isolation")
    for name, arrays in (("support", s), ("test", t), ("queries", q), ("native", native)):
        _same(arrays["class_names"], classes, name + " classes")
    labels = np.asarray([row["labels"] for row in p["support_rows"]])
    _finite(labels, (ns, nc), "support labels")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Support labels must be binary")
    _same(s["labels"], labels, "support labels")
    dimension = int(result["feature_dimension"])
    for name, array, shape in (("support features", s["features"], (ns, dimension)),
                               ("test features", t["features"], (nt, dimension)),
                               ("positive text", q["positive"], (nc, dimension)),
                               ("negative text", q["negative"], (nc, dimension))):
        _finite(array, shape, name, unit=True)
    for key in ("positive", "negative"):
        _finite(q[key + "_raw"], (nc, dimension), key + " raw text")
        norms = np.linalg.norm(q[key + "_raw"], axis=-1, keepdims=True)
        if (norms <= 0).any() or not np.allclose(q[key + "_raw"] / norms, q[key], atol=2e-5, rtol=2e-5):
            raise ValueError("Raw/unit text vectors disagree")
    _finite(s["logits"], (ns, nc), "support logits")
    _finite(t["logits"], (nt, nc), "test logits")
    for key in NATIVE_METHODS:
        _finite(native[key], (nt, nc), "native " + key)
        if (native[key] < 0).any() or (native[key] > 1).any():
            raise ValueError("Native reference is not a probability")
    expected_original = 1.0 / (1.0 + np.exp(-np.clip(t["logits"].astype(np.float64), -700, 700)))
    if not np.allclose(native["original"], expected_original, atol=2e-5, rtol=2e-5):
        raise ValueError("Native Original probability differs from cached native logits")
    folds = []
    validation_seen = np.zeros(ns, dtype=int)
    if len(p["folds"]) != 5:
        raise ValueError("Expected the exact existing five support folds")
    for fold in p["folds"]:
        train, valid = (np.asarray(fold[key], dtype=np.int64) for key in ("train", "validation"))
        if (train.ndim != 1 or valid.ndim != 1 or not len(train) or not len(valid)
                or len(set(train)) != len(train) or len(set(valid)) != len(valid)
                or set(train) & set(valid) or set(train) | set(valid) != set(range(ns))):
            raise ValueError("Invalid fold partition")
        if set(np.asarray(support_groups)[train]) & set(np.asarray(support_groups)[valid]):
            raise ValueError("Support fold group overlap")
        if not ((labels[train].sum(0) > 0) & (labels[train].sum(0) < len(train))).all():
            raise ValueError("Frozen fold lacks a binary training outcome")
        validation_seen[valid] += 1
        folds.append(dict(train=_readonly(train), validation=_readonly(valid)))
    if not np.all(validation_seen == 1):
        raise ValueError("Support folds do not cover every row exactly once")
    provenance = dict(model=model, dataset=dataset, seed=seed, support_count=ns,
        test_count=nt, feature_dimension=dimension, class_names=list(classes),
        support_group_count=len(set(support_groups)), test_group_count=len(set(test_groups)),
        resampling_unit=entry["resampling_unit"], protocol_path=str(ppath),
        protocol_sha256=sha256(ppath), catalog_path=str(catalog_path), catalog_sha256=sources[str(catalog_path)],
        cache_path=str(cache), cache_result_sha256=sources[str(result_path)],
        feature_definition=result["feature_definition"], logits_definition=result["logits_definition"],
        native_score_domain=result["score_domain"],
        text_normalization="positive/negative are unit vectors; corresponding raw vectors are also exposed",
        text_scale_metadata=dict(cached_global_dot_logit_scale=None,
            native_binary_temperature=1.0 if model == "BiomedCLIP" else None,
            checkpoint_logit_scale_in_cache=False,
            note="BiomedCLIP native baseline omits CLIP logit scale; AlphaRAD baseline uses native FLaS, not global cosine"),
        native_branches=list(NATIVE_METHODS), single_view_branches=["original", "query"],
        two_view_branches=["view", "query_view"], support_folds_sha256=object_sha256(p["folds"]),
        support_identities_sha256=object_sha256([support_paths, support_groups]),
        test_identities_sha256=object_sha256([test_paths, test_groups, classes]),
        test_labels_loaded=False, support_test_group_overlap=0,
        source_scope="completed repaired five-draw caches, not historical single-support adapted references")
    label_path = Path(p["test_manifest"]["path"])
    if not label_path.is_absolute():
        label_path = ppath.parent / label_path
    return DataUnit(model, dataset, seed, _readonly(s["features"]), _readonly(labels),
        _readonly(q["positive"]), _readonly(q["negative"]), _readonly(q["positive_raw"]),
        _readonly(q["negative_raw"]), support_paths, support_groups, tuple(folds),
        _readonly(t["features"]), test_paths, classes, test_groups, _readonly(s["logits"]),
        _readonly(t["logits"]), {key: _readonly(native[key]) for key in NATIVE_METHODS},
        str(ppath), sha256(ppath), sources, provenance,
        dict(path=str(label_path.absolute()), sha256=p["test_manifest"]["sha256"]))


def load_test_labels(unit: DataUnit) -> np.ndarray:
    """Evaluator-only annotation read; runner must first verify all fit commits."""
    for path, expected in unit.source_artifact_sha256.items():
        _checked(path, expected, {})
    path = _checked_spec(unit._test_manifest, Path(unit.protocol_path).parent, {})
    rows = _read_json(path)["rows"]
    _same([row["image"] for row in rows], unit.test_paths, "test label manifest paths")
    # Manifest group keys differ across historical exports. When present they
    # must agree; the protocol/cache group identities were already required.
    if all("group_id" in row or "group" in row for row in rows):
        _same([row.get("group_id", row.get("group")) for row in rows], unit.test_groups,
              "test label manifest groups")
    labels = np.asarray([row["labels"] for row in rows])
    _finite(labels, (len(unit.test_paths), len(unit.classes)), "test labels")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Test labels must be binary")
    return _readonly(labels)


def prepare_manifest(output: str | Path, catalog_path=DEFAULT_CATALOG,
                     source_run=DEFAULT_SOURCE_RUN) -> dict:
    """CPU-only, all-20-unit validation; existing identical output is reusable."""
    units = []
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                unit = load_unit(model, dataset, seed, catalog_path, source_run)
                units.append(unit.manifest())
                print(f"Validated {unit.unit_id}: {len(unit.support_paths)} support / {len(unit.test_paths)} test; labels sealed", flush=True)
    manifest = dict(version="taskres-sstext-frozen-data-v1", status="ready", unit_count=len(units),
                    test_labels_loaded=False, source_run=str(Path(source_run).absolute()),
                    catalog_sha256=sha256(catalog_path), interface_sha256=sha256(__file__), units=units)
    manifest["manifest_sha256"] = object_sha256(manifest)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if _read_json(output) != manifest:
            raise FileExistsError(f"Refusing to overwrite different frozen manifest: {output}")
    else:
        with output.open("x") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=HERE / "protocol_ready.json")
    args = parser.parse_args()
    manifest = prepare_manifest(args.output, args.catalog, args.source_run)
    print(json.dumps(dict(status=manifest["status"], units=manifest["unit_count"],
                          manifest_sha256=manifest["manifest_sha256"], output=str(args.output))))


if __name__ == "__main__":
    main()
