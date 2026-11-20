"""Small shared helpers for frozen feature workers; never inspect test labels."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

MEDICAL = Path(__file__).resolve().parents[2]
BRANCHES = ("original", "query", "view", "query_view", "positive_query")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_arguments(description, default_batch_size=8):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--limit", type=int, default=0, help="Limit test images only; keep all supports")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.gpu not in (4, 5, 6, 7):
        parser.error("This experiment is authorized only on physical GPUs 4–7")
    if args.batch_size < 1 or args.limit < 0 or args.workers < 0:
        parser.error("Invalid batch size, limit, or workers")
    args.protocol = args.protocol.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if any((args.output / name).exists() for name in ("support.npz", "test.npz", "result.json")):
        raise FileExistsError("Use a fresh output directory; frozen caches are not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    # Controller may bind an acquired device by UUID; never replace that binding.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "4")
    return args


def _manifest_rows(value, root, support=False):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and ("rows" in value or "support_rows" in value):
        return value.get("support_rows", value.get("rows"))
    spec = value if isinstance(value, dict) else {"path": value}
    path = Path(spec.get("path") or spec.get("manifest"))
    if not path.is_absolute():
        path = root / path
    if spec.get("sha256") and sha(path) != spec["sha256"]:
        raise ValueError(f"Manifest changed: {path}")
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data["support_rows"] if support and "support_rows" in data else data["rows"]


def load_protocol(path):
    """Read support labels and test identities; discard any legacy test labels."""
    config = json.loads(Path(path).read_text())
    classes = config["class_names"]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Invalid ordered classes")
    # Authenticate the fixed source manifests without parsing their test labels.
    for key in ("support_manifest", "test_manifest"):
        source = config.get(key)
        if isinstance(source, dict) and source.get("sha256"):
            if sha(source["path"]) != source["sha256"]:
                raise ValueError(f"Frozen {key} source changed")
    support_value = config.get("support_rows", config.get("support_manifest", config.get("support")))
    test_value = config.get("test_images", config.get("test_rows", config.get("test_manifest", config.get("test"))))
    support_raw = _manifest_rows(support_value, Path(path).parent, support=True)
    test_raw = _manifest_rows(test_value, Path(path).parent)

    def identity(row):
        if isinstance(row, str):
            row = {"image": row}
        image = row.get("image") or row.get("image_path")
        if isinstance(image, str) and image.startswith("["):
            image = ast.literal_eval(image)[0]
        if not image or not Path(image).is_file():
            raise FileNotFoundError(image)
        patient_id = row.get("patient_id") or ""
        if not patient_id and config["dataset"] == "rsna_pneumonia":
            # Existing audited author-default512 RSNA conversion uses its source
            # PatientID as the basename; no filename-based IDs on other datasets.
            patient_id = Path(image).stem
        return {"image": str(image), "patient_id": str(patient_id),
                "group": str(row.get("group") or row.get("group_id") or "")}

    support = []
    for row in support_raw:
        item = identity(row)
        if row.get("byte_sha256") and sha(item["image"]) != row["byte_sha256"]:
            raise ValueError("Support image checksum changed")
        labels = row.get("labels", row.get("label"))
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        if len(labels) != len(classes) or any(float(v) not in (0., 1.) for v in labels):
            raise ValueError("Every support image requires an ordered binary label vector")
        item["labels"] = [int(v) for v in labels]
        support.append(item)
    test = [identity(row) for row in test_raw]
    for name, rows in (("support", support), ("test", test)):
        if not rows or len({r["image"] for r in rows}) != len(rows):
            raise ValueError(f"Empty or duplicated {name} images")
    if {r["image"] for r in support} & {r["image"] for r in test}:
        raise ValueError("Support/test image overlap")
    support_ids = {r["patient_id"] for r in support if r["patient_id"]}
    test_ids = {r["patient_id"] for r in test if r["patient_id"]}
    if support_ids & test_ids:
        raise ValueError("Known support/test patient overlap")
    if float(config.get("eta", .3)) != .3:
        raise ValueError("First-round query strength is fixed at eta=0.3")
    return config, support, test


def identity_arrays(rows, classes):
    import numpy as np
    return {"image_paths": np.asarray([r["image"] for r in rows]),
            "patient_ids": np.asarray([r["patient_id"] for r in rows]),
            "groups": np.asarray([r["group"] for r in rows]),
            "class_names": np.asarray(classes)}


def unit(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or (norms < 1e-12).any():
        raise ValueError("Nonfinite or zero embedding")
    return value / norms


def save_features(path, rows, classes, arrays, support=False):
    import numpy as np
    values = {**identity_arrays(rows, classes), **arrays}
    for key in ("features", "local", "logits"):
        values[key] = np.asarray(values[key], dtype=np.float32)
        if len(values[key]) != len(rows) or not np.isfinite(values[key]).all():
            raise ValueError(f"Invalid cached {key}")
    if values["features"].ndim != 2 or values["local"].shape != (len(rows), len(classes), values["features"].shape[-1]):
        raise ValueError("Feature/cache dimension mismatch")
    if values["logits"].shape != (len(rows), len(classes)):
        raise ValueError("Expected native binary logits [N,C]")
    for key in ("features", "local"):
        np.testing.assert_allclose(np.linalg.norm(values[key], axis=-1), 1., atol=2e-5, rtol=1e-5)
    if support:
        values["labels"] = np.asarray([r["labels"] for r in rows], dtype=np.uint8)
    np.savez_compressed(path, **values)


def progress(args, model, stage, done=0, total=0):
    write_json(args.output / "status.json", {"model": model, "stage": stage, "done": done,
        "total": total, "time_unix": time.time(), "gpu": args.gpu, "smoke": bool(args.limit)})
    print(f"{model} {stage} {done}/{total}", flush=True)


def assert_prediction_parity(actual, expected, name):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5,
                               err_msg=f"Frozen prediction parity failed: {name}")
    return {"passed": True, "max_abs_error": float(np.max(np.abs(actual - expected))),
            "atol": 2e-5, "rtol": 2e-5}
