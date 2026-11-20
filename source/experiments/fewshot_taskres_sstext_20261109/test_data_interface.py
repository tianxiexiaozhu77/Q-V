"""Synthetic CPU tests for cache authentication, ordering and label isolation."""
import json
from pathlib import Path

import numpy as np
import pytest

try:
    from . import data_interface as data
except ImportError:
    import data_interface as data


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": data.sha256(path)}


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


@pytest.fixture
def fixture(tmp_path):
    rng = np.random.default_rng(42)
    model, dataset, seed = "BiomedCLIP", "rsna_pneumonia", 42
    classes = ["Pneumonia"]
    ns, nt, dim = 10, 8, 4
    spaths, tpaths = [f"/support/{i}" for i in range(ns)], [f"/test/{i}" for i in range(nt)]
    sgroups, tgroups = [f"NIH:s{i}" for i in range(ns)], [f"NIH:t{i}" for i in range(nt)]
    labels = np.asarray([[i % 2] for i in range(ns)])
    test_ref = write_json(tmp_path / "sealed_test.json", {
        "rows": [{"image": path, "labels": [i % 2], "group_id": tgroups[i]} for i, path in enumerate(tpaths)]})
    support_rows = [{"image": path, "labels": labels[i].tolist(), "group_id": sgroups[i],
                     "patient_id": sgroups[i]} for i, path in enumerate(spaths)]
    folds = [{"train": [i for i in range(ns) if i not in (2*k, 2*k+1)],
              "validation": [2*k, 2*k+1]} for k in range(5)]
    p = dict(status="frozen", dataset=dataset, support_seed=seed, class_names=classes,
             support_rows=support_rows, support_groups=sgroups, test_images=tpaths,
             test_groups=tgroups, test_patient_ids=tgroups, test_manifest=test_ref,
             support_count=ns, test_count=nt, folds=folds)
    ppath = tmp_path / "protocol.json"
    pref = write_json(ppath, p)
    dataset_ref = write_json(tmp_path / "dataset.json", {"sealed_test_content": "must not deserialize"})
    entry = dict(dataset_ref, class_names=classes, seeds={"42": pref}, support_count_per_seed=ns,
                 test_count=nt, test_group_count=nt, resampling_unit="patient")
    catalog_path = tmp_path / "catalog.json"
    write_json(catalog_path, dict(status="frozen", models=[model], support_seeds=[seed], datasets={dataset: entry}))
    source_run = tmp_path / "run"
    cache = source_run / f"{model}_{dataset}" / "cache" / "seed_42"
    cache.mkdir(parents=True)
    sf, tf = unit_vectors(rng.normal(size=(ns, dim))), unit_vectors(rng.normal(size=(nt, dim)))
    positive, negative = unit_vectors(rng.normal(size=(1, dim))), unit_vectors(rng.normal(size=(1, dim)))
    slogits, tlogits = sf @ (positive - negative).T, tf @ (positive - negative).T
    common = dict(class_names=np.asarray(classes))
    np.savez(cache / "support.npz", **common, image_paths=np.asarray(spaths), groups=np.asarray(sgroups),
             patient_ids=np.asarray(sgroups), features=sf, labels=labels, logits=slogits)
    np.savez(cache / "test.npz", **common, image_paths=np.asarray(tpaths), groups=np.asarray(tgroups),
             patient_ids=np.asarray(tgroups), features=tf, logits=tlogits)
    np.savez(cache / "queries.npz", **common, positive=positive, negative=negative,
             positive_raw=positive * 2, negative_raw=negative * 3)
    native = {key: 1 / (1 + np.exp(-tlogits)) for key in data.NATIVE_METHODS}
    np.savez(cache / "native_predictions.npz", **common, image_paths=np.asarray(tpaths),
             groups=np.asarray(tgroups), patient_ids=np.asarray(tgroups), **native)
    result = dict(status="complete", model=model, dataset=dataset, seed=seed, protocol_sha256=pref["sha256"],
                  catalog_sha256=data.sha256(catalog_path), support_count=ns, test_count=nt, full_test_count=nt,
                  class_names=classes, feature_dimension=dim, feature_definition="synthetic normalized globals",
                  logits_definition="positive minus negative cosine", score_domain="probability",
                  test_labels_used=False, test_boxes_used=False,
                  artifact_sha256={name: data.sha256(cache / name) for name in data.REQUIRED_ARTIFACTS})
    write_json(cache / "result.json", result)
    return dict(model=model, dataset=dataset, seed=seed, catalog_path=catalog_path, source_run=source_run,
                cache=cache, ppath=ppath, test_path=Path(test_ref["path"]))


def load(fixture):
    return data.load_unit(**{key: fixture[key] for key in ("model", "dataset", "seed", "catalog_path", "source_run")})


def rewrite_npz(fixture, filename, callback):
    path = fixture["cache"] / filename
    with np.load(path) as arrays:
        values = {key: arrays[key] for key in arrays.files}
    callback(values)
    np.savez(path, **values)
    result_path = fixture["cache"] / "result.json"
    result = json.loads(result_path.read_text())
    result["artifact_sha256"][filename] = data.sha256(path)
    write_json(result_path, result)


def rewrite_protocol(fixture, callback):
    ppath = fixture["ppath"]
    p = json.loads(ppath.read_text())
    callback(p)
    pref = write_json(ppath, p)
    catalog = json.loads(fixture["catalog_path"].read_text())
    catalog["datasets"][fixture["dataset"]]["seeds"][str(fixture["seed"])] = pref
    write_json(fixture["catalog_path"], catalog)
    result_path = fixture["cache"] / "result.json"
    result = json.loads(result_path.read_text())
    result.update(protocol_sha256=pref["sha256"], catalog_sha256=data.sha256(fixture["catalog_path"]))
    write_json(result_path, result)


def test_fit_loader_does_not_deserialize_test_labels_or_dataset(fixture, monkeypatch):
    original = data._read_json
    forbidden = {fixture["test_path"], fixture["catalog_path"].parent / "dataset.json"}
    def guarded(path):
        if Path(path) in forbidden:
            raise AssertionError("test annotations opened during fitting")
        return original(path)
    monkeypatch.setattr(data, "_read_json", guarded)
    loaded = load(fixture)
    assert loaded.support_features.shape == (10, 4)
    assert loaded.test_features.shape == (8, 4)
    assert len(loaded.folds) == 5
    assert loaded.manifest()["test_labels_loaded"] is False
    assert "test_labels" not in loaded.__dict__
    assert not loaded.support_features.flags.writeable
    assert not loaded.test_features.flags.writeable
    assert set(loaded.native_predictions) == set(data.NATIVE_METHODS)


def test_scoring_label_loader_verifies_labels_after_separate_call(fixture):
    loaded = load(fixture)
    y = data.load_test_labels(loaded)
    np.testing.assert_array_equal(y, [[i % 2] for i in range(8)])
    assert not y.flags.writeable
    fixture["test_path"].write_text('{"rows": []}')
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        data.load_test_labels(loaded)


@pytest.mark.parametrize("artifact", data.REQUIRED_ARTIFACTS)
def test_modified_artifact_rejected_before_array_use(fixture, artifact):
    path = fixture["cache"] / artifact
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load(fixture)


@pytest.mark.parametrize("artifact", ("test.npz", "queries.npz", "native_predictions.npz"))
def test_authenticated_test_labels_still_rejected(fixture, artifact):
    rewrite_npz(fixture, artifact, lambda z: z.update(labels=np.zeros((8, 1))))
    with pytest.raises(ValueError, match="contains test labels"):
        load(fixture)


def test_test_order_cannot_silently_change(fixture):
    rewrite_npz(fixture, "test.npz", lambda z: z.update(image_paths=z["image_paths"][::-1]))
    with pytest.raises(ValueError, match="test paths"):
        load(fixture)


def test_class_order_cannot_silently_change(fixture):
    rewrite_npz(fixture, "queries.npz", lambda z: z.update(class_names=np.asarray(["Wrong"])))
    with pytest.raises(ValueError, match="queries classes"):
        load(fixture)


def test_support_labels_must_match_frozen_protocol(fixture):
    rewrite_npz(fixture, "support.npz", lambda z: z.update(labels=1-z["labels"]))
    with pytest.raises(ValueError, match="support labels"):
        load(fixture)


@pytest.mark.parametrize("change", ("nonfinite", "not_unit"))
def test_invalid_features_are_rejected_without_normalizing_them(fixture, change):
    def mutate(z):
        z["features"][0] *= 2
        if change == "nonfinite":
            z["features"][0, 0] = np.nan
    rewrite_npz(fixture, "test.npz", mutate)
    with pytest.raises(ValueError, match="Nonfinite|not unit-normalized"):
        load(fixture)


def test_support_test_group_overlap_is_rejected(fixture):
    rewrite_protocol(fixture, lambda p: p["support_groups"].__setitem__(0, p["test_groups"][0]))
    with pytest.raises(ValueError, match="support/test overlap"):
        load(fixture)


def test_fold_partition_is_exact(fixture):
    rewrite_protocol(fixture, lambda p: p["folds"][0]["train"].append(0))
    with pytest.raises(ValueError, match="Invalid fold partition"):
        load(fixture)


def test_smoke_cache_cannot_enter_full_comparison(fixture):
    result_path = fixture["cache"] / "result.json"
    result = json.loads(result_path.read_text())
    result["status"] = "smoke_complete"
    write_json(result_path, result)
    with pytest.raises(ValueError, match="noncomplete"):
        load(fixture)


def test_preflight_manifest_is_deterministic_and_immutable(fixture, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "MODELS", (fixture["model"],))
    monkeypatch.setattr(data, "DATASETS", (fixture["dataset"],))
    monkeypatch.setattr(data, "SEEDS", (fixture["seed"],))
    output = tmp_path / "ready.json"
    first = data.prepare_manifest(output, fixture["catalog_path"], fixture["source_run"])
    second = data.prepare_manifest(output, fixture["catalog_path"], fixture["source_run"])
    assert first == second
    assert first["test_labels_loaded"] is False
    payload = dict(first)
    digest = payload.pop("manifest_sha256")
    assert data.object_sha256(payload) == digest
    output.write_text("{}")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        data.prepare_manifest(output, fixture["catalog_path"], fixture["source_run"])
