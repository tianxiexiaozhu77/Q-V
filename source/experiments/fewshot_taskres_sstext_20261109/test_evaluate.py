"""Synthetic checks of the all-unit label barrier and paired seed aggregation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from . import evaluate as evaluation
except ImportError:
    import evaluate as evaluation


def test_mean_is_of_seed_metrics_and_paired_seed_differences():
    rng = np.random.default_rng(45)
    points = rng.uniform(.3, .9, size=(5, len(evaluation.METHODS), 2, 3))
    replicates = np.repeat(points[:, None], 12, axis=1)
    replicates[:, :, 1, :, :] += np.linspace(-.02, .02, 12)[None, :, None, None]
    result = evaluation.summarize_arrays("Synthetic", "synthetic", points, replicates,
                                         ("a", "b"), "image_exact_duplicate_group")
    q = evaluation.METHODS.index("query")
    taskres = evaluation.METHODS.index("taskres")
    qrow = next(row for row in result["means"] if row["method"] == "query")
    assert qrow["auroc_mean_percent"] == pytest.approx(points[:, q, :, 0].mean() * 100)
    assert qrow["auroc_support_sd_pp"] == pytest.approx(points[:, q, :, 0].mean(1).std(ddof=1) * 100)
    comparison = next(row for row in result["comparisons"] if row["target"] == "query" and row["reference"] == "taskres")
    expected = (points[:, q, :, 0].mean(1) - points[:, taskres, :, 0].mean(1)).mean() * 100
    assert comparison["auroc_delta_pp"] == pytest.approx(expected)
    assert comparison["comparison_scope"] == "matched_single_view"
    two_view = next(row for row in result["comparisons"] if row["target"] == "query_view")
    assert two_view["comparison_scope"] == "two_vs_one_views"
    # The fixed-class macro must propagate an invalid class; silently dropping
    # classes/seeds would yield a different estimand and spuriously 12 valid draws.
    replicates[0, 3, q, 1, 0] = np.nan
    result = evaluation.summarize_arrays("Synthetic", "synthetic", points, replicates,
                                         ("a", "b"), "image_exact_duplicate_group")
    qrow = next(row for row in result["means"] if row["method"] == "query")
    assert qrow["auroc_percent_valid_replicates"] == 11
    assert qrow["auroc_percent_invalid_replicates"] == 1


def test_no_labels_without_all_unit_barrier(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("test labels were opened before all predictions committed")
    monkeypatch.setattr(evaluation.data, "load_test_labels", forbidden)
    with pytest.raises(ValueError, match="barrier is missing"):
        evaluation.evaluate_run(tmp_path, bootstrap_replicates=8)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "MODELS", ("BiomedCLIP",))
    monkeypatch.setattr(evaluation, "DATASETS", ("rsna_pneumonia",))
    run = tmp_path / "run"
    run.mkdir()
    source = tmp_path / "locked_source.py"
    source.write_text("# synthetic fixed implementation\n")
    sources = {str(source): evaluation.data.sha256(source)}
    config = dict(methods=list(evaluation.METHODS), models=list(evaluation.MODELS),
                  datasets=list(evaluation.DATASETS), seeds=list(evaluation.SEEDS),
                  comparisons=[list(pair) for pair in evaluation.COMPARISONS],
                  bootstrap=dict(replicates=24, seed=17, unit_by_dataset={"rsna_pneumonia": "patient"}))
    snapshot_units = {f"BiomedCLIP_rsna_pneumonia/seed_{seed}":
                      dict(unit_id=f"BiomedCLIP_rsna_pneumonia/seed_{seed}", fake_authenticated_seed=seed)
                      for seed in evaluation.SEEDS}
    snapshot = dict(config=config, source_sha256=sources, unit_manifests=snapshot_units)
    evaluation.write_json(run / "experiment_snapshot.json", snapshot)
    snapshot_digest = evaluation.data.sha256(run / "experiment_snapshot.json")
    units, verified, labels_opened = {}, [], []
    paths, groups, classes = tuple(f"/test/{i}" for i in range(20)), tuple(f"NIH:{i // 2}" for i in range(20)), ("Pneumonia",)
    labels = np.asarray([[i % 2] for i in range(20)])
    barrier = dict(status="all_predictions_committed", units={}, run_snapshot_sha256=snapshot_digest)
    for seed in evaluation.SEEDS:
        rng = np.random.default_rng(seed)
        unit_id = f"BiomedCLIP_rsna_pneumonia/seed_{seed}"
        probabilities = rng.uniform(.1, .9, size=(20, 1))
        native = {method: probabilities + i * .01 for i, method in enumerate(evaluation.data.NATIVE_METHODS)}
        manifest = dict(unit_id=unit_id, fake_authenticated_seed=seed)
        unit = SimpleNamespace(model="BiomedCLIP", dataset="rsna_pneumonia", seed=seed,
            test_paths=paths, test_groups=groups, classes=classes,
            _test_manifest={"path": "sealed", "sha256": "same_frozen_manifest"},
            native_predictions=native, provenance={"resampling_unit": "patient"},
            manifest=lambda manifest=manifest: manifest)
        units[unit_id] = unit
        folder = run / unit_id
        folder.mkdir(parents=True)
        scores = dict(native)
        for method in ("feature_text_zero_shot", "taskres", "sstext_plus"):
            scores[method] = rng.normal(size=(20, 1))
        np.savez(folder / "predictions.npz", image_paths=np.asarray(paths), class_names=np.asarray(classes),
                 group_ids=np.asarray(groups), **scores)
        evaluation.write_json(folder / "fit_info.json", {"test_labels_opened": False})
        np.savez(folder / "taskres_state.npz", weights=np.zeros((1, 3)))
        np.savez(folder / "sstext_state.npz", weights=np.zeros((1, 3)))
        commit = dict(unit_id=unit_id, unit_manifest_sha256=evaluation.data.object_sha256(manifest),
                      run_snapshot_sha256=snapshot_digest, source_sha256=sources,
                      artifacts={name: evaluation.data.sha256(folder / name) for name in
                                 ("fit_info.json", "predictions.npz", "taskres_state.npz", "sstext_state.npz")},
                      test_labels_opened=False)
        evaluation.write_json(folder / "fit_commit.json", commit)
        barrier["units"][unit_id] = {"fit_commit_sha256": evaluation.data.sha256(folder / "fit_commit.json")}
    evaluation.write_json(run / "predictions_complete.json", barrier)
    def fake_load(model, dataset, seed, *args):
        unit_id = f"{model}_{dataset}/seed_{seed}"
        verified.append(unit_id)
        return units[unit_id]
    def fake_labels(unit):
        assert set(verified) == set(units), "test labels opened before every unit was authenticated"
        labels_opened.append(unit.seed)
        return labels
    monkeypatch.setattr(evaluation.data, "load_unit", fake_load)
    monkeypatch.setattr(evaluation.data, "load_test_labels", fake_labels)
    return SimpleNamespace(run=run, units=units, verified=verified, labels_opened=labels_opened,
                           source=source, labels=labels)


def test_end_to_end_synthetic_shared_plan_and_reusable_output(synthetic_run, monkeypatch):
    run = synthetic_run.run
    actual_bootstrap = evaluation.engine.paired_cluster_bootstrap
    plans = []
    def record_plan(*args, **kwargs):
        plans.append(kwargs["plan"])
        return actual_bootstrap(*args, **kwargs)
    monkeypatch.setattr(evaluation.engine, "paired_cluster_bootstrap", record_plan)
    result = evaluation.evaluate_run(run, bootstrap_replicates=24, bootstrap_seed=17)
    assert result["status"] == "complete"
    assert result["all_predictions_committed_before_test_labels"]
    assert len(synthetic_run.labels_opened) == 1
    assert len(plans) == 5 and all(plan is plans[0] for plan in plans)
    assert len(result["means"]) == len(evaluation.METHODS)
    assert len(result["per_seed"]) == len(evaluation.METHODS) * 5
    for filename in ("means.csv", "per_seed.csv", "comparisons.csv", "per_class.csv", "RESULTS.json", "RESULTS.txt"):
        assert (run / filename).is_file()
    units = list(synthetic_run.units)
    with np.load(run / units[0] / "bootstrap_metrics.npz") as archive:
        assert archive["replicate_metrics"].shape == (24, len(evaluation.METHODS), 1, 3)
    resumed = evaluation.evaluate_run(run, bootstrap_replicates=24, bootstrap_seed=17)
    assert resumed == result
    assert len(synthetic_run.labels_opened) == 1  # verified resume never reopens labels
    (run / "means.csv").write_text("corrupt\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluation.evaluate_run(run, bootstrap_replicates=24, bootstrap_seed=17)


def test_missing_one_commit_blocks_all_labels(synthetic_run):
    barrier_path = synthetic_run.run / "predictions_complete.json"
    barrier = evaluation.read_json(barrier_path)
    barrier["units"].pop(next(iter(barrier["units"])))
    evaluation.write_json(barrier_path, barrier)
    with pytest.raises(ValueError, match="Incomplete/mismatched"):
        evaluation.evaluate_run(synthetic_run.run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened


@pytest.mark.parametrize("artifact", ("fit_info.json", "predictions.npz", "taskres_state.npz", "sstext_state.npz"))
def test_late_unit_artifact_corruption_blocks_all_labels(synthetic_run, artifact):
    last_unit = list(synthetic_run.units)[-1]
    path = synthetic_run.run / last_unit / artifact
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluation.evaluate_run(synthetic_run.run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened


def test_native_controls_cannot_change_even_with_updated_commit(synthetic_run):
    run = synthetic_run.run
    last_unit = list(synthetic_run.units)[-1]
    folder = run / last_unit
    with np.load(folder / "predictions.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["query"][0, 0] += .01
    np.savez(folder / "predictions.npz", **arrays)
    commit = evaluation.read_json(folder / "fit_commit.json")
    commit["artifacts"]["predictions.npz"] = evaluation.data.sha256(folder / "predictions.npz")
    evaluation.write_json(folder / "fit_commit.json", commit)
    barrier = evaluation.read_json(run / "predictions_complete.json")
    barrier["units"][last_unit]["fit_commit_sha256"] = evaluation.data.sha256(folder / "fit_commit.json")
    evaluation.write_json(run / "predictions_complete.json", barrier)
    with pytest.raises(ValueError, match="Native control differs"):
        evaluation.evaluate_run(run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened


def test_source_lock_change_blocks_all_labels(synthetic_run):
    synthetic_run.source.write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluation.evaluate_run(synthetic_run.run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened


def rewrite_snapshot_and_commit_hashes(run, change):
    """Keep outer byte hashes valid to test semantic snapshot/commit checks."""
    snapshot_path = run / "experiment_snapshot.json"
    snapshot = evaluation.read_json(snapshot_path)
    change(snapshot)
    evaluation.write_json(snapshot_path, snapshot)
    digest = evaluation.data.sha256(snapshot_path)
    barrier = evaluation.read_json(run / "predictions_complete.json")
    barrier["run_snapshot_sha256"] = digest
    for unit_id in barrier["units"]:
        path = run / unit_id / "fit_commit.json"
        commit = evaluation.read_json(path)
        commit["run_snapshot_sha256"] = digest
        evaluation.write_json(path, commit)
        barrier["units"][unit_id]["fit_commit_sha256"] = evaluation.data.sha256(path)
    evaluation.write_json(run / "predictions_complete.json", barrier)


@pytest.mark.parametrize("key", ("methods", "models", "datasets", "seeds", "comparisons"))
def test_frozen_scope_cannot_differ_from_evaluator(synthetic_run, key):
    def change(snapshot):
        snapshot["config"][key] = snapshot["config"][key][:-1]
    rewrite_snapshot_and_commit_hashes(synthetic_run.run, change)
    with pytest.raises(ValueError, match="Frozen evaluation configuration mismatch: " + key):
        evaluation.evaluate_run(synthetic_run.run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened


@pytest.mark.parametrize("replicates,seed", ((25, 17), (24, 18)))
def test_bootstrap_options_must_match_frozen_config(synthetic_run, replicates, seed):
    with pytest.raises(ValueError, match="Frozen bootstrap replicates/seed"):
        evaluation.evaluate_run(synthetic_run.run, bootstrap_replicates=replicates, bootstrap_seed=seed)
    assert not synthetic_run.labels_opened


def test_bootstrap_grouping_must_match_authenticated_protocol(synthetic_run):
    def change(snapshot):
        snapshot["config"]["bootstrap"]["unit_by_dataset"]["rsna_pneumonia"] = "image"
    rewrite_snapshot_and_commit_hashes(synthetic_run.run, change)
    with pytest.raises(ValueError, match="bootstrap grouping differs"):
        evaluation.evaluate_run(synthetic_run.run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened


def test_commit_source_subset_cannot_evade_full_snapshot_lock(synthetic_run):
    run = synthetic_run.run
    last_unit = list(synthetic_run.units)[-1]
    commit_path = run / last_unit / "fit_commit.json"
    commit = evaluation.read_json(commit_path)
    commit["source_sha256"] = {}
    evaluation.write_json(commit_path, commit)
    barrier = evaluation.read_json(run / "predictions_complete.json")
    barrier["units"][last_unit]["fit_commit_sha256"] = evaluation.data.sha256(commit_path)
    evaluation.write_json(run / "predictions_complete.json", barrier)
    with pytest.raises(ValueError, match="source locks differ from frozen snapshot"):
        evaluation.evaluate_run(run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened


def test_snapshot_unit_manifest_must_match_both_commit_and_cache(synthetic_run):
    def change(snapshot):
        last_unit = list(snapshot["unit_manifests"])[-1]
        snapshot["unit_manifests"][last_unit]["fake_authenticated_seed"] = -1
    rewrite_snapshot_and_commit_hashes(synthetic_run.run, change)
    with pytest.raises(ValueError, match="unit provenance differs"):
        evaluation.evaluate_run(synthetic_run.run, bootstrap_replicates=24, bootstrap_seed=17)
    assert not synthetic_run.labels_opened
