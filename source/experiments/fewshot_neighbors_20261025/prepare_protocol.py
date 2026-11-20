#!/usr/bin/env python3
"""Freeze CheXchoNet supports and patient-grouped OOF folds for fair neighbors.

Preparation may inspect the inherited labels to verify identity and isolation.
The default public loader exposes only test identities; test labels are kept in
a separate asset and require include_test_labels=True after fits are committed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STABILITY = ROOT / "experiments/chexchonet_stability_20261024"
PARENT = ROOT / "experiments/query_path_multibackbone_20261025"
LEGACY = ROOT / "experiments/fewshot_native_query"
SOURCE = PARENT / "protocol_v1/catalog.json"
MODELS = ("BiomedCLIP", "AlphaRAD")
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SEEDS = (42, 43, 44, 45, 46)
SINGLE_METHODS = ("original", "query", "prototype", "prototype_mix", "tip_adapter", "gda",
                  "simnl", "project_mix", "linear_probe")
COMPETITORS = SINGLE_METHODS[2:]
METHODS = SINGLE_METHODS + tuple(method + "_view" for method in SINGLE_METHODS)
VERSION = "chexchonet-fair-neighbors-five-support-seeds-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(spec):
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("Frozen asset changed: " + spec["path"])
    return json.loads(Path(spec["path"]).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError("Refusing to modify frozen asset: " + str(path))
    else:
        with path.open("x") as handle:
            handle.write(text)
    return ref(path)


def _parent_module():
    path = PARENT / "prepare_protocol.py"
    spec = importlib.util.spec_from_file_location("neighbors_parent_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_folds(rows, seed, task):
    """Five disjoint folds, each holding out one positive and one negative."""
    if len(rows) != 10 or len({r["patient_id"] for r in rows}) != 10:
        raise ValueError("Require ten different support patients")
    positive = sorted((i for i, row in enumerate(rows) if row["label"] == 1),
                      key=lambda i: (rows[i]["patient_id"], rows[i]["image_path"]))
    negative = sorted((i for i, row in enumerate(rows) if row["label"] == 0),
                      key=lambda i: (rows[i]["patient_id"], rows[i]["image_path"]))
    if len(positive) != 5 or len(negative) != 5:
        raise ValueError("Require five positive and five negative support patients")
    fold_seed = 20261025 + 1009 * int(seed) + 1000003 * (TASKS.index(task) + 1)
    rng = random.Random(fold_seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    folds = []
    for pos, neg in zip(positive, negative):
        validation = sorted((pos, neg))
        folds.append(dict(train=[i for i in range(10) if i not in validation], validation=validation))
    return folds, fold_seed


def audit_folds(rows, folds):
    coverage, audits = [], []
    for i, fold in enumerate(folds):
        train, validation = fold["train"], fold["validation"]
        tr_groups = {rows[j]["patient_id"] for j in train}
        va_groups = {rows[j]["patient_id"] for j in validation}
        tr_pos = sum(rows[j]["label"] for j in train)
        va_pos = sum(rows[j]["label"] for j in validation)
        if (len(train) != 8 or len(validation) != 2 or tr_pos != 4 or va_pos != 1 or
                tr_groups & va_groups or set(train) & set(validation) or sorted(train + validation) != list(range(10))):
            raise ValueError("Invalid patient-grouped balanced support fold")
        coverage.extend(validation)
        audits.append(dict(fold=i, train_patients=8, validation_patients=2,
                           train_positive=4, train_negative=4, validation_positive=1,
                           validation_negative=1, train_validation_patient_overlap=0))
    if len(folds) != 5 or sorted(coverage) != list(range(10)):
        raise ValueError("OOF folds must cover each support patient exactly once")
    return audits


def comparison_definitions():
    comparisons = []
    for suffix, condition in (("", "single"), ("_view", "dual")):
        for comparator in COMPETITORS:
            comparisons.append(dict(name="query" + suffix + "_minus_" + comparator + suffix,
                                    high="query" + suffix, low=comparator + suffix,
                                    view_condition=condition, role="primary_same_view_competitor"))
        comparisons.append(dict(name="query" + suffix + "_minus_original" + suffix,
                                high="query" + suffix, low="original" + suffix,
                                view_condition=condition, role="descriptive_native_baseline"))
    for method in SINGLE_METHODS:
        comparisons.append(dict(name=method + "_view_minus_" + method, high=method + "_view", low=method,
                                view_condition="dual_minus_single", role="descriptive_view_increment"))
    return comparisons


def prepare(output, algorithm_plan, source=SOURCE):
    output = Path(output).resolve()
    catalog_path = output / "catalog.json"
    if catalog_path.exists():
        raise FileExistsError("Frozen catalog exists; use --verify: " + str(catalog_path))
    algorithm_plan = Path(algorithm_plan).resolve()
    plan_value = json.loads(algorithm_plan.read_text())
    if not isinstance(plan_value, dict) or not plan_value:
        raise ValueError("An explicit neighbor algorithm/grid plan is required before freezing")
    parent_module = _parent_module()
    parent_module.verify_frozen(source)
    loaded, sources, support_checks = parent_module.validate_inherited(source)
    catalog = copy.deepcopy(loaded["master"])
    classes = loaded["classes"]
    test = loaded["test"]
    test_identity = dict(image_paths=[r["image_path"] for r in test],
                         patient_ids=[r["patient_id"] for r in test], task_keys=list(TASKS), class_names=classes)
    pool_identity = dict(image_paths=[r["image_path"] for r in loaded["pool"]],
                         patient_ids=[r["patient_id"] for r in loaded["pool"]])
    test_patients = set(test_identity["patient_ids"])
    pool_lookup = {p: i for i, p in enumerate(pool_identity["image_paths"])}
    support_tasks, fold_audits = {}, {}
    for seed in SEEDS:
        support_tasks[str(seed)], fold_audits[str(seed)] = {}, {}
        for task in TASKS:
            rows = [dict(image=r["image_path"], image_path=r["image_path"], patient_id=r["patient_id"],
                         group_id=r["patient_id"], labels=[int(r[task])], label=int(r[task]))
                    for r in loaded["supports"][seed][task]]
            if test_patients.intersection(r["patient_id"] for r in rows):
                raise ValueError("Support patient overlaps fixed test")
            folds, fold_seed = make_folds(rows, seed, task)
            fold_audits[str(seed)][task] = audit_folds(rows, folds)
            support_tasks[str(seed)][task] = dict(
                rows=rows, support_rows=rows, groups=[r["patient_id"] for r in rows], folds=folds,
                support_pool_indices=[pool_lookup[r["image_path"]] for r in rows], fold_seed=fold_seed,
                support_seed=seed, task=task, support_images=10, support_patients=10,
                support_positive=5, support_negative=5, support_test_patient_overlap=0)
    assets = {
        "pool_identity": write_json(output / "pool_identity.json", pool_identity),
        "test_identity": write_json(output / "test_identity.json", test_identity),
        "test_labels": write_json(output / "test_labels.json", dict(test_identity, labels=[[int(r[t]) for t in TASKS] for r in test])),
        "support_folds": write_json(output / "support_folds.json", dict(seeds=support_tasks)),
        "algorithm_plan": ref(algorithm_plan),
    }
    legacy_protocol = LEGACY / "protocol_v1/rsna_pneumonia.json"
    legacy = json.loads(legacy_protocol.read_text())
    algorithm_sources = {str(p): sha(p) for p in (LEGACY / "methods.py", LEGACY / "simnl_adapter.py", legacy_protocol)}
    for name in ("neighbor_methods.py", "SOURCES_PROJECT_MIX.json"):
        path = algorithm_plan.parent / name
        if path.exists():
            algorithm_sources[str(path)] = sha(path)
    sources.update(algorithm_sources)
    sources[str(Path(__file__).resolve())] = sha(__file__)
    sources[str(PARENT / "prepare_protocol.py")] = sha(PARENT / "prepare_protocol.py")
    sources[str(algorithm_plan)] = sha(algorithm_plan)
    selection = dict(data="support_only", folds=5, group="patient", validation_positive=1, validation_negative=1,
                     metric="pooled_oof_trapezoidal_auprc", same_folds_across_models_and_methods=True,
                     fit_views=["identity"], select_once_for_single_and_dual=True,
                     test_labels_allowed_in_fit=False,
                     tie="Use the predeclared candidate order/tolerance in each frozen algorithm; never inspect test outcomes.",
                     fold_algorithm="Sort positive and negative rows by patient/image, independently shuffle with Python random.Random(20261025 + 1009*support_seed + 1000003*(task_index+1)), pair one of each per fold.")
    catalog["neighbor_assets"] = assets
    catalog["experiment"] = dict(
        version=VERSION, created_at_utc=datetime.now(timezone.utc).isoformat(),
        study_type="exploratory_same_support_same_view_neighbor_comparison",
        inherited_data_catalog=str(Path(source).resolve()), inherited_data_catalog_sha256=sha(source),
        models=list(MODELS), task_order=list(TASKS), support_seeds=list(SEEDS),
        methods=list(METHODS), single_methods=list(SINGLE_METHODS), competitors=list(COMPETITORS),
        comparisons=comparison_definitions(), eta=.3, center_crop_ratio=.8, views=["identity", "center80"],
        view_fusion="Equal weights in each method's declared native score domain. AlphaRAD Q averages logits before sigmoid; BiomedCLIP Q averages native probabilities; score methods average margins.",
        method_fit_policy="Use only task-specific original-view supports; select once with shared folds, refit all ten supports, apply the same fitted model to both test views.",
        support_policy="Each task receives only its own ten labelled support images; cross-task feature caching does not increase its label budget.",
        selection=selection, legacy_fixed_grids=legacy["fixed_grids"], algorithm_sources_sha256=algorithm_sources,
        no_backbone_training=True, all_methods_training_free=False, test_parameter_selection=False,
        all_models_tasks_methods_seeds_retained=True, score_labels_after_all_task_fits_committed=True,
        fit_bundles_per_model=15, bootstrap=copy.deepcopy(catalog["protocol"]["bootstrap"]),
        summary_policy="Metrics per seed, then five-seed mean and sample SD; three-task macro within each seed/draw. No prediction ensemble or cross-model mean.",
        uncertainty_scope="Patient-bootstrap intervals condition on five fixed supports drawn from the existing 30-patient candidate pool. Supports may overlap. No multiplicity correction or arbitrary-future-support guarantee.",
        data_use_history="CheXchoNet has already been evaluated in fixed-method transfer and query-path extension. This is a new comparator study on those existing images, not a previously unseen cohort.",
        selection_note="All methods, views, tasks and support seeds are retained. Any later highlighted method/task is an exploratory selection.",
        isolation_scope="Support/test patient separation is verified; pretraining-patient overlap and near duplicates are not fully audited.",
    )
    audit = dict(status="passed", inherited_catalog=ref(source), supports_changed=False, test_changed=False,
                 models=list(MODELS), methods=list(METHODS), task_order=list(TASKS), support_seeds=list(SEEDS),
                 test_images=7407, test_patients=2470, candidate_images=87, candidate_patients=30,
                 support_test_patient_overlap=0, inherited_support_checks=support_checks, folds=fold_audits,
                 source_files_sha256=sources, algorithm_plan=assets["algorithm_plan"],
                 test_labels_separate=True, default_loader_opens_test_labels=False)
    catalog["neighbor_audit"] = write_json(output / "PROTOCOL_AUDIT.json", audit)
    write_json(catalog_path, catalog)
    return dict(status="frozen", catalog=str(catalog_path), sha256=sha(catalog_path),
                models=list(MODELS), methods=len(METHODS), support_folds=75)


def load_protocol(path, include_test_labels=False):
    """Read fit-facing assets; default never opens the test-label asset/CSVs."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "catalog.json"
    catalog = json.loads(path.read_text())
    experiment = catalog["experiment"]
    if (experiment["version"] != VERSION or experiment["models"] != list(MODELS) or
            experiment["methods"] != list(METHODS) or experiment["task_order"] != list(TASKS) or
            experiment["support_seeds"] != list(SEEDS) or experiment["comparisons"] != comparison_definitions()):
        raise ValueError("Unexpected frozen neighbor comparison scope")
    if sha(experiment["inherited_data_catalog"]) != experiment["inherited_data_catalog_sha256"]:
        raise ValueError("Parent data catalog changed")
    audit = checked(catalog["neighbor_audit"])
    if audit["status"] != "passed" or audit["support_test_patient_overlap"] != 0:
        raise ValueError("Protocol isolation/fold audit incomplete")
    assets = catalog["neighbor_assets"]
    identity, pool_identity, task_records, algorithm = [checked(assets[key]) for key in
        ("test_identity", "pool_identity", "support_folds", "algorithm_plan")]
    if (identity["task_keys"] != list(TASKS) or len(identity["image_paths"]) != 7407 or
            len(set(identity["patient_ids"])) != 2470 or len(pool_identity["image_paths"]) != 87 or
            len(set(pool_identity["patient_ids"])) != 30 or set(pool_identity["patient_ids"]) & set(identity["patient_ids"])):
        raise ValueError("Frozen test/pool identity changed")
    supports, groups, folds, bundles = {}, {}, {}, {}
    for seed in SEEDS:
        supports[seed], groups[seed], folds[seed], bundles[seed] = {}, {}, {}, {}
        for task in TASKS:
            bundle = task_records["seeds"][str(seed)][task]
            rows = bundle["support_rows"]
            if bundle["rows"] != rows:
                raise ValueError("Support row aliases differ")
            expected, fold_seed = make_folds(rows, seed, task)
            if bundle["folds"] != expected or bundle["fold_seed"] != fold_seed:
                raise ValueError("Shared support CV folds changed")
            audit_folds(rows, expected)
            if bundle["groups"] != [r["patient_id"] for r in rows]:
                raise ValueError("Support patient group order differs")
            for i, row in zip(bundle["support_pool_indices"], rows):
                if (row["image_path"] != pool_identity["image_paths"][i] or row["patient_id"] != pool_identity["patient_ids"][i] or
                        row["labels"] != [row["label"]] or row["patient_id"] in set(identity["patient_ids"])):
                    raise ValueError("Task-only support identity differs")
            supports[seed][task], groups[seed][task], folds[seed][task], bundles[seed][task] = rows, bundle["groups"], expected, bundle
    sources = {str(path): sha(path), catalog["neighbor_audit"]["path"]: catalog["neighbor_audit"]["sha256"]}
    sources.update({assets[key]["path"]: assets[key]["sha256"] for key in
                    ("test_identity", "pool_identity", "support_folds", "algorithm_plan")})
    for source, digest in experiment["algorithm_sources_sha256"].items():
        if sha(source) != digest:
            raise ValueError("Frozen algorithm source changed: " + source)
        sources[source] = digest
    result = dict(catalog=catalog, master=catalog, path=path, sources=sources, classes=identity["class_names"],
                  test=[dict(image_path=p, patient_id=g) for p, g in zip(identity["image_paths"], identity["patient_ids"])],
                  pool=[dict(image_path=p, patient_id=g) for p, g in zip(pool_identity["image_paths"], pool_identity["patient_ids"])],
                  test_identity=identity, pool_identity_source=assets["pool_identity"],
                  supports=supports, groups=groups, folds=folds, support_tasks=bundles,
                  test_labels_source=assets["test_labels"], algorithm_plan=algorithm,
                  selection=experiment["selection"], legacy_fixed_grids=experiment["legacy_fixed_grids"])
    if include_test_labels:
        labelled = checked(assets["test_labels"])
        if any(labelled[key] != identity[key] for key in identity):
            raise ValueError("Test labels/identity alignment differs")
        labels = labelled["labels"]
        if len(labels) != 7407 or any(len(row) != 3 or any(v not in (0, 1) for v in row) for row in labels):
            raise ValueError("Invalid test labels")
        result["test_labels"] = labels
        sources[assets["test_labels"]["path"]] = assets["test_labels"]["sha256"]
    return result


def verify_frozen(path, include_test_labels=False):
    data = load_protocol(path, include_test_labels=include_test_labels)
    return dict(status="verified", catalog=str(data["path"]), sha256=sha(data["path"]),
                test_labels_loaded=include_test_labels, models=list(MODELS), methods=len(METHODS), support_folds=75)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=HERE / "protocol_v1")
    parser.add_argument("--algorithm-plan", type=Path, default=HERE / "METHODS_PLAN.json")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--include-test-labels", action="store_true")
    args = parser.parse_args()
    result = verify_frozen(args.output, args.include_test_labels) if args.verify else prepare(args.output, args.algorithm_plan, args.source)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
