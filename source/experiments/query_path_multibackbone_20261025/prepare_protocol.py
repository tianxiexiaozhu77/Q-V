#!/usr/bin/env python3
"""Freeze inherited CheXchoNet identities for exploratory query-path comparisons.

No images, support draws, labels, or historical source files are changed. The
inherited `protocol` block describes those assets; `experiment` defines this
new single-view Original / R / Q evaluation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STABILITY = ROOT.parent / "chexchonet_stability_20261024"
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SEEDS = (42, 43, 44, 45, 46)
MODEL_CONTROLS = {
    "AlphaRAD": dict(control_name="readout_only", contrast_type="query_minus_readout_only", symbol="R"),
    "RadZero": dict(control_name="readout_only", contrast_type="query_minus_readout_only", symbol="R"),
    "CARZero": dict(control_name="fixed_aggregation", contrast_type="query_minus_fixed_aggregation", symbol="F"),
    "KAD": dict(control_name="fixed_aggregation", contrast_type="query_minus_fixed_aggregation", symbol="F"),
    "MedKLIP": dict(control_name="fixed_aggregation", contrast_type="query_minus_fixed_aggregation", symbol="F"),
    "MAVL": dict(control_name="fixed_aggregation", contrast_type="query_minus_fixed_aggregation", symbol="F"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def inherited_loader():
    path = STABILITY / "infer_multiseed.py"
    spec = importlib.util.spec_from_file_location("query_multibackbone_inherited_data", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_inherited(source):
    """Recheck ordered identities, strict negatives, support bytes and hashes."""
    source = Path(source).resolve()
    loaded = inherited_loader().load_protocol(source)
    catalog = loaded["master"]
    sources = dict(loaded["sources"])
    # Also verify provenance references which the inference data loader does
    # not itself consume (the source preparation code, original audits, etc.).
    for name, expected in catalog["source_files_sha256"].items():
        actual = sha(name)
        if actual != expected:
            raise ValueError("Inherited source changed: " + name)
        sources[name] = actual
    for path_key, hash_key in (("candidate_pool_csv", "candidate_pool_sha256"),
                               ("common_test_csv", "common_test_csv_sha256"),
                               ("split_audit", "split_audit_sha256")):
        actual = sha(catalog[path_key])
        if actual != catalog[hash_key]:
            raise ValueError("Inherited auxiliary asset changed: " + path_key)
        sources[catalog[path_key]] = actual
    test_patients = {r["patient_id"] for r in loaded["test"]}
    test_images = {r["image_path"] for r in loaded["test"]}
    checks = []
    for seed in SEEDS:
        for task in TASKS:
            rows = loaded["supports"][seed][task]
            patient_overlap = sorted({r["patient_id"] for r in rows} & test_patients)
            image_overlap = sorted({r["image_path"] for r in rows} & test_images)
            if patient_overlap or image_overlap:
                raise ValueError("Support/test overlap")
            checks.append(dict(seed=seed, task=task, support_images=len(rows),
                               support_patients=len({r["patient_id"] for r in rows}),
                               positive_images=sum(r[task] for r in rows),
                               negative_images=sum(1 - r[task] for r in rows),
                               support_test_patient_overlap=0, support_test_image_overlap=0))
    return loaded, sources, checks


def freeze(source, output, candidate_models):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Frozen protocol already exists; use --verify: {output}")
    loaded, sources, checks = validate_inherited(source)
    if len(candidate_models) != len(set(candidate_models)) or not set(candidate_models) <= set(MODEL_CONTROLS):
        raise ValueError("Models require a unique audited control definition")
    parent = loaded["master"]
    protocol = copy.deepcopy(parent)
    experiment = dict(
        version="chexchonet-multibackbone-query-path-v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        study_type="exploratory_multibackbone_mechanism_extension",
        inherited_data_catalog=str(Path(source).resolve()),
        inherited_data_catalog_sha256=sha(source),
        candidate_models=list(candidate_models),
        applicable_models_policy="Freeze the applicable model list in the run plan after interface/parity audit, before full test scoring.",
        task_order=list(TASKS), support_seeds=list(SEEDS), eta=.3,
        views=["identity"], model_controls={name: MODEL_CONTROLS[name] for name in candidate_models},
        primary_comparison="query_minus_model_specific_control",
        additional_comparisons=["query_minus_original", "model_specific_control_minus_original"],
        branch_definition={"original": "aggregation q / readout q",
                           "readout_only": "aggregation q / readout q_prime",
                           "query": "native forward with q_prime",
                           "fixed_aggregation": "Keep q_prime decoder residual/FFN/native classifier but replace visual cross-attention outputs at every layer with those cached from the original-q forward."},
        model_interface_policy="AlphaRAD/RadZero support a separate aggregation/readout query (R). CARZero/KAD/MedKLIP/MAVL use fixed-aggregation F, a different intervention, never described as R. Report Q-R and Q-F separately and do not average them across models.",
        native_endpoint_policy="Original and Q must reproduce each model's frozen native forward with q and q_prime. R uses Q's identical readout query. F must pass an unmodified-q identity test and hook-coverage audit.",
        support_policy="Exactly reuse the five existing task-specific support sets; each task sees only its own 10 labelled support images.",
        no_gradient_training=True, no_test_parameter_selection=True,
        report_all_models_tasks_seeds_and_branches=True,
        outcome_selection_note="All evaluated outcomes, including negative and null Q-R differences, are retained. Any model highlighted after viewing results is an exploratory selection and requires subsequent independent confirmation; it is not a prespecified successful replication.",
        bootstrap=copy.deepcopy(parent["protocol"]["bootstrap"]),
        summary_policy="Compute each seed's metric separately. Report five-seed mean/sample SD and paired patient-bootstrap CI for mean differences. Macro is the equally weighted mean of the three task metrics, not pooled predictions.",
        uncertainty_scope="Patient CIs condition on five fixed support sets drawn from the reserved 30-patient pool. Supports can overlap. No multiplicity correction; no claim about arbitrary future support draws.",
        data_use_history="CheXchoNet has previously been evaluated with Original/Q/V/Q+V after method fixing. This is a new query-path analysis on those existing images, not a newly unseen clinical cohort.",
        isolation_scope="Support/test patient disjointness is verified; pretraining-patient overlap and near-duplicate audit are not established.",
    )
    protocol["experiment"] = experiment
    sources[str(STABILITY / "infer_multiseed.py")] = sha(STABILITY / "infer_multiseed.py")
    sources[str(Path(__file__).resolve())] = sha(__file__)
    audit = dict(status="complete", created_at_utc=experiment["created_at_utc"],
                 inherited_catalog=str(Path(source).resolve()), inherited_catalog_sha256=sha(source),
                 support_draws_changed=False, test_rows_changed=False,
                 test_images=len(loaded["test"]), test_patients=len({r["patient_id"] for r in loaded["test"]}),
                 task_order=list(TASKS), per_seed_task=checks, source_files_sha256=sources,
                 limitations=[experiment["uncertainty_scope"], experiment["data_use_history"], experiment["isolation_scope"], experiment["outcome_selection_note"]])
    audit_path = output.parent / "PROTOCOL_AUDIT.json"
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
    write_json(audit_path, audit)
    protocol["query_path_audit"] = str(audit_path)
    protocol["query_path_audit_sha256"] = sha(audit_path)
    write_json(output, protocol)
    return dict(catalog=str(output), sha256=sha(output), audit=str(audit_path), checks=len(checks), status="complete")


def verify_frozen(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    experiment = value["experiment"]
    if experiment["views"] != ["identity"] or not experiment["model_controls"]:
        raise ValueError("Unexpected query-path controls or views")
    for model, control in experiment["model_controls"].items():
        if control != MODEL_CONTROLS[model]:
            raise ValueError("Unexpected model-specific intervention: " + model)
    if sha(experiment["inherited_data_catalog"]) != experiment["inherited_data_catalog_sha256"]:
        raise ValueError("Parent data catalog changed")
    if sha(value["query_path_audit"]) != value["query_path_audit_sha256"]:
        raise ValueError("Protocol audit changed")
    audit = json.loads(Path(value["query_path_audit"]).read_text())
    for source, expected in audit["source_files_sha256"].items():
        if sha(source) != expected:
            raise ValueError("Frozen source changed: " + source)
    validate_inherited(experiment["inherited_data_catalog"])
    return dict(status="verified", catalog=str(path), sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=STABILITY / "protocol_v1/catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "protocol_v1/catalog.json")
    parser.add_argument("--candidate-models", nargs="+", default=list(MODEL_CONTROLS))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify_frozen(args.output) if args.verify else freeze(args.source, args.output, args.candidate_models)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
