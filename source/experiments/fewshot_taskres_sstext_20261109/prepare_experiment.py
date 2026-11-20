#!/usr/bin/env python3
"""CPU-only readiness checks and frozen configuration; never fit real supports."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from data_interface import DEFAULT_CATALOG, DEFAULT_SOURCE_RUN, MODELS, DATASETS, SEEDS, prepare_manifest, sha256
from experiment import HERE, ROOT, METHODS, read_json, verify_repositories, write_json, now


def configuration():
    return dict(version="taskres-sstext-comparison-v1", models=list(MODELS), datasets=list(DATASETS),
        seeds=list(SEEDS), catalog_path=str(DEFAULT_CATALOG), source_run=str(DEFAULT_SOURCE_RUN),
        methods=list(METHODS),
        taskres=dict(epoch_candidates=[0,20,50,100,200], learning_rate_candidates=[0.0002,0.002],
                     residual_scale=0.5, schedule_epochs=200, batch_size=256,
                     weight_decay=0.0005, warmup_lr=0.00001),
        sstext_plus=dict(text_lambda="adaptive_shots", balanced_importance=False, repel=True),
        text_base_policy="Use authenticated positive_raw/negative_raw: AlphaRAD raw projected native text; BiomedCLIP native unit text. No extra base normalization before fitting.",
        feature_policy="Single original-view frozen global features from the exact existing five-support comparison caches; no additional support views.",
        scale_policy="Pretrained global contrastive logit_scale, frozen from checkpoint; AlphaRAD FLaS scale is not substituted for its global scale.",
        taskres_selection="One LR/epoch per backbone-dataset-support draw by pooled grouped-support OOF macro trapezoidal AUPRC; no test tuning.",
        bootstrap=dict(replicates=2000, seed=20261024,
                       unit_by_dataset={"rsna_pneumonia":"patient", "chestxdet10":"image_exact_duplicate_group"},
                       scope="Test-group uncertainty conditional on five fixed support draws; no multiple-comparison correction."),
        comparisons=[[a,b] for a,b in [("taskres","original"),("sstext_plus","original"),
                      ("taskres","feature_text_zero_shot"),("sstext_plus","feature_text_zero_shot"),
                      ("query","original"),("query","taskres"),("query","sstext_plus"),
                      ("query_view","taskres"),("query_view","sstext_plus")]],
        interpretation="Feature-level binary/OvR ports, not reproductions of the authors' original data/augmentation pipelines; Q uses the native readout, AlphaRAD baselines use pooled features. Q+V has two views and is not a matched-view competitor ranking.")


def backbone_scales():
    specs = {
        "BiomedCLIP": (ROOT/"baselines/BiomedCLIP/checkpoints/open_clip_pytorch_model.bin", "logit_scale", None),
        "AlphaRAD": (ROOT/"AlphaRAD/local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar", "logit_scale", "model"),
    }
    result = {}
    for model, (path, key, parent) in specs.items():
        digest = sha256(path)
        provenance = {str(path): digest}
        verified_units = []
        # The completed cache results do not have a source_sha256 field.
        # Authenticate the actual chain: each result pins its reuse manifest,
        # whose model source map pins the checkpoint. Check the per-cache audit
        # independently and retain every file hash for the run source snapshot.
        for dataset in DATASETS:
            for seed in SEEDS:
                unit_id = f"{model}_{dataset}/seed_{seed}"
                cache_root = DEFAULT_SOURCE_RUN/f"{model}_{dataset}/cache"
                cache_result = cache_root/f"seed_{seed}/result.json"
                cached = read_json(cache_result)
                if (cached.get("status") != "complete" or cached.get("model") != model
                        or cached.get("dataset") != dataset or cached.get("seed") != seed):
                    raise ValueError("Invalid cache identity for checkpoint verification: " + unit_id)
                audit_path = Path(cached["reuse_audit"]).resolve()
                if audit_path != (cache_root/"reuse_audit.json").resolve():
                    raise ValueError("Unexpected cache reuse audit path: " + unit_id)
                audit = read_json(audit_path)
                manifest_path = Path(audit["reuse_manifest"]).resolve()
                manifest_digest = sha256(manifest_path)
                if (cached.get("reuse_manifest_sha256") != manifest_digest
                        or audit.get("reuse_manifest_sha256") != manifest_digest):
                    raise ValueError("Cache reuse manifest differs from its saved digest: " + unit_id)
                manifest = read_json(manifest_path)
                manifest_checkpoint = manifest.get("models", {}).get(model, {}).get(str(path))
                audit_checkpoint = audit.get("files", {}).get(str(path))
                if manifest_checkpoint != digest or audit_checkpoint != digest:
                    raise ValueError("Checkpoint differs from authenticated cache provenance: " + unit_id)
                metadata = cached.get("metadata", {})
                if model == "AlphaRAD" and metadata.get("checkpoint_sha256") != digest:
                    raise ValueError("AlphaRAD checkpoint differs from cache metadata: " + unit_id)
                if model == "BiomedCLIP" and metadata.get("adapter_metadata", {}).get("checkpoint") != str(path):
                    raise ValueError("BiomedCLIP checkpoint path differs from cache metadata: " + unit_id)
                provenance.update({str(cache_result): sha256(cache_result),
                                   str(audit_path): sha256(audit_path),
                                   str(manifest_path): manifest_digest})
                verified_units.append(unit_id)
        # Existing published local checkpoints already used by the cached models;
        # AlphaRAD bundles its OmegaConf training config and needs full loading.
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=(model != "AlphaRAD"))
        state = checkpoint[parent] if parent else checkpoint
        value = state[key].detach().cpu().float()
        if value.numel() != 1 or not torch.isfinite(value.exp()) or value.exp() <= 0:
            raise ValueError("Invalid frozen checkpoint scale")
        result[model] = dict(checkpoint=str(path), checkpoint_sha256=digest,
            state_dict_path=f"{parent + '.' if parent else ''}{key}",
            natural_log=float(value), multiplier=float(value.exp()),
            cache_provenance_verified=True, cache_units_verified=verified_units,
            provenance_sha256=provenance,
            role="frozen pretrained global contrastive scale; affects fitting, not selected on supports or test")
        del checkpoint, state
    return result


def prepare():
    data = prepare_manifest(HERE/"protocol_ready.json")
    repos = verify_repositories()
    scales = backbone_scales()
    config = configuration()
    write_json(HERE/"experiment_config.json", config)
    packages = {name:importlib.metadata.version(name) for name in
                ("torch","numpy","scikit-learn","scipy","omegaconf")}
    environment = dict(python=platform.python_version(), packages=packages,
        cuda_build=torch.version.cuda, cuda_initialized=torch.cuda.is_initialized(),
        execution_environment=str(HERE/".venv"),
        isolation="venv with system-site-packages; reuses installed PyTorch/CUDA; exact used package versions locked")
    if environment["cuda_initialized"]:
        raise RuntimeError("Preparation must not initialize a CUDA context")
    write_json(HERE/"environment.json", environment)
    freeze = subprocess.check_output([str(HERE/".venv/bin/python"),"-m","pip","freeze"],text=True)
    (HERE/"requirements-resolved.txt").write_text(freeze)
    files = list(HERE.glob("*.py")) + list(HERE.glob("*.sh")) + [HERE/"experiment_config.json", HERE/"protocol_ready.json",
            HERE/"environment.json", HERE/"requirements.txt"]
    files += sorted((HERE/"sources").rglob("*.py")) + sorted((HERE/"sources").rglob("*.json"))
    files += [ROOT/"experiments/chexchonet_stability_20261024/cluster_bootstrap.py"]
    files += [Path(path) for repo in repos.values() for path in repo["files"]]
    sources = {str(path.resolve()):sha256(path) for path in files}
    for scale in scales.values():
        sources.update(scale["provenance_sha256"])
    ready = dict(status="ready_not_run", prepared_at=now(), config=config,
                 source_sha256=sources, repositories=repos, backbone_scales=scales,
                 environment=environment, unit_manifests={u["unit_id"]:u for u in data["units"]},
                 data_manifest_sha256=sha256(HERE/"protocol_ready.json"),
                 formal_fits_started=False, test_labels_opened=False, gpu_execution_tested=False)
    write_json(HERE/"READY.json",ready)
    print("Ready: 20 validated units, both official sources pinned; no formal fitting or GPU job started.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
