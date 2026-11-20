#!/usr/bin/env python3
"""Replay preserved phase6 computations without the obsolete campaign cutoff.

Use the chosen model's local_setup/python_local.sh. --dry-run validates sources
and prints the plan without importing the model, querying a GPU, or writing a run.
The legacy workers require both original and query-only controls even in
--mode original; read the original branch for the unadapted-model result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "experiments/query_search_20261021_152010"
CODE = CAMPAIGN / "code_phase6"
ALPHA_DATASETS = (
    "chest14", "chestxdet10", "chestxdet10_gt", "rsna_pneumonia",
    "siim_pneumothorax", "node21", "qatacovid_seg", "shenzhenxray",
    "montgomery", "covidkaggle",
)
RAD_DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")
# The archived schema requires a finite number. Runtime deadline callbacks and
# the RadZero deadline alarm are disabled below; this value is only a placeholder.
DEADLINE_PLACEHOLDER = 1_000_000_000_000.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("alpharad", "radzero"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", type=int, choices=range(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("original", "four-arm"), default="four-arm")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    allowed = ALPHA_DATASETS if args.model == "alpharad" else RAD_DATASETS
    if args.dataset not in allowed:
        parser.error(f"Dataset must be one of {allowed}; CheXpert uses its separate existing script")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    output = args.output.expanduser().resolve()
    receipt = output.with_name(output.name + ".launch.json")
    if output.exists() or receipt.exists():
        parser.error("Use a new output path; archived results will not be overwritten")
    source = CODE / f"eval_{args.model}_candidates.py"
    tracked = [Path(__file__).resolve(), source, CODE / "query_candidates.py",
               CODE / "view_geometry.py", CODE / "vision_cache.py"]
    for path in tracked:
        if not path.is_file():
            parser.error(f"Required preserved source is missing: {path}")
    names = ["original"] if args.mode == "original" else [
        "original", "uniform_eta03", "tta_center_original", "tta_center_refined"]
    worker_args = [str(source), "--dataset", args.dataset, "--output", str(output),
        "--deadline", str(DEADLINE_PLACEHOLDER), "--batch-size",
        "8" if args.model == "alpharad" else "16", "--workers", str(args.workers),
        "--candidate-module", str(CODE / "query_candidates.py"),
        "--vision-cache-root", str(CAMPAIGN / "assets/vision_tokens_fp32"),
        "--cache-mode", "read", "--candidate-names", *names]
    if args.model == "alpharad":
        worker_args.append("--save-full-maps")
    plan = {
        "model": args.model, "dataset": args.dataset, "mode": args.mode,
        "physical_gpu": args.gpu, "worker_argv": worker_args,
        "source_sha256": {str(path): digest(path) for path in tracked},
        "deadline_disabled": True, "deadline_field_is_placeholder": True,
        "computational_code_modified": False,
        "legacy_required_controls": ["original", "uniform_eta03"],
        "note": "Only execution deadline handling changes; preserved queries, support/test manifests, model, scoring, geometry, caches and mandatory original-reference checks are retained.",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(ROOT / "AlphaRAD/local_setup"))
    from run_pretrained import reserve_gpu, release_gpu
    lease, gpu_info = reserve_gpu(args.gpu)
    if lease is None:
        raise RuntimeError(f"GPU is occupied; no process was started: {gpu_info}")
    old_argv, old_timer = sys.argv, signal.setitimer
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_info["uuid"]
        sys.dont_write_bytecode = True
        output.parent.mkdir(parents=True, exist_ok=True)
        plan.update(started_at=datetime.now(timezone.utc).isoformat(),
                    gpu_info=gpu_info, python=sys.executable, status="starting")
        receipt.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
        spec = importlib.util.spec_from_file_location(f"paper_phase6_{args.model}", source)
        worker = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = worker
        spec.loader.exec_module(worker)
        if args.model == "radzero":
            worker.CAMPAIGN_DEADLINE = DEADLINE_PLACEHOLDER
            worker.check_deadline = lambda _deadline: None
            def no_deadline_alarm(which, seconds, interval=0.0):
                if which == signal.ITIMER_REAL:
                    return old_timer(which, 0.0, 0.0)
                return old_timer(which, seconds, interval)
            signal.setitimer = no_deadline_alarm
        else:
            def signal_only(_deadline):
                if worker.STOP_REASON is not None:
                    raise worker.DeadlineReached(worker.STOP_REASON)
            worker.check_deadline = signal_only
        sys.argv = worker_args
        worker.main()
        plan.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        plan.update(status="failed_or_interrupted", error=repr(error),
                    finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        signal.setitimer = old_timer
        sys.argv = old_argv
        release_gpu(lease)
        if receipt.exists():
            receipt.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
