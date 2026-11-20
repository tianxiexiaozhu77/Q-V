#!/usr/bin/env python3
"""Schedule the post-development eta/crop sensitivity campaign.

The campaign is deliberately a fixed grid, not a validation-set search.  It
uses the existing frozen support/test manifests and writes one immutable
directory per model, dataset, eta, and crop ratio.  ``--plan-only`` performs
all path and manifest checks without loading CUDA models; ``--smoke`` limits
each worker to a small prefix for a safe first pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CATALOG_CHEXPERT = ROOT / "baselines/evaluation/query_transfer_20261022_v1/catalog.json"
CATALOG_CHEXPERT_ALPHA = ROOT / "AlphaRAD/local_setup/chexpert_query_data_20261022_shot5_seed42/catalog.json"
CATALOG_CHEXCHONET = ROOT / "baselines/evaluation/chexchonet_20261023/catalog.json"
CATALOG_CHEXCHONET_NATIVE = ROOT / "AlphaRAD/local_setup/chexchonet_20261023/catalog.json"
MODEL_ROOTS = {
    "BiomedCLIP": ROOT / "baselines/protocols/alpharad_binary/BiomedCLIP",
    "MedKLIP": ROOT / "baselines/protocols/chexchonet/MedKLIP",
    "CARZero": ROOT / "baselines/CARZero",
}
MODEL_ROOTS_CHEXPERT = {
    "MedKLIP": ROOT / "baselines/MedKLIP",
}


def slug(value):
    return format(float(value), ".6g").replace(".", "p").replace("-", "m")


def base_job(model, dataset, eta, ratio, weight, output, smoke, batch_size, workers):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "QUERY_REFINEMENT_ETA": str(eta), "QUERY_REFINEMENT_CROP_RATIO": str(ratio),
           "QUERY_REFINEMENT_VIEW_WEIGHT": str(weight)}
    if model in MODEL_ROOTS:
        model_root = MODEL_ROOTS_CHEXPERT.get(model, MODEL_ROOTS[model]) if dataset == "chexpert" else MODEL_ROOTS[model]
        if dataset == "chexpert":
            command = ["bash", str(model_root / "local_setup/python_local.sh"),
                       str(ROOT / "baselines/local_setup/evaluate_query_transfer.py"),
                       "--model", model, "--model-root", str(model_root),
                       "--catalog", str(CATALOG_CHEXPERT), "--output", str(output),
                       "--datasets", "chexpert", "--batch-size", str(batch_size)]
        else:
            command = ["bash", str(model_root / "local_setup/python_local.sh"),
                       str(ROOT / "baselines/local_setup/evaluate_chexchonet.py"),
                       "--model", model, "--model-root", str(MODEL_ROOTS[model]),
                       "--catalog", str(CATALOG_CHEXCHONET), "--output", str(output),
                       "--batch-size", str(batch_size)]
        if smoke:
            command += ["--smoke", str(smoke)]
        return command, env
    if model == "AlphaRAD":
        python = ROOT / "AlphaRAD/local_setup/python_local.sh"
        if dataset == "chexpert":
            worker = ROOT / "AlphaRAD/local_setup/eval_chexpert_query_refinement.py"
            command = ["bash", str(python), str(worker), "--catalog", str(CATALOG_CHEXPERT_ALPHA),
                       "--output", str(output), "--eta", str(eta), "--crop-ratio", str(ratio),
                       "--view-weight", str(weight), "--batch-size", str(batch_size),
                       "--workers", str(workers)]
            if smoke:
                command += ["--limit-test", str(smoke)]
        else:
            worker = ROOT / "AlphaRAD/local_setup/eval_chexchonet.py"
            command = ["bash", str(python), str(worker), "--catalog", str(CATALOG_CHEXCHONET_NATIVE),
                       "--output", str(output), "--mode", "paired-four-arm", "--task", "all",
                       "--eta", str(eta), "--crop-ratio", str(ratio), "--view-weight", str(weight),
                       "--batch-size", str(batch_size), "--workers", str(workers)]
            if smoke:
                command += ["--smoke-batches", "1"]
        return command, env
    if model == "RadZero":
        python = ROOT / "RadZero_i2cl/local_setup/python_local.sh"
        if dataset == "chexpert":
            worker = ROOT / "experiments/sensitivity_grid/eval_radzero_chexpert.py"
            eta_values = ["0"] if float(eta) == 0.0 else ["0", str(eta)]
            command = ["bash", str(python), str(worker), "--catalog", str(CATALOG_CHEXPERT),
                       "--output", str(output), "--etas", *eta_values, "--crop-ratios", str(ratio),
                       "--view-weight", str(weight), "--batch-size", str(batch_size),
                       "--workers", str(workers)]
            if smoke:
                command += ["--limit", str(smoke)]
        else:
            worker = ROOT / "RadZero_i2cl/local_setup/eval_chexchonet.py"
            command = ["bash", str(python), str(worker), "--catalog", str(CATALOG_CHEXCHONET_NATIVE),
                       "--output", str(output), "--mode", "paired-four-arm", "--tasks",
                       "slvh", "dlv", "composite_slvh_dlv", "--eta", str(eta),
                       "--crop-ratio", str(ratio), "--view-weight", str(weight),
                       "--batch-size", str(batch_size), "--workers", str(workers)]
            if smoke:
                command += ["--limit", str(smoke)]
        return command, env
    raise ValueError(model)


def main(args):
    if args.max_workers < 1 or not args.gpus or args.max_workers > len(args.gpus):
        raise ValueError("max-workers must be between 1 and the number of GPUs")
    if set(args.models) - {"AlphaRAD", "RadZero", *MODEL_ROOTS}:
        raise ValueError("Unknown model")
    if set(args.datasets) - {"chexpert", "chexchonet"}:
        raise ValueError("Unknown dataset")
    args.output.mkdir(parents=True, exist_ok=False)
    # Independent slices: eta sensitivity at the fixed crop, then crop
    # sensitivity at the historical eta.  This avoids an unplanned Cartesian
    # search while covering both development choices.
    trials = []
    for eta in args.etas:
        trials.append((float(eta), .8))
    for ratio in args.crop_ratios:
        pair = (.3, float(ratio))
        if pair not in trials:
            trials.append(pair)
    jobs = []
    for model in args.models:
        for dataset in args.datasets:
            for eta, ratio in trials:
                name = f"{model}_{dataset}_eta{slug(eta)}_crop{slug(ratio)}"
                output = args.output / name
                command, env = base_job(model, dataset, eta, ratio, args.view_weight,
                                        output, args.smoke, args.batch_size, args.workers)
                jobs.append({"model": model, "dataset": dataset, "eta": eta,
                             "crop_ratio": ratio, "output": str(output),
                             "command": command, "env": {k: env[k] for k in
                             ("QUERY_REFINEMENT_ETA", "QUERY_REFINEMENT_CROP_RATIO",
                              "QUERY_REFINEMENT_VIEW_WEIGHT")}})
    plan = {"status": "planned", "models": args.models, "datasets": args.datasets,
            "trials": [{"eta": eta, "crop_ratio": ratio, "view_weight": args.view_weight}
                       for eta, ratio in trials], "jobs": jobs,
            "note": "Fixed post-development sensitivity slices; no test-driven selection."}
    (args.output / "PLAN.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"jobs": len(jobs), "trials": len(trials), "plan": str(args.output / "PLAN.json")}, indent=2))
    if args.plan_only:
        return
    running = []
    free_gpus = list(args.gpus)
    log_root = args.output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        # Workers create their own output directory with exist_ok=False as an
        # integrity guard.  Keep scheduler logs outside those directories.
        log = log_root / (Path(job["output"]).name + ".log")
        env = os.environ.copy()
        env.update({k: v for k, v in job["env"].items()})
        if not free_gpus:
            raise RuntimeError("Internal scheduler error: no free GPU slot")
        gpu = free_gpus.pop(0)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        handle = log.open("w")
        proc = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        job.update(pid=proc.pid, gpu=env["CUDA_VISIBLE_DEVICES"], log=str(log), status="running")
        running.append((proc, handle, job, gpu))
        while len(running) >= args.max_workers:
            for index, (child, stream, item, gpu) in enumerate(running):
                code = child.poll()
                if code is not None:
                    stream.close(); item.update(status="complete" if code == 0 else "failed", returncode=code)
                    free_gpus.append(gpu)
                    running.pop(index); break
            else:
                time.sleep(5)
    while running:
        for index, (child, stream, item, gpu) in enumerate(running):
            code = child.poll()
            if code is not None:
                stream.close(); item.update(status="complete" if code == 0 else "failed", returncode=code)
                free_gpus.append(gpu)
                running.pop(index); break
        else:
            time.sleep(5)
    plan["status"] = "complete" if all(j.get("status") == "complete" for j in jobs) else "completed_with_failures"
    (args.output / "PLAN.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["AlphaRAD", "RadZero", "BiomedCLIP", "MedKLIP", "CARZero"])
    parser.add_argument("--datasets", nargs="+", default=["chexpert", "chexchonet"])
    parser.add_argument("--etas", type=float, nargs="+", default=[0., .1, .2, .3, .5, .7])
    parser.add_argument("--crop-ratios", type=float, nargs="+", default=[.6, .7, .8, .9])
    parser.add_argument("--view-weight", type=float, default=.5)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    main(parser.parse_args())
