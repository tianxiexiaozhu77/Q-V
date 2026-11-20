#!/usr/bin/env python3
"""Compare frozen AlphaRAD and fixed support-query refinement on idle GPUs 3–7."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from run_pretrained import LOCKS, ROOT, now, release_gpu, reserve_gpu, terminate_group, write_json


DEFAULT_DATASETS = ["chest14", "rsna_pneumonia", "siim_pneumothorax", "chestxdet10",
                    "chestxdet10_gt", "node21", "qatacovid_seg", "shenzhenxray",
                    "montgomery", "covidkaggle"]
METRICS = [("macro_auroc", "AUROC"), ("macro_average_precision", "AP"),
           ("macro_auprc", "AUPRC"), ("pointing_game", "Pointing Game"), ("dice", "Dice")]
SOURCE_FILES = ["local_setup/run_query_refinement.py", "local_setup/eval_query_refinement.py",
                "local_setup/prepare_query_refinement_data.py",
                "local_setup/query_refinement.py", "local_setup/run_pretrained.py",
                "local_setup/eval_pretrained.py", "local_setup/activate.sh",
                "local_setup/python_local.sh", "scripts/eval_query_refinement.sh",
                "cxrclip/base_evaluator.py", "cxrclip/evaluator.py", "cxrclip/evaluator_utils.py",
                "cxrclip/model/baseclip.py", "cxrclip/model/cxrclip_plus_dqn.py",
                "cxrclip/data/data_utils.py", "cxrclip/data/datasets/image_classification.py"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_sources(output):
    records = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records[relative] = sha256(source)
        if sha256(target) != records[relative]:
            raise ValueError(f"Source changed while taking snapshot: {source}")
    write_json(output / "source_sha256.json", records)
    return records


def verify_sources(records):
    for relative, expected in records.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Source changed during frozen campaign: {relative}")


def _comparison(data, key):
    baseline, refined = data.get("baseline", {}).get(key), data.get("refined", {}).get(key)
    if baseline is None or refined is None:
        return "—"
    return f"{100 * baseline:.2f} → {100 * refined:.2f} ({100 * (refined - baseline):+.2f})"


def report(output, state, catalog):
    lines = ["AlphaRAD 原模型与支持集查询修正：同一批隔离评估图像上的直接对照。",
             f"状态：{state['status']}；更新（UTC）：{now()}",
             f"eta={state['eta']}；模型全冻结，无梯度更新；只使用检查空闲的物理 GPU 3–7。",
             "eta 默认 0.3 来自既有 RadZero 探索结果，本轮不按 AlphaRAD 测试分数选参数。",
             "各指标：原模型% → 修正后%（差值/百分点）。",
             "dataset | status | support | evaluation | AUROC | AP | AUPRC | Pointing Game | Dice",
             "--- | --- | --- | --- | --- | --- | --- | --- | ---"]
    if state["limit_test"]:
        lines.insert(0, "SMOKE TEST ONLY：这是有限样本检查，不是完整评估结果。")
    fields = ["dataset", "task", "status", "support_count", "evaluation_count", "eta",
              "metric", "baseline_percent", "refined_percent", "delta_pp", "is_smoke", "reason"]
    records = []
    for name, job in state["jobs"].items():
        entry = catalog["datasets"][name]
        support_count, count = entry["support_count"], entry["actual_count"]
        if state["limit_test"]:
            count = min(count, state["limit_test"])
        common = {"dataset": name, "task": entry["task"], "status": job["status"],
                  "support_count": support_count, "evaluation_count": count,
                  "eta": state["eta"], "is_smoke": bool(state["limit_test"]),
                  "reason": job.get("reason", "")}
        if job["status"] == "complete":
            data = json.loads((output / name / "result.json").read_text())
            common.update(support_count=data["support_count"], evaluation_count=data["rows"])
            values = [_comparison(data, key) for key, _ in METRICS]
            lines.append(f"{name} | complete | {data['support_count']} | {data['rows']} | "
                         + " | ".join(values))
            for key, label in METRICS:
                baseline, refined = data["baseline"].get(key), data["refined"].get(key)
                if baseline is not None and refined is not None:
                    records.append({**common, "metric": label,
                        "baseline_percent": 100 * baseline, "refined_percent": 100 * refined,
                        "delta_pp": 100 * (refined - baseline)})
        else:
            lines.append(f"{name} | {job['status']} | {support_count} | {count} | — | — | — | — | —")
            records.append(common)
    lines.extend(["", "AP 为 Average Precision；AUPRC 使用梯形积分。",
                  "支持图像/患者从本轮评估中排除；原模型也在同一隔离评估集重算，不能直接与此前完整集数字相减。",
                  "QaTaCOV19 沿用作者阳性样本协议，两分支各自在评估标签上搜索最佳 Dice 阈值；并非独立验证集阈值。",
                  "病例分组、数据来源和协议限制见 catalog.json / 各项 data_audit.json；不保证逐数据集提升。",
                  "完整指标表：COMPARISON.csv；模型与原作者输出等价性见各项 zero_eta_check.json。"])
    temporary = output / "RESULTS.txt.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output / "RESULTS.txt")
    temporary = output / "COMPARISON.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "COMPARISON.csv")
    write_json(output / "status.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "local_setup/query_refinement_data_20261021/catalog.json")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--gpus", type=int, nargs="+", choices=range(3, 8), default=list(range(3, 8)))
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are preserved")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "runs/pretrained_20261021_122700")
    parser.add_argument("--eta", type=float, default=.3)
    parser.add_argument("--method", choices=("closed_form", "reaggregation", "cached_multistep", "reaggregation_only", "uncertainty", "jackknife", "tangent_jackknife", "scqt", "ecqa", "ecqt", "sepcal", "tdqa", "tdqa_reagg", "support_select", "cgt", "cct", "scct", "scct2", "scct3", "sgtt", "sgttm", "smgt", "cvsgtt"),
                        default="closed_form")
    parser.add_argument("--reaggregation-max-iter", type=int, default=3)
    parser.add_argument("--reaggregation-query-tol", type=float, default=1e-4)
    parser.add_argument("--reaggregation-patience", type=int, default=1)
    parser.add_argument("--covariance-shrinkage", type=float, default=.2)
    parser.add_argument("--covariance-ridge", type=float, default=1e-3)
    parser.add_argument("--reliability-power", type=float, default=1.)
    parser.add_argument("--jackknife-stability-power", type=float, default=1.)
    parser.add_argument("--jackknife-kappa", type=float, default=8.)
    parser.add_argument("--scqt-geometry", choices=("linear", "tangent"), default="tangent")
    parser.add_argument("--evidence-temperature", type=float, default=1.)
    parser.add_argument("--evidence-weight-floor", type=float, default=.25)
    parser.add_argument("--evidence-source", choices=("grounding", "logit", "mixed"), default="grounding")
    parser.add_argument("--separation-temperature", type=float, default=1.)
    parser.add_argument("--reliability-floor", type=float, default=.25)
    parser.add_argument("--token-weight", type=float, default=.2)
    parser.add_argument("--eta-candidates", type=float, nargs="+", default=(0., .1, .2, .3))
    parser.add_argument("--tdqa-temperature", type=float, default=1.)
    parser.add_argument("--tdqa-reliability-power", type=float, default=1.)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-test", type=int, default=0, help="Smoke test only")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.eta) or args.eta <= 0 or args.batch_size < 1 or args.workers < 0
            or args.limit_test < 0 or args.reaggregation_max_iter < 1
            or args.reaggregation_max_iter > 32 or not math.isfinite(args.reaggregation_query_tol)
            or args.reaggregation_query_tol < 0 or args.reaggregation_patience < 1
            or args.reaggregation_patience > args.reaggregation_max_iter
            or not math.isfinite(args.covariance_shrinkage) or not 0 <= args.covariance_shrinkage <= 1
            or not math.isfinite(args.covariance_ridge) or args.covariance_ridge < 0
            or not math.isfinite(args.reliability_power) or args.reliability_power < 0
            or not math.isfinite(args.jackknife_stability_power) or args.jackknife_stability_power < 0
            or not math.isfinite(args.jackknife_kappa) or args.jackknife_kappa < 0
            or not math.isfinite(args.evidence_temperature) or args.evidence_temperature <= 0
            or not math.isfinite(args.evidence_weight_floor) or not 0 < args.evidence_weight_floor <= 1
            or not args.eta_candidates or any(not math.isfinite(x) or x < 0 for x in args.eta_candidates)
            or len(set(args.eta_candidates)) != len(args.eta_candidates) or 0.0 not in args.eta_candidates
            or not math.isfinite(args.tdqa_temperature) or args.tdqa_temperature <= 0
            or not math.isfinite(args.tdqa_reliability_power) or args.tdqa_reliability_power < 0
            or not math.isfinite(args.separation_temperature) or args.separation_temperature <= 0
            or not math.isfinite(args.reliability_floor) or not 0 < args.reliability_floor <= 1
            or not math.isfinite(args.token_weight) or args.token_weight < 0):
        parser.error("Invalid eta, batch size, workers, or smoke limit")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.gpus)) != len(args.gpus):
        parser.error("Duplicate datasets or GPUs are not allowed")
    args.catalog, args.output, args.baseline_root = (path.expanduser().resolve() for path in
                                                    (args.catalog, args.output, args.baseline_root))
    catalog = json.loads(args.catalog.read_text())
    if catalog.get("status") != "complete":
        raise ValueError("Support/evaluation catalog must be complete before this campaign")
    for name in args.datasets:
        if name not in catalog["datasets"]:
            raise ValueError(f"Unknown dataset: {name}")
    catalog = {**catalog, "datasets": {name: catalog["datasets"][name] for name in args.datasets}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "catalog.json", catalog)
    source_records = freeze_sources(args.output)
    LOCKS.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_key = hashlib.sha256(str(args.output).encode()).hexdigest()
    controller_lock = (LOCKS / f"query_controller_{run_key}.lock").open("a+")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "running", "started_at": now(), "controller_pid": os.getpid(),
             "controller_hostname": socket.gethostname(), "gpus": args.gpus,
             "catalog": str(args.output / "catalog.json"), "input_catalog": str(args.catalog),
             "eta": args.eta, "method": args.method,
             "reaggregation_max_iter": args.reaggregation_max_iter,
             "reaggregation_query_tol": args.reaggregation_query_tol,
             "reaggregation_patience": args.reaggregation_patience,
             "covariance_shrinkage": args.covariance_shrinkage,
             "covariance_ridge": args.covariance_ridge,
             "reliability_power": args.reliability_power, "token_weight": args.token_weight,
             "batch_size": args.batch_size, "workers": args.workers,
             "limit_test": args.limit_test, "baseline_root": str(args.baseline_root),
             "jobs": {name: {"status": "pending"} for name in args.datasets}}
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"Results: {args.output / 'RESULTS.txt'}", flush=True)
    report(args.output, state, catalog)
    try:
        while not stopping:
            for gpu, (proc, name, lease, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                path = args.output / name / "result.json"
                result = json.loads(path.read_text()) if code == 0 and path.exists() else {}
                complete = result.get("status") == "complete" and result.get("dataset") == name
                job = state["jobs"][name]
                job.update(status="complete" if complete else "failed", exit_code=code, finished_at=now())
                if not complete:
                    failure = args.output / name / "failure.json"
                    job["reason"] = (json.loads(failure.read_text()).get("error", "worker failed")
                                     if failure.exists() else f"Exit {code}; see run.log")
                release_gpu(lease)
                log.close()
                del active[gpu]
                print(f"{name}: {job['status']}", flush=True)
            ready = [name for name, job in state["jobs"].items() if job["status"] == "pending"]
            for gpu in args.gpus:
                if stopping or gpu in active or not ready:
                    continue
                lease, info = reserve_gpu(gpu)
                state.setdefault("last_gpu_checks", {})[str(gpu)] = info
                if lease is None:
                    continue
                name = ready.pop(0)
                job_dir = args.output / name
                log = None
                try:
                    verify_sources(source_records)
                    job_dir.mkdir(exist_ok=False)
                    write_json(job_dir / "catalog.json", {**catalog, "datasets": {name: catalog["datasets"][name]}})
                    command = ["bash", str(ROOT / "local_setup/python_local.sh"),
                        str(ROOT / "local_setup/eval_query_refinement.py"),
                        "--catalog", str(job_dir / "catalog.json"), "--dataset", name,
                        "--output", str(job_dir), "--eta", str(args.eta),
                        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                        "--baseline-root", str(args.baseline_root), "--limit-test", str(args.limit_test),
                        "--method", args.method,
                        "--reaggregation-max-iter", str(args.reaggregation_max_iter),
                        "--reaggregation-query-tol", str(args.reaggregation_query_tol),
                        "--reaggregation-patience", str(args.reaggregation_patience),
                        "--covariance-shrinkage", str(args.covariance_shrinkage),
                        "--covariance-ridge", str(args.covariance_ridge),
                        "--reliability-power", str(args.reliability_power)]
                    command += ["--jackknife-stability-power", str(args.jackknife_stability_power),
                                "--jackknife-kappa", str(args.jackknife_kappa),
                                "--scqt-geometry", args.scqt_geometry,
                                "--evidence-temperature", str(args.evidence_temperature),
                                "--evidence-weight-floor", str(args.evidence_weight_floor),
                                "--evidence-source", args.evidence_source,
                                "--separation-temperature", str(args.separation_temperature),
                                "--reliability-floor", str(args.reliability_floor),
                                "--token-weight", str(args.token_weight),
                                "--eta-candidates", *[str(value) for value in args.eta_candidates],
                                "--tdqa-temperature", str(args.tdqa_temperature),
                                "--tdqa-reliability-power", str(args.tdqa_reliability_power)]
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": info["uuid"],
                           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1",
                           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
                    log = (job_dir / "run.log").open("w")
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                except BaseException:
                    if log is not None:
                        log.close()
                    release_gpu(lease)
                    raise
                active[gpu] = (proc, name, lease, log)
                state["jobs"][name] = {"status": "running", "started_at": now(), "pid": proc.pid,
                                       "gpu": info, "command": command}
                print(f"Started {name} on physical GPU {gpu}, PID {proc.pid}", flush=True)
                report(args.output, state, catalog)
            if not active and all(job["status"] != "pending" for job in state["jobs"].values()):
                state["status"] = ("complete" if all(job["status"] == "complete" for job in state["jobs"].values())
                                   else "finished_with_failures")
                break
            report(args.output, state, catalog)
            time.sleep(10)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        for proc, name, lease, log in active.values():
            if proc.poll() is None:
                terminate_group(proc, signal.SIGTERM)
        for proc, name, lease, log in active.values():
            try:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate_group(proc, signal.SIGKILL)
                    proc.wait()
            finally:
                release_gpu(lease)
                log.close()
            state["jobs"][name].update(status="interrupted", finished_at=now())
        if stopping:
            state["status"] = "interrupted"
        state["finished_at"] = now()
        try:
            report(args.output, state, catalog)
        finally:
            release_gpu(controller_lock)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
