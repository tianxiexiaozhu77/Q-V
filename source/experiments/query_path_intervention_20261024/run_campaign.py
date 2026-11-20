#!/usr/bin/env python3
"""Run fixed five-seed AlphaRAD interventions on checked idle GPUs 4--7."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
sys.path.insert(0, str(MEDICAL / "AlphaRAD/local_setup"))
from run_pretrained import reserve_gpu, release_gpu


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["chestxdet10", "rsna_pneumonia"],
                   choices=["chestxdet10", "rsna_pneumonia"])
    p.add_argument("--gpus", nargs="+", type=int, default=[4,5,6,7])
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--skip-statistics", action="store_true")
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= {4,5,6,7}:
        p.error("Use unique physical GPUs from 4--7")
    if len(set(args.datasets)) != len(args.datasets):
        p.error("Duplicate dataset")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    files = [HERE / n for n in ("run_campaign.py", "infer.py", "intervention.py")]
    source = {str(path): sha(path) for path in files}
    protocol = dict(version="native-query-path-intervention-v1", model="AlphaRAD",
        created_utc=datetime.now(timezone.utc).isoformat(), support_seeds=[42,43,44,45,46],
        datasets=args.datasets, eta=.3, views=1, bootstrap_replicates=2000, bootstrap_seed=20261024,
        source_sha256=source, resampling_units=dict(chestxdet10="image_exact_duplicate_group", rsna_pneumonia="patient"),
        primary_comparison="query - readout_only", secondary_comparisons=["query - aggregation_only", "query - same_feature_score", "each - original", "query - aggregation_only - readout_only + original"],
        branches=dict(original=["original", "original"], readout_only=["original", "refined"],
                      aggregation_only=["refined", "original"], query=["refined", "refined"]),
        branch_pair_order=["attention_query", "readout_query"],
        spatial_definition="Native attention-query scaled cosine map. Readout-only map equals Original; aggregation-only map equals Q by construction.",
        test_cohort_role="Existing development cohorts; NOT independent validation",
        selection="Fixed before intervention predictions; no outcome-based branch/support/eta selection",
        ci_scope="Test group sampling conditional on five fixed support sets; no multiplicity correction")
    write(args.output / "PROTOCOL.json", protocol)
    state = dict(status="running", datasets={}, smoke_only=args.smoke_only, controller_pid=os.getpid())
    write(args.output / "status.json", state)
    envbase = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    def unit(dataset):
        lease, info = None, None
        last_report = 0
        while lease is None:
            for gpu in args.gpus:
                lease, info = reserve_gpu(gpu)
                if lease is not None:
                    break
            if lease is None:
                if time.monotonic()-last_report > 30:
                    print(f"Waiting for an idle GPU: {dataset}", flush=True)
                    last_report = time.monotonic()
                time.sleep(5)
        env = dict(envbase, CUDA_VISIBLE_DEVICES=info["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        stages = [("smoke", args.output / ("smoke_AlphaRAD_"+dataset), 16)]
        if not args.smoke_only:
            stages.append(("full", args.output / ("AlphaRAD_"+dataset), 0))
        receipts = []
        try:
            for stage, output, count in stages:
                for filename, digest in source.items():
                    if sha(filename) != digest:
                        raise ValueError("Inference source changed after protocol freeze: "+filename)
                cmd = ["bash", str(MEDICAL / "AlphaRAD/local_setup/python_local.sh"),
                    str(HERE / "infer.py"), "--dataset", dataset, "--output", str(output),
                    "--gpu", str(info["physical_index"]), "--batch-size", "8", "--smoke", str(count)]
                log = args.output / "logs" / f"{dataset}_{stage}.log"
                print(json.dumps(dict(stage=stage, dataset=dataset, gpu=info["physical_index"], log=str(log))), flush=True)
                with log.open("x") as handle:
                    proc = subprocess.Popen(cmd, env=env, cwd=MEDICAL, stdout=handle,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = proc.wait()
                    except BaseException:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=30)
                        raise
                if code:
                    raise RuntimeError(f"{dataset} {stage} failed ({code}); see {log}")
                record = json.loads((output / "result.json").read_text())
                if record["status"] != ("smoke_complete" if count else "complete"):
                    raise ValueError("Inference receipt incomplete")
                receipts.append(dict(stage=stage, path=str(output / "result.json"), sha256=sha(output / "result.json")))
            return dict(status="complete", gpu=info, stages=receipts)
        finally:
            release_gpu(lease)

    with ThreadPoolExecutor(max_workers=min(len(args.datasets),len(args.gpus))) as pool:
        jobs = {pool.submit(unit, d): d for d in args.datasets}
        for future in as_completed(jobs):
            dataset = jobs[future]
            try:
                state["datasets"][dataset] = future.result()
            except Exception as exc:
                state["datasets"][dataset] = dict(status="failed", error=str(exc))
            write(args.output / "status.json", state)
    if any(r["status"] != "complete" for r in state["datasets"].values()):
        state["status"] = "failed"
        write(args.output / "status.json", state)
        raise SystemExit(1)
    if not args.smoke_only and not args.skip_statistics:
        state["status"] = "statistics"
        write(args.output / "status.json", state)
        code = subprocess.call([sys.executable, str(HERE / "evaluate.py"), "--run", str(args.output),
                                "--datasets", *args.datasets], env=dict(envbase, CUDA_VISIBLE_DEVICES=""), cwd=MEDICAL)
        if code:
            state["status"] = "statistics_failed"
            write(args.output / "status.json", state)
            raise SystemExit(code)
    for filename, digest in source.items():
        if sha(filename) != digest:
            raise ValueError("Inference source changed during campaign: "+filename)
    state["status"] = "smoke_complete" if args.smoke_only else ("inference_complete" if args.skip_statistics else "complete")
    write(args.output / "status.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
