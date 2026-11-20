#!/usr/bin/env python3
"""Run the preregistered same-source SIIM holdout after preparation.

This experiment does not establish patient identity or external-hospital
independence; its provenance and exclusions are fixed in the supplied catalog.
"""
from __future__ import annotations
import argparse
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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=HERE / "protocol_siim_v1/catalog.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs="+", type=int, default=[4,5,6,7])
    p.add_argument("--skip-statistics", action="store_true")
    args = p.parse_args()
    if not args.gpus or not set(args.gpus) <= {4,5,6,7} or len(set(args.gpus)) != len(args.gpus):
        p.error("Distinct physical GPUs from 4--7 required")
    args.output, args.protocol = args.output.resolve(), args.protocol.resolve()
    files = [args.protocol, MEDICAL / "experiments/fewshot_native_query/methods.py",
        MEDICAL / "AlphaRAD/local_setup/query_refinement.py",
        MEDICAL / "AlphaRAD/cxrclip/model/cxrclip_plus_dqn.py"] + [HERE / name for name in
        ("infer_siim.py", "intervention.py", "fit_s_siim.py", "run_siim.py")]
    sources = {str(path): sha(path) for path in files}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    write(args.output / "FROZEN_RUN_PLAN.json", dict(
        dataset="siim_new_holdout", source_sha256=sources, support_seeds=[42,43,44,45,46], eta=.3,
        views=1, primary_contrast="query - readout_only", secondary_contrasts=["query - aggregation_only", "query - same_feature_score", "each - original", "query - aggregation_only - readout_only + original"],
        classification_metrics=["AUROC", "trapezoidal_AUPRC"],
        spatial_metrics=["positive_image_pixel_AUROC", "positive_image_mask_PG"],
        bootstrap_replicates=2000, bootstrap_seed=20261024,
        resampling_unit="DICOM identifier group; NOT verified patient", masks_used_for_fitting=False,
        threshold_selection="None; spatial rank and peak metrics only", external_validation=False))
    state = dict(status="waiting_for_gpu", controller_pid=os.getpid(), stages=[])
    write(args.output / "status.json", state)
    lease = None
    envbase = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    def stage(name, cmd, env):
        for filename, digest in sources.items():
            if sha(filename) != digest:
                raise ValueError("Frozen inference/fit source changed: "+filename)
        state["status"] = name
        write(args.output / "status.json", state)
        log = args.output / "logs" / (name+".log")
        print(json.dumps(dict(stage=name, log=str(log))), flush=True)
        with log.open("x") as handle:
            proc = subprocess.Popen(cmd, env=env, cwd=MEDICAL, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait()
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
                raise
        state["stages"].append(dict(stage=name, exit_code=code, log=str(log)))
        if code:
            raise RuntimeError(f"{name} failed ({code}); see {log}")
        write(args.output / "status.json", state)

    try:
        while lease is None:
            for gpu in args.gpus:
                lease, info = reserve_gpu(gpu)
                if lease is not None:
                    break
            if lease is None:
                time.sleep(5)
        state["gpu"] = info
        env = dict(envbase, CUDA_VISIBLE_DEVICES=info["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        for name, out, count in (("smoke", args.output / "smoke_AlphaRAD_siim_new_holdout", 16),
                                 ("full", args.output / "AlphaRAD_siim_new_holdout", 0)):
            stage(name, ["bash", str(MEDICAL / "AlphaRAD/local_setup/python_local.sh"),
                str(HERE / "infer_siim.py"), "--protocol", str(args.protocol), "--output", str(out),
                "--gpu", str(info["physical_index"]), "--batch-size", "8", "--smoke", str(count)], env)
            record = json.loads((out / "result.json").read_text())
            if record["status"] != ("smoke_complete" if count else "complete"):
                raise ValueError("Incomplete inference receipt")
        release_gpu(lease)
        lease = None
        cpuenv = dict(envbase, CUDA_VISIBLE_DEVICES="")
        stage("fit_s", [sys.executable, str(HERE / "fit_s_siim.py"), "--protocol", str(args.protocol),
                       "--run", str(args.output / "AlphaRAD_siim_new_holdout")], cpuenv)
        if not args.skip_statistics:
            stage("evaluation", [sys.executable, str(HERE / "evaluate_siim.py"), "--protocol", str(args.protocol),
                                 "--run", str(args.output)], cpuenv)
        state["status"] = "predictions_complete" if args.skip_statistics else "complete"
        write(args.output / "status.json", state)
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        write(args.output / "status.json", state)
        raise
    finally:
        if lease is not None:
            release_gpu(lease)


if __name__ == "__main__":
    main()
