#!/usr/bin/env python3
"""Explicit launcher for the frozen 20-unit supplement; no background auto-start."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from experiment import (HERE, ROOT, expected_units, now, read_json, sha256,
                        verify_fit_commit, verify_sources, write_json)


def acquire_gpu(index):
    """Lease a specifically requested idle GPU; never terminate existing jobs."""
    root = Path(tempfile.gettempdir()) / f"medical-gpu-locks-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    gpu_uuid = subprocess.check_output(["nvidia-smi","-i",str(index),"--query-gpu=uuid",
                                        "--format=csv,noheader"],text=True,timeout=15).strip()
    lease = (root/f"{gpu_uuid}.lock").open("a+")
    try:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None, {"index":index,"reason":"leased"}
    try:
        raw = subprocess.check_output(["nvidia-smi","-i",str(index),"--query-gpu=uuid,memory.used",
                                       "--format=csv,noheader,nounits"],text=True,timeout=15).strip()
        seen, memory = [part.strip() for part in raw.split(",")]
        apps = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader,nounits"],text=True,timeout=15)
        busy = [line for line in apps.splitlines() if line.split(",")[0].strip()==gpu_uuid]
        info = dict(index=index,uuid=gpu_uuid,memory_mib=int(memory),processes=busy,hostname=socket.gethostname())
        if seen!=gpu_uuid or busy or int(memory)>512:
            release_gpu(lease)
            return None, {**info,"reason":"gpu_in_use"}
        lease.seek(0);lease.truncate()
        lease.write(str(os.getpid())+"\n");lease.flush()
        return lease,info
    except BaseException:
        release_gpu(lease)
        raise


def release_gpu(lease):
    fcntl.flock(lease,fcntl.LOCK_UN)
    lease.close()


def worker_environment(gpu=None):
    return {**os.environ,"CUDA_VISIBLE_DEVICES":"" if gpu is None else gpu["uuid"],
            "CUDA_DEVICE_ORDER":"PCI_BUS_ID","PYTHONUNBUFFERED":"1","PYTHONNOUSERSITE":"1",
            "PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
            "OPENBLAS_NUM_THREADS":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8"}


def check_environment(ready):
    for name, version in ready["environment"]["packages"].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError("Runtime dependency changed since readiness check: "+name)


def signal_owned_process(process, signum):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass  # The owned process exited between poll and killpg.


def write_barrier(run,snapshot):
    """Unlock evaluator only after every frozen-scope prediction is committed."""
    snapshot_sha = sha256(run/"experiment_snapshot.json")
    units={}
    for unit_id in expected_units(snapshot["config"]):
        commit = verify_fit_commit(run/unit_id,snapshot,snapshot_sha,snapshot["unit_manifests"][unit_id])
        if commit["unit_id"] != unit_id:
            raise ValueError("Fit commit belongs to another unit")
        units[unit_id] = {"fit_commit_sha256":sha256(run/unit_id/"fit_commit.json")}
    barrier_path = run/"predictions_complete.json"
    stable = dict(status="all_predictions_committed",units=units,
                  run_snapshot_sha256=snapshot_sha,test_labels_opened=False)
    if barrier_path.exists():
        previous = read_json(barrier_path)
        if any(previous.get(key) != value for key,value in stable.items()):
            raise ValueError("Existing prediction barrier differs from committed fits")
        return  # Preserve its digest so completed evaluation remains resumable.
    write_json(barrier_path,{**stable,"completed_at":now()})


def launch(args):
    ready = read_json(args.ready)
    if ready["status"]!="ready_not_run" or ready["formal_fits_started"]:
        raise ValueError("Expected prepared, unstarted experiment configuration")
    verify_sources(ready["source_sha256"])
    check_environment(ready)
    run=args.output.resolve()
    if args.resume:
        snapshot=read_json(run/"experiment_snapshot.json")
        if snapshot["source_sha256"]!=ready["source_sha256"] or snapshot["config"]!=ready["config"]:
            raise ValueError("Resume requires unchanged experiment configuration and code")
        if snapshot["execution"]["device"]!=args.device:
            raise ValueError("Resume cannot mix CPU and GPU fitting")
    else:
        run.mkdir(parents=True,exist_ok=False)
        snapshot={**ready,"status":"execution_snapshot","formal_fits_started":True,
                  "started_at":now(),"execution":dict(device=args.device,gpus=args.gpus,workers=args.workers)}
        write_json(run/"experiment_snapshot.json",snapshot)
    (run/"logs").mkdir(exist_ok=True)
    controller=(run/"controller.lock").open("a+")
    fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    units=expected_units(snapshot["config"])
    snapshot_sha=sha256(run/"experiment_snapshot.json")
    state=dict(status="fitting",started_at=now(),controller_pid=os.getpid(),jobs={})
    pending=[]
    for unit in units:
        if (run/unit/"fit_commit.json").exists():
            verify_fit_commit(run/unit,snapshot,snapshot_sha,snapshot["unit_manifests"][unit])
            state["jobs"][unit]={"status":"complete_reused"}
        else:
            pending.append(unit)
            state["jobs"][unit]={"status":"pending"}
    active={}
    stopping=False

    def stop(signum,frame):
        nonlocal stopping
        stopping=True

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def save():
        state["updated_at"]=now()
        write_json(run/"status.json",state)

    def start(unit,lease=None,gpu=None):
        spec=snapshot["unit_manifests"][unit]
        command=[sys.executable,str(HERE/"fit_unit.py"),"--run",str(run),
                 "--model",spec["model"],"--dataset",spec["dataset"],"--seed",str(spec["seed"]),
                 "--device",args.device]
        handle=(run/"logs"/(unit.replace("/","_")+".log")).open("a")
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=worker_environment(gpu),stdout=handle,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            handle.close()
            if lease is not None:release_gpu(lease)
            raise
        active[unit]=dict(process=process,log=handle,lease=lease,gpu=gpu)
        state["jobs"][unit]=dict(status="running",pid=process.pid,device=args.device,gpu=gpu,started_at=now())
        print("Started",unit,args.device, "GPU "+str(gpu["index"]) if gpu else "",flush=True)

    try:
        while pending or active:
            if stopping:raise InterruptedError("Controller interrupted")
            for unit,item in list(active.items()):
                code=item["process"].poll()
                if code is None:continue
                item["log"].close()
                if item["lease"] is not None:release_gpu(item["lease"])
                del active[unit]
                if code:
                    state["jobs"][unit].update(status="failed",returncode=code)
                    raise RuntimeError("Worker failed; inspect logs: "+unit)
                verify_fit_commit(run/unit,snapshot,snapshot_sha,snapshot["unit_manifests"][unit])
                state["jobs"][unit].update(status="complete",finished_at=now())
                print("Completed",unit,flush=True)
            if args.device=="cpu":
                while pending and len(active)<args.workers:start(pending.pop(0))
            else:
                used={item["gpu"]["index"] for item in active.values()}
                for index in args.gpus:
                    if not pending or len(active)>=args.workers:break
                    if index in used:continue
                    lease,gpu=acquire_gpu(index)
                    if lease is None:
                        state.setdefault("busy_gpus",{})[str(index)]=gpu
                    else:start(pending.pop(0),lease,gpu)
                if pending and not active:
                    raise RuntimeError("Requested GPUs are occupied; no existing process was stopped. Free cards, then resume.")
            save()
            if active:time.sleep(1)
        verify_sources(snapshot["source_sha256"])
        write_barrier(run,snapshot)
        state["status"]="predictions_committed"
        save()
        # All GPU leases have been released. Bootstrap is CPU-only.
        if not args.fit_only:
            state["status"]="scoring"
            save()
            command=[sys.executable,str(HERE/"evaluate.py"),"--run",str(run)]
            with (run/"logs/evaluation.log").open("a") as handle:
                process=subprocess.Popen(command,cwd=ROOT,env=worker_environment(),stdout=handle,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                active["evaluation"]=dict(process=process,log=handle,lease=None,gpu=None)
                while process.poll() is None:
                    if stopping:raise InterruptedError("Controller interrupted during scoring")
                    time.sleep(1)
                del active["evaluation"]
                if process.returncode:raise RuntimeError("Evaluation failed; inspect logs/evaluation.log")
            state["status"]="complete"
        state["finished_at"]=now()
        save()
        print("Campaign",state["status"],str(run),flush=True)
    except BaseException as exc:
        # Only terminate child processes owned by this controller, never other
        # users' jobs or processes returned by nvidia-smi.
        for item in active.values():
            signal_owned_process(item["process"],signal.SIGTERM)
        for unit,item in active.items():
            try:item["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_owned_process(item["process"],signal.SIGKILL)
                item["process"].wait()
            item["log"].close()
            if item["lease"] is not None:release_gpu(item["lease"])
            state["jobs"].setdefault(unit,{})["status"]="interrupted"
        state.update(status="interrupted" if stopping else "failed",error=repr(exc))
        save()
        raise
    finally:
        fcntl.flock(controller,fcntl.LOCK_UN)
        controller.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ready",type=Path,default=HERE/"READY.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--gpus",type=int,nargs="+",default=[0,1,2,3])
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--fit-only",action="store_true")
    args=p.parse_args()
    if args.workers<1 or len(set(args.gpus))!=len(args.gpus) or any(g<0 for g in args.gpus):
        p.error("Require positive workers and distinct nonnegative GPU IDs")
    launch(args)


if __name__=="__main__":
    main()
