#!/usr/bin/env python3
"""Evaluate the eight verified pretrained baselines on idle GPUs 4–7."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "AlphaRAD/local_setup"))
from run_pretrained import reserve_gpu, release_gpu
from evaluate_baseline import sha, write

MODELS = ["BiomedCLIP", "BioViL-T", "MedKLIP", "MAVL", "KAD", "MaCo", "CARZero", "CheXzero"]
MODEL_ROOTS = {name: ROOT / ("protocols/alpharad_binary/" + name if name in ("BiomedCLIP", "BioViL-T") else name) for name in MODELS}
STOP = None


def now():
    return datetime.now(timezone.utc).isoformat()


def handle_stop(signum, _frame):
    global STOP
    STOP = signum


def report(output, catalog, state):
    records = []
    for key, entry in catalog["datasets"].items():
        for reference in entry["comparison_references"]:
            path = Path(reference["result"])
            if sha(path) != reference["sha256"]:
                raise ValueError("Prior comparison result changed: " + str(path))
            old = json.loads(path.read_text())
            branches = old.get("branches") or {name: value["metrics"] for name, value in old["candidates"].items()}
            for branch, label in (("original", "原始"), ("tta_center_refined", "查询修正＋双视图")):
                records.append({"dataset": key, "task": entry["task"], "rows": entry["count"],
                    "model": reference["model"] + " " + label, "status": "reference", "metrics": branches[branch],
                    "result": str(path), "support_usage": "few-shot" if branch != "original" else "none"})
        for name, job in state["models"].items():
            folder = job.get("full_output")
            path = Path(folder) / key / "result.json" if folder else None
            if path and path.is_file():
                result = json.loads(path.read_text())
                records.append({"dataset": key, "task": entry["task"], "rows": result.get("rows", entry["count"]),
                    "model": name, "status": result["status"], "metrics": result.get("metrics", {}),
                    "reason": result.get("reason", result.get("error")), "result": str(path), "support_usage": "none"})
            else:
                records.append({"dataset": key, "task": entry["task"], "rows": entry["count"],
                    "model": name, "status": job["status"], "metrics": {}, "support_usage": "none"})
    write(output / "RESULTS.json", {"updated_at": now(), "status": state["status"], "records": records,
        "scope": catalog["scope"], "support_policy": "Pretrained baselines never see support images or labels. References use their previously fixed support queries.",
        "comparison_limits": "Identical held-out image IDs, labels and class counts; models retain native image processing. Some models have fixed vocabulary; unsupported full tasks are not scored on partial class subsets.",
        "dense_protocol": "Native published spatial maps only; original-coordinate PG and exploratory positive-only test-threshold Dice. Native model preprocessing/geometry is recorded separately."})
    lines = ["AlphaRAD/RadZero 固定测试集：公开预训练baseline复测", "",
        f"状态：{state['status']}；更新时间UTC：{now()}",
        "11个数据集、13个评估范围（包括SIIM两种测试范围）。测试样本和标签固定，不重新抽支持集。",
        "baseline仅加载公开已训练checkpoint，不训练，不使用支持集；参考改进方法使用既有带标签支持集。",
        "AUROC/AUPRC/PG/Dice单位为%。AUPRC为PR曲线梯形面积，AP另存JSON。",
        "unsupported表示模型无可核实的对应类别/任务实现，未以少类别宏平均冒充完整结果。",
        "各模型保留原生预处理；KAD用疾病名，知识查询模型保留作者词典。",
        "BiomedCLIP/BioViL-T分类为逐疾病正负二元softmax；其他模型保留各自原生评分。",
        "RSNA/SIIM匹配身份及标签，DICOM/PNG像素解码并非完全相同；Dice归一化顺序及浮点阈值差异见独立审计。", "",
        "| 数据集/范围 | 模型/方法 | 状态 | 图片数 | AUROC | AUPRC | PG | Dice |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        values = [r["metrics"].get(k) for k in ("macro_auroc", "macro_auprc", "pointing_game", "dice")]
        fields = ["—" if v is None else f"{100*v:.2f}" for v in values]
        lines.append(f"| {r['dataset']} | {r['model']} | {r['status']} | {r['rows']} | " + " | ".join(fields) + " |")
    reasons = [r for r in records if r.get("reason")]
    if reasons:
        lines += ["", "未完成/不支持项原因："]
        lines += [f"- {r['model']}/{r['dataset']}: {r['reason']}" for r in reasons]
    lines += ["", "说明：表中的既有查询修正结果曾用于探索性方法筛选，不能视为全新独立最终测试。",
        "QaTa分割保留positive-only及测试阈值搜索设置；热图实现、插值和无覆盖区域见各adapter metadata。",
        "KAD空间评估将作者512可视化脚本的最后层attention迁移至224模型；论文多层平均及作者示例的查询集合不同，详见KAD_GROUNDING_PROTOCOL_AUDIT.txt。",
        "MaCo热图存在并列峰值：统一PG取首个argmax，作者PG接受任一并列最大像素；详见MACO_GROUNDING_PROTOCOL_AUDIT.txt。",
        "CheXpert骨折仅1个阳性测试患者；各任务的患者隔离信息和样本局限保留在固定manifest中。"]
    (output / "RESULTS.txt").write_text("\n".join(lines) + "\n")


def ready(model):
    root = MODEL_ROOTS[model]
    path = root / "local_setup/READY.json"
    if not path.is_file(): return None
    value = json.loads(path.read_text())
    if value.get("ready") is not True: return None
    if value.get("adapter_sha256") != sha(root / "local_setup/adapter.py"):
        raise ValueError("READY adapter hash differs: " + model)
    if value.get("manifest_sha256") != sha(root / "local_setup/manifest.json"):
        raise ValueError("READY manifest hash differs: " + model)
    if value.get("launcher_sha256") and value["launcher_sha256"] != sha(root / "local_setup/python_local.sh"):
        raise ValueError("READY launcher hash differs: " + model)
    return {"path": str(path), "sha256": sha(path), "value": value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "evaluation/matched_phase6_20261022/catalog.json")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--gpus", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(i not in (4, 5, 6, 7) for i in args.gpus):
        parser.error("Only physical GPUs 4–7 are authorized")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    catalog = json.loads(args.catalog.read_text())
    state = {"status": "running", "started_at": now(), "controller_pid": os.getpid(),
        "gpus": args.gpus, "protocol": "fixed verified native + binary presence/absence for BiomedCLIP/BioViL-T", "model_roots": {k: str(v) for k,v in MODEL_ROOTS.items()}, "catalog": str(args.catalog), "catalog_sha256": sha(args.catalog),
        "models": {name: {"status": "waiting_for_verified_assets", "attempts": []} for name in args.models}}
    write(args.output / "status.json", state)
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    active = {}

    def launch(name, stage, gpu, lease, info):
        job = state["models"][name]
        attempt = len(job["attempts"])
        output = args.output / f"{name}_{stage}_{attempt}"
        command = ["bash", str(MODEL_ROOTS[name] / "local_setup/python_local.sh"),
            str(ROOT / "local_setup/evaluate_baseline_buffered.py"), "--model-root", str(MODEL_ROOTS[name]),
            "--catalog", str(args.catalog), "--output", str(output), "--batch-size", str(args.batch_size)]
        if stage == "smoke": command += ["--smoke", "2"]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=info["uuid"], PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        logfile = args.output / "logs" / f"{name}_{stage}_{attempt}.log"
        log = logfile.open("w")
        try:
            proc = subprocess.Popen(command, cwd=MODEL_ROOTS[name], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            if gpu not in active:
                release_gpu(lease)
            raise
        active[gpu] = {"name": name, "stage": stage, "proc": proc, "lease": lease, "info": info, "log": log}
        attempt_info = {"stage": stage, "started_at": now(), "pid": proc.pid, "output": str(output), "command": command, "gpu": info, "log": str(logfile)}
        job["attempts"].append(attempt_info)
        job.update(status="running_" + stage, **{stage + "_output": str(output)})
        print(f"START {name}/{stage} GPU{gpu} PID={proc.pid}", flush=True)

    try:
        while True:
            if STOP is not None: raise InterruptedError(f"Signal {STOP}")
            for gpu, item in list(active.items()):
                proc = item["proc"]
                code = proc.poll()
                if code is None: continue
                name = item["name"]
                job = state["models"][name]
                job["attempts"][-1].update(returncode=code, finished_at=now())
                item["log"].close()
                if code == 0 and item["stage"] == "smoke":
                    launch(name, "full", gpu, item["lease"], item["info"])
                else:
                    release_gpu(item["lease"])
                    del active[gpu]
                    job["status"] = "complete" if code == 0 else "waiting_for_fix"
                    print(f"FINISH {name}/{item['stage']} returncode={code}", flush=True)
            for name, job in state["models"].items():
                if job["status"] not in ("waiting_for_verified_assets", "waiting_for_fix"): continue
                try:
                    receipt = ready(name)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    job["readiness_error"] = str(error)
                    continue
                if not receipt: continue
                if job["status"] == "waiting_for_fix" and receipt["sha256"] == job.get("ready_receipt", {}).get("sha256"):
                    continue
                for gpu in args.gpus:
                    if gpu in active: continue
                    lease, info = reserve_gpu(gpu)
                    job["last_gpu_check"] = info
                    if lease is None: continue
                    job["ready_receipt"] = receipt
                    job.pop("readiness_error", None)
                    launch(name, "smoke", gpu, lease, info)
                    break
            done = all(job["status"] in ("complete", "skipped") for job in state["models"].values())
            state["updated_at"] = now()
            if done: state.update(status="complete", finished_at=now())
            write(args.output / "status.json", state)
            report(args.output, catalog, state)
            if done: break
            time.sleep(10)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
            error=repr(error), traceback=traceback.format_exc(), finished_at=now())
        # Only Popen-owned child groups started by this controller are handled.
        for item in active.values():
            if item["proc"].poll() is None:
                try: os.killpg(item["proc"].pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for item in active.values():
            try: item["proc"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(item["proc"].pid, signal.SIGKILL)
                item["proc"].wait()
            release_gpu(item["lease"])
            item["log"].close()
        write(args.output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
