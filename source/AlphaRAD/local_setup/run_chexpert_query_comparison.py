#!/usr/bin/env python3
"""Run the released CheXpert baseline and fixed phase6 comparison on GPUs 6–7."""
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

from eval_pretrained import ASSETS, sha256, write_json
from run_pretrained import reserve_gpu, release_gpu

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("original", "uniform_eta03", "tta_center_original", "tta_center_refined")
STOP_SIGNAL = None


def now():
    return datetime.now(timezone.utc).isoformat()


def request_stop(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def summarize(output, checkpoint):
    """Compare methods only on identical retained images, with a full-set control."""
    import numpy as np
    from eval_query_refinement import classification_metrics

    full_dir = output / "full_baseline/chexpert"
    paired_dir = output / "paired"
    full = json.loads((full_dir / "result.json").read_text())
    paired = json.loads((paired_dir / "result.json").read_text())
    if paired.get("status") != "complete":
        raise ValueError("Paired evaluation is not complete")
    predictions = full_dir / "predictions" / checkpoint.stem / "chexpert"
    full_paths = json.loads((predictions / "image_paths.json").read_text())
    paired_paths = json.loads((paired_dir / "image_paths.json").read_text())
    if len(set(full_paths)) != len(full_paths) or len(set(paired_paths)) != len(paired_paths):
        raise ValueError("Duplicate prediction image paths")
    lookup = {path: index for index, path in enumerate(full_paths)}
    indices = [lookup[path] for path in paired_paths]
    full_labels = np.load(predictions / "labels.npy", allow_pickle=False)
    labels = np.load(paired_dir / "labels.npy", allow_pickle=False)
    scores = np.load(predictions / "predictions.npy", allow_pickle=False).reshape(full_labels.shape)
    original = np.load(paired_dir / "original_probabilities.npy", allow_pickle=False)
    np.testing.assert_array_equal(labels, full_labels[indices])
    np.testing.assert_allclose(original, scores[indices], atol=2e-6, rtol=1e-5)
    classes = json.loads((predictions / "class_names.json").read_text())
    entry = json.loads((output / "data/catalog.json").read_text())["datasets"]["chexpert"]
    if classes != entry["class_names"] or labels.shape != (len(paired_paths), len(classes)):
        raise ValueError("Prediction class order or shape differs")
    split_path = Path(entry["split_audit_path"])
    if sha256(split_path) != entry["split_audit_sha256"]:
        raise ValueError("Split audit changed after catalog preparation")
    split = json.loads(split_path.read_text())
    if (split["class_names"] != classes or split["full"]["image_count"] != len(full_paths)
            or split["evaluation"]["image_count"] != len(paired_paths)
            or split["support"]["image_count"] != entry["support_count"]
            or split["evaluation"]["positive_count"] != labels.sum(0).tolist()):
        raise ValueError("Split audit counts differ from predictions")
    check = {"passed": True, "matched_images": len(paired_paths),
        "matching": "Exact image paths and class labels; full-set baseline subsetted to held-out patients",
        "probability_max_abs_error": float(np.max(np.abs(original - scores[indices]))),
        "probabilities_bitwise_equal": bool(np.array_equal(original, scores[indices])),
        "atol": 2e-6, "rtol": 1e-5,
        "batch_boundary_note": "Patient exclusion changes batches; agreement checked within original helper tolerance."}
    write_json(output / "baseline_parity.json", check)
    branches = {}
    for name in BRANCHES:
        metrics = paired["branches"][name]
        if "metrics" in metrics:
            metrics = metrics["metrics"]
        values = np.load(paired_dir / f"{name}_probabilities.npy", allow_pickle=False)
        computed = classification_metrics(labels, values)
        for key in ("macro_auroc", "macro_auprc", "macro_average_precision"):
            if metrics.get(key) is None or not np.isclose(computed[key], metrics[key], atol=1e-12, rtol=0):
                raise ValueError(f"Saved metric/prediction disagreement: {name}/{key}")
        branches[name] = computed
        branches[name]["per_class_by_name"] = dict(zip(classes, computed["per_class"]))
    baseline = branches["original"]
    deltas = {name: {key: (metrics[key] - baseline[key]) * 100
                     for key in ("macro_auroc", "macro_auprc", "macro_average_precision")}
              for name, metrics in branches.items()}
    full_metrics = full["classification"]
    report = {"status": "complete", "finished_at": now(), "dataset": "chexpert",
        "model": "AlphaRAD", "class_names": classes, "class_count": len(classes),
        "full_author_test_count": len(full_paths), "paired_test_count": len(paired_paths),
        "support_count": entry["support_count"], "patient_isolation": entry.get("scope"),
        "split_counts": {name: split[name] for name in ("full", "support", "evaluation")},
        "split_audit_path": str(split_path), "split_audit_sha256": entry["split_audit_sha256"],
        "limitations": split["limitations"],
        "checkpoint_sha256": full["checkpoint_sha256"], "eta": 0.3,
        "method": "Fixed phase6 query refinement + original/center80 equal native-logit averaging",
        "selection": "Support-only query calculation; no optimizer, no evaluation-driven parameter selection",
        "full_author_baseline": full_metrics, "paired": branches, "delta_percentage_points": deltas,
        "baseline_parity": check,
        "comparison_scope": "Method gains are paired against the baseline on the same held-out patient subset; full 668-image baseline is a separate reference.",
        "auprc_definition": "Trapezoidal precision-recall curve area; author PR_AUROC is average precision (AP)."}
    source_snapshot = json.loads((output / "source_sha256.json").read_text())
    changed = [path for path, digest in source_snapshot.items() if sha256(path) != digest]
    if changed:
        raise ValueError("Source files changed during the run: " + ", ".join(changed))
    report["source_integrity"] = {"passed": True, "files_checked": len(source_snapshot),
                                  "source_sha256": source_snapshot}
    write_json(output / "RESULTS.json", report)
    titles = {"original": "原始 AlphaRAD", "uniform_eta03": "仅查询修正（η=0.3）",
        "tta_center_original": "仅原图/中心裁剪平均", "tta_center_refined": "查询修正＋双视图平均（phase6）"}
    lines = ["AlphaRAD CheXpert：原始模型与固定 phase6 方法", "",
        f"状态：完成；{len(classes)} 类；支持集 {entry['support_count']} 张；患者隔离后测试集 {len(paired_paths)} 张。",
        f"完整集：{split['full']['image_count']} 张/{split['full']['patient_count']} 患者；支持集：{split['support']['image_count']} 张/{split['support']['patient_count']} 患者；测试集：{split['evaluation']['image_count']} 张/{split['evaluation']['patient_count']} 患者。",
        "全部方案使用相同测试图片和标签。未更新模型参数，查询修正使用有标签支持集。",
        "固定 η=0.3；标准预处理后的原图＋中心80%宽高裁剪，原始logits等权平均后sigmoid。",
        "AUPRC为PR曲线梯形面积；AP另列（作者代码PR_AUROC字段对应AP）。单位：%；增益：百分点。", "",
        "| 同一患者隔离测试集上的方法 | AUROC | AUPRC | AP | ΔAUROC | ΔAUPRC |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in BRANCHES:
        metrics, delta = branches[name], deltas[name]
        lines.append(f"| {titles[name]} | {metrics['macro_auroc']*100:.2f} | {metrics['macro_auprc']*100:.2f} | {metrics['macro_average_precision']*100:.2f} | {delta['macro_auroc']:+.2f} | {delta['macro_auprc']:+.2f} |")
    lines += ["", f"完整作者测试集（{len(full_paths)}张）原始 AlphaRAD：AUROC {full_metrics['macro_auroc']*100:.2f}%，AUPRC {full_metrics['macro_auprc_trapezoidal']*100:.2f}%，AP {full_metrics['macro_average_precision']*100:.2f}%。",
        "完整测试集原始结果仅作参考，不与排除支持患者后的改进结果直接相减。",
        f"全量原始预测按路径取子集后，与配对原始预测通过一致性核验，最大概率误差 {check['probability_max_abs_error']:.3g}。", "",
        "| 类别 | 测试阳性图片 | 测试阳性患者 | 原始AUROC | 改进AUROC | ΔAUROC | 原始AUPRC | 改进AUPRC | ΔAUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    refined = branches["tta_center_refined"]
    for index, (name, base, change) in enumerate(zip(classes, baseline["per_class"], refined["per_class"])):
        positives = split["evaluation"]["positive_count"][index]
        positive_patients = split["evaluation"]["positive_patient_count"][index]
        if base["auroc"] is None:
            lines.append(f"| {name} | {positives} | {positive_patients} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(f"| {name} | {positives} | {positive_patients} | {base['auroc']*100:.2f} | {change['auroc']*100:.2f} | {(change['auroc']-base['auroc'])*100:+.2f} | {base['auprc']*100:.2f} | {change['auprc']*100:.2f} | {(change['auprc']-base['auprc'])*100:+.2f} |")
    lines += ["", "限制：本机只有完整的作者测试集，因此本次从中抽有标签支持样本，再按患者排除，属于新的少样本患者留出评估。"]
    if "Fracture" in classes:
        index = classes.index("Fracture")
        lines.append(f"Fracture（骨折）在保留测试集中仅 {split['evaluation']['positive_count'][index]} 张阳性图片、{split['evaluation']['positive_patient_count'][index]} 位阳性患者，其指标和涨幅非常不稳定，不能作为可靠泛化结论。")
    lines += ["支持集同一患者的不同视角可能作为不同类别锚点；这些患者的所有图片均从测试集排除。", "原始 split audit 限制："]
    lines.extend(f"- {item}" for item in split["limitations"])
    lines.append(f"运行结束时全部 {len(source_snapshot)} 个记录源码 SHA256 与启动时一致。")
    (output / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--checkpoint", type=Path, default=ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(gpu not in (6, 7) for gpu in args.gpus):
        parser.error("Only distinct physical GPUs 6 and 7 are authorized")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Require positive batch size and nonnegative workers")
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    state = {"status": "preparing", "started_at": now(), "controller_pid": os.getpid(),
        "physical_gpus": args.gpus, "output": str(args.output), "deadline": None, "jobs": {}}
    status_file = args.output / "status.json"
    write_json(status_file, state)
    active = {}
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        prepare = [sys.executable, str(ROOT / "local_setup/prepare_chexpert_query_data.py"), "--output", str(args.output / "data")]
        state["prepare_command"] = prepare
        with (args.output / "logs/prepare.log").open("w") as log:
            subprocess.run(prepare, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        common = ["--checkpoint", str(args.checkpoint), "--batch-size", str(args.batch_size), "--workers", str(args.workers)]
        jobs = {
            "full_baseline": [sys.executable, str(ROOT / "local_setup/eval_pretrained.py"),
                "--catalog", str(args.output / "data/full_catalog.json"), "--dataset", "chexpert",
                "--output", str(args.output / "full_baseline/chexpert"), *common],
            "paired": [sys.executable, str(ROOT / "local_setup/eval_chexpert_query_refinement.py"),
                "--catalog", str(args.output / "data/catalog.json"), "--output", str(args.output / "paired"), *common],
        }
        sources = [Path(__file__), ROOT / "scripts/eval_chexpert_query_refinement.sh",
            ROOT / "local_setup/prepare_chexpert_query_data.py", ROOT / "local_setup/eval_chexpert_query_refinement.py",
            ROOT / "local_setup/eval_pretrained.py", ROOT / "local_setup/query_refinement.py",
            ROOT / "local_setup/eval_query_refinement.py", ROOT / "local_setup/run_pretrained.py"]
        write_json(args.output / "source_sha256.json", {str(path): sha256(path) for path in sources})
        state["jobs"] = {name: {"status": "pending", "command": command,
            "physical_gpu": args.gpus[index % len(args.gpus)], "log": str(args.output / "logs" / f"{name}.log")}
            for index, (name, command) in enumerate(jobs.items())}
        state["status"] = "running"
        write_json(status_file, state)
        while any(job["status"] in ("pending", "running") for job in state["jobs"].values()):
            if STOP_SIGNAL is not None:
                raise InterruptedError(f"Controller received signal {STOP_SIGNAL}")
            for name, info in list(active.items()):
                proc, lease, log = info
                code = proc.poll()
                if code is None:
                    continue
                job = state["jobs"][name]
                job.update(status="complete" if code == 0 else "failed", returncode=code, finished_at=now())
                release_gpu(lease)
                log.close()
                del active[name]
                print(f"{name}: {job['status']} (returncode={code})", flush=True)
            for name, job in state["jobs"].items():
                if job["status"] != "pending":
                    continue
                if any(state["jobs"][running]["physical_gpu"] == job["physical_gpu"] for running in active):
                    continue
                lease, gpu_info = reserve_gpu(job["physical_gpu"])
                job["last_gpu_check"] = gpu_info
                if lease is None:
                    continue
                log = None
                try:
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu_info["uuid"]
                    env["PYTHONUNBUFFERED"] = "1"
                    log = Path(job["log"]).open("w")
                    proc = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    release_gpu(lease)
                    if log is not None:
                        log.close()
                    raise
                active[name] = (proc, lease, log)
                job.update(status="running", pid=proc.pid, started_at=now(), gpu_info=gpu_info)
                print(f"{name}: started PID {proc.pid} on physical GPU {job['physical_gpu']}", flush=True)
            state["updated_at"] = now()
            write_json(status_file, state)
            if active or any(job["status"] == "pending" for job in state["jobs"].values()):
                time.sleep(5)
        failed = [name for name, job in state["jobs"].items() if job["status"] != "complete"]
        if failed:
            raise RuntimeError("Evaluation jobs failed: " + ", ".join(failed))
        state["status"] = "summarizing"
        write_json(status_file, state)
        summarize(args.output, args.checkpoint)
        state.update(status="complete", finished_at=now())
        write_json(status_file, state)
        print(f"Complete: {args.output / 'RESULTS.txt'}", flush=True)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error=str(error), traceback=traceback.format_exc(), finished_at=now())
        # Only children started and still owned by this controller are signaled.
        # Each was started in a fresh session; no other server jobs are touched.
        for name, (proc, lease, log) in active.items():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for name, (proc, lease, log) in active.items():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            state["jobs"][name].update(status="interrupted", returncode=proc.returncode, finished_at=now())
            release_gpu(lease)
            log.close()
        write_json(status_file, state)
        raise


if __name__ == "__main__":
    main()
