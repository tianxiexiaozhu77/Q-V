#!/usr/bin/env python3
"""Evaluate support-only query candidates through the unchanged frozen RadZero.

One visual encoder pass serves every candidate in each batch. The two mandatory
references (original and uniform_eta03) must reproduce the historical query run.
This entry owns its deadline; it never invokes the earlier overnight deadline.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

CODE = Path(__file__).resolve().parent
MEDICAL = CODE.parents[2]
RADZERO = MEDICAL / "RadZero_i2cl"
SETUP = RADZERO / "local_setup"
DEFAULT_ADAPTERS = RADZERO / "runs/overnight_v2_20261020_2250/selected_query_adapter"
CAMPAIGN_DEADLINE = datetime.fromisoformat("2026-10-22T10:00:00+08:00").timestamp()
DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")
REQUIRED = ("original", "uniform_eta03")
METRICS = ("macro_auroc", "macro_auprc", "macro_average_precision")
sys.dont_write_bytecode = True
sys.path.insert(0, str(SETUP))
sys.path.insert(0, str(CODE))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DeadlineReached(RuntimeError):
    pass


class RunInterrupted(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_deadline(deadline):
    if time.time() >= deadline:
        raise DeadlineReached("The requested campaign deadline has been reached")


def reference_directory(requested, dataset, metadata):
    if requested is None:
        return Path(metadata["source_query_run"])
    root = Path(requested).expanduser().resolve()
    for directory in (root, root / dataset):
        if (directory / "query_adapter.json").is_file():
            recorded = json.loads((directory / "query_adapter.json").read_text())
            if recorded.get("dataset") == dataset:
                return Path(recorded["source_query_run"])
    for directory in (root, root / dataset, root / "query_refinement" / dataset):
        if (directory / "manifest.json").is_file():
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest.get("dataset") == dataset and "predictions" in manifest:
                return directory
    raise FileNotFoundError(f"Cannot resolve the historical {dataset} query run under {root}")


def load_inputs(args):
    """Read verified support arrays and unlabelled evaluation identities only."""
    import numpy as np
    from eval_query_adapter import verify_base
    from extract_frozen_features import public_spec, strict_baseline_check
    from i2cl_data import load_spec
    from overnight_query_refinement import verified_cache
    from query_adapter import QueryAdapter

    check_deadline(args.deadline)
    adapter_path = DEFAULT_ADAPTERS / args.dataset / "query_adapter.npz"
    adapter = QueryAdapter.load(adapter_path)
    verify_base(adapter)
    if adapter.metadata["dataset"] != args.dataset or adapter.eta != .3:
        raise ValueError("Expected the historical dataset adapter with eta=0.3")
    reference = reference_directory(args.baseline_run, args.dataset, adapter.metadata)
    reference_path = reference / "manifest.json"
    if sha256(reference_path) != adapter.metadata["source_query_manifest_sha256"]:
        raise ValueError("Historical query-run manifest changed")
    historical = json.loads(reference_path.read_text())
    if historical["status"] != "complete" or historical["dataset"] != args.dataset:
        raise ValueError("Historical query run is incomplete or belongs to another dataset")
    cache = Path(historical["cache"])
    if sha256(cache / "manifest.json") != adapter.metadata["cache_manifest_sha256"]:
        raise ValueError("Historical feature-cache manifest changed")
    cached = verified_cache(cache)
    cache_manifest = cached["manifest"]
    for key in ("class_names", "prompts", "support_identity", "test_identity", "checkpoint"):
        if cache_manifest[key] != historical[key] or historical[key] != adapter.metadata[key]:
            raise ValueError(f"Historical adapter, cache, and query run differ: {key}")
    if cache_manifest["dataset"] != args.dataset:
        raise ValueError("Feature cache belongs to another dataset")
    for recorded, digest in adapter.metadata["split_sha256"].items():
        if sha256(recorded) != digest:
            raise ValueError(f"Historical split changed: {recorded}")
    split_dir = Path(adapter.metadata["split_dir"])
    support = load_spec(split_dir, args.dataset, "support")
    if public_spec(support) != cached["support_metadata"]:
        raise ValueError("Support identifiers/prompts differ from the frozen feature cache")
    np.testing.assert_array_equal(support["labels"], cached["support_labels"])
    np.testing.assert_array_equal(adapter.original, cached["text_features"])
    for field in ("image_paths", "ids", "groups"):
        if set(cached["support_metadata"][field]) & set(cached["test_metadata"][field]):
            raise ValueError(f"Support/test overlap: {field}")
    references = {}
    for name, eta_key in (("original", "eta_0"), ("uniform_eta03", "eta_0p3")):
        record = historical["predictions"][eta_key]
        directory = reference / record["directory"]
        path = directory / "logits.npy"
        if sha256(path) != record["logits_sha256"]:
            raise ValueError(f"Historical reference logits changed: {name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (len(cached["test_metadata"]["image_paths"]), len(adapter.class_names))
        if values.shape != expected_shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Invalid historical reference logits: {name}")
        table = directory / "predictions.csv"
        if sha256(table) != record["predictions_sha256"]:
            raise ValueError(f"Historical prediction table changed: {name}")
        references[name] = {"array": values, "csv": table, "logits_path": path}
    strict_baseline_check(references["original"]["array"], cached["test_base_logits"])
    return {"adapter": adapter, "adapter_path": adapter_path, "reference": reference,
            "historical": historical, "cache": cache, "cached": cached,
            "support": support, "references": references, "split_dir": split_dir}


def fit_and_validate(inputs, selected_names=None):
    import numpy as np
    from query_candidates import fit_candidates

    cached = inputs["cached"]
    fitted = fit_candidates(cached["support_class_local"], cached["support_labels"],
                            cached["text_features"], num_heads=1,
                            groups=cached["support_metadata"]["groups"])
    names, queries, metadata = fitted["names"], fitted["queries"], fitted["metadata"]
    if (len(names) != len(set(names)) or names[:2] != list(REQUIRED)
            or len(metadata) != len(names) or queries.dtype != np.float32
            or queries.shape != (len(names), *cached["text_features"].shape)
            or not np.isfinite(queries).all()):
        raise ValueError("Invalid shared candidate schema or query array")
    for name in names:
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
            raise ValueError(f"Unsafe candidate name: {name}")
    # These two candidates must literally preserve the established reference vectors.
    np.testing.assert_array_equal(queries[0], inputs["adapter"].original)
    np.testing.assert_array_equal(queries[1], inputs["adapter"].refined)
    if selected_names:
        missing = set(selected_names) - set(names)
        if missing:
            raise ValueError(f"Unknown candidate names: {sorted(missing)}")
        selected = set(REQUIRED) | set(selected_names)
        indices = [i for i, name in enumerate(names) if name in selected]
        names = [names[i] for i in indices]
        queries = queries[indices]
        metadata = [metadata[i] for i in indices]
    return {"names": names, "queries": queries, "metadata": metadata}


def reference_table_check(path, spec, expected_logits):
    """After fitting/inference, verify every original CSV identity and label."""
    import numpy as np
    count = 0
    with Path(path).open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= len(spec["image_paths"]) or int(row["index"]) != index:
                raise ValueError("Historical prediction table index/count mismatch")
            if row["image_path"] != spec["image_paths"][index]:
                raise ValueError("Historical prediction table path order mismatch")
            labels = [float(row[f"label:{name}"]) for name in spec["class_names"]]
            values = [float(row[f"logit:{name}"]) for name in spec["class_names"]]
            np.testing.assert_array_equal(labels, spec["labels"][index])
            np.testing.assert_array_equal(np.asarray(values, dtype=np.float32), expected_logits[index])
            count += 1
    if count != len(spec["image_paths"]):
        raise ValueError("Historical prediction table does not cover the complete test split")
    return {"passed": True, "paths_and_labels_checked": count}


def evaluate_metrics(spec, logits):
    import numpy as np
    from sklearn.metrics import average_precision_score
    from batch_metrics import compute_metrics

    result = compute_metrics(spec, logits)
    values = []
    for index, name in enumerate(spec["class_names"]):
        y = np.asarray(spec["labels"])[:, index]
        ap = float(average_precision_score(y, logits[:, index])) if y.sum() else None
        result["per_class"][name]["average_precision"] = ap
        if ap is not None:
            values.append(ap)
    result["macro_average_precision"] = float(np.mean(values)) if values else None
    result["macro_average_precision_defined_classes"] = len(values)
    result["average_precision_method"] = "sklearn average_precision_score; distinct from trapezoidal macro_auprc"
    return result


def run(args):
    import numpy as np
    import torch
    from extract_frozen_features import image_loader, public_spec, strict_baseline_check, pool_original_matching
    from infer_batch import CachedInference, load_model
    from i2cl_data import load_spec
    from overnight_query_refinement import QueryRefinementMatcher

    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("No assigned CUDA GPU available")
    inputs = load_inputs(args)
    check_deadline(args.deadline)
    fitted = fit_and_validate(inputs, args.candidate_names)
    names, queries = fitted["names"], fitted["queries"]
    cached = inputs["cached"]
    full_metadata = cached["test_metadata"]
    count = min(args.limit_test, len(full_metadata["image_paths"])) if args.limit_test else len(full_metadata["image_paths"])
    test_metadata = {**full_metadata, **{key: full_metadata[key][:count] for key in ("image_paths", "ids", "groups")}}
    np.savez_compressed(args.output / "queries.npz", names=np.asarray(names), queries=queries)
    write_json(args.output / "candidate_metadata.json", dict(zip(names, fitted["metadata"])))
    config = {"model": "radzero", "dataset": args.dataset, "status": "loading_model",
        "created_at": now(), "deadline": args.deadline, "candidate_names": names,
        "support_count": len(inputs["support"]["image_paths"]), "rows": count,
        "is_smoke": bool(args.limit_test), "fit_split": "support_only",
        "test_labels_used_for_fit": False, "trainable_base_parameters": 0,
        "candidate_selection": "Fixed candidate formulas; selecting among reported test outcomes is exploratory test-set selection.",
        "cache_manifest": str(inputs["cache"] / "manifest.json"),
        "cache_manifest_sha256": sha256(inputs["cache"] / "manifest.json"),
        "reference_run": str(inputs["reference"]), "split_dir": str(inputs["split_dir"]),
        "split_sha256": cached["manifest"]["split_sha256"],
        "support_identity": cached["manifest"]["support_identity"],
        "test_identity": cached["manifest"]["test_identity"],
        "checkpoint": inputs["adapter"].metadata["checkpoint"],
        "class_names": full_metadata["class_names"], "prompts": full_metadata["prompts"],
        "batch_size": args.batch_size, "workers": args.workers,
        "precision": "float32; TF32 disabled", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": {str(path): sha256(path) for path in (Path(__file__).resolve(), CODE / "query_candidates.py")},
        "queries_sha256": sha256(args.output / "queries.npz"),
        "historical_code_modified": False, "completed_images": 0}
    write_json(args.output / "run_config.json", config)
    write_json(args.output / "progress.json", {"status": "loading_model", "completed_images": 0, "total_images": count})
    check_deadline(args.deadline)
    model, processor, tokenizer = load_model(torch.device("cuda"))
    model = model.float().requires_grad_(False).eval()
    helper = CachedInference(model, tokenizer, list(inputs["adapter"].prompts))
    np.testing.assert_allclose(helper.features.detach().cpu().numpy(), inputs["adapter"].original, atol=1e-6, rtol=0.)
    matcher = QueryRefinementMatcher(model, queries[:1], (0.,))
    gpu_queries = torch.as_tensor(queries, dtype=torch.float32, device="cuda")
    arrays = {}
    for name in names:
        directory = args.output / "candidates" / name
        directory.mkdir(parents=True)
        values = np.lib.format.open_memmap(directory / "logits.npy", mode="w+", dtype=np.float32,
                                         shape=(count, len(full_metadata["class_names"])))
        values.fill(np.nan)
        arrays[name] = values
    checks = {name: {"passed": True, "checked_images": 0, "max_abs_error": 0.,
                    "bitwise_equal": True, "atol": 1e-6, "rtol": 0.} for name in REQUIRED}
    offset, last_print = 0, 0
    with torch.inference_mode():
        for pixels, _, indices in image_loader(test_metadata, processor, args):
            check_deadline(args.deadline)
            stop = offset + len(pixels)
            if not np.array_equal(indices.numpy(), np.arange(offset, stop)):
                raise ValueError("Image order differs from verified test metadata")
            aligned = model.forward_vision_model(pixels.to("cuda", dtype=torch.float32, non_blocking=True))["vision_tokens"]
            vision = matcher.prepare_vision(aligned)
            for name, query in zip(names, gpu_queries):
                check_deadline(args.deadline)
                output = matcher.match_prepared(vision, query, return_maps=offset == 0)
                arrays[name][offset:stop] = output["logits"].cpu().numpy()
                if offset == 0:
                    np.savez_compressed(args.output / "candidates" / name / "first_batch_maps.npz",
                                        similarity_scores=output["similarity_scores"].cpu().numpy(),
                                        image_paths=np.asarray(test_metadata["image_paths"][:stop]))
            if offset == 0:
                # Independent existing pooling helper uses the same encoded tokens.
                _, official_logits, reconstruction_error = pool_original_matching(matcher.loss, helper.features, aligned)
                proof = strict_baseline_check(arrays["original"][:stop], official_logits.cpu().numpy(), atol=2e-5)
                checks["original_helper"] = {**proof, "images": stop,
                    "pooled_reconstruction_max_error": reconstruction_error, "extra_visual_encoder_passes": 0}
            for name in REQUIRED:
                proof = strict_baseline_check(arrays[name][offset:stop], inputs["references"][name]["array"][offset:stop])
                checks[name]["max_abs_error"] = max(checks[name]["max_abs_error"], proof["max_abs_error"])
                checks[name]["bitwise_equal"] &= proof["bitwise_equal"]
                checks[name]["checked_images"] = stop
            offset = stop
            write_json(args.output / "baseline_checks.json", checks)
            write_json(args.output / "progress.json", {"status": "inference", "completed_images": offset,
                "total_images": count, "candidate_count": len(names), "updated_at": now()})
            if offset == len(pixels) or offset - last_print >= 500 or offset == count:
                for value in arrays.values():
                    value.flush()
                print(f"{args.dataset}: {offset}/{count}, {len(names)} candidates", flush=True)
                last_print = offset
    if offset != count:
        raise ValueError("Incomplete candidate inference")
    for value in arrays.values():
        value.flush()
        if not np.isfinite(value).all():
            raise ValueError("Partial or nonfinite candidate logits")
    check_deadline(args.deadline)
    # Only now are test labels loaded; query fitting has already finished.
    spec = load_spec(inputs["split_dir"], args.dataset, "test")
    if public_spec(spec) != full_metadata:
        raise ValueError("Evaluation identities changed after inference")
    for name in REQUIRED:
        checks[name]["historical_table"] = reference_table_check(inputs["references"][name]["csv"], spec,
                                                                inputs["references"][name]["array"])
    score_spec = {**spec, "labels": spec["labels"][:count], "image_paths": spec["image_paths"][:count]}
    baseline_metrics = evaluate_metrics(score_spec, arrays["original"])
    candidates = {}
    np.save(args.output / "labels.npy", score_spec["labels"], allow_pickle=False)
    write_json(args.output / "image_paths.json", score_spec["image_paths"])
    for name, metadata in zip(names, fitted["metadata"]):
        check_deadline(args.deadline)
        values = arrays[name]
        directory = args.output / "candidates" / name
        metrics = evaluate_metrics(score_spec, values)
        probabilities = torch.sigmoid(torch.from_numpy(np.asarray(values).copy()).double()).numpy().astype(np.float32)
        np.save(directory / "probabilities.npy", probabilities, allow_pickle=False)
        with (directory / "predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "image_path", *[f"label:{c}" for c in spec["class_names"]],
                             *[f"logit:{c}" for c in spec["class_names"]], *[f"score:{c}" for c in spec["class_names"]]])
            writer.writerows([i, path, *label.tolist(), *logit.tolist(), *probability.tolist()]
                             for i, (path, label, logit, probability) in enumerate(zip(score_spec["image_paths"],
                                  score_spec["labels"], values, probabilities)))
        delta = {key: None if metrics[key] is None or baseline_metrics[key] is None
                 else 100 * (metrics[key] - baseline_metrics[key]) for key in METRICS}
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "delta_pp.json", delta)
        candidates[name] = {"metrics": metrics, "delta_pp": delta, "metadata": metadata,
                            "directory": str(directory), "logits_sha256": sha256(directory / "logits.npy")}
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Unexpected trainable base-model parameter")
    check_deadline(args.deadline)
    result = {"status": "complete", "model": "radzero", "dataset": args.dataset,
        "task": "classification", "rows": count, "support_count": config["support_count"],
        "is_smoke": bool(args.limit_test), "candidates": candidates, "baseline_checks": checks,
        "completed_at": now(), "elapsed_seconds": time.time() - started,
        "trainable_base_parameters": 0, "test_labels_used_for_fit": False,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "config": config}
    write_json(args.output / "baseline_checks.json", checks)
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", {"status": "complete", "completed_images": count,
        "total_images": count, "is_smoke": bool(args.limit_test), "updated_at": now()})
    print(json.dumps({"dataset": args.dataset, "status": "complete", "rows": count,
        "candidates": {name: {key: entry["metrics"][key] for key in METRICS} for name, entry in candidates.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, help="Historical query run, parent, or selected adapter directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--deadline", type=float, required=True, help="Unix epoch, at most 2026-10-22 10:00 Asia/Shanghai")
    parser.add_argument("--limit-test", type=int, default=0, help="Smoke only; never counted as full evaluation")
    parser.add_argument("--candidate-names", nargs="+", help="Optional subset; original and uniform_eta03 are always included")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or args.limit_test < 0
            or not math.isfinite(args.deadline) or args.deadline > CAMPAIGN_DEADLINE):
        parser.error("Invalid batch/workers/limit/deadline, or deadline exceeds user authorization")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    handlers = {number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    def stop(number, _frame):
        if number == signal.SIGALRM:
            raise DeadlineReached("Campaign deadline timer expired")
        raise RunInterrupted(f"Evaluation interrupted by signal {number}")
    for number in handlers:
        signal.signal(number, stop)
    try:
        check_deadline(args.deadline)
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline - time.time()))
        run(args)
    except BaseException as error:
        status = ("deadline_reached" if isinstance(error, DeadlineReached) else
                  "interrupted" if isinstance(error, (RunInterrupted, KeyboardInterrupt)) else "failed")
        progress_path = args.output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        failure = {**progress, "status": status, "model": "radzero", "dataset": args.dataset,
                   "is_smoke": bool(args.limit_test), "updated_at": now(), "error": repr(error),
                   "traceback": traceback.format_exc()}
        write_json(args.output / "failure.json", failure)
        write_json(progress_path, failure)
        write_json(args.output / "result.json", failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
