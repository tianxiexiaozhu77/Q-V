#!/usr/bin/env python3
"""CPU-only dependency, support-control and archived resource audit for replay."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "experiments/query_search_20261021_152010"
CODE = CAMPAIGN / "code_phase6"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_cache(directory, model):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["rows"] == manifest["cached_rows"]
    cursor = 0
    for shard in manifest["shards"]:
        assert shard["start"] == cursor
        assert (directory / shard["file"]).stat().st_size == shard["bytes"]
        cursor = shard["end"]
    assert cursor == manifest["rows"]
    for control in ("original", "uniform_eta03"):
        assert manifest["controls"][control]["passed"]
        assert manifest["controls"][control]["images_checked"] == manifest["rows"]
    if model == "alpharad":
        records = manifest["identity"]["input_file_fingerprints"]
        for rec in records:
            stat = Path(rec["path"]).stat()
            assert (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (
                rec["bytes"], rec["mtime_ns"], rec["ctime_ns"])
    else:
        for value in manifest["image_ids"]:
            path, size, mtime, ctime = json.loads(value)
            stat = Path(path).stat()
            assert (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (size, mtime, ctime)
    return {"path": str(directory), "rows": manifest["rows"],
            "shards": len(manifest["shards"]), "status": "passed", "shard_check": "existence_and_size",
            "input_check": "all_file_paths_sizes_mtime_ctime", "manifest_sha256": sha(directory / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("alpharad", "radzero"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "AlphaRAD/local_setup"))
    from run_pretrained import reserve_gpu, release_gpu  # import only; never called
    source = CODE / f"eval_{args.model}_candidates.py"
    spec = importlib.util.spec_from_file_location(f"cpu_phase6_{args.model}", source)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    worker.check_deadline = lambda _value: None
    import numpy as np
    import torch
    torch.set_num_threads(4)
    result = {"model": args.model, "worker_import": "passed", "gpu_used": False,
              "source_sha256": sha(source), "tasks": []}
    if args.model == "alpharad":
        from eval_query_refinement import audit_csv
        base = worker.DEFAULT_BASELINE
        original_sources = json.loads((base / "source_sha256.json").read_text())
        for rel, expected in original_sources.items():
            assert sha(ROOT / "AlphaRAD" / rel) == expected, rel
        checkpoint = worker.ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar"
        assert sha(checkpoint) == worker.CHECKPOINT_SHA
        catalog = json.loads((base / "catalog.json").read_text())
        fit, plan = worker.load_candidate_components(CODE / "query_candidates.py")
        for name, entry in catalog["datasets"].items():
            directory = base / name
            prior = json.loads((directory / "result.json").read_text())
            assert prior["status"] == "complete" and not prior["is_smoke"] and prior["eta"] == .3
            assert prior["entry"] == entry
            sr, sp, sl = audit_csv(entry["support_csv_path"], entry["support_csv_sha256"], entry["class_names"])
            tr, tp, tl = audit_csv(entry["csv_path"], entry["csv_sha256"], entry["class_names"], entry["task"] == "classification")
            assert len(tr) == entry["actual_count"] and len(sr) == entry["support_count"]
            assert not set(sp) & set(tp)
            md = json.loads((directory / "query_adapter.json").read_text())
            assert sha(directory / "query_adapter.npz") == md["query_file_sha256"]
            cached = np.load(directory / "query_adapter.npz", allow_pickle=False)
            features = np.load(directory / "support_class_local.npy", allow_pickle=False)
            queries = fit(features, np.asarray(sl), cached["original_queries"], num_heads=md["num_heads"])
            np.testing.assert_array_equal(queries["queries"][:2], cached["queries"])
            assert all(plan(n)["views"] == ["identity", "center80"] for n in ("tta_center_original", "tta_center_refined"))
            files = ["image_paths.json", "baseline_probabilities.npy", "refined_probabilities.npy", "similarity_map_examples.npz"]
            if entry["task"] != "classification": files += ["baseline_grounding_map.npy", "refined_grounding_map.npy"]
            for filename in files: assert (directory / filename).is_file(), filename
            cache = json.loads((CAMPAIGN / "runs/phase6" / f"alpharad_{name}" / "vision_cache.json").read_text())
            result["tasks"].append({"dataset": name, "support_count": len(sr), "controls": "bitwise_equal", "cache": check_cache(Path(cache["path"]), args.model)})
            print(name, "passed", flush=True)
    else:
        for name in worker.DATASETS:
            opt = SimpleNamespace(dataset=name, baseline_run=None, deadline=1e12)
            inputs = worker.load_inputs(opt)
            gf, _ = worker.load_global_support(inputs, name)
            fitted = worker.fit_and_validate(inputs, ["original", "uniform_eta03", "tta_center_original", "tta_center_refined"], CODE / "query_candidates.py", gf)
            assert fitted["names"] == ["original", "uniform_eta03", "tta_center_original", "tta_center_refined"]
            for filename in ("support_metadata.json", "test_metadata.json"):
                md = json.loads((inputs["cache"] / filename).read_text())
                for relative in md["image_paths"]: assert (Path(md["data_root"]) / relative).is_file(), relative
            config = json.loads((CAMPAIGN / "runs/phase6" / f"radzero_{name}" / "run_config.json").read_text())
            result["tasks"].append({"dataset": name, "support_count": len(inputs["cached"]["support_labels"]), "controls": "bitwise_equal", "cache": check_cache(Path(config["vision_cache"]["path"]), args.model)})
            print(name, "passed", flush=True)
    result["status"] = "passed"
    result["limitations"] = ["No GPU inference executed", "Archived vision shard bytes were not rehashed; existence, sizes, coverage and recorded controls were checked", "No claim of fresh visual-encoder numerical verification"]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
