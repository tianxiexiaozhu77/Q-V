#!/usr/bin/env python3
"""Fit the existing same-feature score control on support labels only.

This worker never opens evaluation_manifest or test labels/masks. Its five folds
hold out one positive and one negative support group each, in stored order.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
METHODS = MEDICAL / "experiments/fewshot_native_query/methods.py"
sys.path.insert(0, str(METHODS.parent))
from methods import run_score_methods, ALPHAS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def folds_for_binary_support(labels, groups):
    y = np.asarray(labels)
    if y.shape != (10, 1) or not np.isin(y, (0, 1)).all():
        raise ValueError("Expected ten binary supports")
    if len(groups) != 10 or len(set(groups)) != 10:
        raise ValueError("Ten distinct support groups required")
    pos, neg = np.flatnonzero(y[:, 0] == 1), np.flatnonzero(y[:, 0] == 0)
    if len(pos) != 5 or len(neg) != 5:
        raise ValueError("Expected five positives and five negatives")
    return [dict(train=np.setdiff1d(np.arange(10), [p, n]), validation=np.asarray([p, n]))
            for p, n in zip(pos, neg)]


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="Dataset inference directory")
    args = p.parse_args()
    args.protocol, args.run = args.protocol.resolve(), args.run.resolve()
    catalog = json.loads(args.protocol.read_text())
    receipt = json.loads((args.run / "result.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("Only completed full inference can be fitted/scored")
    if receipt["protocol_sha256"] != sha(args.protocol):
        raise ValueError("Inference used a different frozen support/test protocol")
    support_file, test_file = args.run / "support_union_features.npz", args.run / "test_features.npz"
    for filename in (support_file, test_file):
        if receipt["artifact_sha256"][filename.name] != sha(filename):
            raise ValueError("Frozen inference feature artifact changed: " + str(filename))
    sources = {str(p): sha(p) for p in (args.protocol, args.run / "result.json", support_file, test_file, METHODS, Path(__file__))}
    support, test = load_npz(support_file), load_npz(test_file)
    if any(k in test for k in ("labels", "masks", "rle")):
        raise ValueError("Test feature cache must be label-free")
    if support["class_names"].tolist() != catalog["class_names"] or test["class_names"].tolist() != catalog["class_names"]:
        raise ValueError("Class order mismatch")
    lookup = {str(path): i for i, path in enumerate(support["image_paths"])}
    if len(lookup) != len(support["image_paths"]):
        raise ValueError("Duplicate union image")
    planned = []
    for seed in (42,43,44,45,46):
        folder = args.run / f"seed_{seed}"
        if (folder / "score_predictions.npz").exists() or (folder / "score_fit.json").exists():
            raise FileExistsError("S artifacts already exist; use a fresh inference output")
        rows = catalog["seeds"][str(seed)]["support_rows"]
        indices = np.asarray([lookup[row["image"]] for row in rows])
        y = np.asarray([row["labels"] for row in rows])
        groups = [row["group_id"] for row in rows]
        np.testing.assert_array_equal(support["labels"][indices], y)
        np.testing.assert_array_equal(support["group_ids"][indices], np.asarray(groups))
        if set(groups) & set(test["group_ids"].tolist()):
            raise ValueError("Support and test header groups overlap")
        folds = folds_for_binary_support(y, groups)
        sl, tl = support["class_local"][indices], test["class_local"]
        fitted = run_score_methods(support_features=sl[:, 0], support_local=sl, labels=y,
            support_logits=support["logits"][indices], groups=groups,
            test_features=tl[:, 0], test_local=tl, test_logits=test["logits"],
            methods=["class_local_difference_score_oof"], folds=folds)
        scores = fitted["scores"]["class_local_difference_score_oof"]
        prediction = load_npz(folder / "predictions.npz")
        for key in ("image_paths", "class_names", "group_ids"):
            np.testing.assert_array_equal(test[key], prediction[key])
        np.testing.assert_allclose(expit(test["logits"]), prediction["original"], atol=2e-6, rtol=2e-5)
        info = dict(support_seed=seed, protocol_sha256=sha(args.protocol), sources=sources,
            methods=fitted["fit_info"], alpha_grid=list(ALPHAS),
            support_only_folds=[{k:v.tolist() for k,v in f.items()} for f in folds],
            global_feature_arguments="Class-local slice fills unused global arguments; only the class-local method is executed",
            spatial_output="Adapted score has no native map; system comparison retains Original map",
            test_labels_read=False, test_masks_read=False)
        planned.append((folder, scores, info))
    # All support-only fits finish before any evaluation script may open labels.
    for folder, scores, info in planned:
        np.savez_compressed(folder / "score_predictions.npz", same_feature_score=expit(scores),
            logits=scores, **{k:test[k] for k in ("image_paths", "group_ids", "class_names")})
        info["predictions_sha256"] = sha(folder / "score_predictions.npz")
        write(folder / "score_fit.json", info)
    write(args.run / "S_FIT_RECEIPT.json", dict(status="complete", test_labels_read=False,
        test_masks_read=False, seeds=[42,43,44,45,46], sources=sources,
        artifacts={str(s): {name: sha(args.run / f"seed_{s}" / name)
            for name in ("score_predictions.npz", "score_fit.json")} for s in (42,43,44,45,46)}))
    print("S five-seed support-only fitting complete", flush=True)


if __name__ == "__main__":
    main()
