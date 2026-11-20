"""CPU-only integration tests against the frozen four-unit provenance snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

import cache_multiseed as worker


class CacheContractTest(unittest.TestCase):
    def test_all_twenty_native_fits_and_exports(self):
        """Rebuild actual fits from authenticated union data, never load a model."""
        self.assertFalse(torch.cuda.is_initialized())
        sys.path[:0] = [str(worker.MEDICAL / "baselines/local_setup"),
                       str(worker.ALPHA / "local_setup")]
        snapshot = worker.read(worker.HERE / "cache_reuse_sources_v1.json")
        catalog = worker.read(worker.HERE / "protocol_v1/catalog.json")
        with tempfile.TemporaryDirectory(prefix="multiseed-cache-cpu-") as temporary:
            for model in worker.MODELS:
                for dataset in worker.DATASETS:
                    with self.subTest(model=model, dataset=dataset):
                        source = snapshot["units"][f"{model}_{dataset}"]
                        stable, first = Path(source["stability"]["path"]), Path(source["first_round"]["path"])
                        for name in ("support_union.json", "support_union_features.npy"):
                            worker.checked(stable / name, source["stability"]["files"][name])
                        union = [worker.normalized_support(row) for row in worker.read(stable / "support_union.json")]
                        supports = {}
                        for seed in worker.SEEDS:
                            spec = catalog["datasets"][dataset]["seeds"][str(seed)]
                            protocol = worker.read(worker.json_spec(spec, worker.HERE))
                            supports[seed] = [worker.normalized_support(row) for row in protocol["support_rows"]]
                        with np.load(first / "queries.npz", allow_pickle=False) as text:
                            positive, negative = (torch.tensor(text[key]) for key in ("positive_raw", "negative_raw"))
                            prompts = [text[key].tolist() for key in ("positive_prompts", "negative_prompts")]
                            classes = text["class_names"].tolist()
                        features = np.load(stable / "support_union_features.npy", allow_pickle=False)
                        if model == "BiomedCLIP":
                            local = np.broadcast_to(features[:, None], (len(features), len(classes), features.shape[-1])).copy()
                            global_features = features
                        else:
                            local = features
                            # This test exercises query fits and cache contracts;
                            # the global comparator vector never enters that fit.
                            global_features = features[:, 0]
                        output = Path(temporary) / f"{model}_{dataset}"
                        output.mkdir()
                        for seed in worker.SEEDS:
                            (output / f"seed_{seed}").mkdir()
                        args = argparse.Namespace(model=model, output=output)
                        job = dict(union=union, supports=supports, classes=classes, stable=stable)
                        arrays = dict(features=global_features, local=local,
                                      logits=np.zeros((len(union), len(classes)), dtype=np.float32))
                        queries, checks = worker.export_support(args, job, arrays, positive, negative, *prompts)
                        self.assertEqual(set(checks), {str(seed) for seed in worker.SEEDS})
                        lookup = {row["image"]: i for i, row in enumerate(union)}
                        for seed in worker.SEEDS:
                            folder = output / f"seed_{seed}"
                            with np.load(folder / "support.npz", allow_pickle=False) as support:
                                self.assertEqual(support["image_paths"].tolist(), [r["image"] for r in supports[seed]])
                                np.testing.assert_array_equal(support["labels"], [r["labels"] for r in supports[seed]])
                            with np.load(folder / "queries.npz", allow_pickle=False) as saved:
                                self.assertEqual(saved["class_names"].tolist(), classes)
                                np.testing.assert_allclose(saved["positive"], worker.unit(positive.numpy()), atol=1e-7, rtol=1e-7)
                                np.testing.assert_array_equal(saved["positive_query"], queries[seed]["positive_query"].numpy())
                            ids = [lookup[r["image"]] for r in supports[seed]]
                            labels = np.asarray([r["labels"] for r in supports[seed]], dtype=np.float32)
                            per_class = local[ids]
                            if model == "BiomedCLIP":
                                per_class = worker.unit(per_class)
                            means = (per_class * labels[..., None]).sum(0) / labels.sum(0)[:, None]
                            expected = worker.unit(worker.unit(positive.numpy()) + .3 * worker.unit(means))
                            if model == "BiomedCLIP":
                                expected *= np.linalg.norm(positive.numpy(), axis=-1, keepdims=True)
                            np.testing.assert_allclose(queries[seed]["positive_query"].numpy(), expected, atol=2e-7, rtol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_sha_rejects_modified_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"frozen")
            digest = worker.sha(path)
            worker.checked(path, digest)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                worker.checked(path, digest)

    def test_fit_rejects_wrong_support_order_and_wrong_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "seed_42"
            folder.mkdir()
            np.savez_compressed(folder / "query_fit.npz", queries=np.array([[1., 0.]], dtype=np.float32),
                                support_union_indices=np.array([2, 4]))
            job = dict(stable=root)
            with self.assertRaises(AssertionError):
                worker.verify_fit(job, 42, dict(queries=np.array([[1., 0.]])), np.array([4, 2]))
            with self.assertRaises(AssertionError):
                worker.verify_fit(job, 42, dict(queries=np.array([[0., 1.]])), np.array([2, 4]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
