"""CPU-only tests for sealed identities, support fitting inputs, and ownership."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import infer_siim as worker


class SIIMInputTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index in range(13):
            path = self.root / f"image_{index}.png"
            path.write_bytes(f"unique_image_{index}".encode())
            self.rows.append(dict(image=str(path), group_id=f"patient_{index}",
                                  byte_sha256=worker.sha(path)))
        self.support = [{**row, "labels": [int(i < 5)]} for i, row in enumerate(self.rows[:10])]
        self.identity = self.root / "test_identity.json"
        self.identity.write_text(json.dumps({"rows": self.rows[10:]}))
        self.catalog = dict(dataset="siim_new_holdout", class_names=["Pneumothorax"], eta=.3,
            test_count=3, test_identity_manifest=dict(path=self.identity.name, sha256=worker.sha(self.identity)),
            evaluation_manifest=dict(path="never_open_evaluation.json", sha256="does-not-exist"),
            seeds={str(seed): dict(support_rows=deepcopy(self.support)) for seed in worker.SEEDS})
        self.protocol = self.root / "catalog.json"
        self.save()

    def save(self):
        self.protocol.write_text(json.dumps(self.catalog))

    def test_loading_does_not_open_or_require_evaluation_manifest(self):
        result = worker.load_protocol(self.protocol, smoke=2)
        self.assertEqual(len(result["test"]), 2)
        self.assertEqual(len(result["test_all"]), 3)
        self.assertEqual(len(result["union"]), 10)
        self.assertFalse(any("evaluation" in path for path in result["sources"]))
        self.assertTrue(all("labels" not in row for row in result["test_all"]))

    def test_rejects_patient_overlap(self):
        self.catalog["seeds"]["42"]["support_rows"][0]["group_id"] = "patient_10"
        self.save()
        with self.assertRaisesRegex(ValueError, "share a DICOM"):
            worker.load_protocol(self.protocol)

    def test_rejects_changed_bytes_and_test_labels(self):
        Path(self.rows[0]["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            worker.load_protocol(self.protocol)
        leaked = deepcopy(self.rows[10:])
        leaked[0]["labels"] = [1]
        self.identity.write_text(json.dumps({"rows": leaked}))
        self.catalog["test_identity_manifest"]["sha256"] = worker.sha(self.identity)
        self.save()
        with self.assertRaisesRegex(ValueError, "no labels"):
            worker.load_protocol(self.protocol)

    def test_rejects_unbalanced_support_and_label_disagreement(self):
        self.catalog["seeds"]["42"]["support_rows"][0]["labels"] = [0]
        self.save()
        with self.assertRaisesRegex(ValueError, "five positive"):
            worker.load_protocol(self.protocol)
        self.catalog["seeds"]["42"]["support_rows"][5]["labels"] = [1]
        self.save()
        with self.assertRaisesRegex(ValueError, "Cross-seed"):
            worker.load_protocol(self.protocol)

    def test_rejects_declared_pixel_duplicate_with_different_png_bytes(self):
        rows = deepcopy(self.rows[10:])
        rows[0]["pixel_sha256"] = "same_decoded_pixels"
        self.identity.write_text(json.dumps({"rows": rows}))
        self.catalog["test_identity_manifest"]["sha256"] = worker.sha(self.identity)
        self.catalog["seeds"]["42"]["support_rows"][0]["pixel_sha256"] = "same_decoded_pixels"
        self.save()
        with self.assertRaisesRegex(ValueError, "share a DICOM"):
            worker.load_protocol(self.protocol)

    def test_existing_output_is_never_modified_on_start_failure(self):
        output = self.root / "existing"
        output.mkdir()
        failure = output / "failure.json"
        failure.write_text("original preserved result")
        with patch.object(worker, "run") as run:
            with self.assertRaises(FileExistsError):
                worker.run_owned(SimpleNamespace(output=output))
            run.assert_not_called()
        self.assertEqual(failure.read_text(), "original preserved result")
        self.assertEqual([p.name for p in output.iterdir()], ["failure.json"])

    def test_owned_new_output_records_failure(self):
        output = self.root / "new"
        with patch.object(worker, "run", side_effect=RuntimeError("deliberate")):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                worker.run_owned(SimpleNamespace(output=output))
        record = json.loads((output / "failure.json").read_text())
        self.assertEqual(record["error"], "deliberate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
