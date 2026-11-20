"""Temporary-directory checks; never reads or mutates the real campaign queue."""
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import queue_append as qa


def job(root, identifier):
    return {"id": identifier, "model": "test", "dataset": identifier,
            "output": str(Path(root)/"outputs"/identifier), "argv": ["python", "never_launched.py"]}


def add_child(path, identifier):
    qa.append_jobs(path, [job(Path(path).parent, identifier)])


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="queue-append-test-")
        self.root = Path(self.temporary.name)
        self.queue = self.root/"queue_v2.json"
        self.queue.write_text(json.dumps({"note": "preserve me", "jobs": []}))

    def tearDown(self):
        self.temporary.cleanup()

    def test_append_and_idempotent_existing_output(self):
        first, second = job(self.root, "first"), job(self.root, "second")
        result = qa.append_jobs(self.queue, [first])
        self.assertEqual(result["added_ids"], ["first"])
        Path(first["output"]).mkdir(parents=True)
        result = qa.append_jobs(self.queue, [first, second, second])
        self.assertEqual(result["added_ids"], ["second"])
        self.assertEqual(result["skipped_ids"], ["first", "second"])
        saved = qa.read_json(self.queue)
        self.assertEqual(saved, {"note": "preserve me", "jobs": [first, second]})
        self.assertTrue((self.root/".queue_v2.append.lock").is_file())

    def test_conflict_is_atomic_and_type_sensitive(self):
        first = {**job(self.root, "first"), "is_smoke": False}
        qa.append_jobs(self.queue, [first])
        before = self.queue.read_bytes()
        changed = {**first, "is_smoke": 0}
        with self.assertRaises(qa.QueueConflict):
            qa.append_jobs(self.queue, [job(self.root, "new"), changed])
        self.assertEqual(self.queue.read_bytes(), before)
        occupied = job(self.root, "occupied")
        Path(occupied["output"]).mkdir(parents=True)
        with self.assertRaises(qa.QueueConflict):
            qa.append_jobs(self.queue, [occupied])
        collision = {**job(self.root, "other"), "output": first["output"]}
        with self.assertRaises(qa.QueueConflict):
            qa.append_jobs(self.queue, [collision])

    def test_multi_process_append_preserves_every_record(self):
        context = multiprocessing.get_context("fork")
        processes = [context.Process(target=add_child, args=(str(self.queue), f"p{i}")) for i in range(8)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        saved = qa.read_json(self.queue)["jobs"]
        self.assertEqual(len(saved), 8)
        self.assertEqual({item["id"] for item in saved}, {f"p{i}" for i in range(8)})

    def test_guard_and_outside_writer_prevent_commit(self):
        before = self.queue.read_bytes()
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("STOP before atomic replacement")
        with self.assertRaises(RuntimeError):
            qa.append_jobs(self.queue, [job(self.root, "new")], guard=guard)
        self.assertEqual(self.queue.read_bytes(), before)
        count = 0
        external = {"jobs": [], "external": True}
        def outside_writer():
            nonlocal count
            count += 1
            if count == 3:
                self.queue.write_text(json.dumps(external))
        with self.assertRaises(qa.QueueConflict):
            qa.append_jobs(self.queue, [job(self.root, "new")], guard=outside_writer)
        self.assertEqual(qa.read_json(self.queue), external)

    def test_source_validation_dryrun_and_fsync_order(self):
        source = self.root/"worker.py"
        source.write_text("# fixture")
        new = {**job(self.root, "new"), "source_sha256": {str(source): qa.file_sha256(source)}}
        before = self.queue.read_bytes()
        qa.append_jobs(self.queue, [new], dry_run=True)
        self.assertEqual(self.queue.read_bytes(), before)
        events = []
        fsync, replace = qa.os.fsync, qa.os.replace
        def sync_call(fd):
            events.append("fsync")
            return fsync(fd)
        def replace_call(src, dst):
            events.append("replace")
            return replace(src, dst)
        with patch.object(qa.os, "fsync", sync_call), patch.object(qa.os, "replace", replace_call):
            qa.append_jobs(self.queue, [new])
        self.assertEqual(events, ["fsync", "replace", "fsync"])
        source.write_text("# changed")
        bad = {**new, "id": "another", "output": str(self.root/"another")}
        with self.assertRaises(qa.QueueConflict):
            qa.append_jobs(self.queue, [bad])


if __name__ == "__main__":
    unittest.main(verbosity=2)
