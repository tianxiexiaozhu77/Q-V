"""Synthetic full-result fixtures exercise gate decisions without GPUs/daemons."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pilot_gate as gate
from queue_append import QueueError, read_json, file_sha256


REPORTER_PATH = Path(__file__).with_name("campaign_report.py")
spec = importlib.util.spec_from_file_location("gate_test_real_reporter", REPORTER_PATH)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pilot-gate-test-")
        self.root = Path(self.temporary.name)
        (self.root/"controller_v2").mkdir()
        self.deadline, self.now = 5000., 1000.
        self.sources = {}
        for phase in ("phase1", "phase9"):
            directory = self.root/("code_"+phase)
            directory.mkdir()
            for name in ("query_candidates.py", "eval_radzero_candidates.py", "eval_alpharad_candidates.py"):
                path = directory/name
                path.write_text(f"# synthetic unused source {phase}/{name}\n")
            self.sources[phase] = {str(path): file_sha256(path) for path in directory.iterdir()}
        self.baseline = [self.make_job("phase1", task) for task in reporter.TASKS]
        self.pilots = [self.make_job("phase9", task) for task in gate.PILOTS]
        self.jobs = self.baseline+self.pilots
        self.write("queue_v2.json", {"note": "keep order", "jobs": self.jobs})
        self.controller = {"status": "running", "pid": 999999,
                           "deadline": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
                           "jobs": {job["id"]: {"status": "complete", "job": copy.deepcopy(job)} for job in self.jobs}}
        self.write("controller_v2/state.json", self.controller)
        for job in self.baseline:
            self.write_result(job, {"original": (0., 0.), "uniform_eta03": (.1, .1)})
        for index, job in enumerate(self.pilots):
            improvement = 1.5 if index < 3 else .2
            self.write_result(job, {"original": (0., 0.), "uniform_eta03": (.1, .1),
                                    "shared_candidate": (improvement, improvement)})
        self.refresh_summary()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, data):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def make_job(self, phase, task):
        model, dataset = task
        output = self.root/"runs"/phase/f"{model}_{dataset}"
        worker = self.root/("code_"+phase)/f"eval_{model}_candidates.py"
        return {"id": f"{phase}_{model}_{dataset}", "model": model, "dataset": dataset,
                "is_smoke": False, "require_result": True, "cwd": str(self.root),
                "output": str(output), "command_marker": str(worker),
                "source_sha256": self.sources[phase],
                "argv": ["python", str(worker), "--dataset", dataset, "--output", str(output),
                         "--deadline", str(self.deadline), "--batch-size", "16" if model == "radzero" else "8",
                         "--candidate-module", str(self.root/("code_"+phase)/"query_candidates.py"),
                         "--cache-mode", "read", "--vision-cache-root", str(self.root/"assets") ]}

    def write_result(self, job, candidates):
        task = (job["model"], job["dataset"])
        metric_names = reporter.METRICS[reporter.TASKS[task]]
        values = {name: {"metrics": {metric: .5+improvements[i]/100
                                     for i, metric in enumerate(metric_names)}} for name, improvements in candidates.items()}
        proofs = {name: {"passed": True, "checked_images": 100, "images_checked": 100}
                  for name in ("original", "uniform_eta03")}
        if job["model"] == "radzero":
            checks = {**proofs, "original_helper": {"passed": True}}
        else:
            checks = {"formal_prediction_controls": proofs, "author_helper": {"passed": True}}
        result = {"status": "complete", "is_smoke": False, "model": job["model"], "dataset": job["dataset"],
                  "task": reporter.TASKS[task], "rows": 100, "support_count": 10,
                  "candidates": values, "baseline_checks": checks, "config": {"source_sha256": job["source_sha256"]}}
        path = Path(job["output"])
        path.mkdir(parents=True, exist_ok=True)
        (path/"result.json").write_text(json.dumps(result))
        if job["model"] == "alpharad":
            (path/"data_audit.json").write_text(json.dumps({"source_sha256": job["source_sha256"]}))

    def refresh_summary(self):
        phase, _ = reporter.summarize_phase(self.root/"runs/phase9")
        self.write("summary.json", {"expected_tasks": 15, "expected_pairs": 14, "phases": [phase]})

    def run_gate(self, **kwargs):
        return gate.run_once(self.root, phases=["phase9"], deadline=self.deadline,
                             now=lambda: self.now, controller_live=lambda root, state: True,
                             reporter=reporter, **kwargs)

    def test_pass_appends_remaining_ten_with_all_candidates_and_idempotence(self):
        before = copy.deepcopy(self.jobs)
        result = self.run_gate()
        self.assertEqual(result["phases"]["phase9"]["status"], "expanded")
        jobs = read_json(self.root/"queue_v2.json")["jobs"]
        self.assertEqual(jobs[:len(before)], before)
        self.assertEqual(len(jobs), len(before)+10)
        expected = set(reporter.TASKS)-set(gate.PILOTS)
        self.assertEqual({(j["model"], j["dataset"]) for j in jobs[len(before):]}, expected)
        for job in jobs[len(before):]:
            self.assertNotIn("--candidate-names", job["argv"])
            self.assertEqual(gate._argument(job["argv"], "--cache-mode"), "read")
            self.assertFalse(Path(job["output"]).exists())
        again = self.run_gate()
        self.assertEqual(again["phases"]["phase9"]["status"], "already_queued")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], jobs)

    def test_cross_candidate_winners_cannot_be_mixed_and_both_metrics_required(self):
        for index, job in enumerate(self.pilots):
            self.write_result(job, {"original": (0., 0.), "uniform_eta03": (.1, .1),
                "candidate_a": ((1.5, 1.5) if index < 4 else (2., -.1)),
                "candidate_b": ((-.1, 2.) if index < 4 else (1.5, 1.5))})
        self.refresh_summary()
        result = self.run_gate()
        self.assertEqual(result["phases"]["phase9"]["status"], "rejected")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_pending_failed_smoke_and_truncated_pilots_never_expand(self):
        identifier = self.pilots[-1]["id"]
        self.controller["jobs"][identifier]["status"] = "running"
        self.write("controller_v2/state.json", self.controller)
        self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "waiting")
        self.controller["jobs"][identifier]["status"] = "failed"
        self.write("controller_v2/state.json", self.controller)
        self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "rejected")
        self.controller["jobs"][identifier]["status"] = "complete"
        self.write("controller_v2/state.json", self.controller)
        path = Path(self.pilots[-1]["output"])/"result.json"
        original = read_json(path)
        for changed in ({**original, "is_smoke": True}, {**original, "rows": 32}, {**original, "support_count": 20}):
            path.write_text(json.dumps(changed))
            self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "error")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_stale_or_tampered_summary_and_bad_control_fail_closed(self):
        path = Path(self.pilots[-1]["output"])/"result.json"
        result = read_json(path)
        result["candidates"]["shared_candidate"]["metrics"]["macro_auroc"] = .49
        path.write_text(json.dumps(result))
        self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "waiting")
        result["baseline_checks"]["original"]["passed"] = False
        path.write_text(json.dumps(result))
        self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "error")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_wrong_pairs_and_counts_are_rejected_by_gate_contract(self):
        phase, _ = reporter.summarize_phase(self.root/"runs/phase9")
        changed = copy.deepcopy(phase)
        record = changed["candidates"]["shared_candidate"]
        record["per_pair_min_gain_pp"]["alpharad/wrong"] = record["per_pair_min_gain_pp"].pop("alpharad/node21")
        with self.assertRaises(QueueError):
            gate.eligible_candidates(changed)
        changed = copy.deepcopy(phase)
        changed["candidates"]["shared_candidate"]["main_metric_count"] = 5
        with self.assertRaises(QueueError):
            gate.eligible_candidates(changed)

    def test_source_mismatch_and_existing_output_cannot_append(self):
        target = self.root/"runs/phase9/alpharad_chest14"
        target.mkdir(parents=True)
        self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "error")
        target.rmdir()
        (self.root/"code_phase9/query_candidates.py").write_text("# changed")
        self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "error")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_stop_deadline_terminal_and_dead_controller_exit_without_append(self):
        (self.root/"STOP_v2").touch()
        self.assertEqual(self.run_gate()["status"], "stopped")
        (self.root/"STOP_v2").unlink()
        self.now = self.deadline
        self.assertEqual(self.run_gate()["status"], "stopped")
        self.now = 1000.
        self.controller["status"] = "stopped"
        self.write("controller_v2/state.json", self.controller)
        self.assertEqual(self.run_gate()["status"], "stopped")
        self.controller["status"] = "running"
        self.write("controller_v2/state.json", self.controller)
        result = gate.run_once(self.root, phases=["phase9"], deadline=self.deadline,
                               now=lambda: self.now, controller_live=lambda root, state: False,
                               reporter=reporter)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_dry_run_config_scope_and_mutated_template(self):
        result = self.run_gate(dry_run=True)
        self.assertEqual(result["phases"]["phase9"]["status"], "eligible_dry_run")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)
        with self.assertRaises(QueueError):
            gate.run_once(self.root, phases=["phase6"], reporter=reporter)
        self.pilots[-1]["argv"] += ["--candidate-names", "shared_candidate"]
        self.write("queue_v2.json", {"jobs": self.jobs})
        self.assertEqual(self.run_gate()["phases"]["phase9"]["status"], "error")

    def test_changed_evidence_or_stop_between_decision_and_commit_cancels(self):
        append = gate.append_jobs
        path = Path(self.pilots[-1]["output"])/"result.json"
        original = path.read_bytes()
        def mutate_before_append(*args, **kwargs):
            value = read_json(path)
            value["rows"] = 32
            path.write_text(json.dumps(value))
            return append(*args, **kwargs)
        with patch.object(gate, "append_jobs", mutate_before_append):
            result = self.run_gate()
        self.assertEqual(result["status"], "error")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)
        path.write_bytes(original)
        def stop_before_append(*args, **kwargs):
            (self.root/"STOP_v2").touch()
            return append(*args, **kwargs)
        with patch.object(gate, "append_jobs", stop_before_append):
            result = self.run_gate()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
