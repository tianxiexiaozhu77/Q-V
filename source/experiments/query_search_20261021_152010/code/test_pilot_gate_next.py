"""CPU-only synthetic gate tests; all queue mutations are inside temp dirs."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pilot_gate_next as gate
from queue_append import QueueError, read_json, file_sha256

spec = importlib.util.spec_from_file_location("next_gate_real_reporter", Path(__file__).with_name("campaign_report.py"))
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pilot-gate-next-test-")
        self.root = Path(self.temp.name)
        self.deadline, self.now = 5000., 1000.
        self.build()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2))

    def build(self, phase="phase15", keep_global=False):
        self.phase, self.keep_global, self.sources = phase, keep_global, {}
        for name in ("phase6", phase):
            directory = self.root/("code_"+name)
            directory.mkdir(exist_ok=True)
            paths = [directory/"eval_radzero_candidates.py", directory/"eval_alpharad_candidates.py"]
            candidate = (self.root/"code_multiview_prototype/query_candidates.py" if name in ("phase17", "phase17b")
                         else directory/"query_candidates.py")
            candidate.parent.mkdir(exist_ok=True)
            paths.append(candidate)
            for path in paths:
                path.write_text("# synthetic unused source " + str(path) + "\n")
            self.sources[name] = {str(path): file_sha256(path) for path in paths}
        # Deliberately noncanonical task order: expansion must preserve phase6 order.
        self.baseline = [self.job("phase6", task) for task in reversed(list(reporter.TASKS))]
        self.pilots = [self.job(phase, task) for task in gate.PILOTS]
        self.jobs = self.baseline+self.pilots
        self.write("queue_v2.json", {"preserve": "opaque queue metadata", "jobs": self.jobs})
        self.controller = {"status": "running", "pid": 999999,
            "deadline": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
            "jobs": {j["id"]: {"status": "complete", "job": copy.deepcopy(j)} for j in self.jobs}}
        self.write("controller_v2/state.json", self.controller)
        self.freeze = {"status": "complete", "source_sha256": self.sources[phase], "jobs": self.pilots}
        if phase in ("phase17", "phase17b"):
            self.freeze["canonical_candidate_module"] = str(self.root/"code_multiview_prototype/query_candidates.py")
        self.write("audits/"+phase+"_freeze.json", self.freeze)
        for job in self.baseline:
            self.result(job, {"original": (0., 0.), "uniform_eta03": (.1, .1)})
        for i, job in enumerate(self.pilots):
            gain = 1.5 if i < 3 else .2
            self.result(job, {"original": (0., 0.), "uniform_eta03": (.1, .1),
                             "common": (gain, gain), "rejected": (-.1, 2.)})
        self.refresh()

    def job(self, phase, task):
        model, dataset = task
        worker = self.root/("code_"+phase)/f"eval_{model}_candidates.py"
        candidate = (self.root/"code_multiview_prototype/query_candidates.py" if phase in ("phase17", "phase17b")
                     else self.root/("code_"+phase)/"query_candidates.py")
        output = self.root/"runs"/phase/f"{model}_{dataset}"
        argv = ["python", str(worker), "--dataset", dataset, "--output", str(output),
            "--deadline", str(self.deadline), "--batch-size", "16" if model == "radzero" else "8",
            "--candidate-module", str(candidate), "--cache-mode", "read", "--vision-cache-root", str(self.root/"assets")]
        if model == "radzero" and (phase == "phase6" or self.keep_global):
            argv += ["--global-support-features", str(self.root/"global"/dataset/"global_features.npy")]
        if model == "alpharad":
            argv += ["--save-full-maps"]
        return {"id": gate._job_id(phase, task), "model": model, "dataset": dataset,
            "is_smoke": False, "require_result": True, "cwd": str(self.root),
            "output": str(output), "command_marker": str(worker), "argv": argv,
            "source_sha256": self.sources[phase]}

    def result(self, job, candidates):
        task = (job["model"], job["dataset"])
        metrics = reporter.METRICS[reporter.TASKS[task]]
        values = {name: {"metrics": {key: .5+gains[i]/100 for i,key in enumerate(metrics)}}
                  for name,gains in candidates.items()}
        proof = {name: {"passed": True, "checked_images": 100, "images_checked": 100} for name in gate.CONTROLS}
        checks = ({**proof, "original_helper": {"passed": True}} if task[0] == "radzero" else
                  {"formal_prediction_controls": proof, "author_helper": {"passed": True}})
        result = {"status": "complete", "is_smoke": False, "model": task[0], "dataset": task[1],
            "task": reporter.TASKS[task], "rows": 100, "support_count": 10,
            "candidates": values, "baseline_checks": checks, "config": {"source_sha256": job["source_sha256"]}}
        path = Path(job["output"])
        path.mkdir(parents=True, exist_ok=True)
        (path/"result.json").write_text(json.dumps(result))
        if task[0] == "alpharad":
            (path/"data_audit.json").write_text(json.dumps({"source_sha256": job["source_sha256"]}))

    def refresh(self):
        phase, _ = reporter.summarize_phase(self.root/"runs"/self.phase)
        self.write("summary.json", {"expected_tasks": 15, "expected_pairs": 14, "phases": [phase]})

    def run_gate(self, **kwargs):
        return gate.run_once(self.root, [self.phase], self.deadline, now=lambda: self.now,
            controller_live=lambda root,state: True, reporter=reporter, **kwargs)

    def decision(self, **kwargs):
        return self.run_gate(**kwargs)["phases"][self.phase]

    def test_default_is_dry_run_and_retains_all_candidate_evidence(self):
        before = (self.root/"queue_v2.json").read_bytes()
        decision = self.decision()
        self.assertEqual(decision["status"], "eligible_dry_run")
        self.assertEqual(decision["passing_shared_candidates"], ["common"])
        self.assertEqual(set(decision["candidate_evidence"]), {"original", "uniform_eta03", "common", "rejected"})
        self.assertEqual(decision["append_result"]["new_job_count"], 10)
        self.assertEqual((self.root/"queue_v2.json").read_bytes(), before)
        self.assertFalse((self.root/"pilot_gate_next_state.json").exists())
        self.assertFalse((self.root/"runs"/self.phase/"radzero_OpenI").exists())

    def test_commit_keeps_order_actual_cli_and_is_exactly_idempotent(self):
        result = self.decision(commit=True)
        self.assertEqual(result["status"], "expanded")
        jobs = read_json(self.root/"queue_v2.json")["jobs"]
        self.assertEqual(jobs[:20], self.jobs)
        self.assertEqual([gate._task(j) for j in jobs[20:]], [gate._task(j) for j in self.baseline if gate._task(j) not in gate.PILOTS])
        for job in jobs[20:]:
            self.assertNotIn("--candidate-names", job["argv"])
            self.assertNotIn("--global-support-features", job["argv"])
            self.assertEqual(job["source_sha256"], self.sources[self.phase])
            self.assertIn("code_"+self.phase, job["command_marker"])
        self.assertEqual(self.decision(commit=True)["status"], "already_queued")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], jobs)

    def test_phase17_canonical_module_and_future_phases(self):
        for phase in ("phase12", "phase14", "phase16", "phase18", "phase17", "phase17b"):
            self.build(phase)
            self.assertEqual(self.decision()["status"], "eligible_dry_run")
        self.assertEqual(self.decision(commit=True)["status"], "expanded")
        jobs = read_json(self.root/"queue_v2.json")["jobs"]
        for job in jobs[20:]:
            self.assertEqual(gate._argument(job["argv"], "--candidate-module"), self.freeze["canonical_candidate_module"])

    def test_optional_global_input_uses_corresponding_phase6_dataset(self):
        self.build("phase12", keep_global=True)
        self.assertEqual(self.decision(commit=True)["status"], "expanded")
        for job in read_json(self.root/"queue_v2.json")["jobs"][20:]:
            if job["model"] == "radzero":
                self.assertEqual(gate._argument(job["argv"], "--global-support-features"),
                    str(self.root/"global"/job["dataset"]/"global_features.npy"))

    def test_old_controls_cannot_authorize_expansion(self):
        for job in self.baseline:
            self.result(job, {"original": (0.,0.), "uniform_eta03": (2.,2.)})
        for job in self.pilots:
            self.result(job, {"original": (0.,0.), "uniform_eta03": (2.,2.), "bad": (-.1,-.1)})
        self.refresh()
        decision = self.decision(commit=True)
        self.assertEqual(decision["status"], "rejected")
        self.assertFalse(decision["candidate_evidence"]["uniform_eta03"]["eligible"])
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_no_stitching_and_two_metric_requirement(self):
        for i,job in enumerate(self.pilots):
            self.result(job, {"original": (0.,0.), "uniform_eta03": (.1,.1),
                "a": (1.5,1.5) if i<4 else (2.,-.1), "b": (-.1,-.1) if i<4 else (1.5,1.5)})
        self.refresh()
        self.assertEqual(self.decision(commit=True)["status"], "rejected")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_three_pairs_at_one_point_is_required_not_just_positive_average(self):
        for i,job in enumerate(self.pilots):
            gain = 2. if i < 2 else .99
            self.result(job, {"original": (0.,0.), "uniform_eta03": (.1,.1), "common": (gain,gain)})
        self.refresh()
        decision = self.decision()
        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(decision["candidate_evidence"]["common"]["positive_pairs"], 5)
        self.assertEqual(decision["candidate_evidence"]["common"]["maximum_possible_pairs_ge_1pp"], 11)
        self.result(self.pilots[2], {"original": (0.,0.), "uniform_eta03": (.1,.1), "common": (1.,1.)})
        self.refresh()
        self.assertEqual(self.decision()["status"], "eligible_dry_run")

    def test_missing_freeze_or_pending_pilot_waits_failed_pilot_rejects(self):
        path = self.root/"audits"/(self.phase+"_freeze.json")
        content = path.read_bytes();path.unlink()
        self.assertEqual(self.decision()["status"], "waiting")
        path.write_bytes(content)
        ident = self.pilots[-1]["id"]
        for status,expected in (("running","waiting"),("failed","rejected")):
            self.controller["jobs"][ident]["status"] = status
            self.write("controller_v2/state.json", self.controller)
            self.assertEqual(self.decision()["status"], expected)

    def test_smoke_truncated_missing_controls_and_false_fullN_fail_closed(self):
        path = Path(self.pilots[-1]["output"])/"result.json"
        original = read_json(path)
        bad = copy.deepcopy(original);bad["baseline_checks"]["original"]["checked_images"] = 99
        for result in ({**original,"is_smoke":True},{**original,"rows":99}, {**original,"support_count":20},bad):
            path.write_text(json.dumps(result))
            self.assertEqual(self.decision(commit=True)["status"], "error")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_stale_summary_and_wrong_pilot_pairs(self):
        path = Path(self.pilots[-1]["output"])/"result.json"
        result = read_json(path);result["candidates"]["common"]["metrics"]["macro_auroc"] = .49
        path.write_text(json.dumps(result))
        self.assertEqual(self.decision()["status"], "waiting")
        phase,_ = reporter.summarize_phase(self.root/"runs"/self.phase)
        phase["candidates"]["common"]["per_pair_min_gain_pp"]["wrong"] = 1.
        with self.assertRaises(QueueError):gate.candidate_evidence(phase)

    def test_bad_phase_audit_cli_source_and_existing_output_fail_closed(self):
        self.freeze["jobs"][0]["argv"] += ["--candidate-names", "common"]
        self.write("audits/"+self.phase+"_freeze.json", self.freeze)
        self.assertEqual(self.decision()["status"], "error")
        self.build()
        target = self.root/"runs"/self.phase/"radzero_OpenI";target.mkdir()
        self.assertEqual(self.decision(commit=True)["status"], "error")
        target.rmdir()
        Path(next(iter(self.sources[self.phase]))).write_text("# altered source")
        self.assertEqual(self.decision(commit=True)["status"], "error")

    def test_conflicting_existing_id_rejected_identical_partial_append_skipped(self):
        frozen = gate.frozen_templates(self.root,self.phase,self.jobs,self.deadline)
        expected = gate.expansion_jobs(self.root,self.phase,self.baseline,frozen)
        self.write("queue_v2.json", {"jobs": self.jobs+expected[:1]})
        result = self.decision(commit=True)
        self.assertEqual(result["append_result"]["new_job_count"], 9)
        self.assertEqual(result["append_result"]["skipped_ids"], [expected[0]["id"]])
        jobs = read_json(self.root/"queue_v2.json")["jobs"]
        jobs[-1]["argv"] += ["--changed"]
        self.write("queue_v2.json", {"jobs": jobs})
        self.assertEqual(self.decision(commit=True)["status"], "error")

    def test_evidence_race_and_stop_are_checked_inside_shared_lock(self):
        path = self.root/"audits"/(self.phase+"_freeze.json")
        original = path.read_bytes();append = gate.append_jobs
        def change(*args,**kwargs):
            data = read_json(path);data["changed"] = True;path.write_text(json.dumps(data))
            return append(*args,**kwargs)
        with patch.object(gate,"append_jobs",change):
            self.assertEqual(self.run_gate(commit=True)["status"], "error")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)
        path.write_bytes(original)
        def stop(*args,**kwargs):
            (self.root/"STOP_v2").touch();return append(*args,**kwargs)
        with patch.object(gate,"append_jobs",stop):
            self.assertEqual(self.run_gate(commit=True)["status"], "stopped")
        self.assertEqual(read_json(self.root/"queue_v2.json")["jobs"], self.jobs)

    def test_stop_deadline_terminal_and_dead_controller(self):
        (self.root/"STOP_v2").touch();self.assertEqual(self.run_gate()["status"], "stopped")
        (self.root/"STOP_v2").unlink();self.now=self.deadline
        self.assertEqual(self.run_gate()["status"], "stopped")
        self.now=1000.;self.controller["status"]="complete"
        self.write("controller_v2/state.json",self.controller)
        self.assertEqual(self.run_gate()["status"], "stopped")
        self.controller["status"]="running";self.write("controller_v2/state.json",self.controller)
        self.assertEqual(gate.run_once(self.root,[self.phase],self.deadline,now=lambda:1000.,
            controller_live=lambda *_:False,reporter=reporter)["status"],"stopped")

    def test_cli_default_never_commits_and_old_phase_is_not_supported(self):
        with patch.object(gate,"run_once",return_value={"status":"finished"}) as run:
            gate.main(["--no-state-file"]);self.assertIs(run.call_args.kwargs["commit"],False)
            gate.main(["--commit","--no-state-file"]);self.assertIs(run.call_args.kwargs["commit"],True)
        for phases in (["phase9"],["phase15","phase15"],[],"phase15"):
            with self.assertRaises(QueueError):gate.run_once(self.root,phases,self.deadline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
