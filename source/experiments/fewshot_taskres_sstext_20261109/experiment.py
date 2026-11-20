"""Shared file/provenance helpers; importing this module does not use CUDA."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from data_interface import object_sha256, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("original", "query", "view", "query_view", "feature_text_zero_shot",
           "taskres", "sstext_plus")
REPOS = {
    "TaskRes": (ROOT / "baselines/TaskRes", "https://github.com/geekyutao/TaskRes",
                "5195359661d6bef661090c33b81aacfc7d7cc181"),
    "SS-Text": (ROOT / "baselines/SS-Text", "https://github.com/jusiro/SS-Text",
                "4bc0c9e3b7b9e0f090079bcc51d7aa852806e91f"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_sources(sources):
    for path, digest in sources.items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError("Frozen experiment source/artifact changed: " + path)


def verify_repositories():
    output = {}
    for name, (path, url, commit) in REPOS.items():
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                         "--untracked-files=no"], text=True).strip()
        if actual != commit or dirty:
            raise ValueError(f"Official source checkout differs from pinned commit: {name}")
        files = subprocess.check_output(["git", "-C", str(path), "ls-files"], text=True).splitlines()
        output[name] = dict(path=str(path), url=url, commit=commit,
                           files={str(path / filename): sha256(path / filename) for filename in files})
    return output


def expected_units(config):
    return [f"{model}_{dataset}/seed_{seed}" for model in config["models"]
            for dataset in config["datasets"] for seed in config["seeds"]]


def load_snapshot(run):
    path = Path(run) / "experiment_snapshot.json"
    snapshot = read_json(path)
    verify_sources(snapshot["source_sha256"])
    return snapshot


def verify_fit_commit(folder, snapshot, snapshot_sha256, unit_manifest=None):
    folder = Path(folder)
    commit = read_json(folder / "fit_commit.json")
    if commit["run_snapshot_sha256"] != snapshot_sha256 or commit["test_labels_opened"] is not False:
        raise ValueError("Fit commit has wrong run or test-label boundary")
    if commit["source_sha256"] != snapshot["source_sha256"]:
        raise ValueError("Fit commit source snapshot differs")
    if unit_manifest is not None and commit["unit_manifest_sha256"] != object_sha256(unit_manifest):
        raise ValueError("Fit data provenance differs from frozen unit")
    required = {"fit_info.json", "predictions.npz", "taskres_state.npz", "sstext_state.npz"}
    if set(commit["artifacts"]) != required:
        raise ValueError("Fit commit missing fitted states or predictions")
    for name, digest in commit["artifacts"].items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError("Committed fit artifact changed: " + name)
    return commit
