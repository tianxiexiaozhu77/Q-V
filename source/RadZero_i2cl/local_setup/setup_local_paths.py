"""Link existing server resources without copying or changing model/data files."""
import hashlib
import json
from pathlib import Path

from asset_paths import ASSETS_DIR, PROJECT_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def link(target, source):
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(f"Refusing to replace {target}")
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def main():
    workspace = PROJECT_ROOT.parent
    original = workspace / "RadZero"
    for folder in ("outputs", "splits", ".cache/huggingface", ".cache/matplotlib"):
        (ASSETS_DIR / folder).mkdir(parents=True, exist_ok=True)
    link(ASSETS_DIR / "data", workspace / "data/radzero/prepared")
    link(ASSETS_DIR / "data_images", workspace / "data/radzero/raw")
    link(ASSETS_DIR / "pretrained/StanfordAIMI/dinov2-base-xray-224",
         original / "dependencies/dinov2-base-xray-224")
    link(ASSETS_DIR / "pretrained/sentence-transformers/all-mpnet-base-v2",
         original / "dependencies/all-mpnet-base-v2")
    link(PROJECT_ROOT / ".venv", workspace / ".envs/radzero")
    model = ASSETS_DIR / "model"
    model.mkdir(exist_ok=True)
    source = original / "checkpoints/RadZero"
    for path in source.iterdir():
        if path.is_file():
            link(model / path.name, path)
    provenance = json.loads((PROJECT_ROOT / "local_setup/provenance.json").read_text())
    declared = next(item for item in provenance["models"] if item["repo"] == "Deepnoid/RadZero")
    previous = original / "runs/i2cl_latest/implementation_audit.json"
    expected = json.loads(previous.read_text())["base_checkpoint_sha256"]
    actual = sha256(source / "model.safetensors")
    if actual != expected:
        raise ValueError("Existing weights differ from the audited RadZero checkpoint")
    manifest = {
        "repo": declared["repo"], "revision": declared["revision"],
        "source": str(source),
        "provenance": "Reused local published weights, checked against the previous experiment audit; revision recorded in the transferred project's provenance.json. No new download.",
        "reference_audit": str(previous.resolve()),
        "files": [{"path": "model.safetensors", "hash": actual, "hash_algorithm": "sha256"}],
    }
    (model / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"assets": str(ASSETS_DIR), "data": str((ASSETS_DIR / "data").resolve()),
              "model": str(source), "environment": str((PROJECT_ROOT / ".venv").resolve()),
              "base_checkpoint_sha256": actual, "training_settings_changed": False}
    (ASSETS_DIR / "path_setup.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
