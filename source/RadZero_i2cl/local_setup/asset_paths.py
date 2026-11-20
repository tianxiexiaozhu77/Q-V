"""Paths for the transferred project; preserve the source training settings."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(
    os.environ.get("RADZERO_ASSETS", PROJECT_ROOT / "local_assets")
).expanduser().resolve()
