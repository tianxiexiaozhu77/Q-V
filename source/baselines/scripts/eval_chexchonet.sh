#!/usr/bin/env bash
set -euo pipefail
chexcho_baselines="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
chexcho_run="$chexcho_baselines/runs/chexchonet_$(date +%Y%m%d_%H%M%S)_$$"
exec python3 "$chexcho_baselines/local_setup/run_chexchonet.py" --output "$chexcho_run" "$@"
