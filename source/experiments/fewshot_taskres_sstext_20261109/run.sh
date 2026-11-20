#!/usr/bin/env bash
set -euo pipefail
taskres_sstext_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$taskres_sstext_dir/python_local.sh" "$taskres_sstext_dir/run_campaign.py" "$@"
