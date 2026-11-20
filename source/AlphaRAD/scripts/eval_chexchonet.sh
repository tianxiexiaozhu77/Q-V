#!/usr/bin/env bash
set -euo pipefail
alpharad_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$alpharad_root"
exec bash local_setup/python_local.sh local_setup/eval_chexchonet.py "$@"
