#!/usr/bin/env bash
set -euo pipefail
taskres_sstext_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
taskres_sstext_python="${TASKRES_SSTEXT_BASE_PYTHON:-/opt/anaconda3/bin/python}"
if [[ ! -x "$taskres_sstext_dir/.venv/bin/python" ]]; then
  "$taskres_sstext_python" -m venv --system-site-packages "$taskres_sstext_dir/.venv"
fi
"$taskres_sstext_dir/.venv/bin/python" -m pip install -r "$taskres_sstext_dir/requirements.txt"
CUDA_VISIBLE_DEVICES='' bash "$taskres_sstext_dir/python_local.sh" -c \
 'import torch,numpy,sklearn,omegaconf; assert not torch.cuda.is_initialized(); print("Solver environment ready:",torch.__version__,numpy.__version__,sklearn.__version__)'
