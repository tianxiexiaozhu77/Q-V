#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$root/../.cache/uv" TMPDIR="$root/../.cache/tmp"
export UV_LOCK_TIMEOUT=3600 UV_HTTP_TIMEOUT=120
export NO_PROXY="${NO_PROXY:-},pypi.org,files.pythonhosted.org"
export no_proxy="$NO_PROXY"
uv="$root/../.tools/bin/uv"
"$uv" venv --allow-existing --python /usr/bin/python3.10 "$root/.venv"
"$uv" pip install --python "$root/.venv/bin/python" 'torch==2.2.2' 'torchvision==0.17.2' 'gdown==6.4.2' 'setuptools<81' -r "$root/local_setup/requirements_eval.txt"
"$root/.venv/bin/python" - "$root" <<'PY'
import hashlib,json,pathlib,sys,urllib.request
root=pathlib.Path(sys.argv[1]);spec=json.loads((root/'local_setup/manifest.json').read_text())['text_encoder_assets']
dest=root/'local_assets/BioClinicalMPBERT';dest.mkdir(parents=True,exist_ok=True)
for name,expected in spec['files'].items():
 path=dest/name
 if not path.is_file():
  url=spec['repo']+'/resolve/'+spec['commit']+'/'+name
  with urllib.request.urlopen(url,timeout=120) as response:data=response.read()
  if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('Text asset hash mismatch: '+name)
  path.write_bytes(data)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Text asset hash mismatch: '+name)
PY
if [[ ! -f "$root/pretrain_model/CARZero_best_model.ckpt" ]]; then
 "$root/.venv/bin/python" "$root/local_setup/download_ranges.py" 'https://drive.usercontent.google.com/download?id=1kYF-k5otW5DHwz1En5d_ScV3zu2E27Ch&export=download&confirm=t' "$root/pretrain_model/CARZero_best_model.ckpt"
fi
