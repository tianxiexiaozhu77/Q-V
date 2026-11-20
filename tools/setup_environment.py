#!/usr/bin/env python3
"""Print/install a model's isolated environment; never changes an existing venv."""
import argparse,json,shlex,subprocess
from pathlib import Path
RELEASE=Path(__file__).resolve().parents[1]
specs={s['name']:s for s in json.loads((RELEASE/'environments/index.json').read_text())}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model',choices=sorted(specs))
p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--execute',action='store_true',help='Run installation; default only prints commands')
a=p.parse_args();s=specs[a.model];venv=a.workspace.resolve()/s['path'];python=venv/'bin/python'
commands=[['uv','python','install',s['python']],['uv','venv','--seed','--python',s['python'],str(venv)],
[str(python),'-m','pip','install','torch=='+s['torch']]+(['torchvision=='+s['torchvision']] if s['torchvision'] else [])+['--index-url','https://download.pytorch.org/whl/'+s['index']],
[str(python),'-m','pip','install','-r',str(RELEASE/s['requirements'])],
[str(python),'-m','pip','check'],
[str(python),'-c','import sys,torch,numpy; print(sys.version); print(torch.__version__, torch.version.cuda, numpy.__version__)']]
if a.execute and venv.exists():raise SystemExit(f'Refusing to modify existing environment: {venv}')
for command in commands:
 print(shlex.join(command),flush=True)
 if a.execute:subprocess.run(command,check=True)
