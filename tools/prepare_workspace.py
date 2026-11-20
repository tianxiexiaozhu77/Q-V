#!/usr/bin/env python3
"""Restore pinned upstream checkouts then overlay release code into a NEW workspace."""
import argparse,json,shlex,shutil,subprocess
from pathlib import Path
RELEASE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
root=a.workspace.resolve()
if root.exists():raise SystemExit('Choose a new workspace; existing research files will not be overwritten.')
repos=json.loads((RELEASE/'resources/MODEL_ASSETS.json').read_text())
repos += [dict(repository='https://github.com/geekyutao/TaskRes',revision='5195359661d6bef661090c33b81aacfc7d7cc181',checkout_path='baselines/TaskRes'),dict(repository='https://github.com/jusiro/SS-Text',revision='4bc0c9e3b7b9e0f090079bcc51d7aa852806e91f',checkout_path='baselines/SS-Text')]
for s in repos:
 dest=root/s['checkout_path']
 for cmd in [['git','clone',s['repository'],str(dest)],['git','-C',str(dest),'checkout','--detach',s['revision']]]:
  print(shlex.join(cmd),flush=True)
  if a.execute:
   dest.parent.mkdir(parents=True,exist_ok=True);subprocess.run(cmd,check=True)
print('Overlay release source/ into',root)
if a.execute:shutil.copytree(RELEASE/'source',root,dirs_exist_ok=True)
print('Weights, datasets, protocols, source compatibility patches and authenticated caches still need preparation; see README.md.')
