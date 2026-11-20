#!/usr/bin/env python3
"""Rebuild generated tables from aggregate snapshots in an isolated directory."""
import argparse, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'build')
args=parser.parse_args()
output=args.output.resolve()
if output.exists(): raise SystemExit('Output already exists; choose a new --output directory.')
with tempfile.TemporaryDirectory(prefix='paper-tables-') as tmp:
    work=Path(tmp)/'source';shutil.copytree(ROOT/'source',work)
    for name in ['build_supplement.py','build_classification_table.py']:
        subprocess.run([sys.executable,str(work/'latex_ours/scripts'/name)],check=True)
    files=[]
    for p in (work/'latex_ours/sec').rglob('*.tex'):
        original=ROOT/'source'/p.relative_to(work)
        # Only scripts' generated files; other tables are frozen source snapshots.
        if p.name == 'e_sensitivity.tex': continue
        if p.parent.name!='supplement_tables' and not p.name.startswith(('classification_comparison','component_ablations')): continue
        generated=p.read_bytes(); matches=original.exists() and generated==original.read_bytes()
        files.append(dict(file=str(p.relative_to(work)),matches_snapshot=matches,sha256=hashlib.sha256(generated).hexdigest()))
    output.mkdir(parents=True)
    shutil.copytree(work/'latex_ours/sec',output/'sec')
    shutil.copytree(work/'latex_ours/build',output/'audits')
    (output/'verification.json').write_text(json.dumps(files,indent=2)+'\n')
    failures=[r for r in files if (ROOT/'source'/r['file']).exists() and not r['matches_snapshot']]
    print(json.dumps({'checked_files':len(files),'snapshot_mismatches':failures},indent=2))
    if failures:raise SystemExit(1)
