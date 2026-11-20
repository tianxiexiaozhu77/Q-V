#!/usr/bin/env python3
"""Maintainer-only: collect aggregate evidence and environment versions."""
import hashlib,json,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]; ROOT=OUT.parent
selected=[
'experiments/sensitivity_grid/campaign_20261108_v3/sensitivity_grid_summary_corrected.csv',
'experiments/qata_validation_threshold_20261024/runs/validation_selected_20261024_v1/evaluation/RESULTS.csv']
for rel in selected:
    p=OUT/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
envs=json.loads((ROOT/'baselines/paper_draft_zh/ENVIRONMENTS_VERIFIED.json').read_text())
(OUT/'ENVIRONMENTS.json').write_text(json.dumps([{k:r[k] for k in ['model','python','packages','torch_build'] if k in r} for r in envs],indent=2)+'\n')
v=ROOT/'baselines/paper_draft_zh/verified_results.json'
d=json.loads(v.read_text());records=[]
for r in d['records']:
    row={k:r[k] for k in ['model','dataset','status','source_sha256','evaluation_manifest_sha256'] if k in r}
    for k in ['source_path','evaluation_manifest']:
        if k in r:
            p=Path(r[k]);row[k]=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name
    records.append(row)
(OUT/'RESULT_PROVENANCE.json').write_text(json.dumps({'original_verified_results_sha256':hashlib.sha256(v.read_bytes()).hexdigest(),'records':records},indent=2)+'\n')
manifest=[]
for p in sorted((OUT/'source').rglob('*')):
    if p.is_file():manifest.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
import re
issues=[]
for p in sorted((OUT/'source').rglob('*')):
    if p.suffix not in {'.py','.sh'}:continue
    for n,line in enumerate(p.read_text().splitlines(),1):
        reasons=[]
        if re.search(r'/(storage|home|tmp|nfs)/',line):reasons.append('machine_path')
        if re.search(r'\{4, 5, 6, 7\}|Only.*GPU|authorized',line):reasons.append('local_policy_candidate')
        if reasons:issues.append(dict(path=str(p.relative_to(OUT)),line=n,kind=','.join(reasons),code=line.strip()))
(OUT/'PORTABILITY_AUDIT.json').write_text(json.dumps(issues,indent=2)+'\n')
