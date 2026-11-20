#!/usr/bin/env python3
"""Collect manuscript-linked code without changing the research workspace."""
import csv, hashlib, json, re, shutil, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'latex_ours'
def copy(p):
    p = Path(p)
    if not p.is_file() or p.is_symlink(): return
    dest = OUT / 'source' / p.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
# Preserve relative source layout: native adapters import across these directories.
experiments = ['fewshot_multiseed_20261024','fewshot_native_query','fewshot_neighbors_20261025',
 'fewshot_taskres_sstext_20261109','query_path_intervention_20261024',
 'query_path_multibackbone_20261025','qata_validation_threshold_20261024',
 'qata_multiseed_20261024','chexchonet_stability_20261024','chexchonet_tools',
 'rsna_chest_stability_20261024','table2_completion_20261103',
 'query_resource_completion_20261103','rsna_fixed_patient_cleanup_20261110','sensitivity_grid']
dirs = [ROOT/'experiments'/n for n in experiments]
dirs += [ROOT/'experiments/query_search_20261021_152010'/n for n in ['code','code_phase6']]
dirs += [ROOT/'baselines/local_setup', ROOT/'baselines/scripts',ROOT/'baselines/paper_draft_zh/scripts',
 ROOT/'AlphaRAD/local_setup',ROOT/'AlphaRAD/scripts',ROOT/'RadZero_i2cl/local_setup']
models = ['BiomedCLIP','BioViL-T','CheXzero','MedKLIP','MAVL','KAD','MaCo','CARZero']
dirs += [ROOT/'baselines'/m/'local_setup' for m in models]
for p in (ROOT/'baselines/protocols').glob('*/*/local_setup'): dirs.append(p)
skip = {'runs','prior','integration','__pycache__','.venv','logs','paper_before_supplement','sources','checkpoints','data','cache','source_snapshot','source_snapshots','backup','backups'}
for d in dirs:
    if not d.exists(): continue
    for base, children, files in os.walk(d):
        children[:] = [c for c in children if c not in skip and not c.startswith('.')]
        for name in files:
            p=Path(base)/name
            if p.suffix in {'.py','.sh'}: copy(p)
for n in ['build_verified_results.py','audit_manuscript_tables.py','neighbor_comparison_tables.py','query_path_extension_tables.py']:
    copy(ROOT/'baselines/paper_draft_zh'/n)
for n in ['build_supplement.py','build_classification_table.py','export_taskres_sstext_csv.py']: copy(PAPER/'scripts'/n)
# Snapshot aggregate inputs only; no images, patient-level predictions or model weights.
audit=json.loads((PAPER/'build/SUPPLEMENT_BUILD_AUDIT.json').read_text())
for rel in audit['sources_sha256']:
    if rel.endswith('.csv'): copy(ROOT/rel)
verified=json.loads((ROOT/'baselines/paper_draft_zh/verified_results.json').read_text())
# Rendering needs only these fields. Original provenance hashes remain in inventory.
clean={k:verified[k] for k in ['status','included_models','included_datasets','summary']}
fields=['model','dataset','task','status','rows','support_count','class_names','branches','reason','delta_pp','change']
clean['records']=[{k:r[k] for k in fields if k in r} for r in verified['records']]
p=OUT/'source/baselines/paper_draft_zh/verified_results.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean,indent=2)+'\n')
for p in (PAPER/'supplement_data').glob('*'):
    if p.suffix in {'.csv','.txt'} and not p.name.startswith('token_mlp'): copy(p)
# Follow active input statements and ignore commented-out tables.
def uncomment(s): return re.sub(r'(?<!\\)%[^\n]*','',s)
seen=set(); tables=[]
aux=(PAPER/'build/main_with_supplement.aux').read_text()
numbers=dict(re.findall(r'\\newlabel\{(tab:[^{}@]+)\}\{\{([^{}]+)\}',aux))
def visit(rel):
    p=PAPER/rel
    if p in seen: return
    seen.add(p); text=p.read_text();copy(p)
    active=uncomment(text)
    for match in re.finditer(r'\\input\{([^}]+)\}|\\begin\{table\*?\}.*?\\end\{table\*?\}',active,re.S):
        if match.group(1):
            child=match.group(1)+'.tex'
            if (PAPER/child).exists():visit(child)
        else:
            block=match.group(); label=re.search(r'\\label\{([^}]+)\}',block)
            if not label: raise ValueError('Unlabelled table in '+str(p))
            label=label.group(1); index=len(tables)+1
            dest=OUT/'tables'/f'{index:02d}_{label.replace(":","_")}.tex';dest.parent.mkdir(exist_ok=True);dest.write_text(block+'\n')
            tables.append(dict(number=numbers.get(label,'unknown'),label=label,source=str(p.relative_to(ROOT)),file=str(dest.relative_to(OUT))))
visit('main_with_supplement.tex')
for row in tables:
    s=row['source'];l=row['label']
    if l in ['tab:overall','tab:components'] or '/a_' in s or '/d_counts' in s:
        family='fixed_support'; entry='baselines/paper_draft_zh/build_verified_results.py';gen='latex_ours/scripts/build_classification_table.py' if l in ['tab:overall','tab:components'] else 'latex_ours/scripts/build_supplement.py'
    elif l=='tab:competitors' or '/c_development' in s or l=='tab:supp-taskres-sstext':
        family='five_support_seeds';entry='experiments/fewshot_multiseed_20261024/run_campaign.py; experiments/fewshot_taskres_sstext_20261109/run_campaign.py';gen='latex_ours/scripts/build_supplement.py' if '/c_development' in s else 'manual LaTeX, aggregate CSV supplied'
    elif l=='tab:paths' or '/b_alpha' in s:
        family='alpharad_paths';entry='experiments/query_path_intervention_20261024/run_campaign.py; experiments/query_path_intervention_20261024/run_siim.py';gen='latex_ours/scripts/build_supplement.py' if '/b_alpha' in s else 'manual LaTeX, aggregate CSV supplied'
    elif '/b_cardiac' in s:
        family='cardiac_paths';entry='experiments/query_path_multibackbone_20261025/run_campaign.py';gen='latex_ours/scripts/build_supplement.py'
    elif 'sensitivity' in l:
        family='sensitivity';entry='experiments/sensitivity_grid/run_sensitivity_campaign.py';gen='manual LaTeX'
    elif 'validation-segmentation' in l:
        family='validation_segmentation';entry='experiments/qata_validation_threshold_20261024/run_campaign.py';gen='manual LaTeX'
    else:family='protocol_documentation';entry='see source LaTeX and native adapters';gen='manual LaTeX, not an experimental metric table'
    row.update(family=family,entry=entry,generator=gen)
with (OUT/'TABLE_INDEX.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
(OUT/'TABLE_INDEX.json').write_text(json.dumps(tables,indent=2)+'\n')
# Record code portability issues without silently rewriting authenticated historical code.
issues=[];manifest=[]
for p in sorted((OUT/'source').rglob('*')):
    if not p.is_file():continue
    b=p.read_bytes();rel=str(p.relative_to(OUT));manifest.append(dict(path=rel,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    if p.suffix in {'.py','.sh'}:
        for i,line in enumerate(b.decode().splitlines(),1):
            reasons=[]
            if re.search(r'/(storage|home|tmp|nfs)/',line): reasons.append('machine_path')
            if re.search(r'\{4, 5, 6, 7\}|Only.*GPU|authorized',line): reasons.append('gpu_policy')
            if reasons:issues.append(dict(path=rel,line=i,kind=','.join(reasons),code=line.strip()))
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'PORTABILITY_AUDIT.json').write_text(json.dumps(issues,indent=2)+'\n')
print(json.dumps(dict(tables=len(tables),files=len(manifest),portability_findings=len(issues))))

import runpy
runpy.run_path(str(OUT/'tools/collect_evidence.py'))
