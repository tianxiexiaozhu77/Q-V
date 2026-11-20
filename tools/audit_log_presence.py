"""Read-only presence/hash audit of manuscript experiment evidence."""
import csv,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'github_release'
groups={
'fixed_phase6_controller':'experiments/query_search_20261021_152010/controller_v2/logs',
'fixed_phase6':'experiments/query_search_20261021_152010/runs/phase6',
'fixed_baselines':'baselines/runs/query_transfer_20261022_224826',
'fixed_cardiac_baselines':'baselines/runs/chexchonet_20261024_000447',
'fixed_cardiac_alpha_radzero':'experiments/chexchonet_comparison_20261023_233926',
'fixed_chexpert_alpha':'AlphaRAD/runs/chexpert_phase6_20261022_170123_2309385',
'fixed_completion':'experiments/table2_completion_20261103',
'fixed_query_resources':'experiments/query_resource_completion_20261103',
'fixed_rsna_cleanup':'experiments/rsna_fixed_patient_cleanup_20261110',
'five_seeds':'experiments/fewshot_multiseed_20261024/runs/full_20261024_v2',
'additional_adapters':'experiments/fewshot_taskres_sstext_20261109/runs/full_20261109_v1',
'alpha_paths':'experiments/query_path_intervention_20261024/runs/diagnostic_20261024_v1',
'siim_paths':'experiments/query_path_intervention_20261024/runs/siim_holdout_20261024_v1',
'cardiac_paths':'experiments/query_path_multibackbone_20261025/runs/full_20261025_v1',
'segmentation':'experiments/qata_validation_threshold_20261024/runs/validation_selected_20261024_v1',
'sensitivity':'experiments/sensitivity_grid/campaign_20261108_v3'}
summary={};files=[]
for group,rel in groups.items():
 logs=[];records=[]
 for base,ds,names in os.walk(ROOT/rel):
  ds[:]=[d for d in ds if d not in {'prior','integration','source_snapshot','source_snapshots','__pycache__','.git'}]
  for name in names:
   p=Path(base)/name
   if p.suffix in {'.log','.out','.err'} or name in {'stdout.txt','stderr.txt'}:kind='log';logs.append(p)
   elif p.suffix in {'.json','.csv'} and any(s in name.lower() for s in ['result','audit','summary','comparison','status','mean_sd','per_seed','corrected']):kind='record';records.append(p)
   else:continue
   files.append(dict(group=group,kind=kind,path=str(p.relative_to(ROOT)),bytes=p.stat().st_size))
 summary[group]=dict(directory=rel,exists=(ROOT/rel).exists(),logs=len(logs),nonempty_logs=sum(p.stat().st_size>0 for p in logs),records=len(records),example_logs=[str(p.relative_to(ROOT)) for p in logs[:3]])
verified=json.loads((ROOT/'baselines/paper_draft_zh/verified_results.json').read_text());checks=[]
for r in verified['records']:
 p=Path(r['source_path']);present=p.is_file()
 checks.append(dict(model=r['model'],dataset=r['dataset'],status=r['status'],path=str(p.relative_to(ROOT)),exists=present,sha256_matches=present and hashlib.sha256(p.read_bytes()).hexdigest()==r['source_sha256']))
report=dict(groups=summary,fixed_source_checks=checks,scope='Presence and source hash audit only; log existence does not prove successful or complete inference.')
(OUT/'LOG_PRESENCE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'LOG_FILES.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['group','kind','path','bytes']);w.writeheader();w.writerows(files)
print('Group log counts:', {k:v['logs'] for k,v in summary.items()})
print('Fixed records:',len(checks),'complete:',sum(c['status']=='complete' for c in checks),'missing:',sum(not c['exists'] for c in checks),'hash mismatch:',sum(c['exists'] and not c['sha256_matches'] for c in checks))

tables=json.loads((OUT/'TABLE_INDEX.json').read_text())
rows=[]
for t in tables:
 f=t['family'];n=t['number']
 if f=='fixed_support':keys=[k for k in groups if k.startswith('fixed_')]
 elif f=='five_support_seeds':keys=['additional_adapters'] if n=='C.9' else (['five_seeds','additional_adapters'] if n=='2' else ['five_seeds'])
 elif f=='alpharad_paths':keys=['siim_paths'] if n in ['B.5','B.6'] else ['alpha_paths']
 elif f=='cardiac_paths':keys=['cardiac_paths']
 elif f=='validation_segmentation':keys=['segmentation']
 elif f=='sensitivity':keys=['sensitivity']
 else:keys=[]
 rows.append(dict(table=n,label=t['label'],classification='experiment_batch_logs_found' if keys else 'protocol_table_no_independent_run_log',log_groups=';'.join(keys),directories=';'.join(groups[k] for k in keys),note='Batch-level presence only; not proof every job succeeded. RSNA rescoring has JSON audit but no standalone text log.' if f=='fixed_support' else ''))
with (OUT/'TABLE_LOG_INDEX.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
