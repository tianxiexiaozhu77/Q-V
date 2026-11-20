#!/usr/bin/env python3
"""Summarize the fixed 90-job sensitivity plan from saved metrics (no inference)."""
import argparse,csv,hashlib,json,math
from pathlib import Path
from statistics import mean
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);a=p.parse_args()
run=a.run.resolve();plan=json.loads((run/'PLAN.json').read_text());sources={}
if a.output.exists():raise SystemExit('Use a new output directory.')
assert plan['status']=='complete',plan['status']
assert len(plan['jobs'])==90
models={'AlphaRAD','RadZero','BiomedCLIP','MedKLIP','CARZero'}
trials={(e,.8) for e in [0.,.1,.2,.3,.5,.7]}|{(.3,c) for c in [.6,.7,.9]}
expected={(m,d,e,c) for m in models for d in ['chexpert','chexchonet'] for e,c in trials}
seen=set();rows=[]
def read(path):
 b=path.read_bytes();sources[str(path.relative_to(run))]=hashlib.sha256(b).hexdigest();d=json.loads(b)
 assert d['status']=='complete',path
 assert not d.get('smoke',False) and not d.get('is_smoke',False) and not d.get('smoke_batches',0),path
 return d
def difference(original,adapted):
 return [100*(float(adapted['macro_'+k])-float(original['macro_'+k])) for k in ['auroc','auprc']]
for job in plan['jobs']:
 key=(job['model'],job['dataset'],float(job['eta']),float(job['crop_ratio']))
 assert key in expected and key not in seen,key
 seen.add(key);assert job['status']=='complete' and job.get('returncode',0)==0
 folder=run/Path(job['output']).name
 files=[folder/'result.json'] if (folder/'result.json').exists() else sorted(folder.glob('*/result.json'))
 assert len(files) in [1,3],folder
 deltas=[]
 for f in files:
  d=read(f)
  if 'records' in d:
   # RadZero/CheXpert: reference is eta=0 IDENTITY, not identity at target eta.
   selected=[r for r in d['records'] if float(r['eta'])==key[2] and float(r['crop_ratio'])==key[3]]
   baseline=[r for r in d['records'] if float(r['eta'])==0 and float(r['crop_ratio'])==key[3]]
   assert len(selected)==len(baseline)==1
   delta=difference(baseline[0]['identity_eta_metrics'],selected[0]['metrics'])
  else:
   branches=d.get('branches',d.get('arms'))
   assert branches is not None,f
   delta=difference(branches['original'],branches['tta_center_refined'])
  assert all(math.isfinite(v) for v in delta)
  deltas.append(delta)
 if key[1]=='chexchonet' and len(files)==3:
  assert {f.parent.name for f in files}=={'chexchonet_slvh','chexchonet_dlv','chexchonet_composite_slvh_dlv'}
 rows.append(dict(model=key[0],dataset=key[1],eta=key[2],crop_ratio=key[3],delta_auroc_pp=mean(d[0] for d in deltas),delta_auprc_pp=mean(d[1] for d in deltas),task_count=3 if key[1]=='chexchonet' else 1))
assert seen==expected
if a.reference:
 with a.reference.open(newline='') as f:ref={(r['model'],r['dataset'],float(r['eta']),float(r['crop_ratio'])):r for r in csv.DictReader(f)}
 assert set(ref)==expected
 for row in rows:
  old=ref[row['model'],row['dataset'],row['eta'],row['crop_ratio']]
  for k in ['delta_auroc_pp','delta_auprc_pp']:assert abs(row[k]-float(old[k]))<1e-9,(row,old)
a.output.mkdir(parents=True)
def write(name,data):
 with (a.output/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('sensitivity_grid_summary.csv',rows)
by_model=[]
for m in sorted(models):
 for d in ['chexpert','chexchonet']:
  group=[r for r in rows if r['model']==m and r['dataset']==d]
  by_model.append(dict(model=m,dataset=d,mean_delta_auroc_pp=mean(r['delta_auroc_pp'] for r in group),mean_delta_auprc_pp=mean(r['delta_auprc_pp'] for r in group)))
write('model_dataset_summary.csv',by_model)
slices=[]
for field,values,other,fixed in [('eta',[0.,.1,.2,.3,.5,.7],'crop_ratio',.8),('crop_ratio',[.6,.7,.8,.9],'eta',.3)]:
 for value in values:
  group=[r for r in rows if r[field]==value and r[other]==fixed];assert len(group)==10
  slices.append(dict(slice_type=field,value=value,n_model_dataset=10,mean_delta_auroc_pp=mean(r['delta_auroc_pp'] for r in group),mean_delta_auprc_pp=mean(r['delta_auprc_pp'] for r in group)))
write('parameter_summary.csv',slices)
(a.output/'AUDIT.json').write_text(json.dumps(dict(rows=90,sources_sha256=sources,plan_sha256=hashlib.sha256((run/'PLAN.json').read_bytes()).hexdigest(),reference_checked=bool(a.reference),mean_delta_auroc_pp=mean(r['delta_auroc_pp'] for r in rows),mean_delta_auprc_pp=mean(r['delta_auprc_pp'] for r in rows)),indent=2)+'\n')
print('Summarized 90 jobs; reference matched.' if a.reference else 'Summarized 90 jobs.')
