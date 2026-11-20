#!/usr/bin/env python3
"""Build supplementary tables from saved results; never runs model inference."""
from pathlib import Path
import argparse, csv, hashlib, json, math, re, statistics
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixed-only', action='store_true', help='Rebuild only fixed-support tables and records, preserving repeated-support/path outputs.')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'latex_ours'; TABLES=OUT/'sec/supplement_tables'; DATA=OUT/'supplement_data'
BUILD=OUT/'build'
for p in (TABLES,DATA,BUILD):p.mkdir(parents=True,exist_ok=True)
SOURCES={}; COUNTS={}; NUMERIC=0
MODELS=['AlphaRAD','RadZero','BiomedCLIP','BioViL-T','CheXzero','MedKLIP','MAVL','KAD','MaCo','CARZero']
TASKS=['slvh','dlv','composite_slvh_dlv','macro_three_tasks']
TN={'slvh':'CSLV','dlv':'CDLV','composite_slvh_dlv':'CCOM','macro_three_tasks':'Task macro'}
DN={'chest14':'ChestXray14','chestxdet10':'ChestXDet10 (classification)','chestxdet10_gt':'ChestXDet10 (PG)','rsna_pneumonia':'RSNA','siim_pneumothorax':'SIIM (1,704)','siim_radzero_holdout':'SIIM (1,694)','chexpert':'CheXpert','shenzhenxray':'Shenzhen','montgomery':'Montgomery','covidkaggle':'CovidKaggle','node21':'Node21 (PG)','qatacovid_seg':r'QaTaCOV19 (Dice$^\dagger$)','chexchonet_slvh':'CheXchoNet / CSLV','chexchonet_dlv':'CheXchoNet / CDLV','chexchonet_composite_slvh_dlv':'CheXchoNet / CCOM','siim_new_holdout':'SIIM new holdout'}
MN={'original':'Original','view':'V','original_view':'V','query':'Q','query_view':'Q+V','readout_only':'R','aggregation_only':'A','fixed_aggregation':'F','score_retained_original_map':'S','positive_query':'Positive-support-only adaptation','class_local_difference_score_oof':'Score S','prototype_score_only':'Prototype','prototype_score_mix_oof':'Prototype mix','tip_adapter_ovr_oof':'Tip-Adapter','gda_ovr_ks_oof':'GDA','simnl_feature_ovr':'SimNL','prototype':'Prototype','prototype_mix':'Prototype mix','tip_adapter':'Tip-Adapter','gda':'GDA','simnl':'SimNL','project_mix':r'P\&M','linear_probe':'Linear probe'}
for k in ['prototype','prototype_mix','tip_adapter','gda','simnl','project_mix','linear_probe']:MN[k+'_view']=MN[k]+r' (2V)'
OLD_ORDER=['original','prototype_score_only','prototype_score_mix_oof','tip_adapter_ovr_oof','gda_ovr_ks_oof','simnl_feature_ovr','positive_query','class_local_difference_score_oof','query','view','query_view']
NEW_ORDER=['original','query','prototype','prototype_mix','tip_adapter','gda','simnl','project_mix','linear_probe','original_view','query_view','prototype_view','prototype_mix_view','tip_adapter_view','gda_view','simnl_view','project_mix_view','linear_probe_view']
def raw(path):
 p=ROOT/path; b=p.read_bytes();SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b.decode()
def readcsv(path,export):
 rows=list(csv.DictReader(raw(path).splitlines()))
 # Keep historical source files intact while applying the manuscript scope to
 # both rendered tables and packaged numerical records.
 rows=[r for r in rows if 'model' not in r or r['model'] in MODELS]
 assert rows,path
 # Results tables contain no image identifiers or filesystem paths. Keep only
 # AUROC/AUPRC, PG and supporting grouping fields in packaged CSV exports.
 keys=[k for k in rows[0] if not k.startswith('average_precision')]
 if 'metric' in keys:rows_export=[r for r in rows if r['metric'] in ['auroc','auprc','pg']]
 else:rows_export=rows
 for r in rows_export:
  assert all('/storage/' not in str(v) and '/home/' not in str(v) for v in r.values())
 writecsv(export,keys,[{k:r[k] for k in keys} for r in rows_export])
 return rows

def writecsv(name,keys,rows):
 with (DATA/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def num(v,n=2,sign=False):
 global NUMERIC
 NUMERIC+=1; v=float(v);assert math.isfinite(v)
 if abs(v)<0.5*10**(-n):v=0.0
 return f'{v:+.{n}f}' if sign else f'{v:.{n}f}'
def ms(mean,sd,n=2):return '$'+num(mean,n)+r'\pm'+num(sd,n)+'$'
def ci(v,lo,hi,n=2):return '$'+num(v,n,True)+r'\;['+num(lo,n,True)+','+num(hi,n,True)+']$'
def pair(a,b):return num(a)+' / '+num(b)
def table(label,caption,header,rows,cols,size='small'):
 assert all(isinstance(r,str) or len(r)==len(cols) for r in rows),label
 lines=[r'\begin{table*}[tp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}','\\'+size,r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+cols+'@{}}',r'\toprule',header+r' \\',r'\midrule']
 lines+= [r if isinstance(r,str) else ' & '.join(r)+r' \\' for r in rows]
 lines +=[r'\bottomrule',r'\end{tabular*}',r'\end{table*}']
 return '\n'.join(lines)+'\n'
def save(name,s): (TABLES/name).write_text(s)
def blocks(tables):return '\n'.join(tables)+'\n'
def filtermetrics(rows):return [r for r in rows if r.get('metric','auroc') in ['auroc','auprc']]
def index(rows,fields):
 out={}
 for r in filtermetrics(rows):
  k=tuple(r[f] for f in fields);assert k not in out,k;out[k]=r
 return out

def write_fixed_metadata(records):
 rr=[]
 for ds in DN:
  rs=[r for r in records if r['dataset']==ds]
  if not rs:continue
  r=rs[0];classes=r['class_names']
  rr.append([DN[ds],str(len(classes)),str(r['support_count']),f"{r['rows']:,}"])
 save('d_counts.tex',blocks([table('tab:supp-counts',r'\textbf{Fixed-support evaluation counts.} Counts are images. Separate SIIM protocols and spatial tasks are listed explicitly; cardiac labels share the same test images. Patient identities are not inferred from image counts.','Dataset / task & Classes & Support images & Test images',rr,'lrrr')]))
 # Sidecar contains label names without patient/image identifiers.
 labels=[]
 for r in records:
  for i,c in enumerate(r['class_names']):labels.append({'model':r['model'],'dataset':r['dataset'],'class_index':i,'class_name':c})
 writecsv('evaluated_label_orders.csv',list(labels[0]),labels)

# A. Complete fixed-support evaluations.
verified=json.loads(raw('baselines/paper_draft_zh/verified_results.json'))
records=[r for r in verified['records'] if r['status']=='complete']
assert verified['included_models']==MODELS
assert set(verified['included_datasets'])==set(DN)-{'siim_new_holdout'}
assert {r['model'] for r in records}==set(MODELS)
assert len(records)==verified['summary']['counts']['complete']
assert sum(r['task']=='classification' for r in records)==verified['summary']['complete_task_counts']['classification']
flat=[];tabs=[]
for model in MODELS:
 rs=[r for r in records if r['model']==model];rr=[]
 for r in rs:
  measures=['macro_auroc','macro_auprc'] if r['task']=='classification' else ['pointing_game' if r['task']=='grounding' else 'dice']
  cells=[]
  for branch in ['original','query_refined','tta_center_original','tta_center_refined']:
   vals=[100*r['branches'][branch][m] for m in measures]
   cells.append(' / '.join(num(v) for v in vals))
   for m,v in zip(measures,vals):flat.append({'model':model,'dataset':r['dataset'],'task':r['task'],'test_images':r['rows'],'support_images':r['support_count'],'branch':branch,'metric':m,'score_percent':v})
  rr.append([DN[r['dataset']]]+cells)
 tabs.append(table('tab:supp-fixed-'+model.lower(),r'\textbf{'+model+r': complete fixed-support results.} Classification cells are AUROC / AUPRC (\%); PG and Dice are single percentages. $\dagger$: historical test-mask threshold selection. All four branches are retained, including declines.','Dataset / task & Original & Q & V & Q+V',rr,'lcccc'))
save('a_fixed.tex',blocks(tabs));writecsv('fixed_support_all.csv',list(flat[0]),flat)
# Coverage matrix includes NR separately from unsupported.
lookup={(r['model'],r['dataset']):r for r in verified['records']}; rr=[]
for ds in DN:
 if ds=='siim_new_holdout':continue
 rr.append([DN[ds]]+[('Y' if lookup[(m,ds)]['status']=='complete' else 'NA') if (m,ds) in lookup else 'NR' for m in MODELS])
header='Task & Alpha & Rad & BioCLIP & BioViL & CheX0 & MedK & MAVL & KAD & MaCo & CAR0'
save('a_coverage.tex',blocks([table('tab:supp-coverage',r'\textbf{Coverage.} Y: completed; NA: no verified spatial-response implementation in the current evaluation; NR: not run in this evaluation. Abbreviations follow the backbone order in Appendix A. Neither NA nor NR is assigned a score.',header,rr,'l'+'c'*len(MODELS),'footnotesize')]))
COUNTS['fixed_support_tasks']=len(records);COUNTS['fixed_support_branch_metric_values']=len(flat)

if args.fixed_only:
 write_fixed_metadata(records)
 COUNTS['formatted_numeric_values']=NUMERIC
 report={'sources_sha256':SOURCES,'counts':COUNTS,'no_model_inference':True,
         'scope':'fixed-support tables, counts, and label orders only',
         'table_files':['a_fixed.tex','a_coverage.tex','d_counts.tex']}
 (BUILD/'FIXED_SUPPORT_BUILD_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(COUNTS,indent=2))
 raise SystemExit(0)

# B. Original AlphaRAD interventions, including saved SIIM holdout.
diag='experiments/query_path_intervention_20261024/runs/diagnostic_20261024_v1/evaluation/'
siim='experiments/query_path_intervention_20261024/runs/siim_holdout_20261024_v1/evaluation/'
means=[];contrasts=[]
for base,key in [(diag,'development'),(siim,'siim')]:
 means+=readcsv(base+'MEAN_SD_COMPARISON.csv','path_'+key+'_means.csv')
 contrasts+=readcsv(base+'PAIRED_COMPARISONS.csv','path_'+key+'_comparisons.csv')
 readcsv(base+'PER_SEED_RESULTS.csv','path_'+key+'_seeds.csv')
sp=readcsv(diag+'SPATIAL_PAIRED_COMPARISONS.csv','path_development_spatial_comparisons.csv')
tabs=[]
for ds in ['rsna_pneumonia','chestxdet10','siim_new_holdout']:
 rr=[]
 for r in means:
  if r['dataset']==ds:rr.append([MN[r['method']]]+[ms(r[m+'_mean_percent'],r[m+'_sd_pp']) for m in ['auroc','auprc']])
 tabs.append(table('tab:supp-path-'+ds,r'\textbf{AlphaRAD / '+DN[ds]+r': all path branches.} Five-support-seed mean $\pm$ sample SD; absolute scores are percentages. All branches use one view.','Branch & AUROC & AUPRC',rr,'lcc'))
 rr=[]
 for r in contrasts:
  if r['dataset']==ds:rr.append([MN[r['target']]+' $-$ '+MN[r['reference']]]+[ci(r[m+'_delta_pp'],r[m+'_ci_low_pp'],r[m+'_ci_high_pp']) for m in ['auroc','auprc']])
 tabs.append(table('tab:supp-path-ci-'+ds,r'\textbf{AlphaRAD / '+DN[ds]+r': all prespecified classification contrasts.} Five-support mean differences and paired 95\% intervals (pp), conditional on the five supports.',r'Contrast & $\Delta$AUROC [95\% interval] & $\Delta$AUPRC [95\% interval]',rr,'lcc','footnotesize'))
save('b_alpha.tex',blocks(tabs))
base='experiments/query_path_multibackbone_20261025/runs/full_20261025_v1/'
pm=readcsv(base+'MEAN_SD_COMPARISON.csv','cardiac_paths_means.csv');pc=readcsv(base+'PAIRED_COMPARISONS.csv','cardiac_paths_comparisons.csv');ps=readcsv(base+'PER_SEED_RESULTS.csv','cardiac_paths_seeds.csv')
pmi=index(pm,['model','task','branch','metric']);pci=index(pc,['model','task','comparison','metric']);tabs=[]
for model in ['AlphaRAD','RadZero','CARZero','MedKLIP','MAVL','KAD']:
 control='readout_only' if model in ['AlphaRAD','RadZero'] else 'fixed_aggregation'
 rr=[]
 for task in TASKS:
  for branch in ['original',control,'query']:
   rr.append([TN[task],MN[branch]]+[ms(pmi[model,task,branch,m]['mean_percent'],pmi[model,task,branch,m]['support_seed_sd_pp'],4) for m in ['auroc','auprc']])
 tabs.append(table('tab:supp-cardiac-path-'+model.lower(),r'\textbf{'+model+r' / CheXchoNet: every path branch.} Mean $\pm$ support-seed SD (\%). Task macro averages the same three tasks and adds no independent cohort.','Task & Branch & AUROC & AUPRC',rr,'llcc'))
 rr=[]
 for task in TASKS:
  for comparison in ['query_minus_'+control,'query_minus_original',control+'_minus_original']:
   high,low=comparison.split('_minus_')
   rr.append([TN[task],MN[high]+' $-$ '+MN[low]]+[ci(pci[model,task,comparison,m]['mean_delta_pp'],pci[model,task,comparison,m]['ci_low_pp'],pci[model,task,comparison,m]['ci_high_pp'],4) for m in ['auroc','auprc']])
 tabs.append(table('tab:supp-cardiac-path-ci-'+model.lower(),r'\textbf{'+model+r' / CheXchoNet: every path contrast.} Mean paired differences [95\% patient-bootstrap intervals] in pp. R and F are different interventions; see Appendix B.',r'Task & Contrast & $\Delta$AUROC [95\% interval] & $\Delta$AUPRC [95\% interval]',rr,'llcc'))
save('b_cardiac.tex',blocks(tabs))
COUNTS['cardiac_path_task_branch_results']=len([r for r in ps if r['task']!='macro_three_tasks' and r['metric']=='auroc'])
COUNTS['cardiac_path_contrasts_including_macro']=len(pci)//2

# C. Primary two-dataset few-shot comparison: means, all contrasts, every seed.
base='experiments/fewshot_multiseed_20261024/runs/full_20261024_v2/'
om=readcsv(base+'MEAN_SD_COMPARISON.csv','fewshot_development_means.csv');oc=readcsv(base+'PAIRED_COMPARISONS.csv','fewshot_development_comparisons.csv');os=readcsv(base+'PER_SEED_RESULTS.csv','fewshot_development_seeds.csv')
osi=index(os,['model','dataset','method','seed']);omi=index(om,['model','dataset','method']);mean_tabs=[];ci_tabs=[];seed_tabs=[]
for model in ['BiomedCLIP','AlphaRAD']:
 for ds in ['rsna_pneumonia','chestxdet10']:
  key=model.lower()+'-'+ds;title=model+' / '+DN[ds]
  rr=[]
  for method in OLD_ORDER:
   r=omi[model,ds,method];rr.append([MN[method]]+[ms(r[m+'_mean_percent'],r[m+'_sd_pp']) for m in ['auroc','auprc']])
  mean_tabs.append(table('tab:supp-oldmeans-'+key,r'\textbf{'+title+r': original 11-method/branch comparison.} Mean $\pm$ sample SD over support seeds 42--46. Scores are percentages. V and Q+V use two views; other rows use one.','Method & AUROC & AUPRC',rr,'lcc'))
  rr=[]
  for r in oc:
   if r['model']==model and r['dataset']==ds:rr.append([MN[r['target']]+' $-$ '+MN[r['reference']]]+[ci(r[m+'_delta_pp'],r[m+'_ci_low_pp'],r[m+'_ci_high_pp']) for m in ['auroc','auprc']])
  ci_tabs.append(table('tab:supp-oldci-'+key,r'\textbf{'+title+r': all prespecified method contrasts.} Mean differences and conditional paired 95\% intervals (pp). Cross-view comparisons do not have a matched view budget.',r'Contrast & $\Delta$AUROC [95\% interval] & $\Delta$AUPRC [95\% interval]',rr,'lcc','footnotesize'))
  rr=[]
  for method in OLD_ORDER:
   rr.append([MN[method]]+[pair(osi[model,ds,method,str(seed)]['auroc_percent'],osi[model,ds,method,str(seed)]['auprc_percent']) for seed in range(42,47)])
  seed_tabs.append(table('tab:supp-oldseeds-'+key,r'\textbf{'+title+r': every support seed.} Each cell is AUROC / AUPRC (\%). These are independent fits evaluated on the same test list, not a prediction ensemble.','Method & Seed 42 & Seed 43 & Seed 44 & Seed 45 & Seed 46',rr,'lccccc'))
save('c_development_means.tex',blocks(mean_tabs));save('c_development_ci.tex',blocks(ci_tabs));save('c_development_seeds.tex',blocks(seed_tabs))
COUNTS['fewshot_development_means']=len(om);COUNTS['fewshot_development_contrasts']=len(oc);COUNTS['fewshot_development_seed_results']=len(os)

# C. CheXchoNet: complete seven-competitor matched-view evaluation.
base='experiments/fewshot_neighbors_20261025/runs/full_20261025_v1/'
nm=readcsv(base+'MEAN_SD_COMPARISON.csv','fewshot_cardiac_means.csv');nc=readcsv(base+'PAIRED_COMPARISONS.csv','fewshot_cardiac_comparisons.csv');ns=readcsv(base+'PER_SEED_RESULTS.csv','fewshot_cardiac_seeds.csv')
nmi=index(nm,['model','task','branch','metric']);nci=index(nc,['model','task','comparison','metric']);nsi=index(ns,['model','task','branch','support_seed','metric'])
mean_tabs=[];ci_tabs=[];seed_tabs=[]
for model in ['BiomedCLIP','AlphaRAD']:
 for view,order in [('one',NEW_ORDER[:9]),('two',NEW_ORDER[9:])]:
  rr=[]
  for branch in order:
   rr.append([MN[branch]]+[ms(nmi[model,task,branch,m]['mean_percent'],nmi[model,task,branch,m]['support_seed_sd_pp']) for task in TASKS[:3] for m in ['auroc','auprc']])
  header=r'Method & \multicolumn{2}{c}{CSLV} & \multicolumn{2}{c}{CDLV} & \multicolumn{2}{c}{CCOM} \\'+'\n'+r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'+'\n'+r' & AUROC & AUPRC & AUROC & AUPRC & AUROC & AUPRC'
  mean_tabs.append(table('tab:supp-newmeans-'+model.lower()+'-'+view,r'\textbf{'+model+r': all '+view+r'-view cardiac comparisons.} Mean $\pm$ sample SD over five support sets, in percent. All methods share support labels, folds, and test patients. 2V denotes two views.',header,rr,'lcccccc','footnotesize'))
 rr=[]
 for branch in NEW_ORDER:rr.append([MN[branch]]+[ms(nmi[model,'macro_three_tasks',branch,m]['mean_percent'],nmi[model,'macro_three_tasks',branch,m]['support_seed_sd_pp']) for m in ['auroc','auprc']])
 mean_tabs.append(table('tab:supp-newmacro-'+model.lower(),r'\textbf{'+model+r': three-task macro averages.} Average the three task metrics within each seed before computing the mean and SD. This summary does not create another dataset.','Method & Macro AUROC & Macro AUPRC',rr,'lcc'))
 for task in TASKS:
  comparisons=list(dict.fromkeys(r['comparison'] for r in nc if r['model']==model and r['task']==task))
  rr=[]
  for comparison in comparisons:
   r=nci[model,task,comparison,'auroc'];high,low=r['high_branch'],r['low_branch']
   rr.append([MN[high]+' $-$ '+MN[low]]+[ci(nci[model,task,comparison,m]['mean_delta_pp'],nci[model,task,comparison,m]['ci_low_pp'],nci[model,task,comparison,m]['ci_high_pp']) for m in ['auroc','auprc']])
  ci_tabs.append(table('tab:supp-newci-'+model.lower()+'-'+task,r'\textbf{'+model+' / '+TN[task]+r': all prespecified contrasts.} Mean differences [paired 95\% patient-bootstrap intervals], in pp. Includes all same-view competitors, comparisons with Original/V, and each method\textquotesingle s view effect.',r'Contrast & $\Delta$AUROC [95\% interval] & $\Delta$AUPRC [95\% interval]',rr,'lcc','footnotesize'))
  if task!='macro_three_tasks':
   rr=[]
   for branch in NEW_ORDER:
    rr.append([MN[branch]]+[pair(nsi[model,task,branch,str(seed),'auroc']['metric_percent'],nsi[model,task,branch,str(seed),'auprc']['metric_percent']) for seed in range(42,47)])
   seed_tabs.append(table('tab:supp-newseeds-'+model.lower()+'-'+task,r'\textbf{'+model+' / '+TN[task]+r': every support seed and view condition.} Each cell is AUROC / AUPRC (\%). Seed variation is reported separately from conditional test-patient bootstrap uncertainty.','Method & Seed 42 & Seed 43 & Seed 44 & Seed 45 & Seed 46',rr,'lccccc'))
save('c_cardiac_means.tex',blocks(mean_tabs));save('c_cardiac_ci.tex',blocks(ci_tabs));save('c_cardiac_seeds.tex',blocks(seed_tabs))
COUNTS['fewshot_cardiac_task_means']=len([r for r in nm if r['metric']=='auroc' and r['task']!='macro_three_tasks'])
COUNTS['fewshot_cardiac_contrasts_including_macro']=len(nci)//2
COUNTS['fewshot_cardiac_seed_results']=len([r for r in ns if r['metric']=='auroc' and r['task']!='macro_three_tasks'])

# D. Label dictionaries and fixed-support counts.
write_fixed_metadata(records)
# Verify reported sample means/SD against individual support runs.
for model in ['BiomedCLIP','AlphaRAD']:
 for ds in ['rsna_pneumonia','chestxdet10']:
  for method in OLD_ORDER:
   r=omi[model,ds,method]
   for metric in ['auroc','auprc']:
    values=[float(osi[model,ds,method,str(seed)][metric+'_percent']) for seed in range(42,47)]
    assert abs(statistics.mean(values)-float(r[metric+'_mean_percent']))<1e-9
    assert abs(statistics.stdev(values)-float(r[metric+'_sd_pp']))<1e-9
for model in ['BiomedCLIP','AlphaRAD']:
 for task in TASKS:
  for branch in NEW_ORDER:
   for metric in ['auroc','auprc']:
    values=[float(nsi[model,task,branch,str(seed),metric]['metric_percent']) for seed in range(42,47)]
    r=nmi[model,task,branch,metric]
    assert abs(statistics.mean(values)-float(r['mean_percent']))<1e-9
    assert abs(statistics.stdev(values)-float(r['support_seed_sd_pp']))<1e-9
psi=index(ps,['model','task','branch','support_seed','metric'])
for (model,task,branch,metric),r in pmi.items():
 values=[float(psi[model,task,branch,str(seed),metric]['metric_percent']) for seed in range(42,47)]
 assert abs(statistics.mean(values)-float(r['mean_percent']))<1e-9
 assert abs(statistics.stdev(values)-float(r['support_seed_sd_pp']))<1e-9
COUNTS['formatted_numeric_values']=NUMERIC
report={'sources_sha256':SOURCES,'counts':COUNTS,'means_and_sample_sd_checked_against_seed_rows':True,'no_model_inference':True,'table_files':sorted(p.name for p in TABLES.glob('*.tex'))}
report['path_reporting_scope']={'complete_path_tables':True,'alpharad_datasets':['rsna_pneumonia','chestxdet10','siim_new_holdout'],'cardiac_models':['AlphaRAD','RadZero','CARZero','MedKLIP','MAVL','KAD'],'all_saved_path_branches_and_contrasts_retained':True}
(BUILD/'SUPPLEMENT_BUILD_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(COUNTS,indent=2))
