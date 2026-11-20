#!/usr/bin/env python3
"""Check all displayed numeric cells in main Tables 2 and 4 against CSVs."""
import csv, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source/latex_ours/supplement_data'
index=json.loads((ROOT/'TABLE_INDEX.json').read_text())
def rows(name):
    with (DATA/name).open(newline='') as f:return list(csv.DictReader(f))
def cells(label):
    p=ROOT/next(t['file'] for t in index if t['label']==label)
    result=[]
    for line in p.read_text().splitlines():
        if re.search(r'\d+\.\d+\s*(?:\}|\s)*\s*/',line) and '&' in line:
            result.append([float(v) for v in re.findall(r'(?<![\w.])(\d+\.\d+)',line)])
    return result
means=rows('fewshot_development_means.csv')+rows('fewshot_taskres_sstext_means.csv')
lookup={(r['model'],r['dataset'],r['method']):r for r in means}
methods=['original','prototype_score_only','prototype_score_mix_oof','tip_adapter_ovr_oof',
'gda_ovr_ks_oof','simnl_feature_ovr','taskres','sstext_plus','positive_query',
'class_local_difference_score_oof','query','view','query_view']
settings=[(m,d) for m in ['BiomedCLIP','AlphaRAD'] for d in ['rsna_pneumonia','chestxdet10']]
expected=[[float(f"{float(lookup[m,d,b][metric+'_mean_percent']):.2f}") for m,d in settings for metric in ['auroc','auprc']] for b in methods]
actual=cells('tab:competitors')
assert actual==expected, {'table':2,'actual':actual,'expected':expected}
lookup={(r['dataset'],r['method']):r for r in rows('path_development_means.csv')}
expected=[[float(f"{float(lookup[d,b][metric+'_mean_percent']):.2f}") for d in ['rsna_pneumonia','chestxdet10'] for metric in ['auroc','auprc']] for b in ['original','readout_only','aggregation_only','query']]
actual=cells('tab:paths')
assert actual==expected, {'table':4,'actual':actual,'expected':expected}
print('PASS: main Table 2 (104 values) and Table 4 (16 values) match unrounded CSV aggregates.')
# Validation-selected threshold table: every displayed value, including intervals.
p=ROOT/'source/experiments/qata_validation_threshold_20261024/runs/validation_selected_20261024_v1/evaluation/RESULTS.csv'
with p.open(newline='') as f: result=list(csv.DictReader(f))
text=(ROOT/next(t['file'] for t in index if t['label']=='tab:supp-validation-segmentation')).read_text()
for r in result:
    line=next(line for line in text.splitlines() if line.startswith(r['method']+' &'))
    values=[float(v) for v in re.findall(r'[+-]?\d+\.\d+',line)]
    keys=['threshold','test_dice_percent']
    if r['method']!='Original':keys+=['delta_dice_pp','delta_ci95_lower_pp','delta_ci95_upper_pp']
    assert values==[float(f'{float(r[k]):.2f}') for k in keys],r['method']
print('PASS: validation-threshold segmentation table (17 values).')
# Use the corrected grid, not the stale sensitivity_parameter_summary.csv.
from statistics import mean
p=ROOT/'source/experiments/sensitivity_grid/campaign_20261108_v3/sensitivity_grid_summary_corrected.csv'
with p.open(newline='') as f:grid=list(csv.DictReader(f))
assert len(grid)==90
keys=['delta_auroc_pp','delta_auprc_pp']
def averages(rs):return [float(f'{mean(float(r[k]) for r in rs):.2f}') for k in keys]
text=(ROOT/next(t['file'] for t in index if t['label']=='tab:supp-sensitivity-model')).read_text()
for model in sorted({r['model'] for r in grid}):
    line=next(line for line in text.splitlines() if line.startswith(model+' &'))
    actual=[float(v) for v in re.findall(r'[+-]?\d+\.\d+',line)]
    expected=[v for dataset in ['chexpert','chexchonet'] for v in averages([r for r in grid if r['model']==model and r['dataset']==dataset])]
    assert actual==expected,(model,actual,expected)
line=next(line for line in text.splitlines() if line.startswith('All 90 slices'))
assert [float(v) for v in re.findall(r'[+-]?\d+\.\d+',line)]==averages(grid)
text=(ROOT/next(t['file'] for t in index if t['label']=='tab:supp-sensitivity-slices')).read_text()
for prefix,field,values in [(r'$\eta$', 'eta',[0,.1,.2,.3,.5,.7]),('Crop ratio','crop_ratio',[.6,.7,.8,.9])]:
    line=next(line for line in text.splitlines() if line.startswith(prefix))
    actual=[float(v) for v in re.findall(r'[+-]?\d+\.\d+',line.split('&',1)[1])]
    other,fixed=('crop_ratio',.8) if field=='eta' else ('eta',.3)
    expected=[v for value in values for v in averages([r for r in grid if float(r[field])==value and float(r[other])==fixed])]
    assert actual==expected,(field,actual,expected)
print('PASS: both sensitivity tables (42 values) match the corrected 90-row grid.')
text=(ROOT/next(t['file'] for t in index if t['label']=='tab:supp-taskres-sstext')).read_text()
lookup={(r['model'],r['dataset'],r['reference']):r for r in rows('fewshot_taskres_sstext_comparisons.csv')}
count=0
for line in text.splitlines():
    if not line.startswith(('BiomedCLIP &','AlphaRAD &')):continue
    model,dataset,method,*_= [x.strip() for x in line.split('&')]
    r=lookup[model,{'RSNA':'rsna_pneumonia','ChestXDet10':'chestxdet10'}[dataset],{'TaskRes':'taskres','SS-Text+':'sstext_plus'}[method]]
    values=[float(v) for v in re.findall(r'[+-]?\d+\.\d+',line)]
    expected=[float(f'{float(r[k]):.2f}') for metric in ['auroc','auprc'] for k in [metric+'_delta_pp',metric+'_delta_pp_ci_low',metric+'_delta_pp_ci_high']]
    assert values==expected,(model,dataset,method)
    count+=len(values)
assert count==48
print('PASS: TaskRes / SS-Text+ paired table (48 values).')
