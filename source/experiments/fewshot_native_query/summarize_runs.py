#!/usr/bin/env python3
"""Combine explicit completed/pending run directories, without cherry-picking methods."""
import argparse
import json
from pathlib import Path
from cache_native_common import write_json
from run_suite import summarize

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
jobs={};sources=[]
for run in a.runs:
    state=json.loads((run/'status.json').read_text())
    if state['limit']:
        raise ValueError('Smoke runs cannot enter the formal report')
    for key,job in state['jobs'].items():
        if key in jobs:raise ValueError('Duplicate model/dataset run; select explicit provenance, not best result')
        jobs[key]={**job,'result_file':str((run/key/'evaluation/result.json').resolve())}
    sources.append(str(run.resolve()))
state={'status':'complete' if all(j['status']=='complete' for j in jobs.values()) else 'incomplete',
       'gpus':[4,5,6,7],'jobs':jobs,'source_runs':sources,
       'expected_model_dataset_combinations':6}
if len(jobs)!=6:state['status']='incomplete'
write_json(a.output/'catalog.json',state)
summarize(a.output,state)
print(a.output/'RESULTS.txt')
