#!/usr/bin/env python3
"""Run the independent saved-result audit once the existing campaign finishes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from evaluate_baseline_buffered import write,now

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();run=args.run.resolve()
    write(run/'finalizer_status.json',{'status':'waiting_for_full_evaluation','pid':os.getpid(),'started_at':now()})
    while True:
        state=json.loads((run/'status.json').read_text())
        if state['status']!='running':break
        pid=state['controller_pid'];stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split(') ',1)[1].split()[0]=='Z':
            write(run/'finalizer_status.json',{'status':'controller_exited_before_completion','updated_at':now(),
                    'note':'Inference workers are not interrupted. Resume the recorded controller to continue summary.'})
            return
        time.sleep(15)
    auditor=ROOT/'local_setup/audit_query_transfer.py'
    if not auditor.is_file():raise FileNotFoundError(auditor)
    cmd=['bash',str(ROOT.parent/'AlphaRAD/local_setup/python_local.sh'),str(auditor),
         '--run',str(run),'--require-complete']
    with (run/'final_audit.log').open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
    write(run/'finalizer_status.json',dict(status='audit_passed' if result.returncode==0 else 'audit_needs_attention',
        returncode=result.returncode,campaign_status=state['status'],finished_at=now(),log=str(run/'final_audit.log')))

if __name__=='__main__':main()
