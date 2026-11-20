#!/usr/bin/env python3
"""One leased-GPU feature run followed by support-only fitting and scoring."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from cache_native_common import write_json

p=argparse.ArgumentParser()
p.add_argument('--model',required=True,choices=['BiomedCLIP','AlphaRAD','RadZero'])
p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--gpu',type=int,required=True)
p.add_argument('--limit',type=int,default=0)
a=p.parse_args()
root=Path(__file__).resolve().parent
a.output.mkdir(parents=True,exist_ok=True)
worker={'BiomedCLIP':'cache_biomed.py','AlphaRAD':'cache_alpharad.py','RadZero':'cache_radzero.py'}[a.model]
batch=16 if a.model=='BiomedCLIP' else 8
for stage,command in [
 ('cache',[sys.executable,str(root/worker),'--protocol',str(a.protocol),'--output',str(a.output/'cache'),
           '--gpu',str(a.gpu),'--limit',str(a.limit),'--batch-size',str(batch)]),
 ('evaluate',[sys.executable,str(root/'evaluate.py'),'--protocol',str(a.protocol),
              '--cache',str(a.output/'cache'),'--output',str(a.output/'evaluation')])]:
    write_json(a.output/'status.json',{'stage':stage,'pid':os.getpid()})
    subprocess.run(command,check=True)
write_json(a.output/'status.json',{'stage':'complete','pid':os.getpid()})
