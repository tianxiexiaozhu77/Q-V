#!/usr/bin/env python3
"""Refresh the campaign report while its own controller is active."""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEADLINE = 1792634400
while True:
    result = subprocess.run([sys.executable, str(ROOT/'code/campaign_report.py'), '--verify-protected'], capture_output=True, text=True)
    with (ROOT/'logs/report_watch.log').open('a') as handle:
        handle.write(time.strftime('%Y-%m-%d %H:%M:%S %Z') + ' ' + str(result.returncode) + ' ' + result.stdout + result.stderr)
    state = json.loads((ROOT/'controller_v2/state.json').read_text())
    if time.time() >= DEADLINE + 10 or state.get('status') not in ('running', 'starting'):
        break
    time.sleep(30)
