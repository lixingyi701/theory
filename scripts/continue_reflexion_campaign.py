#!/usr/bin/env python3
"""Bounded stage continuation for the already running Reflexion development job."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from agent_reliability.io import read_json,write_json,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--dev-pid',type=int,required=True);a=p.parse_args()
    dev=Path('runs/reflexion-dev12-20261022');forks=Path('runs/reflexion-forks-20261022')
    status=Path('runs/reflexion-campaign-20261022.json')
    deadline=read_json(dev/'run.json')['created_unix']+7200
    def state(stage,**extra):
        write_json(status,{'stage':stage,'updated_unix':time.time(),'deadline_unix':deadline,
            'max_total_dispatched_calls':5200,'controller_sha256':file_hash(Path(__file__)),**extra})
        print(stage,flush=True)
    state('waiting_for_dev12')
    while time.time()<deadline:
        if (dev/'summary.json').exists() and read_json(dev/'summary.json')['completed_tasks']==12: break
        try: os.kill(a.dev_pid,0)
        except ProcessLookupError: state('dev12_stopped_before_completion');return
        time.sleep(10)
    else: state('campaign_deadline');return
    subprocess.run([sys.executable,'scripts/summarize_reflexion.py','--parent-run',str(dev),'--forks',str(forks),
                    '--output','runs/reflexion-dev12-final-20261022.json'],check=True)
    consumed=sum(len(list(root.glob('**/calls/*/attempt-*.request.json'))) for root in [dev,forks])
    allowance=min(2000,5200-consumed)
    if allowance<=0: state('campaign_call_allowance');return
    # The pilot entry checks native capability, the immutable task manifest,
    # both environment audits, prior requests and remaining campaign limits.
    state('starting_new_scene_pilot',prior_calls=consumed,pilot_max_calls=allowance)
    pilot=Path('runs/reflexion-pilot24-20261022')
    try:
        with Path('runs/setup/reflexion-pilot24.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u','scripts/run_reflexion_pilot.py','--output',str(pilot),
                                   '--max-calls',str(allowance)],stdout=log,stderr=subprocess.STDOUT,
                                  timeout=max(1,int(deadline-time.time())))
    except subprocess.TimeoutExpired:
        state('campaign_deadline');return
    if (pilot/'run.json').exists():
        subprocess.run([sys.executable,'scripts/summarize_reflexion.py','--parent-run',str(pilot),
            '--manifest','repro/reflexion-20261022/pilot24.json','--output','runs/reflexion-pilot24-final-20261022.json'],check=True)
    summary=read_json(pilot/'summary.json') if (pilot/'summary.json').exists() else {}
    state('pilot24_completed' if result.returncode==0 and summary.get('completed_tasks')==24 else 'pilot24_stopped',
          returncode=result.returncode,completed_tasks=summary.get('completed_tasks',0),
          next_stage='review qualifying propositions and freeze calibration protocol')

if __name__=='__main__': main()
