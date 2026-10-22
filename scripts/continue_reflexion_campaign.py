#!/usr/bin/env python3
"""Continue the bounded campaign: raw reference, Qwen interface check, new pilot."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from agent_reliability.io import read_json,write_json,file_hash
from agent_reliability.dataset import load_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--dev-pid',type=int,required=True);a=p.parse_args()
    raw=Path('runs/reflexion-dev12-20261022');chat=Path('runs/reflexion-chat-parsed-dev12-20261022')
    deadline=read_json(raw/'run.json')['created_unix']+7200
    def state(stage,**values):
        write_json(Path('runs/reflexion-campaign-20261022.json'),{'stage':stage,'updated_unix':time.time(),
            'deadline_unix':deadline,'max_total_dispatched_calls':5200,
            'controller_sha256':file_hash(Path(__file__)),**values})
        print(stage,flush=True)
    def remaining_calls():
        roots=[p for p in Path('runs').glob('reflexion-*') if p.is_dir()]
        return 5200-sum(len(list(r.glob('**/calls/*/attempt-*.request.json'))) for r in roots)
    def execute(script,output,limit,extras=()):
        allowance=min(limit,remaining_calls())
        if (output/'run.json').exists():
            allowance=read_json(output/'run.json')['identity']['max_calls']
            used=len(list(output.glob('**/calls/*/attempt-*.request.json')))
            if allowance>used+remaining_calls(): state('reserved_allowance_conflict');return False
        if allowance<=0 or time.time()>=deadline: state('campaign_allowance_exhausted');return False
        state('running_'+output.name,allocated_calls=allowance)
        try:
            with (Path('runs/setup')/Path(script).stem).with_suffix('.log').open('a') as log:
                result=subprocess.run([sys.executable,'-u',script,'--output',str(output),'--max-calls',str(allowance),*extras],
                    stdout=log,stderr=subprocess.STDOUT,timeout=max(1,int(deadline-time.time())))
        except subprocess.TimeoutExpired: state('campaign_deadline');return False
        if result.returncode!=0: state('stage_stopped',script=script,returncode=result.returncode);return False
        return True
    state('waiting_for_raw_reference')
    while time.time()<deadline:
        if (raw/'summary.json').exists() and read_json(raw/'summary.json')['completed_tasks']==12: break
        try: os.kill(a.dev_pid,0)
        except ProcessLookupError: state('raw_reference_stopped');return
        time.sleep(10)
    else: state('campaign_deadline');return
    subprocess.run([sys.executable,'scripts/summarize_reflexion.py','--parent-run',str(raw),
                    '--forks','runs/reflexion-forks-20261022','--output','runs/reflexion-raw-final-20261022.json'],check=True)
    if not execute('scripts/run_reflexion_chat.py',chat,1200,['--input-format','qwen3_non_thinking_actionline']): return
    subprocess.run([sys.executable,'scripts/summarize_reflexion.py','--parent-run',str(chat),
                    '--output','runs/reflexion-chat-parsed-final-20261022.json'],check=True)
    report=read_json(chat/'summary.json');manifest=load_manifest(Path('repro/baseline-20261019/dev12.json'))
    won={r['task_id'] for r in report['rows'] if r['success']}
    kinds={t['task_type'] for t in manifest['tasks'] if t['task_id'] in won}
    if report['completed_tasks']!=12 or report['successes']<6 or len(kinds)<4:
        state('chat_capability_not_met',successes=report['successes'],successful_types=sorted(kinds));return
    pilot=Path('runs/reflexion-pilot24-20261022')
    if not execute('scripts/run_reflexion_pilot.py',pilot,2000,
                   ['--capability-run',str(chat),'--input-format','qwen3_non_thinking_actionline']): return
    subprocess.run([sys.executable,'scripts/summarize_reflexion.py','--parent-run',str(pilot),
        '--manifest','repro/reflexion-20261022/pilot24.json','--output','runs/reflexion-pilot24-final-20261022.json'],check=True)
    state('pilot24_completed',next_stage='review qualifying propositions and freeze calibration protocol')

if __name__=='__main__': main()
