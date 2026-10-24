#!/usr/bin/env python3
"""Diagnostic repair of an observed native procedural error, using shared demos."""
import time
import re
from pathlib import Path
from agent_reliability.io import digest,file_hash,read_json,write_json
from agent_reliability.reflexion import run_attempt,ROOT
from agent_reliability.reference import CompletionTransport
from agent_reliability.dataset import load_manifest
from agent_reliability.cli import provenance

def main():
    dev=Path('runs/reflexion-dev12-20261022');parent=dev/'tasks/181a4c810cec8a32/trial-00'
    result=read_json(parent/'result.json');run=read_json(dev/'run.json')
    if run['identity']['source_hashes']!=provenance()['source_hashes']: raise ValueError('Parent code changed')
    if result['success'] or result['trajectory_sha256']!=digest(read_json(parent/'checkpoint.json')['transitions']):
        raise ValueError('Parent trajectory mismatch')
    memory=list(result['memory_after']);text=memory[-1]
    edits=[('I should have put cup 2 in microwave 1, then heat it.',
            'I should have used heat cup 2 with microwave 1 while holding the cup.'),
           ('In the next trial, I will try to put the cup 2 in the microwave 1 again',
            'In the next trial, I will use the demonstrated command heat cup 2 with microwave 1')]
    fixed=text
    for before,after in edits:
        if fixed.count(before)!=1: raise ValueError('Native plan span changed')
        fixed=fixed.replace(before,after,1)
    demos=read_json(ROOT/'prompts/alfworld_3prompts.json')
    if '> heat apple 1 with microwave 1\n' not in demos['react_heat_1']: raise ValueError('Shared tool example missing')
    task=next(t for t in load_manifest(Path('repro/baseline-20261019/dev12.json'))['tasks'] if t['task_id']==result['task_id'])
    output=Path('runs/reflexion-native-repair-20261022')
    transport=CompletionTransport('http://127.0.0.1:8001/v1','local-development',180)
    if transport._request('/models')!=run['identity']['serving']: raise ValueError('Serving changed')
    identity={'parent_result_sha256':file_hash(parent/'result.json'),'edits':edits,
              'support':{'source':'shared react_heat_1 demonstration','sha256':digest(demos['react_heat_1']),
                         'command_schema':'heat OBJECT with microwave 1'},
              'scope':'selected development diagnostic; not a prospective population-effect estimate',
              'driver_sha256':file_hash(Path(__file__)),'max_calls':500,'max_seconds':1800,
              'deadline_unix':run['created_unix']+7200,'seed':42}
    if (output/'run.json').exists():
        if read_json(output/'run.json')['identity']!=identity: raise ValueError('Identity changed')
    else: write_json(output/'run.json',{'identity':identity,'created_unix':time.time()})
    def allowance():
        if len(list(output.glob('branches/*/trial-*/calls/*/attempt-*.request.json')))>=500:
            raise RuntimeError('Native repair call allowance')
        if time.time()>min(identity['deadline_unix'],read_json(output/'run.json')['created_unix']+1800):
            raise RuntimeError('Native repair time allowance')
    rows=[]
    for branch,start_memory in [('retain_native',memory),('repair_supported',memory[:-1]+[fixed])]:
        current=start_memory;attempts=[];invalid=[]
        for trial in [1,2]:
            directory=output/'branches'/branch/f'trial-{trial:02d}'
            row=run_attempt(task,current,trial,directory,transport,Path('data/alfworld'),42,allowance)
            current=row['memory_after'];attempts.append(row)
            actions=read_json(directory/'checkpoint.json')['transitions']
            invalid.append(sum(x['action'] in ['put cup 2 in microwave 1','put cup 2 in the microwave 1'] for x in actions))
            print({'branch':branch,'trial':trial,'success':row['success'],'invalid_put_actions':invalid[-1]},flush=True)
            if row['success']: break
        rows.append({'branch':branch,'attempts':len(attempts),'success':any(r['success'] for r in attempts),
          'decisions':sum(r['decisions'] for r in attempts),'dispatched_calls':sum(r['dispatched_calls'] for r in attempts),
          'input_tokens':sum(r['input_tokens_known'] for r in attempts),'output_tokens':sum(r['output_tokens_known'] for r in attempts),
          'invalid_put_actions_by_attempt':invalid})
        write_json(output/'summary.json',{'expected_branches':2,'completed_branches':len(rows),'rows':rows,
                   'scope':identity['scope'],'parent_result_sha256':identity['parent_result_sha256']})

if __name__=='__main__': main()
