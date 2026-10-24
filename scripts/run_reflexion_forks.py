#!/usr/bin/env python3
"""Paired attempt-boundary pilot from an audited native reflection span."""
import argparse
import time
from pathlib import Path
from agent_reliability.io import digest,file_hash,read_json,write_json
from agent_reliability.dataset import load_manifest
from agent_reliability.reference import CompletionTransport
from agent_reliability.reflexion import run_attempt
from agent_reliability.cli import provenance

def memories(parent,target):
    original=list(parent['memory_after']);text=original[-1]
    start,end=target['span'];true=target['original_text'];wrong=target['replacement_text']
    if text[start:end]!=true or true==wrong: raise ValueError('Target span mismatch')
    poisoned=text[:start]+wrong+text[end:]
    fixed=poisoned[:start]+true+poisoned[start+len(wrong):]
    if fixed!=text: raise ValueError('Repair changed non-target content')
    return {'original':original,'sham':list(original),'retain_error':original[:-1]+[poisoned],
            'correct_target':original[:-1]+[fixed],'discard_new':original[:-1]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials',type=int,default=2);p.add_argument('--max-calls',type=int,default=1200)
    p.add_argument('--max-seconds',type=int,default=3600);a=p.parse_args()
    target=read_json(a.target)
    if target['sha256']!=digest({k:v for k,v in target.items() if k!='sha256'}): raise ValueError('Target integrity')
    parent_dir=a.parent_run/'tasks'/target['task_id']/f"trial-{target['parent_trial']:02d}"
    parent=read_json(parent_dir/'result.json')
    if parent['success'] or file_hash(parent_dir/'result.json')!=target['parent_result_sha256']: raise ValueError('Parent mismatch')
    if not target['public_evidence_reviewed']: raise ValueError('Public fact evidence required')
    task=next(t for t in load_manifest(Path('repro/baseline-20261019/dev12.json'))['tasks'] if t['task_id']==target['task_id'])
    branches=memories(parent,target);upstream=read_json(a.parent_run/'run.json')['identity']
    if upstream['source_hashes']!=provenance()['source_hashes']: raise ValueError('Parent source mismatch')
    transport=CompletionTransport('http://127.0.0.1:8001/v1','local-development',180)
    if transport._request('/models')!=upstream['serving']: raise ValueError('Parent serving mismatch')
    identity={'parent_identity':parent['identity_sha256'],'target':target,'trials':a.trials,
              'max_calls':a.max_calls,'max_seconds':a.max_seconds,'driver_sha256':file_hash(Path(__file__)),
              'source_hashes':provenance()['source_hashes'],'seed':42,'branches':{k:digest(v) for k,v in branches.items()}}
    if (a.output/'run.json').exists():
        if read_json(a.output/'run.json')['identity']!=identity: raise ValueError('Fork identity changed')
    else: write_json(a.output/'run.json',{'identity':identity,'created_unix':time.time(),'scope':'paired boundary development pilot'})
    def allowance():
        count=len(list(a.output.glob('branches/*/trial-*/calls/*/attempt-*.request.json')))
        if count>=a.max_calls or time.time()-read_json(a.output/'run.json')['created_unix']>=a.max_seconds:
            raise RuntimeError('Fork allowance exhausted')
    rows=[];initial=None
    for branch,memory in branches.items():
        attempts=[]
        for offset in range(a.trials):
            trial=target['parent_trial']+1+offset
            result=run_attempt(task,memory,trial,a.output/'branches'/branch/f'trial-{trial:02d}',transport,Path('data/alfworld'),42,allowance)
            initial=initial or result['initial_observation_sha256']
            if initial!=result['initial_observation_sha256']: raise ValueError('Fork initial observation differs')
            attempts.append(result);memory=result['memory_after']
            print({'branch':branch,'trial':trial,'success':result['success'],'decisions':result['decisions']},flush=True)
            if result['success']: break
        row={'branch':branch,'success':any(r['success'] for r in attempts),'attempts':len(attempts),
             'decisions':sum(r['decisions'] for r in attempts),'calls':sum(r['dispatched_calls'] for r in attempts),
             'input_tokens':sum(r['input_tokens_known'] for r in attempts),'output_tokens':sum(r['output_tokens_known'] for r in attempts)}
        rows.append(row);write_json(a.output/'summary.json',{'expected_branches':len(branches),'completed_branches':len(rows),'rows':rows,
              'task_id':task['task_id'],'matched_initial_observation_sha256':initial,'target_sha256':target['sha256']})

if __name__=='__main__': main()
