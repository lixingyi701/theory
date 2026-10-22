#!/usr/bin/env python3
"""Qwen-template development check; same author functions and generation settings."""
import argparse
import time
from pathlib import Path
from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest,file_hash,read_json,write_json,run_lock
from agent_reliability.reference import CompletionTransport
from agent_reliability.reflexion import author_namespace,run_attempt
from agent_reliability.audit import verify_audit
from agent_reliability.cli import provenance

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('repro/baseline-20261019/dev12.json'))
    p.add_argument('--data-root',type=Path,default=Path('data/alfworld'))
    p.add_argument('--trials',type=int,default=3);p.add_argument('--max-calls',type=int,default=1200)
    p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--stop-after-tasks',type=int)
    p.add_argument('--input-format',choices=['qwen3_non_thinking','qwen3_non_thinking_actionline'],default='qwen3_non_thinking')
    a=p.parse_args();start=time.monotonic();manifest=load_manifest(a.manifest)
    raw=Path('runs/reflexion-dev12-20261022')
    if read_json(raw/'summary.json')['completed_tasks']!=12: raise ValueError('Complete raw reference first')
    preceding=[p for p in Path('runs').glob('reflexion-*') if p.is_dir() and p.resolve()!=a.output.resolve()]
    consumed=sum(len(list(root.glob('**/calls/*/attempt-*.request.json'))) for root in preceding)
    if consumed+a.max_calls>5200: raise ValueError('Whole campaign call allowance')
    campaign_deadline=read_json(raw/'run.json')['created_unix']+7200
    if time.time()>=campaign_deadline: raise ValueError('Campaign deadline')
    wrapper=read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
    if wrapper['model_revision']!='b968826d9c46dd6066d109eabc6255188de91218': raise ValueError('Wrapper revision')
    if a.manifest.name!='dev12.json': raise ValueError('Development entry: holdout use requires a frozen protocol')
    audit=read_json(Path('runs/audit/remote-direct-native-v2-20261022.json'));verify_audit(audit,manifest)
    equivalent=read_json(Path('runs/audit/official-gym-native-v3-20261022.json'));current=provenance()
    if digest({k:v for k,v in equivalent.items() if k!='sha256'})!=equivalent['sha256'] or not equivalent['passed']:
        raise ValueError('Environment audit integrity')
    if equivalent['manifest_sha256']!=manifest['sha256'] or equivalent['native_hashes']!=current['native_hashes']:
        raise ValueError('Environment audit mismatch')
    for name in ['alfworld.py','env.py','audit.py','reference.py']:
        if equivalent['source_hashes'][name]!=current['source_hashes'][name]: raise ValueError('Environment adapter changed')
    transport=CompletionTransport('http://127.0.0.1:8001/v1','local-development',180)
    health=transport._request('/models');_,upstream=author_namespace()
    if health['serving_profile']!='upstream_reflexion_raw_serial_v1' or health['model_revision']!='b968826d9c46dd6066d109eabc6255188de91218': raise ValueError('Serving identity')
    identity={'manifest_sha256':manifest['sha256'],'trials':a.trials,'max_calls':a.max_calls,
              'max_seconds':a.max_seconds,'campaign_deadline_unix':campaign_deadline,'prior_calls':consumed,'input_format':a.input_format,'wrapper':wrapper,'seed':42,'serving':health,'upstream':upstream,
              'source_hashes':current['source_hashes'],'driver_sha256':file_hash(Path(__file__)),
              'audit_sha256':audit['sha256'],'environment_audit_sha256':equivalent['sha256']}
    if (a.output/'run.json').exists():
        if read_json(a.output/'run.json')['identity']!=identity: raise ValueError('Run identity changed')
    else: write_json(a.output/'run.json',{'identity':identity,'created_unix':time.time(),'scope':'development capability and instrumentation'})
    def allowance():
        count=len(list(a.output.glob('tasks/*/trial-*/calls/*/attempt-*.request.json')))
        elapsed=time.time()-read_json(a.output/'run.json')['created_unix']
        if count>=a.max_calls or elapsed>=a.max_seconds or time.time()>=campaign_deadline: raise RuntimeError('Campaign allowance exhausted')
    rows=[]
    for i,task in enumerate(manifest['tasks']):
        memory=[];attempts=[]
        for trial in range(a.trials):
            result=run_attempt(task,memory,trial,a.output/'tasks'/task['task_id']/f'trial-{trial:02d}',transport,a.data_root,42,allowance,input_format=a.input_format)
            memory=result['memory_after'];attempts.append(result)
            print({k:result[k] for k in ['task_id','trial','success','decisions','dispatched_calls']},flush=True)
            if result['success']: break
        row={'task_id':task['task_id'],'success':any(r['success'] for r in attempts),'attempts':len(attempts)}
        rows.append(row)
        write_json(a.output/'summary.json',{'expected_tasks':len(manifest['tasks']),'completed_tasks':len(rows),
            'successes':sum(r['success'] for r in rows),'rows':rows,'elapsed_seconds':time.monotonic()-start,
            'dispatched_calls':len(list(a.output.glob('tasks/*/trial-*/calls/*/attempt-*.request.json'))),
            'saved_responses':len(list(a.output.glob('tasks/*/trial-*/calls/*/response.json')))})
        if a.stop_after_tasks and i+1>=a.stop_after_tasks: break

if __name__=='__main__': main()
