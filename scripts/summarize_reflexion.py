#!/usr/bin/env python3
"""Summarize recorded requests and task outcomes, including partial runs."""
import argparse
from pathlib import Path
from agent_reliability.io import read_json,write_json,digest
from agent_reliability.dataset import load_manifest
from agent_reliability.llm import CallJournal

def costs(root):
    journals=[CallJournal(p,None,295).totals() for p in root.glob('**/calls') if p.is_dir()]
    keys=['dispatched_calls','saved_responses','uncertain_or_unrecorded_attempts','input_tokens_known',
          'output_tokens_known','truncated_responses','usage_missing_responses']
    return {k:sum(j[k] for j in journals) for k in keys}

def summarize(parent,forks=None,manifest_path=Path('repro/baseline-20261019/dev12.json')):
    manifest=load_manifest(manifest_path);tasks=manifest['tasks'];rows=[]
    identity=read_json(parent/'run.json')['identity'];trials=identity['trials']
    if identity['manifest_sha256']!=manifest['sha256']: raise ValueError('Summary manifest mismatch')
    for task in tasks:
        attempts=[read_json(p) for p in sorted((parent/'tasks'/task['task_id']).glob('trial-*/result.json'))]
        success=any(r['success'] for r in attempts)
        rows.append({'task_id':task['task_id'],'scene_id':task['scene_id'],'task_type':task['task_type'],
          'completed':success or len(attempts)==trials,'success':success if success or len(attempts)==trials else None,
          'first_attempt_success':attempts[0]['success'] if attempts else None,'attempts':len(attempts),
          'decisions':sum(r['decisions'] for r in attempts),'reflections':sum(len(r['memory_after'])>len(r['memory_before']) for r in attempts)})
    report={'scope':'development; pinned author functions with Qwen3-8B raw-completion substitution',
      'expected_tasks':len(tasks),'completed_tasks':sum(r['completed'] for r in rows),
      'missing_tasks':sum(not r['completed'] for r in rows),'successes':sum(r['success'] is True for r in rows),
      'first_attempt_successes':sum(r['first_attempt_success'] is True for r in rows),
      'attempts_completed':sum(r['attempts'] for r in rows),'reflections':sum(r['reflections'] for r in rows),
      'tasks':rows,'costs':costs(parent)}
    if forks and (forks/'run.json').exists():
        summary=read_json(forks/'summary.json') if (forks/'summary.json').exists() else {'expected_branches':5,'completed_branches':0,'rows':[]}
        # Compare real fresh calls, not a shared response cache.
        paths=[forks/'branches'/b/'trial-01'/'calls' for b in ['original','sham','correct_target']]
        control={}
        for other,path in zip(['sham','correct_target'],paths[1:]):
            a={p.parent.name:read_json(p)['payload_sha256'] for p in paths[0].glob('*/attempt-000.request.json')}
            b={p.parent.name:read_json(p)['payload_sha256'] for p in path.glob('*/attempt-000.request.json')}
            complete=all((p.parent/'result.json').exists() for p in [paths[0],path])
            control[other]={'original_calls':len(a),'other_calls':len(b),
                            'same_complete_payload_sequence':a==b if complete else None}
        report['forks']={**summary,'costs':costs(forks),'fresh_input_controls':control}
    report['sha256']=digest(report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--forks',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('repro/baseline-20261019/dev12.json'));a=p.parse_args()
    result=summarize(a.parent_run,a.forks,a.manifest);write_json(a.output,result)
    print({k:v for k,v in result.items() if k not in ['tasks','forks']})
    if 'forks' in result: print(result['forks'])

if __name__=='__main__': main()
