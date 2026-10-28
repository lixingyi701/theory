#!/usr/bin/env python3
"""Evaluate frozen paired content contrasts and independently audit raw attempts."""
from pathlib import Path
from collections import defaultdict
import statistics as st
import math
import numpy as np
from agent_reliability.io import digest, file_hash, read_json
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import checked
from prepare_content_variants import CONDITIONS, SEEDS
from run_content_intervention import PROTOCOL, PRIVATE, PUBLIC, validate, T_CRITICAL
from diagnose_second_attempt import trace_features


def contrast(rows,left,right,key,critical=T_CRITICAL):
    lookup={(r['task_id'],r['seed'],r['condition']):r for r in rows}
    scene_tasks=defaultdict(set)
    for r in rows:scene_tasks[r['scene_id']].add(r['task_id'])
    values={}
    for scene,tasks in sorted(scene_tasks.items()):
        task_values=[]
        for task in sorted(tasks):
            task_values.append(st.mean(float(lookup[task,s,left][key])-float(lookup[task,s,right][key]) for s in SEEDS))
        values[scene]=st.mean(task_values)
    ds=list(values.values());mean=st.mean(ds);se=st.stdev(ds)/math.sqrt(len(ds))
    return {'left':left,'right':right,'outcome':key,'difference':mean,'paired_scene_SE':se,
        't95':[mean-critical*se,mean+critical*se],'scene_differences':values}


def smoke():
    # Known crossed fixture tests scene/task/seed weighting and contrast sign.
    rows=[]
    for scene,task in [('a','x'),('a','y'),('b','z')]:
        for seed in SEEDS:
            for condition in CONDITIONS:
                rows.append({'scene_id':scene,'task_id':task,'seed':seed,'condition':condition,
                    'success':condition=='neutralized' and task!='y'})
    c=contrast(rows,'neutralized','paraphrase','success')
    assert abs(c['difference']-.75)<1e-15 and c['scene_differences']=={'a':.5,'b':1.}
    reverse=contrast(rows,'paraphrase','neutralized','success')
    assert reverse['difference']==-c['difference']
    print({'paired_weighting_fixture_passed':True,'new_model_calls':0})


def main():
    p,variants=validate();summary=checked(PRIVATE/'summary.json')
    assert summary['status']=='complete' and summary['completed_attempts']==276
    assert [(r['id'],r['memory_sha256']) for r in summary['rows']]==[(c['id'],c['memory_sha256']) for c in p['cells']]
    rows=[]
    for saved,cell in zip(summary['rows'],p['cells']):
        path=Path(saved['result_path']);assert file_hash(path)==saved['result_file_sha256']
        result=read_json(path);identity=checked(path.parent/'identity.json');cp=read_json(path.parent/'checkpoint.json')
        case=p['cases'][cell['index']];memory=variants[cell['index']]['memories'][cell['condition']]
        assert identity['task']==case['task'] and identity['memory']==memory and identity['seed']==cell['seed'] and identity['trial']==2
        assert result['task_id']==cell['task_id'] and result['success']==saved['success']==bool(cp['transitions'][-1]['won'])
        assert result['memory_before']==memory and digest(memory)==cell['memory_sha256']
        assert result['initial_observation_sha256']==case['initial_observation_sha256']
        assert result['memory_after'][:len(memory)]==memory and len(result['memory_after'])==len(memory)+(not result['success'])
        task_dir=Path(case['task']['gamefile']).parts[-3].split('-');target=task_dir[1].lower()
        trace=trace_features(cp,target)
        request=read_json(path.parent/'calls/actor-0000/attempt-000.request.json')['payload']
        prompt=next(r for r in read_json(PRIVATE/'interface-prompts.json') if r['index']==cell['index'] and r['condition']==cell['condition'])
        assert request['prompt']==prompt['wrapped_prompt']
        # First request seed is independent of condition and identical within the paired seed.
        row={**cell,'success':result['success'],'heated':bool(trace['target_heat_success_steps']),
            'put_then_heat':bool(trace['placed_then_heat_without_pickup']),'physical_no_effect':trace['physical_no_effect'],
            'decisions':result['decisions'],'calls':result['dispatched_calls'],'result_file_sha256':saved['result_file_sha256'],
            'checkpoint_file_sha256':file_hash(path.parent/'checkpoint.json'),'trace':trace,
            'first_request_seed':request['seed'],'truncated_responses':result['truncated_responses']}
        rows.append(row)
    for task in {r['task_id'] for r in rows}:
        for seed in SEEDS:
            assert len({r['first_request_seed'] for r in rows if r['task_id']==task and r['seed']==seed})==1
    comparisons={}
    for left,right in [('neutralized','paraphrase'),('neutralized','original'),('paraphrase','original')]:
        comparisons[left+'_minus_'+right]={key:contrast(rows,left,right,key) for key in ['success','heated','put_then_heat','physical_no_effect','decisions','calls']}
    primary=comparisons['neutralized_minus_paraphrase']['success'];d=np.array(list(primary['scene_differences'].values()))
    rng=np.random.default_rng(2026102802);draws=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(axis=1)
    primary['secondary_whole_scene_bootstrap95']=np.quantile(draws,[.025,.975]).tolist()
    summaries={}
    for condition in CONDITIONS:
        rr=[r for r in rows if r['condition']==condition]
        by_scene=defaultdict(lambda:defaultdict(list))
        for r in rr:by_scene[r['scene_id']][r['task_id']].append(float(r['success']))
        summaries[condition]={'attempts':len(rr),'successes':sum(r['success'] for r in rr),
            'scene_equal_success_fraction':st.mean(st.mean(st.mean(seeds) for seeds in tasks.values()) for tasks in by_scene.values()),
            'heated_attempts':sum(r['heated'] for r in rr),'put_then_heat_attempts':sum(r['put_then_heat'] for r in rr),
            'truncated_responses':sum(r['truncated_responses'] for r in rr)}
    audit=read_json(PRIVATE/'protocol-audit.json');assert audit['passed']
    result=save(PUBLIC/'content-intervention-results-20261028.json',{'protocol_sha256':p['sha256'],
        'summary_file_sha256':file_hash(PRIVATE/'summary.json'),'source_sha256':file_hash(Path(__file__)),
        'attempts':276,'tasks':46,'scenes':24,'conditions':summaries,'primary':primary,'comparisons':comparisons,
        'checks_passed':True,'primary_positive_interval':primary['t95'][0]>0,'parameter_fits':0,'holdout134_calls':0,
        'scope':'Prospective local content effects on reused tasks, not fresh-task budget validation or a refitted predictor',
        'protocol_audit_sha256':file_hash(PRIVATE/'protocol-audit.json')})
    for i,start in enumerate(range(0,len(rows),92)):
        save(PUBLIC/f'content-intervention-observed-{i+1:02d}-20261028.json',{'result_sha256':result['sha256'],'rows':rows[start:start+92]})
    print({'result':result['sha256'],'conditions':summaries,'primary':primary},flush=True)


if __name__=='__main__':
    import sys
    smoke() if '--smoke' in sys.argv else main()
