#!/usr/bin/env python3
"""Frozen native-outcome and paired recovery-curve evaluator; no fitting."""
from pathlib import Path
from collections import defaultdict
import statistics as st,math
from agent_reliability.io import read_json,file_hash,digest,write_json,seed_for
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from prepare_recovery_variants import ROOT,PUBLIC,CONDITIONS,SEEDS
from run_recovery_mechanism import validate
from run_multiround_diagnostic import verify_attempt
from diagnose_second_attempt import trace_features
from summarize_reflexion import costs


def scene_curve(rows,condition,budget):
    scenes=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['condition']==condition:scenes[r['scene_id']][r['task_id']].append(float(r['success_curve'][budget-1]))
    assert len(scenes)==23
    return {s:st.mean(st.mean(values) for values in tasks.values()) for s,tasks in sorted(scenes.items())}


def difference(rows,left,right,budget,critical):
    a=scene_curve(rows,left,budget);b=scene_curve(rows,right,budget);d={s:a[s]-b[s] for s in a}
    mean=st.mean(d.values());se=st.stdev(d.values())/math.sqrt(23)
    return dict(difference=mean,paired_scene_se=se,interval=[mean-critical*se,mean+critical*se],scene_differences=d)


def engineering_checks():
    rows=[]
    for scene in range(23):
        for task in range(10 if scene==0 else 1):
            for seed in SEEDS:
                for c in ['correct','paraphrase']:
                    rows.append(dict(condition=c,scene_id=str(scene),task_id=f'{scene}-{task}',seed=seed,
                        success_curve=[bool(scene==0 and c=='correct')]*4))
    d=difference(rows,'correct','paraphrase',4,2.1);reverse=difference(rows,'paraphrase','correct',4,2.1)
    assert abs(d['difference']-1/23)<1e-12 and abs(reverse['difference']+1/23)<1e-12
    memory=['N1','N2','C'];seen=[]
    for j in range(1,4):seen.append(memory[-3:]);memory=memory+[f'J{j}']
    keep=list(memory);keep[-3]='C'
    assert seen==[['N1','N2','C'],['N2','C','J1'],['C','J1','J2']]
    assert memory[-3:]==['J1','J2','J3'] and keep[-3:]==['C','J2','J3']
    from calibrate_successor_success import features
    def state(t):return [{'type':'N','native':False},{'type':'N','native':False},{'type':t,'native':True}]
    assert features(state('C'),'seven')==features(state('U'),'seven')
    return dict(unequal_task_scene_weights_and_sign=True,common_prefix_retention_boundary=True,
        corrected_removed_target_feature_alias=True,scope='Engineering fixtures only, no Agent or measured task result')


def main():
    p,variants=validate();summary=checked(ROOT/'summary.json')
    assert summary['status']=='complete' and summary['completed_trajectories']==510 and summary['derived_completed']==102
    assert len({r['trajectory_id'] for r in summary['attempts_rows']})<=612
    prompts=read_json(ROOT/'interface-prompts.json');first={(r['index'],r['condition']):r['wrapped_prompt'] for r in prompts}
    actual={}; queue=[]
    for row in summary['attempts_rows']:
        path=Path(row['result_path']);r=read_json(path);identity=checked(path.parent/'identity.json');case=next(c for c in p['cases'] if c['task_id']==row['task_id'])
        assert file_hash(path)==row['result_file_sha256'] and r['success']==row['success']
        assert row['trial']==row['attempt_index']+1 and r['trial']==row['trial']
        cp=read_json(path.parent/'checkpoint.json');assert bool(cp['transitions'][-1]['won'])==r['success']
        verify_attempt(r,path.parent,case,{'seed':row['seed']},r['memory_before'],row['trial'])
        target=Path(case['task']['gamefile']).parts[-3].split('-')[1].lower();trace=trace_features(cp,target)
        request=read_json(path.parent/'calls/actor-0000/attempt-000.request.json')['payload']
        assert request['seed']==seed_for(row['task_id'],row['seed'],row['trial'],0,'actor')
        if row['attempt_index']==1:assert request['prompt']==first[case['index'],row['condition']]
        key=(row['trajectory_id'],row['attempt_index']);assert key not in actual
        actual[key]=dict(row,result=r,heated=bool(trace['target_heat_success_steps']),put_then_heat=bool(trace['placed_then_heat_without_pickup']),
            physical_no_effect=trace['physical_no_effect'],calls=r['dispatched_calls'],truncations=r['truncated_responses'])
        if not r['success']:
            queue.append(dict(trajectory_id=row['trajectory_id'],attempt_index=row['attempt_index'],condition=row['condition'],task_id=row['task_id'],
                scene_id=row['scene_id'],text=r['memory_after'][-1],text_sha256=digest(r['memory_after'][-1]),label=None))
    observed=[];by_id={t['id']:t for t in summary['trajectories']}
    for tr in summary['trajectories']:
        memory=variants[tr['index']]['memories'][tr['condition']];assert digest(memory)==tr['initial_memory_sha256']
        rr=[]
        for j,row in enumerate(tr['attempts'],1):
            a=actual[tr['id'],j];assert a['result']['memory_before']==memory
            assert a['result']['success']==(j==len(tr['attempts']) and tr['success'])
            rr.append(a);memory=a['result']['memory_after']
        assert tr['success'] or len(rr)==4
        observed.append(dict(trajectory_id=tr['id'],task_id=tr['task_id'],scene_id=tr['scene_id'],seed=tr['seed'],condition=tr['condition'],
            success_curve=[any(a['success'] for a in rr if a['attempt_index']<=b) for b in range(1,5)],
            attempts=len(rr),heated=any(a['heated'] for a in rr),put_then_heat=any(a['put_then_heat'] for a in rr),
            result_hashes=[a['result_file_sha256'] for a in rr]))
    for tr in summary['retention_trajectories']:
        parent=by_id[tr['shared_parent_trajectory']];pr=[actual[parent['id'],j] for j in range(1,len(parent['attempts'])+1)]
        if tr['alternative'] is None:
            assert any(a['success'] for a in pr[:3]);curve=[any(a['success'] for a in pr[:3] if a['attempt_index']<=b) for b in range(1,5)]
        else:
            assert len(pr)==4 and not any(a['success'] for a in pr[:3])
            a=actual[tr['id'],4];expected=list(pr[3]['result']['memory_before']);expected[-3]=variants[tr['index']]['memories']['correct'][-1]
            assert a['result']['memory_before']==expected and tr['shared_prefix']==parent['attempts'][:3]
            curve=[False,False,False,a['success']]
        observed.append(dict(trajectory_id=tr['id'],task_id=tr['task_id'],scene_id=tr['scene_id'],seed=tr['seed'],condition='correct_kept',
            success_curve=curve,shared_prefix=tr['shared_parent_trajectory'],extra_attempt=tr['alternative'] is not None))
    assert len(observed)==612 and len({(r['task_id'],r['seed'],r['condition']) for r in observed})==612
    pairs=[('correct','paraphrase'),('correct','delete_entry'),('correct','original'),('correct','remove_target'),('remove_target','delete_entry'),('paraphrase','original'),('correct_kept','correct')]
    contrasts=[]
    for left,right in pairs:
        for budget in range(1,5):
            primary=[left,right] in p['analysis']['primary'] and budget==4
            contrasts.append(dict(left=left,right=right,budget=budget,primary=primary,coverage=.975 if primary else .95,
                **difference(observed,left,right,budget,p['analysis']['t_primary'] if primary else p['analysis']['t_secondary'])))
    curves={c:dict(raw_success_counts=[sum(r['success_curve'][b] for r in observed if r['condition']==c) for b in range(4)],
        scene_equal_curve=[st.mean(scene_curve(observed,c,b).values()) for b in range(1,5)]) for c in CONDITIONS+['correct_kept']}
    forecasts=checked(PUBLIC/'recovery-mechanism-forecasts-20261029.json');assert forecasts['sha256']==p['forecasts_sha256']
    for c,rr in curves.items():
        rr['frozen_forecast']=forecasts['curves'][c]
        rr['forecast_minus_observed']=[a-b for a,b in zip(rr['frozen_forecast'],rr['scene_equal_curve'])]
    for c in contrasts:
        c['frozen_predicted_difference']=forecasts['curves'][c['left']][c['budget']-1]-forecasts['curves'][c['right']][c['budget']-1]
    write_json(ROOT/'failed-write-label-queue.json',dict(protocol_sha256=p['sha256'],scope='All actual failed writes, no outcome-based semantic selection',rows=queue))
    a=read_json(ROOT/'protocol-audit.json');assert a['passed'];cc=costs(ROOT)
    assert cc['dispatched_calls']==cc['saved_responses']==read_json(ROOT/'budget.json')['reserved'] and cc['uncertain_or_unrecorded_attempts']==0
    save(ROOT/'observed.json',dict(protocol_sha256=p['sha256'],rows=observed))
    result=save(PUBLIC/'recovery-mechanism-results-20261029.json',dict(protocol_sha256=p['sha256'],summary_sha256=summary['sha256'],
        tasks=51,scenes=23,base_trajectories=510,derived_trajectories=102,actual_attempts=len(actual),
        curves=curves,contrasts=contrasts,costs=cc,actual_truncated_responses=sum(a['truncations'] for a in actual.values()),
        observed_private_sha256=file_hash(ROOT/'observed.json'),failed_write_queue_sha256=file_hash(ROOT/'failed-write-label-queue.json'),
        semantic_writes_pending=len(queue),action_counts_actual={k:sum(a[k] for a in actual.values()) for k in ['heated','put_then_heat','physical_no_effect']},
        checks_passed=True,source_sha256=file_hash(Path(__file__)),parameter_fits=0,holdout134_calls=0))
    print(dict(result_sha256=result['sha256'],curves=curves,primary=[c for c in contrasts if c['primary']],semantic_writes_pending=len(queue)),flush=True)


if __name__=='__main__':main()
