#!/usr/bin/env python3
"""Independent raw-record audit, scene contrast reconstruction and no-fit diagnosis."""
from pathlib import Path
from collections import Counter, defaultdict
import math,re,time,sys
import numpy as np
from agent_reliability.io import digest,file_hash,read_json,write_json,seed_for
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from verify_second_attempt import extract
from calibrate_successor_success import features
from continue_reflexion_pilot import ledger

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-content-intervention-20261028')
CONDITIONS=['original','neutralized','paraphrase']
OUTCOMES=['success','heated','put_then_heat','physical_no_effect','decisions','calls']
PAIRS=[('neutralized','paraphrase'),('neutralized','original'),('paraphrase','original')]


def main():
    started=time.process_time();audit_path=PUBLIC/'content-intervention-audit-20261028.json'
    assert not audit_path.exists(),'Reuse completed independent analysis; do not silently rerun'
    protocol=checked(PUBLIC/'content-intervention-protocol-20261028.json')
    assert protocol['sha256']=='07afa23379dd0341f99fefeae773a8508769e9577005495f75eed31de0c3422f'
    summary=checked(PRIVATE/'summary.json');closed=checked(PRIVATE/'allowance-closeout.json')
    frozen=checked(PUBLIC/'content-intervention-results-20261028.json')
    assert summary['status']==closed['status']=='complete' and closed['allowance_status']=='closed' and closed['master_exit_code']==0
    assert summary['completed_attempts']==summary['expected_attempts']==276 and frozen['checks_passed']
    assert frozen['summary_file_sha256']==file_hash(PRIVATE/'summary.json')
    for path,h in protocol['protected_files'].items():assert file_hash(Path(path))==h,path
    registry=checked(PUBLIC/'content-intervention-registry-20261028.json');variants=read_json(PRIVATE/'variants.json')
    assert registry['sha256']==protocol['registry_sha256'] and file_hash(PRIVATE/'variants.json')==registry['private_variants_sha256']
    assert protocol['cases']==registry['entries']
    prompts={(r['index'],r['condition']):r for r in read_json(PRIVATE/'interface-prompts.json')}
    old_rows=read_json(Path('runs/reflexion-second-attempt-diagnostic-20261028/input-rows.json'))
    old_index={(r['trajectory_id'],r['trial']):r for r in old_rows}
    rows=[];parts=[]
    for j in range(1,4):
        name=f'content-intervention-observed-{j:02d}-20261028.json';part=checked(PUBLIC/name)
        assert part['result_sha256']==frozen['sha256'] and len(part['rows'])==92
        rows.extend(part['rows']);parts.append({'file':name,'sha256':part['sha256'],'rows':92})
    assert len(rows)==len({r['id'] for r in rows})==276
    assert [r['id'] for r in rows]==[r['id'] for r in protocol['cells']]==[r['id'] for r in summary['rows']]
    extra={};raw_results={};physical_traces={}
    for row,saved in zip(rows,summary['rows']):
        case=protocol['cases'][row['index']];v=variants[row['index']];c=row['condition'];path=Path(saved['result_path'])
        assert file_hash(path)==saved['result_file_sha256']==row['result_file_sha256']
        rr=read_json(path);cp=read_json(path.parent/'checkpoint.json');identity=checked(path.parent/'identity.json')
        assert file_hash(path.parent/'checkpoint.json')==row['checkpoint_file_sha256']
        memory=v['memories'][c];assert digest(memory)==row['memory_sha256']==case['variant_memory_sha256'][c]
        assert rr['memory_before']==identity['memory']==memory and identity['task']==case['task']
        assert identity['seed']==row['seed'] and identity['trial']==rr['trial']==2
        assert rr['initial_observation_sha256']==case['initial_observation_sha256']
        assert rr['success']==row['success']==bool(cp['transitions'][-1]['won'])
        assert rr['memory_after'][:len(memory)]==memory and len(rr['memory_after'])==len(memory)+(not rr['success'])
        source=v['variants']['original'];a,b=case['edit']['start'],case['edit']['end']
        replacement=source[a:b] if c=='original' else case['edit']['neutral_replacement' if c=='neutralized' else 'paraphrase_replacement']
        assert source[:a]+replacement+source[b:]==memory[case['native_slot']]
        assert digest(source[a:b])==case['edit']['source_span_sha256']
        tid,trial=case['parent_attempt_id']
        assert case['seven_history']==[features(old_index[tid,j]['state'],'seven') for j in range(1,trial+1)]
        assert features(case['state'],'seven')==case['seven_history'][-1]
        target=Path(case['task']['gamefile']).parts[-3].split('-')[1].lower()
        trace=extract(cp,target)
        for key,value in trace.items():assert row['trace'][key]==value,(row['id'],key)
        assert row['heated']==bool(trace['target_heat_success_steps'])
        assert row['put_then_heat']==bool(trace['placed_then_heat_without_pickup'])
        assert row['physical_no_effect']==trace['physical_no_effect'] and row['decisions']==len(cp['transitions'])
        first=read_json(path.parent/'calls/actor-0000/attempt-000.request.json')['payload']
        assert first['prompt']==prompts[row['index'],c]['wrapped_prompt']
        assert first['seed']==row['first_request_seed']==seed_for(row['task_id'],row['seed'],2,0,'actor')
        nonmicro=[]
        for j,t in enumerate(cp['transitions']):
            if t['action'].startswith('think:'):continue
            m=re.fullmatch(r'heat '+re.escape(target)+r' \d+ with ([a-z]+) \d+',t['submitted_action'])
            if m and m[1]!='microwave':nonmicro.append(j)
        extra[row['id']]={'failure_stage':trace['failure_stage'],'non_microwave_heat':bool(nonmicro),
            'native_type':case['native_type'],'additional_advice_remains':case['additional_target_advice_remains']}
        raw_results[row['id']]=rr
        physical_traces[row['id']]=[{'step':j,'action':t['submitted_action'],'feedback':t['observation'][:180],'won':t['won']}
            for j,t in enumerate(cp['transitions']) if not t['action'].startswith('think:')]
    lookup={(r['task_id'],r['seed'],r['condition']):r for r in rows}
    scene_tasks=defaultdict(set)
    for c in protocol['cases']:scene_tasks[c['scene_id']].add(c['task_id'])
    assert len(scene_tasks)==24 and len(protocol['cases'])==46
    weights={c['task_id']:1/(24*len(scene_tasks[c['scene_id']])*2) for c in protocol['cases']}
    # Independent distribution functions from the already installed, task-private numerical dependency.
    sys.path.insert(0,str(Path('runs/reflexion-seven-compatibility-20261027/numerics')))
    from scipy.stats import t as student_t
    critical=float(student_t.ppf(.975,23));assert abs(critical-protocol['analysis']['t_critical'])<1e-12
    comparisons={};all_p=[]
    for left,right in PAIRS:
        key=left+'_minus_'+right;comparisons[key]={}
        for outcome in OUTCOMES:
            by_scene={}
            for scene,tasks in sorted(scene_tasks.items()):
                by_scene[scene]=sum(sum(float(lookup[task,seed,left][outcome])-float(lookup[task,seed,right][outcome]) for seed in [2180,2181])/2 for task in tasks)/len(tasks)
            d=np.array(list(by_scene.values()));mean=float(d.sum()/24);se=float(np.sqrt(np.sum((d-mean)**2)/(23*24)))
            lo,hi=mean-critical*se,mean+critical*se;expected=frozen['comparisons'][key][outcome]
            assert abs(mean-expected['difference'])<1e-12 and abs(se-expected['paired_scene_SE'])<1e-12
            assert np.allclose([lo,hi],expected['t95'],atol=1e-12,rtol=0)
            assert all(abs(value-expected['scene_differences'][scene])<1e-12 for scene,value in by_scene.items())
            p_value=float(2*student_t.sf(abs(mean/se),23)) if se else (1. if not mean else 0.)
            comparisons[key][outcome]={'difference':mean,'paired_scene_SE':se,'t95':[lo,hi],'p_two_sided':p_value,'scene_differences':by_scene}
            all_p.append((p_value,key,outcome))
    primary=comparisons['neutralized_minus_paraphrase']['success'];d=np.array(list(primary['scene_differences'].values()))
    rng=np.random.default_rng(2026102802);sample=d[rng.integers(0,24,size=(20000,24))].mean(axis=1)
    bootstrap=np.quantile(sample,[.025,.975]).tolist()
    assert np.allclose(bootstrap,frozen['primary']['secondary_whole_scene_bootstrap95'],atol=1e-12,rtol=0)
    running=0.;holm=[]
    for rank,(pv,key,outcome) in enumerate(sorted(all_p)):
        running=max(running,min(1.,(18-rank)*pv));holm.append({'comparison':key,'outcome':outcome,'p_raw':pv,'p_holm_18':running})
    def group(predicate):
        result={}
        for condition in CONDITIONS:
            selected=[r for r in rows if r['condition']==condition and predicate(r)]
            exposure=sum(weights[r['task_id']] for r in selected)
            result[condition]={'attempts':len(selected),'successes':sum(r['success'] for r in selected),
                'weighted_success_fraction':sum(weights[r['task_id']]*r['success'] for r in selected)/exposure,
                'weight_mass':exposure,'heated_attempts':sum(r['heated'] for r in selected),
                'put_then_heat_attempts':sum(r['put_then_heat'] for r in selected),
                'failure_stages':dict(Counter(extra[r['id']]['failure_stage'] for r in selected)),
                'non_microwave_heat_attempts':sum(extra[r['id']]['non_microwave_heat'] for r in selected),
                'success_with_put_then_heat':sum(r['success'] and r['put_then_heat'] for r in selected),
                'decisions':sum(r['decisions'] for r in selected),'calls':sum(r['calls'] for r in selected),
                'physical_no_effect_actions':sum(r['physical_no_effect'] for r in selected)}
        return result
    groups={'all':group(lambda r:True),'native_C':group(lambda r:extra[r['id']]['native_type']=='C'),
        'native_U':group(lambda r:extra[r['id']]['native_type']=='U'),
        'other_advice_remains':group(lambda r:extra[r['id']]['additional_advice_remains']),
        'no_other_advice_marked':group(lambda r:not extra[r['id']]['additional_advice_remains'])}
    for c in CONDITIONS:
        assert abs(groups['all'][c]['weighted_success_fraction']-frozen['conditions'][c]['scene_equal_success_fraction'])<1e-12
    contingencies={}
    for left,right in PAIRS:
        pairs=[(lookup[c['task_id'],seed,left],lookup[c['task_id'],seed,right]) for c in protocol['cases'] for seed in [2180,2181]]
        contingencies[left+'_minus_'+right]={
            'both_success':sum(a['success'] and b['success'] for a,b in pairs),
            'left_only_success':sum(a['success'] and not b['success'] for a,b in pairs),
            'right_only_success':sum(not a['success'] and b['success'] for a,b in pairs),
            'neither_success':sum(not a['success'] and not b['success'] for a,b in pairs),
            'same_success_different_order':sum(a['success']==b['success'] and a['put_then_heat']!=b['put_then_heat'] for a,b in pairs)}
    # Cached reference probabilities: descriptive transfer, not a new prespecified score or refit.
    development=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    config=next(c for c in development['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    predictions=checked(PUBLIC/config['prediction_file']);assert predictions['sha256']==config['prediction_sha256']
    probabilities={(r['trajectory_id'],r['trial']):r['probability'] for r in predictions['predictions']}
    folds={f['scene_id']:f for f in config['folds']};expected_pairs=np.zeros(3);parent_prob={}
    for c in protocol['cases']:
        beta=np.array(folds[c['scene_id']]['coefficients']);base=config['full_model']
        offsets=base['task_scale']*np.array(base['task_offsets'])
        first=1/(1+np.exp(-(np.array(c['seven_history'][0])@beta+offsets)))
        later=1/(1+np.exp(-(np.array(c['seven_history'][-1])@beta+offsets)))
        posterior=np.array(base['task_prior'])*(1-first);posterior/=posterior.sum()
        q=float(posterior@later);assert abs(q-probabilities[tuple(c['parent_attempt_id'])])<1e-12
        parent_prob[c['task_id']]=q
        expected_pairs+=np.array([float(posterior@(math.comb(2,k)*later**k*(1-later)**(2-k))) for k in range(3)])
    old_transfer={c:{'expected_successes':sum(parent_prob[r['task_id']] for r in rows if r['condition']==c),
        'weighted_probability':sum(weights[r['task_id']]*parent_prob[r['task_id']] for r in rows if r['condition']==c),
        'observed_task_pairs_success_counts':dict(Counter(sum(lookup[r['task_id'],seed,c]['success'] for seed in [2180,2181]) for r in protocol['cases']))} for c in CONDITIONS}
    call_counts=Counter();tokens_in=tokens_out=0;trunc=[];requests=list(PRIVATE.glob('**/calls/*/attempt-*.request.json'))
    for path in requests:
        req=read_json(path);response=read_json(path.parent/'response.json');role=path.parent.name.split('-')[0]
        assert digest(req['payload'])==req['payload_sha256']==response['payload_sha256']
        assert req['payload']['temperature']==.7 and req['payload']['top_p']==.8 and req['payload']['top_k']==20
        usage=response['response']['usage'];tokens_in+=usage['prompt_tokens'];tokens_out+=usage['completion_tokens'];call_counts[role]+=1
        if response['response']['choices'][0]['finish_reason']=='length':
            cid=next((row['id'] for row in rows if row['id'] in path.parts),None)
            trunc.append({'cell_id':cid,'role':role,'logical_call':path.parent.name,'completion_tokens':usage['completion_tokens']})
    assert len(requests)==10838==closed['costs']['dispatched_calls']==closed['costs']['saved_responses']
    assert tokens_in==closed['costs']['input_tokens_known'] and tokens_out==closed['costs']['output_tokens_known']
    assert len(trunc)==1 and closed['costs']['uncertain_or_unrecorded_attempts']==0
    total=ledger();assert sum(v['dispatched'] for v in total.values())==sum(v['responses'] for v in total.values())==211234
    examples=[];seen=set()
    candidates=sorted([(c['task_id'],seed) for c in protocol['cases'] for seed in [2180,2181]],key=digest)
    for reason,predicate in [
        ('neutralized_success_paraphrase_failure',lambda a,b:a['success'] and not b['success']),
        ('neutralized_failure_paraphrase_success',lambda a,b:not a['success'] and b['success']),
        ('same_success_different_order',lambda a,b:a['success']==b['success'] and a['put_then_heat']!=b['put_then_heat']),
        ('neutralized_success_after_bad_order',lambda a,b:a['success'] and a['put_then_heat'])]:
        for task,seed in candidates:
            if (task,seed) in seen:continue
            a=lookup[task,seed,'neutralized'];b=lookup[task,seed,'paraphrase']
            if not predicate(a,b):continue
            case=next(c for c in protocol['cases'] if c['task_id']==task)
            examples.append({'selection_reason':reason,'task_id':task,'seed':seed,'case_index':case['index'],
                'focal_original':variants[case['index']]['focal_original'],
                'neutralized_trace':physical_traces[a['id']],'paraphrase_trace':physical_traces[b['id']],
                'neutralized_id':a['id'],'paraphrase_id':b['id']})
            seen.add((task,seed));break
    write_json(PRIVATE/'analysis-case-view.json',examples)
    elapsed=time.process_time()-started;assert elapsed<900
    audit=save(audit_path,{'scope':'Independent raw audit and frozen arithmetic; extra stratification/cached-parameter transfer are postcollection diagnostics',
        'protocol_sha256':protocol['sha256'],'frozen_result_sha256':frozen['sha256'],'summary_sha256':summary['sha256'],
        'closeout_sha256':closed['sha256'],'source_sha256':file_hash(Path(__file__)),
        'independent_trace_parser_sha256':file_hash(Path('scripts/verify_second_attempt.py')),
        'checks_passed':True,'attempts':276,'requests_verified':len(requests),'protected_files':len(protocol['protected_files']),
        'projection_parts':parts,'comparisons':comparisons,'primary_bootstrap95_recomputed':bootstrap,
        'groups':groups,'paired_contingencies':contingencies,'exploratory_Holm18':holm,
        'cached_probability_transfer':old_transfer,'cached_two_independent_seeds_expected_task_counts':expected_pairs.tolist(),
        'cached_transfer_scope':'No fits. Previous scene-held-out MLE components/post-first-failure posterior, copied to same parent/state. Pair count calculation applies only to two distinct seeds within one condition, not six correlated paired branches.',
        'group_weights':'All groups retain original24-scene/task/two-seed start weights, then normalize their included mass',
        'call_roles':dict(call_counts),'truncations':trunc,'global_calls_responses':211234,
        'analysis_new_model_calls':0,'parameter_fits':0,'holdout134_calls':0,'analysis_cpu_seconds':elapsed,
        'private_case_view_sha256':file_hash(PRIVATE/'analysis-case-view.json')})
    state=read_json(PRIVATE/'preparation.json');state['analysis']={'status':'independent_audit_complete','audit_sha256':audit['sha256'],
        'cpu_limit_seconds':900,'cpu_seconds':elapsed,'new_model_calls':0,'parameter_fits':0};write_json(PRIVATE/'preparation.json',state)
    print({'audit':audit['sha256'],'cpu_seconds':elapsed,'primary':primary,'groups':groups,'contingencies':contingencies,
        'old_transfer':old_transfer,'expected_pair_counts':expected_pairs.tolist(),'truncations':trunc,'case_count':len(examples)},flush=True)


if __name__=='__main__':main()
