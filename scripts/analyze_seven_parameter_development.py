#!/usr/bin/env python3
"""Independently aggregate stored development predictions and inspect support."""
from pathlib import Path
from collections import defaultdict
import math,statistics as st
import numpy as np
from agent_reliability.io import digest,file_hash,read_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from calibrate_successor_success import features,success_components
from diagnose_success_regularization import objective
from continue_reflexion_pilot import ledger

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-seven-parameter-development-20261027')


def score(rows,probabilities,predicate=lambda r:True):
    trajectories=defaultdict(list);info={};subset=[]
    for r in rows:
        if not predicate(r):continue
        subset.append(r);tid=r['trajectory_id'];info[tid]=(r['task_id'],r['scene_id'])
        trajectories[tid].append((probabilities[tid,r['trial']]-float(r['success']))**2)
    tasks=defaultdict(list);scenes=defaultdict(list)
    for tid,losses in trajectories.items():tasks[info[tid]].append(st.mean(losses))
    for (task,scene),values in tasks.items():scenes[scene].append(st.mean(values))
    scene={s:st.mean(v) for s,v in sorted(scenes.items())}
    return {'mean_brier':st.mean(scene.values()),'scene_scores':scene,'scenes':len(scene),
        'attempts':len(subset),'observed_successes':sum(r['success'] for r in subset),
        'expected_successes':sum(probabilities[r['trajectory_id'],r['trial']] for r in subset)}


def main():
    protocol=checked(PUBLIC/'seven-parameter-development-protocol-20261027.json')
    result=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    closeout=checked(PRIVATE/'closeout.json');rows=read_json(PRIVATE/'input-events.json')
    assert result['protocol_sha256']==protocol['sha256'] and digest(rows)==protocol['input_rows_sha256']
    assert closeout['status']=='complete' and closeout['fits']==result['parameter_fits']==180
    assert result['elapsed_seconds']<protocol['limits']['seconds']
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    assert len(rows)==1553 and sum(r['success'] for r in rows)==420
    all_ids={(r['trajectory_id'],r['trial']) for r in rows};scores={};predictions={}
    late=lambda r:sum(s['native'] for s in r['state'])>=2 and any(s['type'] in ['E','B'] for s in r['state'])
    for config in result['configs']:
        name=config['config']['name'];data=checked(PUBLIC/config['prediction_file'])
        assert data['sha256']==config['prediction_sha256']
        prob={(r['trajectory_id'],r['trial']):r['probability'] for r in data['predictions']}
        assert set(prob)==all_ids and len(prob)==len(data['predictions'])
        independent=score(rows,prob);tail=score(rows,prob,late)
        assert abs(independent['mean_brier']-config['primary']['mean_brier'])<1e-12
        assert abs(tail['mean_brier']-config['groups']['late_error']['mean_brier'])<1e-12
        assert len(config['folds'])==27 and all(f['training_audit']['rank']==7 for f in config['folds'])
        assert config['full_model']['write_coefficients_reference_A']==protocol['base_model']['write_coefficients_reference_A']
        assert max(a['objective'] for a in config['full_start_audits'])-min(a['objective'] for a in config['full_start_audits'])<1e-5
        assert min(config['hessian_eigenvalues'])>0 and config['finite_difference_max_error']<1e-6
        scores[name]=independent;predictions[name]=prob
    reference=protocol['reference_config'];best=min(scores,key=lambda n:scores[n]['mean_brier'])
    values=[scores[reference]['scene_scores'][s]-scores[best]['scene_scores'][s] for s in sorted(scores[reference]['scene_scores'])]
    gain=st.mean(values);se=st.stdev(values)/math.sqrt(27)
    late_ok=score(rows,predictions[best],late)['mean_brier']<=score(rows,predictions[reference],late)['mean_brier']
    chosen=best if best!=reference and gain>se and late_ok else reference
    assert chosen==result['selection']['selected'] and abs(gain-result['selection']['paired_scene_gain'])<1e-12
    assert abs(se-result['selection']['paired_scene_se'])<1e-12 and late_ok==result['selection']['late_guard_passed']
    selected=predictions[chosen];base=protocol['base_model'];frozen={}
    for tid in sorted({r['trajectory_id'] for r in rows}):
        weights=np.array(base['task_prior'],float)
        for row in sorted([r for r in rows if r['trajectory_id']==tid],key=lambda r:r['trial']):
            pp=success_components(base,row['state']);frozen[tid,row['trial']]=float(weights@pp)
            if not row['success']:weights*=1-pp;weights/=weights.sum()
    new=lambda r:r['source']=='completed_successor_primary'
    error=lambda r:any(s['type'] in ['E','B'] for s in r['state'])
    comparisons={}
    for name,predicate in [('reused100_all',new),('reused100_late_error',lambda r:new(r) and late(r)),
        ('reused100_late_no_error',lambda r:new(r) and sum(s['native'] for s in r['state'])>=2 and not error(r)),
        ('old_controlled_initial_error',lambda r:not new(r) and r['trial']==1 and error(r)),
        ('reused100_A_present',lambda r:new(r) and any(s['type']=='A' for s in r['state']))]:
        comparisons[name]={'frozen_old':score(rows,frozen,predicate),'selected_cv':score(rows,selected,predicate)}
    support={}
    for name,predicate in [('old680',lambda r:not new(r)),('reused873',new)]:
        subset=[r for r in rows if predicate(r) and any(s['type']=='A' for s in r['state'])]
        support[name]={'A_present_attempts':len(subset),'successes':sum(r['success'] for r in subset),'tasks':len({r['task_id'] for r in subset})}
    old=[r for r in rows if not new(r)];x=np.asarray([features(r['state'],'seven') for r in old]);y=np.asarray([r['success'] for r in old],float)
    task_groups=[np.array([i for i,r in enumerate(old) if r['task_id']==t]) for t in sorted({r['task_id'] for r in old})]
    old_value,old_gradient=objective(np.asarray(base['p_coefficients']),x,y,task_groups,base['task_scale'],base['p_ridge'])
    assert np.max(np.abs(old_gradient))<1e-4
    trial_comparison={str(b):{'observed_successes':sum(new(r) and r['trial']==b and r['success'] for r in rows),
        'frozen_old_expected':sum(frozen[r['trajectory_id'],r['trial']] for r in rows if new(r) and r['trial']==b),
        'selected_cv_expected':sum(selected[r['trajectory_id'],r['trial']] for r in rows if new(r) and r['trial']==b)} for b in range(1,5)}
    original=checked(PUBLIC/'successor-complete-diagnostic-metrics-20261027.json')
    for b in range(1,5):assert abs(trial_comparison[str(b)]['frozen_old_expected']-original['success_by_trial'][str(b)]['expected_successes']['full_seven'])<1e-12
    totals=ledger();assert sum(v['dispatched'] for v in totals.values())==sum(v['responses'] for v in totals.values())==200396
    selected_config=next(r for r in result['configs'] if r['config']['name']==chosen)
    audit=save(PUBLIC/'seven-parameter-development-audit-20261027.json',{
        'scope':'Independent result aggregation and reused-development diagnostics; not new independent validation',
        'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],'source_sha256':file_hash(Path(__file__)),
        'passed':True,'configs_checked':6,'prediction_rows_checked':6*1553,'fit_count':180,
        'selection_recomputed':{'best':best,'selected':chosen,'gain':gain,'paired_scene_se':se,'late_guard':late_ok},
        'same_reused_cohort_comparison':comparisons,
        'comparison_limit':'Frozen old parameters used no new100-task outcomes, but selected CV now reuses those records for development. Old680 includes original fitting data; no new independent advantage claim.',
        'reused100_expected_by_trial':trial_comparison,'A_state_support':support,
        'old680_original_objective':old_value,'old680_original_max_gradient':float(np.max(np.abs(old_gradient))),
        'coefficient_comparison':{'frozen_old':base['p_coefficients'],'selected_full_refit':selected_config['full_model']['p_coefficients']},
        'maximum_gradient_check_error':max(c['finite_difference_max_error'] for c in result['configs']),
        'minimum_fold_rank':min(f['training_audit']['rank'] for c in result['configs'] for f in c['folds']),
        'global_calls_responses':200396,'new_model_calls':0,'holdout134_calls':0})
    save(PUBLIC/'seven-parameter-development-candidate-20261027.json',{
        'status':'development_candidate_not_promoted_over_frozen_baseline',
        'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],'audit_sha256':audit['sha256'],
        'config':selected_config['config'],'model':selected_config['full_model'],
        'reason':'Late-state development calibration improves, while early/overall reused-cohort loss regresses; no independent budget validation of this candidate',
        'reused_attempts':1553,'tasks':163,'scenes':27,'new_model_calls':0,'holdout134_calls':0})
    print({k:audit[k] for k in ['sha256','selection_recomputed','reused100_expected_by_trial','A_state_support','old680_original_max_gradient']})
    print('COMPARISON',{k:{n:{z:r[z] for z in ['mean_brier','attempts','observed_successes','expected_successes']} for n,r in v.items()} for k,v in comparisons.items()})


if __name__=='__main__':main()
