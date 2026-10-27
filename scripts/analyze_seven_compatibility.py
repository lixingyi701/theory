#!/usr/bin/env python3
"""Independently verify compatibility witnesses, all solver snapshots and held-scene scores."""
from pathlib import Path
from collections import defaultdict
import math,statistics as st
import numpy as np
from agent_reliability.io import digest,file_hash,read_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from develop_seven_parameterization import predict
from calibrate_successor_success import features
from analyze_seven_reached_risk import independent_score,PREDICATES
from continue_reflexion_pilot import ledger

PUBLIC=Path('repro/reflexion-20261025');PRIVATE=Path('runs/reflexion-seven-compatibility-20261027')
NAMES=['initial_C_N','initial_error','second_attempt','late_error','late_no_error']


def evaluate(rows,coefficients,base):
    pp=predict(rows,{**base,'p_coefficients':coefficients})
    probabilities={(p['trajectory_id'],p['trial']):p['probability'] for p in pp}
    scores={key:independent_score(rows,probabilities,PREDICATES[key]) for key in ['all']+NAMES}
    moments={k:scores[k]['weighted_expected_success_fraction']-scores[k]['weighted_observed_success_fraction'] for k in NAMES}
    beta=np.array(coefficients);penalty=.005*float(beta[1:]@beta[1:])+.5*np.logaddexp(0,-beta[0])+.5*np.logaddexp(0,beta[0])
    j=scores['all']['risk_per_planned_horizon']+float(penalty)/(4*len({r['trajectory_id'] for r in rows}))
    return scores,moments,j


def verify_fit(fit,rows,base,tolerance=None,omit=None,fixed=None):
    scores,moments,j=evaluate(rows,fit['coefficients'],base)
    assert len(fit['coefficients'])==7 and max(abs(b) for b in fit['coefficients'])<=30+1e-8
    assert abs(j-fit['penalized_risk'])<1e-10
    assert abs(scores['all']['risk_per_planned_horizon']-fit['risk'])<1e-10
    for name in NAMES:assert abs(moments[name]-fit['group_probability_minus_observation'][name])<1e-10
    violation=0.
    if tolerance is not None:violation=max(0.,max(abs(moments[k]) for k in NAMES if k!=omit)-tolerance)
    if fixed is not None:violation=max(violation,abs(fit['coefficients'][fixed[0]]-fixed[1]))
    assert abs(violation-fit['constraint_violation'])<1e-9
    assert fit['feasible']==(violation<=1e-6)
    return scores


def main():
    protocol=checked(PUBLIC/'seven-compatibility-protocol-20261027.json')
    result=checked(PUBLIC/'seven-compatibility-results-20261027.json')
    uncertainty=checked(PUBLIC/'seven-compatibility-uncertainty-20261027.json')
    pp=checked(PUBLIC/'seven-compatibility-predictions-20261027.json')
    closeout=checked(PUBLIC/'seven-compatibility-closeout-20261027.json');rows=read_json(PRIVATE/'input-events.json')
    assert digest(rows)==protocol['input_rows_sha256'] and result['protocol_sha256']==protocol['sha256']
    assert closeout['status']=='complete' and closeout['result_sha256']==result['sha256']
    assert result['uncertainty_sha256']==uncertainty['sha256'] and result['prediction_sha256']==pp['sha256']
    assert closeout['parameter_solves']==result['parameter_solves']==94
    for path,h in protocol['protected_files'].items():assert file_hash(Path(path))==h,path
    base=protocol['base_model'];verified=0;converged=0;profile_summaries=[]
    j0=protocol['baseline_full_brier']['penalized_risk']
    for case in result['full_cases']:
        for fit in case['runs']:
            verify_fit(fit,rows,base,case['case']['tolerance'],case['case']['omit']);verified+=1;converged+=fit['optimizer_success']
        feasible=[r for r in case['runs'] if r['feasible']]
        best=min(feasible,key=lambda r:r['penalized_risk']) if feasible else None
        assert best==case['best_feasible']
        witness=best is not None and best['penalized_risk']<=j0+.001
        assert witness==case['near_optimal_witness_found']
        profile_summaries.append({'case':case['case'],'feasible':best is not None,'near_optimal_witness':witness,
            'penalized_risk_cost':None if best is None else best['penalized_risk']-j0})
    center=np.array(protocol['brier_reference']['full_model']['p_coefficients'])
    for profile in result['coordinate_profiles']:
        verify_fit(profile['fit'],rows,base,fixed=(profile['column'],center[profile['column']]+profile['offset']))
        verified+=1;converged+=profile['fit']['optimizer_success']
    all_probabilities={(p['trajectory_id'],p['trial']):p['probability'] for p in pp['predictions']};expected_ids=set()
    for fold in result['folds']:
        scene=fold['held_scene'];train=[r for r in rows if r['scene_id']!=scene];test=[r for r in rows if r['scene_id']==scene]
        for fit in fold['runs']:
            verify_fit(fit,train,base,tolerance=.02);verified+=1;converged+=fit['optimizer_success']
        expected_starts=[next(f for f in protocol['brier_reference']['folds'] if f['held_scene']==scene)['coefficients'],
            next(f for f in protocol['mle_reference']['folds'] if f['scene_id']==scene)['coefficients']]
        assert [f['initial'] for f in fold['runs']]==expected_starts
        valid=[r for r in fold['runs'] if r['feasible'] and r['optimizer_success']]
        chosen=min(valid,key=lambda r:r['penalized_risk']) if valid else None;assert chosen==fold['chosen']
        if chosen is not None:
            for prediction in predict(test,{**base,'p_coefficients':chosen['coefficients']}):
                key=prediction['trajectory_id'],prediction['trial'];expected_ids.add(key)
                assert abs(prediction['probability']-all_probabilities[key])<1e-12
    assert verified==94 and set(all_probabilities)==expected_ids and len(all_probabilities)==len(pp['predictions'])
    complete=len(all_probabilities)==1553;assert complete==result['selection']['complete']==pp['complete']
    reference_pred=checked(PUBLIC/protocol['mle_reference']['prediction_file'])
    ref={(p['trajectory_id'],p['trial']):p['probability'] for p in reference_pred['predictions']}
    ref_scores={k:independent_score(rows,ref,PREDICATES[k]) for k in ['all']+NAMES}
    selection={'complete':complete,'retained_for_development':False,'replace_frozen_baseline':False}
    cv=None
    if complete:
        cv={k:independent_score(rows,all_probabilities,PREDICATES[k]) for k in ['all']+NAMES}
        assert abs(cv['all']['risk_per_planned_horizon']-result['cross_validation']['primary']['risk_per_planned_horizon'])<1e-12
        for k in NAMES:assert abs(cv[k]['weighted_conditional_brier']-result['cross_validation']['groups'][k]['weighted_conditional_brier'])<1e-12
        scenes=sorted(ref_scores['all']['scene_contributions']);delta=[ref_scores['all']['scene_contributions'][s]-cv['all']['scene_contributions'][s] for s in scenes]
        gain=st.mean(delta);se=st.stdev(delta)/math.sqrt(len(delta))
        guards={k:cv[k]['weighted_conditional_brier']<=ref_scores[k]['weighted_conditional_brier'] for k in ['initial_C_N','late_error']}
        best02=next(c for c in result['full_cases'] if c['case']['name']=='all_tol0.02')['best_feasible']
        selection.update(gain=gain,paired_scene_SE=se,guards=guards,retained_for_development=gain>se and all(guards.values()) and best02 is not None)
    assert selection['retained_for_development']==result['selection']['retained_for_development']
    assert abs(selection['gain']-result['selection']['gain'])<1e-12
    assert file_hash(PRIVATE/'bootstrap-draws.npz')==uncertainty['private_draws_file_sha256']
    samples=np.load(PRIVATE/'bootstrap-draws.npz')['residuals'];assert samples.shape==(20000,5)
    moment=np.array([ref_scores[k]['weighted_expected_success_fraction']-ref_scores[k]['weighted_observed_success_fraction'] for k in NAMES])
    assert max(abs(moment[i]-uncertainty['fixed_predictor_residuals'][k]) for i,k in enumerate(NAMES))<1e-12
    se=samples.std(axis=0,ddof=1);critical=float(np.quantile(np.max(np.abs((samples-moment)/se),axis=1),.95))
    assert abs(critical-uncertainty['max_standardized_critical'])<1e-10
    for i,k in enumerate(NAMES):
        assert max(abs(np.quantile(samples[:,i],[.025,.975])-uncertainty['pointwise_percentile95'][k]))<1e-12
        assert max(abs(np.array([moment[i]-critical*se[i],moment[i]+critical*se[i]])-uncertainty['simultaneous_centered95'][k]))<1e-10
    # Descriptive subtype decomposition, no new fitting or posthoc candidate selection.
    subtype=[]
    for native in [1,2,3]:
        for kind in ['B_present','E_without_B','error_absent']:
            def predicate(r,native=native,kind=kind):
                if sum(s['native'] for s in r['state'])!=native:return False
                b=any(s['type']=='B' for s in r['state']);e=any(s['type']=='E' for s in r['state'])
                return b if kind=='B_present' else e and not b if kind=='E_without_B' else not e and not b
            if any(predicate(r) for r in rows):
                item=independent_score(rows,ref,predicate);item.update(native_count=native,memory_subtype=kind)
                item['tasks']=len({r['task_id'] for r in rows if predicate(r)});subtype.append(item)
    # Same task + identical entire seven-feature history implies identical probabilities for all beta.
    sequences=defaultdict(list)
    for r in rows:sequences[r['trajectory_id']].append(r)
    blocks=defaultdict(lambda:defaultdict(list))
    for tid,entries in sequences.items():
        history=[]
        for r in sorted(entries,key=lambda r:r['trial']):
            history.append(tuple(features(r['state'],'seven')))
            b=any(s['type']=='B' for s in r['state']);e=any(s['type']=='E' for s in r['state'])
            if b or e:
                key=r['task_id'],r['trial'],tuple(history);blocks[key]['B' if b else 'E'].append(r)
    matched=[]
    for key,group in blocks.items():
        if not ('B' in group and 'E' in group):continue
        all_rows=group['B']+group['E'];values=[ref[r['trajectory_id'],r['trial']] for r in all_rows]
        assert max(values)-min(values)<1e-12
        matched.append({'task_id':key[0],'trial':key[1],'B_records':len(group['B']),'E_records':len(group['E']),
            'B_mean_success':st.mean(float(r['success']) for r in group['B']),
            'E_mean_success':st.mean(float(r['success']) for r in group['E'])})
    task_diff=defaultdict(list)
    for m in matched:task_diff[m['task_id']].append(m['B_mean_success']-m['E_mean_success'])
    alias={'matched_blocks':len(matched),'tasks':len(task_diff),'B_records':sum(m['B_records'] for m in matched),
        'E_records':sum(m['E_records'] for m in matched),'task_mean_B_minus_E':st.mean(st.mean(v) for v in task_diff.values()) if task_diff else None,
        'blocks':matched,'interpretation':'Identical task and full seven-feature prefix; observed subtype association, not randomized causal effect or an independent sufficiency test'}
    if task_diff:
        values=np.array([st.mean(v) for k,v in sorted(task_diff.items())]);rng=np.random.default_rng(2026102707)
        draws=values[rng.integers(0,len(values),size=(20000,len(values)))].mean(axis=1)
        alias['exploratory_task_bootstrap_percentile95']=np.quantile(draws,[.025,.975]).tolist()
        alias['bootstrap_seed']=2026102707;alias['bootstrap_draws']=20000
        alias['interval_scope']='Exploratory task resampling after subtype inspection; not independent confirmation or causal attribution'
    counts=ledger();assert sum(v['dispatched'] for v in counts.values())==sum(v['responses'] for v in counts.values())==200396
    audit=save(PUBLIC/'seven-compatibility-audit-20261027.json',{'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],
        'source_sha256':file_hash(Path(__file__)),'checks_passed':True,'solver_snapshots_independently_verified':verified,
        'optimizer_converged_snapshots':converged,'full_case_costs':profile_summaries,'reference_cv_scores':ref_scores,'constrained_cv_scores':cv,
        'selection_recomputed':selection,'bootstrap_draws_verified':20000,'subtype_descriptive_scores':subtype,'same_feature_prefix_E_B_diagnostic':alias,
        'global_calls_responses':200396,'new_model_calls':0,'holdout134_calls':0})
    best02=next(c for c in result['full_cases'] if c['case']['name']=='all_tol0.02')['best_feasible']
    candidate=save(PUBLIC/'seven-compatibility-candidate-20261027.json',{'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],'audit_sha256':audit['sha256'],
        'status':'retained_for_development_requires_fresh_validation' if selection['retained_for_development'] else 'not_retained_over_current_estimation',
        'model':None if best02 is None else {**base,'p_coefficients':best02['coefficients'],'estimation':'predictable_Brier_with_five_train_group_moment_bounds','calibration_tolerance':.02},
        'historical_frozen_baseline_replaced':False})
    print({'audit_sha256':audit['sha256'],'candidate_sha256':candidate['sha256'],'verified':verified,'converged':converged,
        'full_case_costs':profile_summaries,'alias_summary':{k:v for k,v in alias.items() if k!='blocks'},
        'subtypes':[{'native':r['native_count'],'type':r['memory_subtype'],'n':r['attempts'],'observed':r['weighted_observed_success_fraction'],
            'predicted':r['weighted_expected_success_fraction']} for r in subtype]},flush=True)


if __name__=='__main__':main()
