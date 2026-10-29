#!/usr/bin/env python3
"""Independent predictable-weight scoring and cached-parameter diagnostics."""
from pathlib import Path
from collections import defaultdict
import math,statistics as st
import numpy as np
from agent_reliability.io import digest,file_hash,read_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from calibrate_successor_success import success_components
from continue_reflexion_pilot import ledger

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-seven-reached-risk-20261027')
PREDICATES={'all':lambda r:True,
    'initial_C_N':lambda r:r['trial']==1 and r['condition'] in ['correct_middle','neutral_full'],
    'initial_error':lambda r:r['trial']==1 and any(s['type'] in ['E','B'] for s in r['state']),
    'second_attempt':lambda r:r['trial']==2,
    'late_error':lambda r:sum(s['native'] for s in r['state'])>=2 and any(s['type'] in ['E','B'] for s in r['state']),
    'late_no_error':lambda r:sum(s['native'] for s in r['state'])>=2 and not any(s['type'] in ['E','B'] for s in r['state'])}


def independent_score(rows,probs,predicate=lambda r:True):
    task_branches=defaultdict(set);scene_tasks=defaultdict(set);metadata={}
    for r in rows:
        task_branches[r['task_id']].add(r['trajectory_id']);scene_tasks[r['scene_id']].add(r['task_id'])
        metadata[r['trajectory_id']]=(r['task_id'],r['scene_id'])
    base={tid:1/(len(scene_tasks)*len(scene_tasks[s])*len(task_branches[t])) for tid,(t,s) in metadata.items()}
    scene={s:0. for s in scene_tasks};exposure=weighted_y=weighted_p=0.;subset=[]
    for r in rows:
        if not predicate(r):continue
        subset.append(r);a=base[r['trajectory_id']]/4;p=probs[r['trajectory_id'],r['trial']]
        scene[r['scene_id']]+=a*(p-float(r['success']))**2
        exposure+=a;weighted_y+=a*float(r['success']);weighted_p+=a*p
    loss=sum(scene.values())
    return {'risk_per_planned_horizon':loss,'scene_contributions':{s:v*len(scene) for s,v in sorted(scene.items())},
        'weighted_conditional_brier':loss/exposure if exposure else None,'reached_weight':exposure,
        'attempts':len(subset),'observed_successes':sum(r['success'] for r in subset),
        'expected_successes':sum(probs[r['trajectory_id'],r['trial']] for r in subset),
        'weighted_observed_success_fraction':weighted_y/exposure if exposure else None,
        'weighted_expected_success_fraction':weighted_p/exposure if exposure else None}


def main():
    protocol=checked(PUBLIC/'seven-reached-risk-protocol-20261027.json')
    result=checked(PUBLIC/'seven-reached-risk-results-20261027.json')
    closed=checked(PUBLIC/'seven-reached-risk-closeout-20261027.json');rows=read_json(PRIVATE/'input-events.json')
    assert digest(rows)==protocol['input_rows_sha256'] and len(rows)==1553
    assert closed['result_sha256']==result['sha256'] and closed['status']=='complete' and result['parameter_fits']==60
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    parent=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    reference=next(c for c in parent['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    configs=[{'mode':'original_task_marginal_likelihood',**reference}]+result['candidates'];outputs={};n=0
    ids={(r['trajectory_id'],r['trial']) for r in rows}
    for config in configs:
        pred=checked(PUBLIC/config['prediction_file']);assert pred['sha256']==config['prediction_sha256']
        probs={(r['trajectory_id'],r['trial']):r['probability'] for r in pred['predictions']}
        assert set(probs)==ids and len(probs)==len(pred['predictions'])==1553;n+=1553
        full={key:independent_score(rows,probs,p) for key,p in PREDICATES.items()}
        expected=result['reference'] if config['mode']=='original_task_marginal_likelihood' else config
        assert abs(full['all']['risk_per_planned_horizon']-expected['primary']['risk_per_planned_horizon'])<1e-12
        for key in PREDICATES:
            if key=='all':continue
            assert abs(full[key]['weighted_conditional_brier']-expected['groups'][key]['weighted_conditional_brier'])<1e-12
        if config['mode']!='original_task_marginal_likelihood':
            seq=defaultdict(list);folds={f['held_scene']:f for f in config['folds']}
            for r in rows:seq[r['trajectory_id']].append(r)
            for tid,entries in seq.items():
                entries.sort(key=lambda r:r['trial']);model={**protocol['base_model'],'p_coefficients':folds[entries[0]['scene_id']]['coefficients']}
                weights=np.array(model['task_prior'],float)
                for r in entries:
                    p=success_components(model,r['state']);assert abs(float(weights@p)-probs[tid,r['trial']])<1e-12
                    if not r['success']:weights*=1-p;weights/=weights.sum()
        panel=[r for r in rows if r['source']=='completed_successor_primary'];assert len(panel)==873
        outputs[config['mode']]={'all_records':full,'reused100':{key:independent_score(panel,probs,p) for key,p in PREDICATES.items()},
            'reused100_by_trial':{str(j):independent_score(panel,probs,lambda r,j=j:r['trial']==j) for j in range(1,5)},
            'full_coefficients':config['full_model']['p_coefficients'],
            'full_start_spread':config.get('full_start_spread'),'maximum_fold_gradient':max(f.get('audit',f.get('training_audit'))['max_gradient'] for f in config['folds'])}
    best=min(result['candidates'],key=lambda c:outputs[c['mode']]['all_records']['all']['risk_per_planned_horizon'])
    a=outputs['original_task_marginal_likelihood']['all_records'];b=outputs[best['mode']]['all_records']
    d=[a['all']['scene_contributions'][s]-b['all']['scene_contributions'][s] for s in sorted(a['all']['scene_contributions'])]
    gain=st.mean(d);se=st.stdev(d)/math.sqrt(len(d));guards={k:b[k]['weighted_conditional_brier']<=a[k]['weighted_conditional_brier'] for k in ['initial_C_N','late_error']}
    selected=best['mode'] if gain>se and all(guards.values()) and best['full_starts_agree'] else 'original_task_marginal_likelihood'
    assert selected==result['selection']['selected_for_development'] and guards==result['selection']['guards']
    assert abs(gain-result['selection']['gain'])<1e-12 and abs(se-result['selection']['paired_scene_SE'])<1e-12
    # No new fits: rescore six previously published parameter settings as an exploratory diagnostic.
    cached=[]
    for config in parent['configs']:
        p=checked(PUBLIC/config['prediction_file']);assert p['sha256']==config['prediction_sha256']
        pp={(r['trajectory_id'],r['trial']):r['probability'] for r in p['predictions']};assert set(pp)==ids
        cached.append({'config':config['config'],**{k:independent_score(rows,pp,PREDICATES[k]) for k in ['all','initial_C_N','late_error']}})
    # Reassess the historical frozen coefficients under the corrected diagnostic weight.
    # The first680 records overlap their original training; only descriptive comparisons here.
    historical={};sequences=defaultdict(list)
    for r in rows:sequences[r['trajectory_id']].append(r)
    for tid,entries in sequences.items():
        weights=np.array(protocol['base_model']['task_prior'],float)
        for r in sorted(entries,key=lambda r:r['trial']):
            p=success_components(protocol['base_model'],r['state']);historical[tid,r['trial']]=float(weights@p)
            if not r['success']:weights*=1-p;weights/=weights.sum()
    panel=[r for r in rows if r['source']=='completed_successor_primary']
    historical_scores={'all_records':{k:independent_score(rows,historical,p) for k,p in PREDICATES.items()},
        'reused100':{k:independent_score(panel,historical,p) for k,p in PREDICATES.items()},
        'scope':'Historical frozen beta versus refitted scene-held-out estimates; reused development diagnostic, not new independent validation. Original680 include frozen-model training records.'}
    counts=ledger();assert sum(v['dispatched'] for v in counts.values())==sum(v['responses'] for v in counts.values())==200396
    audit=save(PUBLIC/'seven-reached-risk-audit-20261027.json',{'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],
        'source_sha256':file_hash(Path(__file__)),'prediction_rows_reaggregated':n,'checks_passed':True,'comparisons':outputs,
        'selection_recomputed':{'best_primary':best['mode'],'gain':gain,'paired_scene_SE':se,'guards':guards,'selected':selected},
        'cached_six_configuration_rescoring':cached,'cached_rescoring_scope':'Post-comparison sensitivity diagnostic only; no parameter fits or candidate selection from this extra table',
        'historical_frozen_reassessment':historical_scores,
        'global_calls_responses':200396,'new_model_calls':0,'holdout134_calls':0})
    print({'sha256':audit['sha256'],'summary':{mode:{'risk':v['all_records']['all']['risk_per_planned_horizon'],
        'initial_C_N':v['all_records']['initial_C_N']['weighted_conditional_brier'],'late_error':v['all_records']['late_error']['weighted_conditional_brier'],
        'expected_by_trial':[v['reused100_by_trial'][str(j)]['expected_successes'] for j in range(1,5)]} for mode,v in outputs.items()},
        'cached_six':[{'config':v['config'],'risk':v['all']['risk_per_planned_horizon'],'initial':v['initial_C_N']['weighted_conditional_brier'],'late':v['late_error']['weighted_conditional_brier']} for v in cached],
        'historical_vs_refit_reused100':{key:{'historical':historical_scores['reused100'][key]['weighted_conditional_brier'],
            'refit':outputs['original_task_marginal_likelihood']['reused100'][key]['weighted_conditional_brier']} for key in PREDICATES}},flush=True)


if __name__=='__main__':main()
