#!/usr/bin/env python3
"""Independent aggregation and outcome-dependent weight diagnosis."""
from pathlib import Path
from collections import defaultdict
import math,statistics as st
import numpy as np
from agent_reliability.io import digest,file_hash,read_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from calibrate_successor_success import features,success_components
from analyze_seven_parameter_development import score
from continue_reflexion_pilot import ledger

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-seven-sequential-objectives-20261027')


def main():
    protocol=checked(PUBLIC/'seven-sequential-objectives-protocol-20261027.json')
    result=checked(PUBLIC/'seven-sequential-objectives-results-20261027.json')
    closeout=checked(PUBLIC/'seven-sequential-objectives-closeout-20261027.json')
    rows=read_json(PRIVATE/'input-events.json');assert digest(rows)==protocol['input_rows_sha256']
    assert result['protocol_sha256']==protocol['sha256'] and closeout['status']=='complete'
    assert closeout['result_sha256']==result['sha256'] and closeout['parameter_fits']==result['parameter_fits']==60
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    lookup={(r['trajectory_id'],r['trial']):r for r in rows};all_ids=set(lookup)
    late=lambda r:sum(s['native'] for s in r['state'])>=2 and any(s['type'] in ['E','B'] for s in r['state'])
    predicates={'all':lambda r:True,'initial_C_N':lambda r:r['trial']==1 and r['condition'] in ['correct_middle','neutral_full'],
        'initial_error':lambda r:r['trial']==1 and any(s['type'] in ['E','B'] for s in r['state']),
        'late_error':late,'late_no_error':lambda r:sum(s['native'] for s in r['state'])>=2 and not any(s['type'] in ['E','B'] for s in r['state']),
        'second_attempt':lambda r:r['trial']==2}
    configs=[{'mode':'original_task_marginal_likelihood',**result['reference']}]+result['candidates']
    outputs={};read_rows=0
    for config in configs:
        data=checked(PUBLIC/config['prediction_file']);assert data['sha256']==config['prediction_sha256']
        probs={(r['trajectory_id'],r['trial']):r['probability'] for r in data['predictions']}
        assert len(probs)==len(data['predictions'])==1553 and set(probs)==all_ids
        assert all(0<=p<=1 for p in probs.values());read_rows+=len(probs)
        scores={key:score(rows,probs,pred) for key,pred in predicates.items()}
        assert abs(scores['all']['mean_brier']-config['primary']['mean_brier'])<1e-12
        for key in ['initial_C_N','initial_error','late_error','late_no_error']:
            assert abs(scores[key]['mean_brier']-config['groups'][key]['mean_brier'])<1e-12
        panel=[r for r in rows if r['source']=='completed_successor_primary']
        assert len(panel)==873 and len({r['task_id'] for r in panel})==100
        panel_scores={key:score(panel,probs,pred) if any(pred(r) for r in panel) else {'attempts':0,'mean_brier':None} for key,pred in predicates.items()}
        by_trial={str(j):score(panel,probs,lambda r,j=j:r['trial']==j) for j in range(1,5)}
        if config['mode']!='original_task_marginal_likelihood':
            folds={f['held_scene']:f for f in config['folds']}
            seq=defaultdict(list)
            for row in rows:seq[row['trajectory_id']].append(row)
            for tid,entries in seq.items():
                entries.sort(key=lambda r:r['trial']);model={**protocol['base_model'],'p_coefficients':folds[entries[0]['scene_id']]['coefficients']}
                weights=np.asarray(model['task_prior'],float)
                for row in entries:
                    pp=success_components(model,row['state']);q=float(weights@pp)
                    assert abs(q-probs[tid,row['trial']])<1e-12
                    if not row['success']:weights*=1-pp;weights/=weights.sum()
        outputs[config['mode']]={'all_records':scores,'reused100':panel_scores,'reused100_by_trial':by_trial,
            'full_coefficients':config['full_model']['p_coefficients'],
            'full_starts_agree':config.get('full_starts_agree',True),
            'maximum_fold_gradient':max(f.get('audit',f.get('training_audit'))['max_gradient'] for f in config['folds'])}
    reference=outputs['original_task_marginal_likelihood']['all_records']
    best=min(result['candidates'],key=lambda c:outputs[c['mode']]['all_records']['all']['mean_brier'])
    candidate=outputs[best['mode']]['all_records'];scenes=sorted(reference['all']['scene_scores'])
    delta=[reference['all']['scene_scores'][s]-candidate['all']['scene_scores'][s] for s in scenes]
    gain=st.mean(delta);se=st.stdev(delta)/math.sqrt(len(delta))
    guards={k:candidate[k]['mean_brier']<=reference[k]['mean_brier'] for k in ['initial_C_N','late_error']}
    selected=best['mode'] if gain>se and all(guards.values()) and best['full_starts_agree'] else 'original_task_marginal_likelihood'
    assert selected==result['selection']['selected_for_development']
    assert abs(gain-result['selection']['reference_minus_candidate_scene_gain'])<1e-12
    assert guards==result['selection']['guards']
    # Exact empirical constant-q target in the initial C/N stratum, with all data weights fixed.
    seq=defaultdict(list);scene_tasks=defaultdict(set);task_branches=defaultdict(set)
    for row in rows:
        seq[row['trajectory_id']].append(row);scene_tasks[row['scene_id']].add(row['task_id']);task_branches[row['task_id']].add(row['trajectory_id'])
    values=[]
    for tid,entries in seq.items():
        first=min(entries,key=lambda r:r['trial'])
        if not predicates['initial_C_N'](first):continue
        base=1/(len(scene_tasks)*len(scene_tasks[first['scene_id']])*len(task_branches[first['task_id']]))
        values.append({'success':int(first['success']),'length':len(entries),'base_weight':base,'features':features(first['state'],'seven')})
    assert len({tuple(v['features']) for v in values})==1
    base_success=sum(v['base_weight']*v['success'] for v in values)/sum(v['base_weight'] for v in values)
    divided_target=sum(v['base_weight']*v['success']/v['length'] for v in values)/sum(v['base_weight']/v['length'] for v in values)
    weight_diagnosis={'attempts':len(values),'first_successes':sum(v['success'] for v in values),
        'first_success_rate_under_fixed_start_weights':base_success,
        'constant_probability_minimizing_length_divided_first_Brier':divided_target,
        'one_shared_initial_feature_vector':values[0]['features'],
        'mean_length_if_first_success':st.mean(v['length'] for v in values if v['success']),
        'mean_length_if_first_failure':st.mean(v['length'] for v in values if not v['success']),
        'interpretation':'Exact target for the first-attempt stratum alone; not the global seven-coefficient optimum. Inverse realized length depends on current/future outcomes, so this trajectory-mean diagnostic is not a proper conditional-probability scoring rule.'}
    counts=ledger();assert sum(v['dispatched'] for v in counts.values())==sum(v['responses'] for v in counts.values())==200396
    audit=save(PUBLIC/'seven-sequential-objectives-audit-20261027.json',{'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],
        'source_sha256':file_hash(Path(__file__)),'prediction_rows_reaggregated':read_rows,'checks_passed':True,
        'comparisons':outputs,'selection_recomputed':{'best_primary':best['mode'],'gain':gain,'paired_scene_SE':se,'guards':guards,'selected':selected},
        'outcome_dependent_weight_diagnosis':weight_diagnosis,'global_calls_responses':200396,'new_model_calls':0,'holdout134_calls':0})
    print({'sha256':audit['sha256'],'weights':weight_diagnosis,'summary':{mode:{'all_brier':v['all_records']['all']['mean_brier'],
        'initial_C_N':v['all_records']['initial_C_N']['mean_brier'],'late_error':v['all_records']['late_error']['mean_brier'],
        'reused100_brier':v['reused100']['all']['mean_brier'],'expected_by_trial':[v['reused100_by_trial'][str(j)]['expected_successes'] for j in range(1,5)]} for mode,v in outputs.items()}},flush=True)


if __name__=='__main__':main()
