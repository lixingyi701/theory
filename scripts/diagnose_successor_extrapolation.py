#!/usr/bin/env python3
"""No-refit sensitivity to extrapolating native-entry count beyond calibration."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from agent_reliability.io import file_hash
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
import shared_recovery_model as model

ROOT=Path('repro/reflexion-20261025')


def kernel(fitted,state,cap_p,cap_g):
    x=np.asarray(model.features(state,fitted['success_family']))
    z=np.asarray(model.features(state,fitted['write_family']))
    if cap_p:x[2]=min(x[2],1/3)
    if cap_g:z[2]=min(z[2],1/3)
    p=1/(1+np.exp(-(x@np.asarray(fitted['success_coefficients'])+fitted['scale']*model.OFFSETS)))
    g=model.softmax(np.r_[z@np.asarray(fitted['write_coefficients_reference_A']),0.])
    return np.c_[p,(1-p[:,None])*g[None,:]]


def forecast_curve(fitted,cap_p,cap_g):
    cache={}
    def recurse(state,left,z):
        if not left:return 0.
        key=(state,left,z)
        if key in cache:return cache[key]
        s=[{'type':k,'native':n} for k,n in state]
        k=kernel(fitted,s,cap_p,cap_g)[z]
        value=k[0]+sum(k[j+1]*recurse((state+((kind,True),))[-3:],left-1,z) for j,kind in enumerate(model.OUTCOMES[1:]))
        cache[key]=float(value);return float(value)
    initial=tuple((k,False) for k in ['N','C','N'])
    return [sum(float(w)*recurse(initial,b,z) for z,w in enumerate(model.PRIOR)) for b in range(1,5)]


def main():
    data=checked(ROOT/'successor-support-diagnostic-20261026.json')
    params=checked(ROOT/'shared-calibration-parameters.json')
    diagnostic=checked(ROOT/'success-regularization-diagnostic-20261026.json')
    training=checked(ROOT/'shared-calibration-events.json')
    base=params['comparison']['selected_model']
    weak={**base,'success_coefficients':next(r['coefficients'] for r in diagnostic['calibration_fits'] if r['ridge']==.01)}
    assert data['parameters_sha256']==params['sha256'] and data['regularization_diagnostic_sha256']==diagnostic['sha256']
    assert params['events_sha256']==training['sha256']
    models={'frozen':base,'weak_ridge':weak}
    for fitted in models.values():
        for row in training['rows']:
            assert sum(s['native'] for s in row['state'])<=1
            assert np.allclose(kernel(fitted,row['state'],True,True),model.component_kernel(fitted,row['state']),atol=1e-14,rtol=0)
    outputs=[]
    for name,fitted in models.items():
        for cap_p,cap_g in [(False,False),(True,False),(False,True),(True,True)]:
            success=defaultdict(list);writes=defaultdict(list)
            for identity in sorted({r['trajectory_id'] for r in data['attempt_rows']}):
                sequence=sorted([r for r in data['attempt_rows'] if r['trajectory_id']==identity],key=lambda r:r['trial'])
                weights=model.PRIOR.copy()
                for i,row in enumerate(sequence):
                    state=row['state'];k=kernel(fitted,state,cap_p,cap_g);p=float(weights@k[:,0])
                    success[row['native_count']].append((p,float(row['success'])))
                    if not row['success']:
                        weights*=1-k[:,0];weights/=weights.sum()
                        if i+1<len(sequence):
                            next_row=sequence[i+1];g=k[0,1:]/(1-k[0,0])
                            integrated=sum(float(g[j]*(weights@kernel(fitted,(state+[{'type':kind,'native':True}])[-3:],cap_p,cap_g)[:,0]))
                                           for j,kind in enumerate(model.OUTCOMES[1:]))
                            actual=float(weights@kernel(fitted,next_row['state'],cap_p,cap_g)[:,0])
                            writes[row['native_count']].append((integrated,actual,float(next_row['success'])))
            scores={str(n):{'attempts':len(v),'successes':int(sum(t[1] for t in v)),
                'expected_successes':float(sum(t[0] for t in v)),
                'mean_brier':float(np.mean([(p-y)**2 for p,y in v]))} for n,v in success.items()}
            if not cap_p and not cap_g:
                for n,r in scores.items():
                    expected=data['success_by_native_count'][n][name]
                    assert abs(r['expected_successes']-expected['expected_successes'])<1e-10
                    assert abs(r['mean_brier']-expected['mean_brier'])<1e-10
            curve=forecast_curve(fitted,cap_p,cap_g)
            assert abs(curve[0]-model.budget_probability(fitted,[{'type':k,'native':False} for k in ['N','C','N']],1))<1e-12
            assert abs(curve[1]-model.budget_probability(fitted,[{'type':k,'native':False} for k in ['N','C','N']],2))<1e-12
            outputs.append({'model':name,'cap_success_count':cap_p,'cap_write_count':cap_g,
                'model_implied_C_N_budget_curve_unscored':curve,
                'success_by_native_count':scores,'bridges_by_native_count':{str(n):{
                    'events':len(v),'integrated_next_expected_successes':float(sum(t[0] for t in v)),
                    'actual_type_next_expected_successes':float(sum(t[1] for t in v)),
                    'next_observed_successes':int(sum(t[2] for t in v))} for n,v in writes.items()}})
    by_a={str(a):{'attempts':sum(any(s['type']=='A' for s in r['state'])==a for r in data['attempt_rows']),
                     'successes':sum(r['success'] and any(s['type']=='A' for s in r['state'])==a for r in data['attempt_rows'])}
          for a in [False,True]}
    attempts=save(ROOT/'successor-support-attempts-20261026.json',{
        'scope':'All217 typed attempts in the selected108 trajectories','source_diagnosis_sha256':data['sha256'],
        'rows':data['attempt_rows']})
    writes=save(ROOT/'successor-support-writes-20261026.json',{
        'scope':'All109 reviewed writes; join to attempts by trajectory_id and trial',
        'source_diagnosis_sha256':data['sha256'],
        'rows':[{k:r[k] for k in ['trajectory_id','trial','new_reflection_sha256','new_type','g','bridges','next_success']} for r in data['write_rows']]})
    projection=save(ROOT/'successor-support-evaluation-20261026.json',{
        **{k:v for k,v in data.items() if k not in ['sha256','attempt_rows','write_rows']},
        'projection_scope':'Full aggregate and trajectory projection; original complete diagnosis remains on server',
        'source_diagnosis_sha256':data['sha256'],'source_diagnosis_file_sha256':file_hash(ROOT/'successor-support-diagnostic-20261026.json'),
        'omitted_fields':['attempt_rows','write_rows'],
        'public_parts':{'successor-support-attempts-20261026.json':attempts['sha256'],
                        'successor-support-writes-20261026.json':writes['sha256']}})
    result=save(ROOT/'successor-extrapolation-sensitivity-20261026.json',{
        'scope':'Postcollection diagnostic ablation; no new fitting or independent validation',
        'operation':'In p and/or G feature vectors only, replace native_count/3 with min(native_count,1)/3; retain actual ordered memory and every other feature',
        'source_sha256':file_hash(Path(__file__)),'diagnosis_sha256':data['sha256'],
        'parameters_sha256':params['sha256'],'regularization_diagnostic_sha256':diagnostic['sha256'],
        'public_evaluation_sha256':projection['sha256'],
        'all463_training_kernels_unchanged_for_both_models':True,
        'original_C_N_exit_probability_by_attempt3':1-forecast_curve(base,False,False)[1],
        'sensitivity_variants':outputs,'A_presence_descriptive':by_a,
        'interpretation':'Capping changes only extrapolation beyond observed native counts. Conditional-state scores and bridges diagnose sensitivity; they are not frozen budget-validation scores. Other features such as A-count can still extrapolate.',
        'new_model_calls':0,'parameter_refits':0,'holdout134_calls':0})
    print({'sha256':result['sha256'],'A_presence':by_a,
           'fourth_attempt':[{'model':r['model'],'cap_p':r['cap_success_count'],'cap_G':r['cap_write_count'],
                             **r['success_by_native_count']['3']} for r in outputs]})


if __name__=='__main__':main()
