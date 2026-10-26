#!/usr/bin/env python3
"""Fixed six-versus-seven success comparison on existing development records."""
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
from agent_reliability.io import digest, file_hash, write_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from diagnose_success_regularization import objective
import shared_recovery_model as original

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-successor-development-calibration-20261026')
FAMILIES=['six','seven']
RIDGE=.01
SCALE=1.5


def features(state,family):
    x=original.features(state,'error_lifetime')
    if family=='seven':
        x=x+[float(any(s['type'] in ['E','B'] for s in state) and any(s['native'] for s in state))]
    elif family!='six':raise ValueError('Unknown success feature set')
    return x


def success_components(fitted,state):
    logit=np.asarray(features(state,fitted['p_feature_set']))@np.asarray(fitted['p_coefficients'])
    return 1/(1+np.exp(-np.clip(logit+fitted['task_scale']*np.asarray(fitted['task_offsets']),-700,700)))


def component_kernel(fitted,state):
    p=success_components(fitted,state)
    z=np.asarray(original.features(state,fitted['write_family']))
    g=original.softmax(np.r_[z@np.asarray(fitted['write_coefficients_reference_A']),0.])
    return np.c_[p,(1-p[:,None])*g[None,:]]


def arrays(rows,family):
    x=np.asarray([features(r['state'],family) for r in rows])
    y=np.asarray([r['success'] for r in rows],float)
    tasks=sorted({r['task_id'] for r in rows})
    groups=[np.asarray([i for i,r in enumerate(rows) if r['task_id']==t]) for t in tasks]
    return x,y,groups,tasks


def fit(rows,family,base,initial=None):
    x,y,groups,_=arrays(rows,family)
    fun=lambda b:objective(b,x,y,groups,SCALE,RIDGE)
    beta,audit=original.bfgs(fun,np.zeros(x.shape[1]) if initial is None else initial)
    fitted={'p_feature_set':family,'p_coefficients':beta.tolist(),'p_ridge':RIDGE,
        'task_scale':SCALE,'task_offsets':original.OFFSETS.tolist(),'task_prior':original.PRIOR.tolist(),
        'write_family':base['write_family'],'write_coefficients_reference_A':base['write_coefficients_reference_A'],
        'kernel_implementation':'scripts/calibrate_successor_success.py:component_kernel'}
    return fitted,{**audit,'objective':fun(beta)[0],'feature_rank':int(np.linalg.matrix_rank(x))}


def grouped_scores(predictions,rows):
    lookup={(r['trajectory_id'],r['trial']):r for r in rows}
    group=defaultdict(list)
    for pred in predictions:
        row=lookup[pred['trajectory_id'],pred['trial']]
        n=sum(s['native'] for s in row['state'])
        item=(pred['probability'],float(row['success']))
        group['native_'+str(n)].append(item)
        error=any(s['type'] in ['E','B'] for s in row['state'])
        group['native_'+str(n)+'_error_'+str(error)].append(item)
        group['source_'+row['source']].append(item)
        if row['trial']==1 and row['condition'] in ['correct_middle','neutral_full']:group['initial_C_N'].append(item)
        if row['trial']==1:group['first_attempt'].append(item)
        if n>=2:group['native_2_or_3'].append(item)
    return {key:{'attempts':len(v),'successes':int(sum(y for p,y in v)),
                'expected_successes':float(sum(p for p,y in v)),
                'mean_prediction_minus_observation':float(np.mean([p-y for p,y in v])),
                'mean_brier':float(np.mean([(p-y)**2 for p,y in v]))} for key,v in group.items()}


def predictions(rows,fitted):
    output=[];trajectories=[]
    for identity in sorted({r['trajectory_id'] for r in rows}):
        sequence=sorted([r for r in rows if r['trajectory_id']==identity],key=lambda r:r['trial'])
        weights=original.PRIOR.copy();losses=[]
        for row in sequence:
            p=success_components(fitted,row['state']);prob=float(weights@p)
            output.append({'trajectory_id':identity,'trial':row['trial'],'probability':prob})
            losses.append((prob-float(row['success']))**2)
            if not row['success']:
                weights*=1-p;weights/=weights.sum()
        trajectories.append(float(np.mean(losses)))
    return output,float(np.mean(trajectories))


def full_training_summary(rows,fitted):
    x,y,groups,tasks=arrays(rows,fitted['p_feature_set'])
    logits=x@np.asarray(fitted['p_coefficients'])[:,None]+SCALE*original.OFFSETS[None,:]
    p=1/(1+np.exp(-logits));posterior={}
    for task,idx in zip(tasks,groups):
        logs=y[idx,None]*(-np.logaddexp(0,-logits[idx]))+(1-y[idx,None])*(-np.logaddexp(0,logits[idx]))
        posterior[task]=original.softmax(np.log(original.PRIOR)+logs.sum(axis=0))
    buckets=defaultdict(list)
    for i,row in enumerate(rows):
        key='native_'+str(sum(s['native'] for s in row['state']))
        item=(float(y[i]),float(original.PRIOR@p[i]),float(posterior[row['task_id']]@p[i]))
        buckets[key].append(item)
        if row['trial']==1 and row['condition'] in ['correct_middle','neutral_full']:buckets['initial_C_N'].append(item)
    return {key:{'attempts':len(v),'successes':int(sum(t[0] for t in v)),
        'prior_expected_successes':float(sum(t[1] for t in v)),
        'posterior_fitted_expected_successes':float(sum(t[2] for t in v))} for key,v in buckets.items()}


def budget_curve(fitted,labels):
    cache={}
    def recur(state,left,z):
        if not left:return 0.
        key=(state,left,z)
        if key in cache:return cache[key]
        k=component_kernel(fitted,[{'type':kind,'native':n} for kind,n in state])[z]
        v=k[0]+sum(k[j+1]*recur((state+((kind,True),))[-3:],left-1,z) for j,kind in enumerate(original.OUTCOMES[1:]))
        cache[key]=float(v);return float(v)
    state=tuple((kind,False) for kind in labels)
    return [sum(float(w)*recur(state,b,z) for z,w in enumerate(original.PRIOR)) for b in range(1,5)]


def main():
    started=time.monotonic()
    old=checked(PUBLIC/'shared-calibration-events.json')
    attempts=checked(PUBLIC/'successor-support-attempts-20261026.json')
    writes=checked(PUBLIC/'successor-support-writes-20261026.json')
    evaluation=checked(PUBLIC/'successor-support-evaluation-20261026.json')
    params=checked(PUBLIC/'shared-calibration-parameters.json')
    assert params['events_sha256']==old['sha256']
    assert params['model_source_sha256']==file_hash(Path('scripts/shared_recovery_model.py'))
    assert evaluation['public_parts']['successor-support-attempts-20261026.json']==attempts['sha256']
    assert evaluation['public_parts']['successor-support-writes-20261026.json']==writes['sha256']
    types={(r['trajectory_id'],r['trial']):r['new_type'] for r in writes['rows']}
    rows=[]
    for source,data in [('original_calibration',old['rows']),('successor_review',attempts['rows'])]:
        for row in data:
            outcome=row['outcome'] if source=='original_calibration' else ('S' if row['success'] else types.get((row['trajectory_id'],row['trial'])))
            rows.append({k:row[k] for k in ['trajectory_id','task_id','scene_id','condition','trial','state','success','result_file_sha256']}|
                        {'source':source,'outcome':outcome})
    assert len(rows)==680 and len({(r['trajectory_id'],r['trial']) for r in rows})==680
    assert sum(r['success'] for r in rows)==137
    assert sum(not r['success'] and r['outcome'] is not None for r in rows)==514
    missing=[r for r in rows if r['outcome'] is None]
    assert len(missing)==29 and all(r['source']=='successor_review' and r['trial']==4 and not r['success'] for r in missing)
    assert not ({r['task_id'] for r in rows if r['source']=='original_calibration'} & {r['task_id'] for r in rows if r['source']=='successor_review'})
    scenes=sorted({r['scene_id'] for r in rows});assert len(scenes)==27
    assert len({r['task_id'] for r in rows})==63 and len({r['trajectory_id'] for r in rows})==360
    PRIVATE.mkdir(parents=True,exist_ok=True);write_json(PRIVATE/'input-events.json',rows)
    before={name:file_hash(PUBLIC/name) for name in ['shared-calibration-parameters.json','shared-calibration-events.json',
             'shared-validation-results.json','four-attempt-validation-results.json','four-attempt-validation-evaluation.json']}
    source=file_hash(Path(__file__))
    manifest=save(PUBLIC/'successor-development-inputs-20261026.json',{
        'scope':'Development reuse of original calibration and previously inspected successor records',
        'sources':{'shared-calibration-events.json':old['sha256'],'successor-support-attempts-20261026.json':attempts['sha256'],
                   'successor-support-writes-20261026.json':writes['sha256'],'successor-support-evaluation-20261026.json':evaluation['sha256']},
        'input_rows_sha256':digest(rows),'private_input_file_sha256':file_hash(PRIVATE/'input-events.json'),
        'attempts':680,'tasks':63,'scenes':27,'trajectories':360,'successes':137,'known_failed_write_types':514,
        'unlabelled_terminal_failures_success_side_only':29,
        'comparison':{'families':FAMILIES,'additional_feature':'has_E_or_B * I(native_count>0)',
                      'ridge':RIDGE,'task_scale':SCALE,'task_prior':original.PRIOR.tolist(),
                      'write_kernel':'unchanged original frozen conditional type model',
                      'primary_development_metric':'mean of27 scene trajectory-mean sequential single-success Brier scores',
                      'selection':'seven only if mean(scene_loss_six-scene_loss_seven)>paired_scene_difference_SE; otherwise six',
                      'maximum_success_fits':60,'cpu_wall_seconds':900},
        'source_sha256':source,'objective_source_sha256':file_hash(Path('scripts/diagnose_success_regularization.py')),
        'new_model_calls':0,'holdout134_calls':0})
    base=params['comparison']['selected_model'];folds=[];oof={f:[] for f in FAMILIES};checks={};full={}
    for family in FAMILIES:
        x,y,groups,_=arrays(rows,family)
        assert all(features(r['state'],family)[:6]==original.features(r['state'],'error_lifetime') for r in rows)
        point=np.linspace(-.2,.2,x.shape[1]);fun=lambda b:objective(b,x,y,groups,SCALE,RIDGE)
        grad=fun(point)[1];numeric=[]
        for j in range(len(point)):
            delta=np.eye(len(point))[j]*1e-5
            numeric.append((fun(point+delta)[0]-fun(point-delta)[0])/(2e-5))
        checks[family]={'max_finite_difference_error':float(np.max(np.abs(grad-np.asarray(numeric))))}
        assert checks[family]['max_finite_difference_error']<1e-6
        for scene in scenes:
            if time.monotonic()-started>900:raise RuntimeError('CPU allowance exhausted')
            train=[r for r in rows if r['scene_id']!=scene];test=[r for r in rows if r['scene_id']==scene]
            assert not ({r['task_id'] for r in train}&{r['task_id'] for r in test})
            fitted,audit=fit(train,family,base);pred,score=predictions(test,fitted);oof[family].extend(pred)
            folds.append({'family':family,'scene_id':scene,'trajectories':len({r['trajectory_id'] for r in test}),
                          'attempts':len(test),'score':score,'optimizer':audit,'coefficients':fitted['p_coefficients']})
        fitted,audit=fit(rows,family,base);restarts=[]
        for initial in [np.full(x.shape[1],.5),np.full(x.shape[1],-.5)]:
            alternate,extra=fit(rows,family,base,initial)
            restarts.append({**extra,'max_coefficient_difference':float(np.max(np.abs(np.asarray(alternate['p_coefficients'])-fitted['p_coefficients'])))})
        full[family]={'model':fitted,'optimizer':audit,'restarts':restarts,
                      'training_groups':full_training_summary(rows,fitted)}
        print({'family':family,'folds_complete':27,'gradient_check':checks[family]},flush=True)
    six={r['scene_id']:r['score'] for r in folds if r['family']=='six'}
    seven={r['scene_id']:r['score'] for r in folds if r['family']=='seven'}
    delta=np.asarray([six[s]-seven[s] for s in scenes]);se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    selected='seven' if float(delta.mean())>se else 'six'
    summaries={f:{'scene_mean_brier':float(np.mean([r['score'] for r in folds if r['family']==f])),
                   'out_of_fold_groups':grouped_scores(oof[f],rows)} for f in FAMILIES}
    prediction_parts={}
    for family in FAMILIES:
        prediction_parts[family]=save(PUBLIC/f'successor-development-predictions-{family}-20261026.json',{
            'scope':'Out-of-fold actual-state conditional success probabilities; development only',
            'input_manifest_sha256':manifest['sha256'],'family':family,'rows':oof[family]})['sha256']
    forecasts={f:{name:budget_curve(full[f]['model'],labels) for name,labels in {
        'correct_middle':['N','C','N'],'neutral_full':['N','N','N'],'repeated_error':['E','N','E'],
        'correct_latest':['E','N','C'],'delete_latest':['E','N'],'correct_oldest':['C','N','E'],'error_middle':['N','E','N']}.items()}
        for f in FAMILIES}
    selection={'selected':selected,'six_minus_seven_mean':float(delta.mean()),'paired_scene_difference_SE':se,
        'scenes_with_seven_lower_loss':int((delta>1e-12).sum()),'scene_order':scenes,'paired_scene_values':delta.tolist(),
        'interpretation':'Prespecified development simplicity rule; no significance or independent-validation claim'}
    for name,h in before.items():assert file_hash(PUBLIC/name)==h,name
    candidate=save(PUBLIC/'successor-development-parameters-20261026.json',{
        'scope':'Development candidate; not a replacement for prior frozen parameters or a validated predictor',
        'input_manifest_sha256':manifest['sha256'],'parent_parameters_sha256':params['sha256'],
        'selection':selection,'models':full,'selected_model':full[selected]['model'],
        'unscored_budget_forecasts':forecasts,'source_sha256':source,'new_model_calls':0,'original_parameters_overwritten':False})
    result=save(PUBLIC/'successor-development-results-20261026.json',{
        'scope':'Fixed six-versus-seven development comparison after validation reuse',
        'input_manifest_sha256':manifest['sha256'],'parameters_sha256':candidate['sha256'],
        'summaries':summaries,'selection':selection,'folds':folds,'prediction_parts':prediction_parts,
        'gradient_checks':checks,'frozen_file_hashes_unchanged':before,'source_sha256':source,
        'elapsed_seconds':time.monotonic()-started,'new_model_calls':0,'holdout134_calls':0,
        'original_parameter_refits':0,'development_success_fits':60,'write_kernel_refits':0})
    print({'sha256':result['sha256'],'parameters_sha256':candidate['sha256'],'selection':selection,
           'summaries':summaries,'training_groups':{f:full[f]['training_groups'] for f in FAMILIES},'forecasts':forecasts},flush=True)


if __name__=='__main__':main()
