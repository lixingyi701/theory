#!/usr/bin/env python3
"""Seven-feature development with predictable reached-attempt weights."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse,time
import numpy as np
from agent_reliability.io import digest,file_hash,read_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from compare_seven_sequential_objectives import pack as legacy_pack,objective,MODES
from develop_seven_parameterization import predict
import shared_recovery_model as original

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-seven-reached-risk-20261027')
PARENT=Path('runs/reflexion-seven-sequential-objectives-20261027')
PROTOCOL=PUBLIC/'seven-reached-risk-protocol-20261027.json'
SOURCE=Path('scripts/compare_seven_reached_risk.py')
DESIGN=Path('docs/experiments/seven-reached-risk-design-20261027.md')
HORIZON=4
BRANCHES={'original_calibration':7,'successor_review':4,'completed_successor_primary':4}
TASKS={'original_calibration':36,'successor_review':27,'completed_successor_primary':100}
LIMITS={'seconds':900,'max_parameter_fits':60,'cpu_workers':2,'new_agent_calls':0}


def start_weights(rows):
    trajectories=defaultdict(list);scene_tasks=defaultdict(set);task_trajectories=defaultdict(set)
    for row in rows:
        trajectories[row['trajectory_id']].append(row)
        scene_tasks[row['scene_id']].add(row['task_id']);task_trajectories[row['task_id']].add(row['trajectory_id'])
    weights={}
    for tid,entries in trajectories.items():
        first=min(entries,key=lambda r:r['trial']);source=first['source'];task=first['task_id'];scene=first['scene_id']
        assert len(task_trajectories[task])==BRANCHES[source]
        assert len({(r['source'],r['task_id'],r['scene_id']) for r in entries})==1
        assert 1<=len(entries)<=HORIZON
        weights[tid]=1/(len(scene_tasks)*len(scene_tasks[scene])*BRANCHES[source])
    assert abs(sum(weights.values())-1)<1e-12
    return weights


def pack(rows):
    data=legacy_pack(rows);weights=start_weights(rows)
    for i,tid in enumerate(data['identities']):
        data['alpha'][i,:]=weights[tid]/HORIZON*data['mask'][i,:]
    # Scaling depends only on the complete predeclared branch cohort, never realized lengths.
    data['count']=HORIZON*len(data['identities'])
    data['actual_attempts']=len(rows)
    assert 0<data['alpha'].sum()<=1+1e-12
    return data


def score(rows,predictions,predicate=lambda r:True):
    probs={(p['trajectory_id'],p['trial']):p['probability'] for p in predictions}
    weights=start_weights(rows);scenes=sorted({r['scene_id'] for r in rows})
    numerator={s:0. for s in scenes};exposure={s:0. for s in scenes};observed=expected=0.;n=0
    for row in rows:
        if not predicate(row):continue
        p=probs[row['trajectory_id'],row['trial']];a=weights[row['trajectory_id']]/HORIZON
        numerator[row['scene_id']]+=a*(p-float(row['success']))**2
        exposure[row['scene_id']]+=a;observed+=a*float(row['success']);expected+=a*p;n+=1
    total=sum(numerator.values());mass=sum(exposure.values())
    return {'risk_per_planned_horizon':total,'scene_contributions':{s:v*len(scenes) for s,v in numerator.items()},
        'reached_weight':mass,'weighted_conditional_brier':total/mass if mass else None,'attempts':n,
        'observed_successes':sum(r['success'] for r in rows if predicate(r)),
        'expected_successes':sum(probs[r['trajectory_id'],r['trial']] for r in rows if predicate(r)),
        'weighted_observed_success_fraction':observed/mass if mass else None,
        'weighted_expected_success_fraction':expected/mass if mass else None}


def groups(rows,predictions):
    predicates={'initial_C_N':lambda r:r['trial']==1 and r['condition'] in ['correct_middle','neutral_full'],
        'initial_error':lambda r:r['trial']==1 and any(s['type'] in ['E','B'] for s in r['state']),
        'second_attempt':lambda r:r['trial']==2,
        'late_error':lambda r:sum(s['native'] for s in r['state'])>=2 and any(s['type'] in ['E','B'] for s in r['state']),
        'late_no_error':lambda r:sum(s['native'] for s in r['state'])>=2 and not any(s['type'] in ['E','B'] for s in r['state'])}
    return {k:score(rows,predictions,pred) for k,pred in predicates.items()}


def checks(rows,base):
    data=pack(rows);points=[np.zeros(7),np.linspace(-.4,.4,7),np.asarray(base['p_coefficients'])];cases=[]
    for mode in MODES:
        for beta in points:
            _,g=objective(beta,data,mode);numeric=[]
            for j in range(7):
                d=np.eye(7)[j]*1e-5;numeric.append((objective(beta+d,data,mode)[0]-objective(beta-d,data,mode)[0])/(2e-5))
            error=float(np.max(np.abs(g-numeric)));assert error<1e-6
            cases.append({'mode':mode,'point':beta.tolist(),'max_gradient_difference':error})
    _,_,pp=objective(points[-1],data,'prequential_brier',return_predictions=True)
    risk=score(rows,pp)['risk_per_planned_horizon']
    value,_=objective(points[-1],data,'prequential_brier',penalty=False)
    assert abs(value/data['count']-risk)<1e-12
    # Truncating to first-attempt prefixes retains the same preplanned identities/weights.
    prefix=pack([r for r in rows if r['trial']==1])
    assert prefix['identities']==data['identities'] and prefix['count']==data['count']
    assert np.array_equal(prefix['alpha'][:,0],data['alpha'][:,0])
    return {'gradients':cases,'objective_vs_score_difference':abs(value/data['count']-risk),
        'prefix_weight_invariance':True,'preplanned_loss_scale':data['count'],'scope':'Engineering checks, not Agent evidence'}


def prepare():
    assert not (PRIVATE/'contract.json').exists(),'Do not reset a reached-risk comparison'
    PRIVATE.mkdir(parents=True,exist_ok=True)
    parent=checked(PUBLIC/'seven-sequential-objectives-protocol-20261027.json')
    old_result=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    seq_result=checked(PUBLIC/'seven-sequential-objectives-results-20261027.json')
    seq_audit=checked(PUBLIC/'seven-sequential-objectives-audit-20261027.json')
    rows=read_json(PARENT/'input-events.json');assert digest(rows)==parent['input_rows_sha256']
    assert len(rows)==1553 and sum(r['success'] for r in rows)==420
    assert {s:len({r['task_id'] for r in rows if r['source']==s}) for s in BRANCHES}==TASKS
    assert len({r['trajectory_id'] for r in rows})==760
    (PRIVATE/'input-events.json').write_bytes((PARENT/'input-events.json').read_bytes())
    protected=dict(parent['protected_files'])
    paths=[SOURCE,DESIGN,PUBLIC/'seven-sequential-objectives-protocol-20261027.json',
        PUBLIC/'seven-sequential-objectives-results-20261027.json',PUBLIC/'seven-sequential-objectives-audit-20261027.json']
    paths += [PUBLIC/c['prediction_file'] for c in seq_result['candidates']]
    for path in paths:protected[path.as_posix()]=file_hash(path)
    for name,h in protected.items():assert file_hash(Path(name))==h,name
    reference=next(c for c in old_result['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    pp=checked(PUBLIC/reference['prediction_file']);assert pp['sha256']==reference['prediction_sha256']
    protocol=save(PROTOCOL,{'scope':'Seven-feature development with predictable reached-attempt weights',
        'parent_protocol_sha256':parent['sha256'],'parent_result_sha256':seq_result['sha256'],'parent_audit_sha256':seq_audit['sha256'],
        'input_rows_sha256':digest(rows),'private_input_file_sha256':file_hash(PRIVATE/'input-events.json'),
        'attempts':1553,'tasks':163,'scenes':27,'trajectories':760,'successes':420,'base_model':parent['base_model'],
        'horizon':HORIZON,'planned_branches_by_source':BRANCHES,'planned_task_counts_by_source':TASKS,
        'weighting':'v_b=1/(numberScenes * tasksInScene * preplannedBranchesInTask); row alpha=v_b/4 if reached, zero if stopped; data scale=4*preplannedTrajectoryCount',
        'modes':MODES,'limits':LIMITS,'folds':parent['folds'],
        'reference':{'config':'ridge0.01_scale1.5','prediction_file':reference['prediction_file'],'prediction_sha256':reference['prediction_sha256']},
        'penalty':'Same ridge .01 on six nonintercept coefficients and symmetric intercept half-count; data scale is preplanned',
        'primary':'Scene-equal risk of sum of reached single-attempt squared losses divided by common preplanned H=4, no inverse realized length',
        'selection':'Lower primary risk among two candidates is retained for development only if reference-minus-candidate scene mean > paired scene SE, initial_C_N and late_error conditional weighted Brier do not increase, and three full starts agree within1e-5. Otherwise retain marginal MLE. Do not select the other candidate after guard failure. No promotion over historical frozen baseline.',
        'reference_scores':{'primary':score(rows,pp['predictions']),'groups':groups(rows,pp['predictions'])},
        'checks':checks(rows,parent['base_model']),'protected_files':protected,'new_model_calls':0,'holdout134_calls':0})
    print({'sha256':protocol['sha256'],'checks':protocol['checks'],'protected':len(protected)},flush=True)


def fit(rows,base,mode,deadline,initial):
    data=pack(rows)
    def fun(beta):
        if time.monotonic()>deadline:raise TimeoutError('Whole reached-risk CPU allowance exhausted')
        value,g=objective(beta,data,mode)
        if not np.isfinite(value) or not np.all(np.isfinite(g)):raise ValueError('Nonfinite objective')
        return value,g
    beta,audit=original.bfgs(fun,np.asarray(initial,float))
    return {**base,'p_coefficients':beta.tolist(),'estimation_objective':mode,'estimation_weighting':'predictable_reached_risk'}, {**audit,'objective':fun(beta)[0]}


def worker(mode):
    protocol=checked(PROTOCOL);contract=checked(PRIVATE/'contract.json');rows=read_json(PRIVATE/'input-events.json')
    assert digest(rows)==protocol['input_rows_sha256'];base=protocol['base_model'];full=[]
    for value in [0.,.2,-.2]:
        model,audit=fit(rows,base,mode,contract['deadline_monotonic'],[value]*7);full.append({'model':model,'audit':audit})
    best=min(full,key=lambda v:v['audit']['objective']);spread=max(r['audit']['objective'] for r in full)-min(r['audit']['objective'] for r in full)
    predictions=[];folds=[]
    for fold in protocol['folds']:
        scene=fold['held_scene'];train=[r for r in rows if r['scene_id']!=scene];test=[r for r in rows if r['scene_id']==scene]
        model,audit=fit(train,base,mode,contract['deadline_monotonic'],np.zeros(7));predictions+=predict(test,model)
        folds.append({'held_scene':scene,'audit':audit,'coefficients':model['p_coefficients']})
    assert len(predictions)==1553
    path=PUBLIC/f'seven-reached-risk-predictions-{mode}-20261027.json'
    pp=save(path,{'protocol_sha256':protocol['sha256'],'mode':mode,'predictions':predictions})
    result={'mode':mode,'parameter_fits':30,'full_model':best['model'],'full_start_audits':[r['audit'] for r in full],
        'full_start_spread':spread,'full_starts_agree':spread<1e-5,'folds':folds,
        'primary':score(rows,predictions),'groups':groups(rows,predictions),'prediction_file':path.name,'prediction_sha256':pp['sha256']}
    save(PRIVATE/f'worker-{mode}.json',result);return result


def run():
    protocol=checked(PROTOCOL);assert protocol['modes']==MODES and protocol['limits']==LIMITS
    assert protocol['horizon']==HORIZON and protocol['planned_branches_by_source']==BRANCHES
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    assert not (PRIVATE/'contract.json').exists(),'Do not reset a CPU contract'
    freeze=(PRIVATE/'freeze-commit.txt').read_text().strip();assert len(freeze)==40
    now=time.monotonic();contract=save(PRIVATE/'contract.json',{'protocol_sha256':protocol['sha256'],'freeze_commit':freeze,
        'started_unix':time.time(),'deadline_monotonic':now+LIMITS['seconds'],'limits':LIMITS})
    with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,MODES))
    reference=protocol['reference_scores'];best=min(results,key=lambda r:r['primary']['risk_per_planned_horizon'])
    scenes=sorted(reference['primary']['scene_contributions'])
    delta=np.array([reference['primary']['scene_contributions'][s]-best['primary']['scene_contributions'][s] for s in scenes])
    gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(scenes)))
    guards={k:best['groups'][k]['weighted_conditional_brier']<=reference['groups'][k]['weighted_conditional_brier'] for k in ['initial_C_N','late_error']}
    use=gain>se and all(guards.values()) and best['full_starts_agree']
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    result=save(PUBLIC/'seven-reached-risk-results-20261027.json',{'scope':'Predictable-weight parameter development, not new Agent validation',
        'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],'reference':reference,'candidates':results,
        'selection':{'best_primary':best['mode'],'gain':gain,'paired_scene_SE':se,'guards':guards,
            'selected_for_development':best['mode'] if use else 'original_task_marginal_likelihood','replace_frozen_baseline':False},
        'parameter_fits':60,'elapsed_seconds':time.monotonic()-now,'new_model_calls':0,'holdout134_calls':0,'protected_files_unchanged':protocol['protected_files']})
    closeout=save(PRIVATE/'closeout.json',{'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],
        'status':'complete','result_sha256':result['sha256'],'parameter_fits':60,'elapsed_seconds':time.monotonic()-now,'new_model_calls':0})
    save(PUBLIC/'seven-reached-risk-closeout-20261027.json',{k:v for k,v in closeout.items() if k!='sha256'})
    print({'sha256':result['sha256'],'selection':result['selection'],'elapsed_seconds':result['elapsed_seconds'],
        'summary':[{'mode':r['mode'],'primary':r['primary'],'groups':r['groups']} for r in results]},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
