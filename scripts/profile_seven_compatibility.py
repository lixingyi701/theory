#!/usr/bin/env python3
"""Bounded seven-coefficient compatibility, uncertainty and scene-held-out diagnosis."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse,sys,time
import numpy as np

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-seven-compatibility-20261027')
sys.path.insert(0,str(PRIVATE/'numerics'))
import scipy
from scipy.optimize import minimize
from agent_reliability.io import digest,file_hash,read_json
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import checked
from compare_seven_reached_risk import pack,predict,score,groups
from compare_seven_sequential_objectives import softmax_rows,objective as previous_objective
import shared_recovery_model as original

PROTOCOL=PUBLIC/'seven-compatibility-protocol-20261027.json'
SOURCE=Path('scripts/profile_seven_compatibility.py')
DESIGN=Path('docs/experiments/seven-compatibility-design-20261027.md')
NAMES=['initial_C_N','initial_error','second_attempt','late_error','late_no_error']
LIMITS={'parameter_solves':94,'computation_seconds':900,'cpu_workers':2,'bootstrap_draws':20000,'new_agent_calls':0}
OPTIONS={'maxiter':300,'ftol':1e-11,'disp':False}
SEED=2026102706
BOUNDS=[(-30.,30.)]*7
CASES=[{'name':'all_tol0','tolerance':0.,'omit':None},
    {'name':'all_tol0.02','tolerance':.02,'omit':None},{'name':'all_tol0.05','tolerance':.05,'omit':None}]
CASES += [{'name':'omit_'+name,'tolerance':.02,'omit':name} for name in NAMES]
AXES={2:'native_count',4:'error_presence',5:'error_lifetime',6:'error_native_interaction'}


def arrays(rows):
    data=pack(rows);index={tid:i for i,tid in enumerate(data['identities'])};masks=np.zeros((5,)+data['mask'].shape)
    for r in rows:
        n=sum(s['native'] for s in r['state']);e=any(s['type'] in ['E','B'] for s in r['state'])
        flags=[r['trial']==1 and r['condition'] in ['correct_middle','neutral_full'],r['trial']==1 and e,
            r['trial']==2,n>=2 and e,n>=2 and not e]
        masks[:,index[r['trajectory_id']],r['trial']-1]=flags
    assert np.array_equal(masks.sum(axis=0),data['mask'])
    data['group_alpha']=masks*data['alpha'][None,:,:]
    data['group_mass']=data['group_alpha'].sum(axis=(1,2));assert min(data['group_mass'])>0
    return data


class Oracle:
    def __init__(self,data,deadline=float('inf')):
        self.data=data;self.deadline=deadline;self.last=None;self.value=None

    def __call__(self,beta):
        if time.monotonic()>self.deadline:raise TimeoutError('Whole compatibility-computation allowance exhausted')
        beta=np.asarray(beta,float)
        if self.last is not None and np.array_equal(beta,self.last):return self.value
        d=self.data;x=d['x'];n,t,_=x.shape
        logw=np.broadcast_to(np.log(original.PRIOR),(n,3)).copy();h=np.zeros((n,3,7))
        q=np.zeros((n,t));prior_q=np.zeros((n,t));jac=np.zeros((n,t,7))
        for j in range(t):
            current=x[:,j];eta=(current@beta)[:,None]+1.5*original.OFFSETS[None,:]
            p=np.exp(-np.logaddexp(0,-eta));w=softmax_rows(logw)
            centered=h-np.einsum('iz,izk->ik',w,h)[:,None,:]
            q[:,j]=np.einsum('iz,iz->i',w,p);prior_q[:,j]=p@original.PRIOR
            jac[:,j]=np.einsum('iz,izk->ik',w,p[:,:,None]*centered)
            jac[:,j]+=np.einsum('iz,iz->i',w,p*(1-p))[:,None]*current
            failed=d['mask'][:,j] & (d['y'][:,j]==0)
            logw[failed]-=np.logaddexp(0,eta[failed]);logw-=logw.max(axis=1,keepdims=True)
            h[failed]-=p[failed,:,None]*current[failed,None,:]
        residual=q-d['y'];risk=float(np.sum(d['alpha']*residual**2))
        grad=np.einsum('ij,ijk->k',2*d['alpha']*residual,jac)
        moments=np.einsum('gij,ij->g',d['group_alpha'],residual)/d['group_mass']
        moment_jac=np.einsum('gij,ijk->gk',d['group_alpha'],jac)/d['group_mass'][:,None]
        b=beta.copy();b[0]=0;penalty=.005*float(b@b)+.5*np.logaddexp(0,-beta[0])+.5*np.logaddexp(0,beta[0])
        penalty_grad=.01*b;penalty_grad[0]+=np.exp(-np.logaddexp(0,-beta[0]))-.5
        jvalue=risk+float(penalty)/d['count'];jgrad=grad+penalty_grad/d['count']
        contribution=np.einsum('gij,ij->g',d['group_alpha'],q-prior_q)/d['group_mass']
        self.last=beta.copy();self.value={'J':jvalue,'gradient':jgrad,'risk':risk,'moments':moments,'moment_jacobian':moment_jac,
            'posterior_contribution':contribution,'q':q,'q_jacobian':jac}
        return self.value


def snapshot(oracle,beta):
    value=oracle(beta)
    return {'coefficients':np.asarray(beta).tolist(),'penalized_risk':value['J'],'risk':value['risk'],
        'group_probability_minus_observation':dict(zip(NAMES,value['moments'].tolist())),
        'failure_posterior_contribution':dict(zip(NAMES,value['posterior_contribution'].tolist()))}


def solve(oracle,initial,tolerance=None,omit=None,fixed=None):
    constraints=[];active=np.array([i for i,k in enumerate(NAMES) if k!=omit])
    if tolerance is not None:
        if tolerance==0:
            constraints.append({'type':'eq','fun':lambda b:oracle(b)['moments'][active],
                'jac':lambda b:oracle(b)['moment_jacobian'][active]})
        else:
            constraints.append({'type':'ineq','fun':lambda b:np.r_[tolerance-oracle(b)['moments'][active],tolerance+oracle(b)['moments'][active]],
                'jac':lambda b:np.r_[-oracle(b)['moment_jacobian'][active],oracle(b)['moment_jacobian'][active]]})
    if fixed is not None:
        column,value=fixed
        constraints.append({'type':'eq','fun':lambda b:np.array([b[column]-value]),'jac':lambda b:np.eye(7)[[column]]})
    result=minimize(lambda b:oracle(b)['J'],np.asarray(initial),jac=lambda b:oracle(b)['gradient'],
        method='SLSQP',bounds=BOUNDS,constraints=constraints,options=OPTIONS)
    state=snapshot(oracle,result.x);m=oracle(result.x)['moments'][active]
    violation=max(0.,float(np.max(np.abs(m))-tolerance)) if tolerance is not None else 0.
    if fixed is not None:violation=max(violation,abs(result.x[fixed[0]]-fixed[1]))
    return {**state,'optimizer_success':bool(result.success),'optimizer_status':int(result.status),'message':str(result.message),
        'iterations':int(result.nit),'constraint_violation':violation,'feasible':violation<=1e-6,
        'initial':np.asarray(initial).tolist(),'coefficient_bound_hit':bool(np.any(np.abs(result.x)>29.999))}


def engineering_checks(rows,base):
    data=arrays(rows);oracle=Oracle(data);errors=[]
    for beta in [np.zeros(7),np.linspace(-.3,.3,7),np.asarray(base['p_coefficients'])]:
        value=oracle(beta);old,grad=previous_objective(beta,data,'prequential_brier')
        assert abs(old/data['count']-value['J'])<1e-12 and np.max(np.abs(grad/data['count']-value['gradient']))<1e-12
        columns=[];objective_numeric=[]
        for j in range(7):
            step=np.eye(7)[j]*1e-5;a=oracle(beta+step);b=oracle(beta-step)
            columns.append((a['moments']-b['moments'])/(2e-5));objective_numeric.append((a['J']-b['J'])/(2e-5))
        me=float(np.max(np.abs(np.array(columns).T-value['moment_jacobian'])))
        je=float(np.max(np.abs(objective_numeric-value['gradient'])));assert me<1e-7 and je<1e-7
        errors.append({'point':beta.tolist(),'moment_jacobian_error':me,'risk_gradient_error':je})
    return {'derivatives':errors,'previous_objective_matches':True,'groups_partition_attempts':True,
        'group_attempt_counts':dict(zip(NAMES,[int(np.count_nonzero(a)) for a in data['group_alpha']]))}


def prepare():
    assert scipy.__version__=='1.15.3'
    assert not (PRIVATE/'contract.json').exists(),'Do not reset an existing compatibility analysis'
    PRIVATE.mkdir(parents=True,exist_ok=True)
    parent=checked(PUBLIC/'seven-reached-risk-protocol-20261027.json')
    rr=checked(PUBLIC/'seven-reached-risk-results-20261027.json')
    audit=checked(PUBLIC/'seven-reached-risk-audit-20261027.json')
    dev=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    rows=read_json(Path('runs/reflexion-seven-reached-risk-20261027/input-events.json'))
    assert digest(rows)==parent['input_rows_sha256'] and len(rows)==1553
    (PRIVATE/'input-events.json').write_bytes(Path('runs/reflexion-seven-reached-risk-20261027/input-events.json').read_bytes())
    mle=next(c for c in dev['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    brier=next(c for c in rr['candidates'] if c['mode']=='prequential_brier')
    protected=dict(parent['protected_files'])
    paths=[SOURCE,DESIGN,PUBLIC/'seven-reached-risk-protocol-20261027.json',PUBLIC/'seven-reached-risk-results-20261027.json',PUBLIC/'seven-reached-risk-audit-20261027.json']
    paths += [PUBLIC/c['prediction_file'] for c in rr['candidates']]
    for path in paths:protected[path.as_posix()]=file_hash(path)
    for name,h in protected.items():assert file_hash(Path(name))==h,name
    oracle=Oracle(arrays(rows))
    result=save(PROTOCOL,{'scope':'Seven-dimensional compatibility diagnostics on reused development records',
        'parent_protocol_sha256':parent['sha256'],'parent_result_sha256':rr['sha256'],'parent_audit_sha256':audit['sha256'],
        'input_rows_sha256':digest(rows),'private_input_file_sha256':file_hash(PRIVATE/'input-events.json'),
        'base_model':parent['base_model'],'attempts':1553,'tasks':163,'scenes':27,'trajectories':760,'successes':420,
        'mle_reference':mle,'brier_reference':brier,'groups':NAMES,'cases':CASES,'profile_axes':AXES,'profile_offsets':[-1.,-.5,.5,1.],
        'limits':LIMITS,'optimizer':{'name':'SciPy SLSQP','options':OPTIONS,'coefficient_bounds':BOUNDS,'feasibility_tolerance':1e-6},
        'dependency':read_json(PRIVATE/'dependency.json'),'checks':engineering_checks(rows,parent['base_model']),
        'full_start_order':'Previous full predictable-Brier beta, previous full MLE beta, zero beta; 8cases*3=24 solves',
        'coordinate_profiles':'Fix one existing coefficient at Brier beta plus offset, reoptimize remaining6 from Brier beta; 4axes*4=16 solves; reuse center',
        'calibration_constraints':'Absolute fixed-start-weight mean prediction-minus-observation within delta for five predetermined groups; empirical moment compatibility only',
        'near_optimal_witness':'Feasible coefficients with J<=previous full Brier J+0.001; J=R+original penalty/3040. No global impossibility claim if search fails',
        'cross_validation':'All five group tolerances fixed0.02; 27whole-scene folds, two TRAINING-ONLY starts from previous Brier and MLE fold betas =54 solves; choose converged feasible lower TRAINING J; never use held outcomes to fit constraints',
        'selection':'Primary MLE-CV minus constrained-CV predictable Brier risk; retain only for development if gain>paired scene SE, initial_C_N and late_error losses do not exceed MLE, all27folds have converged feasible fits and full tol0.02 fit exists. Do not replace historical frozen baseline',
        'bootstrap':{'draws':20000,'seed':SEED,'hierarchy':'resample27scenes, then tasks within each sampledscene; preserve branches/seeds together',
            'predictions':'previous MLE held-scene probabilities FIXED; report pointwise percentile and centered max-standardized simultaneous intervals, not refit uncertainty or independent validation'},
        'baseline_full_brier':snapshot(oracle,brier['full_model']['p_coefficients']),
        'protected_files':protected,'new_model_calls':0,'holdout134_calls':0})
    print({'sha256':result['sha256'],'checks':result['checks'],'protected_files':len(protected),'planned_solves':94},flush=True)


def worker(job):
    protocol=checked(PROTOCOL);contract=checked(PRIVATE/'contract.json');rows=read_json(PRIVATE/'input-events.json')
    assert digest(rows)==protocol['input_rows_sha256'];brier=protocol['brier_reference'];mle=protocol['mle_reference']
    beta=np.array(brier['full_model']['p_coefficients']);kind=job['kind']
    if kind=='full':
        oracle=Oracle(arrays(rows),contract['deadline_monotonic']);case=job['case'];runs=[]
        for initial in [beta,np.array(mle['full_model']['p_coefficients']),np.zeros(7)]:
            runs.append(solve(oracle,initial,case['tolerance'],case['omit']))
        possible=[r for r in runs if r['feasible']];best=min(possible,key=lambda r:r['penalized_risk']) if possible else None
        result={'kind':kind,'case':case,'solves':3,'runs':runs,'best_feasible':best,
            'near_optimal_witness_found':best is not None and best['penalized_risk']<=protocol['baseline_full_brier']['penalized_risk']+.001}
    elif kind=='coordinate':
        oracle=Oracle(arrays(rows),contract['deadline_monotonic']);column=job['column'];value=float(beta[column]+job['offset'])
        result={'kind':kind,'column':column,'axis':AXES[column],'offset':job['offset'],'solves':1,
            'fit':solve(oracle,beta,fixed=(column,value))}
    else:
        scene=job['scene'];train=[r for r in rows if r['scene_id']!=scene];test=[r for r in rows if r['scene_id']==scene]
        oracle=Oracle(arrays(train),contract['deadline_monotonic']);runs=[]
        bfold=next(f for f in brier['folds'] if f['held_scene']==scene)
        mfold=next(f for f in mle['folds'] if f['scene_id']==scene)
        for initial in [bfold['coefficients'],mfold['coefficients']]:runs.append(solve(oracle,initial,tolerance=.02))
        possible=[r for r in runs if r['feasible'] and r['optimizer_success']]
        best=min(possible,key=lambda r:r['penalized_risk']) if possible else None
        pp=[] if best is None else predict(test,{**protocol['base_model'],'p_coefficients':best['coefficients']})
        result={'kind':kind,'held_scene':scene,'solves':2,'runs':runs,'chosen':best,'predictions':pp,'training_only_constraints':True}
    save(PRIVATE/('worker-'+job['name']+'.json'),result)
    return result


def bootstrap(rows,predictions,deadline):
    probabilities={(p['trajectory_id'],p['trial']):p['probability'] for p in predictions};data=arrays(rows)
    task_rows=defaultdict(list)
    for r in rows:task_rows[r['scene_id'],r['task_id']].append(r)
    task_values=defaultdict(list)
    for (scene,task),entries in sorted(task_rows.items()):
        a=np.zeros((2,5));nbranches=len({r['trajectory_id'] for r in entries})
        for r in entries:
            n=sum(s['native'] for s in r['state']);e=any(s['type'] in ['E','B'] for s in r['state'])
            flags=np.array([r['trial']==1 and r['condition'] in ['correct_middle','neutral_full'],r['trial']==1 and e,r['trial']==2,n>=2 and e,n>=2 and not e])
            a[0]+=flags*(probabilities[r['trajectory_id'],r['trial']]-float(r['success']))/(4*nbranches)
            a[1]+=flags/(4*nbranches)
        task_values[scene].append(a)
    blocks=[np.array(v) for k,v in sorted(task_values.items())];aggregate=np.array([b.mean(axis=0) for b in blocks]).sum(axis=0)
    center=aggregate[0]/aggregate[1];rng=np.random.default_rng(SEED);draws=[]
    for draw in range(LIMITS['bootstrap_draws']):
        if time.monotonic()>deadline:raise TimeoutError('Bootstrap exceeded whole computation allowance')
        total=np.zeros((2,5))
        for selected in rng.integers(0,len(blocks),size=len(blocks)):
            block=blocks[selected];total+=block[rng.integers(0,len(block),size=len(block))].mean(axis=0)
        if np.all(total[1]>0):draws.append(total[0]/total[1])
    values=np.array(draws);assert len(values)==20000
    se=values.std(axis=0,ddof=1);critical=float(np.quantile(np.max(np.abs((values-center)/se),axis=1),.95))
    result={'draws':20000,'seed':SEED,'fixed_predictor_residuals':dict(zip(NAMES,center.tolist())),
        'standard_errors':dict(zip(NAMES,se.tolist())),'pointwise_percentile95':dict(zip(NAMES,np.quantile(values,[.025,.975],axis=0).T.tolist())),
        'simultaneous_centered95':{name:[float(center[i]-critical*se[i]),float(center[i]+critical*se[i])] for i,name in enumerate(NAMES)},
        'max_standardized_critical':critical,'interpretation':'Reused-data fixed-CV-predictor sampling diagnostic; no parameter refits, no post-selection coverage guarantee'}
    np.savez_compressed(PRIVATE/'bootstrap-draws.npz',residuals=values)
    result['private_draws_file_sha256']=file_hash(PRIVATE/'bootstrap-draws.npz');return result


def run():
    protocol=checked(PROTOCOL);assert protocol['limits']==LIMITS
    assert not (PRIVATE/'contract.json').exists(),'Do not restart a compatibility campaign'
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    freeze=(PRIVATE/'freeze-commit.txt').read_text().strip();assert len(freeze)==40
    started=time.monotonic();contract=save(PRIVATE/'contract.json',{'protocol_sha256':protocol['sha256'],'freeze_commit':freeze,
        'started_unix':time.time(),'deadline_monotonic':started+900,'limits':LIMITS})
    jobs=[{'name':c['name'],'kind':'full','case':c} for c in CASES]
    jobs += [{'name':f'axis{column}_{offset:g}','kind':'coordinate','column':column,'offset':offset} for column in AXES for offset in [-1.,-.5,.5,1.]]
    jobs += [{'name':'fold'+f['held_scene'],'kind':'fold','scene':f['held_scene']} for f in protocol['brier_reference']['folds']]
    with ProcessPoolExecutor(max_workers=2) as pool:outputs=list(pool.map(worker,jobs))
    assert sum(o['solves'] for o in outputs)==94
    rows=read_json(PRIVATE/'input-events.json');mle_pred=checked(PUBLIC/protocol['mle_reference']['prediction_file'])
    uncertainty=bootstrap(rows,mle_pred['predictions'],contract['deadline_monotonic'])
    unc=save(PUBLIC/'seven-compatibility-uncertainty-20261027.json',{'protocol_sha256':protocol['sha256'],**uncertainty})
    full=[o for o in outputs if o['kind']=='full'];coordinate=[o for o in outputs if o['kind']=='coordinate'];folds=[o for o in outputs if o['kind']=='fold']
    predictions=[p for f in folds for p in f['predictions']];complete=all(f['chosen'] is not None for f in folds)
    forecast=save(PUBLIC/'seven-compatibility-predictions-20261027.json',{'protocol_sha256':protocol['sha256'],
        'complete':complete,'predictions':predictions,'missing_scenes':[f['held_scene'] for f in folds if f['chosen'] is None]})
    selection={'complete':complete,'retained_for_development':False,'replace_frozen_baseline':False}
    scoring=None
    if complete:
        assert len(predictions)==1553;scoring={'primary':score(rows,predictions),'groups':groups(rows,predictions)}
        reference=protocol['mle_reference'];refp=checked(PUBLIC/reference['prediction_file']);refscore=score(rows,refp['predictions']);refgroups=groups(rows,refp['predictions'])
        scenes=sorted(refscore['scene_contributions']);delta=np.array([refscore['scene_contributions'][s]-scoring['primary']['scene_contributions'][s] for s in scenes])
        gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        guards={k:scoring['groups'][k]['weighted_conditional_brier']<=refgroups[k]['weighted_conditional_brier'] for k in ['initial_C_N','late_error']}
        full02=next(o for o in full if o['case']['name']=='all_tol0.02')['best_feasible']
        selection.update(gain=gain,paired_scene_SE=se,guards=guards,retained_for_development=gain>se and all(guards.values()) and full02 is not None)
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    result=save(PUBLIC/'seven-compatibility-results-20261027.json',{'scope':'Seven-coefficient empirical compatibility and held-scene development, not new independent validation',
        'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],'uncertainty_sha256':unc['sha256'],
        'prediction_sha256':forecast['sha256'],'full_cases':full,'coordinate_profiles':coordinate,'folds':[{k:v for k,v in f.items() if k!='predictions'} for f in folds],
        'cross_validation':scoring,'selection':selection,'parameter_solves':94,'elapsed_seconds':time.monotonic()-started,
        'new_model_calls':0,'holdout134_calls':0,'protected_files_unchanged':protocol['protected_files']})
    closed=save(PRIVATE/'closeout.json',{'status':'complete','protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],
        'parameter_solves':94,'elapsed_seconds':time.monotonic()-started,'new_model_calls':0})
    save(PUBLIC/'seven-compatibility-closeout-20261027.json',{k:v for k,v in closed.items() if k!='sha256'})
    print({'sha256':result['sha256'],'selection':selection,'seconds':result['elapsed_seconds'],
        'full_cases':[{'case':o['case']['name'],'witness':o['near_optimal_witness_found'],
            'J':None if o['best_feasible'] is None else o['best_feasible']['penalized_risk'],
            'moments':None if o['best_feasible'] is None else o['best_feasible']['group_probability_minus_observation']} for o in full],
        'residual_pointwise95':uncertainty['pointwise_percentile95'],
        'cross_validation':None if scoring is None else {'risk':scoring['primary']['risk_per_planned_horizon'],
            'groups':{k:v['weighted_conditional_brier'] for k,v in scoring['groups'].items()}}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
