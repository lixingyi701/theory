#!/usr/bin/env python3
"""Compare fixed seven-feature estimation objectives on reused development data."""
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import argparse,time
import numpy as np
from agent_reliability.io import digest,file_hash,read_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from calibrate_successor_success import features
from develop_seven_parameterization import predict,nested_scores,groups
import shared_recovery_model as original

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-seven-sequential-objectives-20261027')
PARENT=Path('runs/reflexion-seven-parameter-development-20261027')
PROTOCOL=PUBLIC/'seven-sequential-objectives-protocol-20261027.json'
DESIGN=Path('docs/experiments/seven-sequential-objectives-design-20261027.md')
SOURCE=Path('scripts/compare_seven_sequential_objectives.py')
MODES=['prequential_log','prequential_brier']
LIMITS={'seconds':900,'max_parameter_fits':60,'cpu_workers':2,'new_agent_calls':0}


def pack(rows):
    """Weights implement attempt->trajectory->task->scene mean exactly."""
    seq=defaultdict(list)
    for row in rows:seq[row['trajectory_id']].append(row)
    identities=sorted(seq);tasks=defaultdict(set);scenes=defaultdict(set)
    for tid in identities:
        seq[tid].sort(key=lambda r:r['trial'])
        first=seq[tid][0];tasks[first['task_id']].add(tid);scenes[first['scene_id']].add(first['task_id'])
        assert [r['trial'] for r in seq[tid]]==list(range(1,len(seq[tid])+1))
        assert not any(r['success'] for r in seq[tid][:-1])
        assert len({(r['task_id'],r['scene_id']) for r in seq[tid]})==1
    shape=(len(identities),max(map(len,seq.values())))
    x=np.zeros(shape+(7,));y=np.zeros(shape);mask=np.zeros(shape,bool);alpha=np.zeros(shape)
    for i,tid in enumerate(identities):
        entries=seq[tid];task=entries[0]['task_id'];scene=entries[0]['scene_id']
        weight=1/(len(scenes)*len(scenes[scene])*len(tasks[task])*len(entries))
        for j,row in enumerate(entries):
            x[i,j]=features(row['state'],'seven');y[i,j]=float(row['success']);mask[i,j]=True;alpha[i,j]=weight
    assert abs(alpha.sum()-1)<1e-12
    return {'x':x,'y':y,'mask':mask,'alpha':alpha,'identities':identities,'count':len(rows)}


def softmax_rows(values):
    values=values-values.max(axis=1,keepdims=True);answer=np.exp(values)
    return answer/answer.sum(axis=1,keepdims=True)


def objective(beta,data,mode,scale=1.5,ridge=.01,penalty=True,return_predictions=False):
    """Differentiate both the current probability and previous failure posterior."""
    if mode not in MODES:raise ValueError(mode)
    x,y,mask,alpha=(data[k] for k in ['x','y','mask','alpha'])
    count=len(x);logweights=np.broadcast_to(np.log(original.PRIOR),(count,3)).copy()
    history_gradient=np.zeros((count,3,7));loss=0.;gradient=np.zeros(7);output=[]
    for j in range(x.shape[1]):
        current=x[:,j];logits=(current@beta)[:,None]+scale*original.OFFSETS[None,:]
        logs=-np.logaddexp(0,-logits);logf=-np.logaddexp(0,logits)
        p=np.exp(logs);weights=softmax_rows(logweights)
        mean_history=np.einsum('iz,izk->ik',weights,history_gradient)
        centered=history_gradient-mean_history[:,None,:]
        probability=np.einsum('iz,iz->i',weights,p)
        derivative=np.einsum('iz,izk->ik',weights,p[:,:,None]*centered)
        derivative+=np.einsum('iz,iz->i',weights,p*(1-p))[:,None]*current
        if mode=='prequential_brier':
            loss_rows=(probability-y[:,j])**2
            grad_rows=2*(probability-y[:,j])[:,None]*derivative
        else:
            joint_s=logweights+logs;joint_f=logweights+logf
            def logsum(a):
                m=a.max(axis=1);return m+np.log(np.exp(a-m[:,None]).sum(axis=1))
            normalization=logsum(logweights)
            loss_rows=-y[:,j]*(logsum(joint_s)-normalization)-(1-y[:,j])*(logsum(joint_f)-normalization)
            score_s=(1-p)[:,:,None]*current[:,None,:]+centered
            score_f=-p[:,:,None]*current[:,None,:]+centered
            grad_rows=-y[:,j,None]*np.einsum('iz,izk->ik',softmax_rows(joint_s),score_s)
            grad_rows-=(1-y[:,j,None])*np.einsum('iz,izk->ik',softmax_rows(joint_f),score_f)
        a=data['count']*alpha[:,j]
        loss+=float(a@loss_rows);gradient+=np.einsum('i,ik->k',a,grad_rows)
        if return_predictions:
            for i in np.flatnonzero(mask[:,j]):output.append({'trajectory_id':data['identities'][i],'trial':j+1,'probability':float(probability[i])})
        failed=mask[:,j] & (y[:,j]==0)
        logweights[failed]+=logf[failed]
        logweights-=np.max(logweights,axis=1,keepdims=True)
        history_gradient[failed]-=p[failed,:,None]*current[failed,None,:]
    if penalty:
        non_intercept=beta.copy();non_intercept[0]=0
        loss+=.5*ridge*float(non_intercept@non_intercept)+.5*np.logaddexp(0,-beta[0])+.5*np.logaddexp(0,beta[0])
        gradient+=ridge*non_intercept;gradient[0]+=np.exp(-np.logaddexp(0,-beta[0]))-.5
    if return_predictions:return float(loss),gradient,output
    return float(loss),gradient


def engineering_checks(rows,base):
    data=pack(rows);cases=[]
    for mode in MODES:
        for beta in [np.zeros(7),np.linspace(-.4,.4,7),np.asarray(base['p_coefficients'])]:
            value,gradient=objective(beta,data,mode);numeric=[]
            for j in range(7):
                d=np.eye(7)[j]*1e-5
                numeric.append((objective(beta+d,data,mode)[0]-objective(beta-d,data,mode)[0])/(2e-5))
            err=float(np.max(np.abs(gradient-numeric)));assert err<1e-6,(mode,err)
            cases.append({'mode':mode,'point':beta.tolist(),'max_finite_difference_error':err})
    _,_,pp=objective(np.asarray(base['p_coefficients']),data,'prequential_brier',return_predictions=True)
    expected={(r['trajectory_id'],r['trial']):r['probability'] for r in predict(rows,base)}
    error=max(abs(r['probability']-expected[r['trajectory_id'],r['trial']]) for r in pp);assert error<1e-12
    # Zero offsets collapse the latent posterior; analytic derivatives reduce to ordinary scores.
    beta=np.linspace(-.2,.2,7);p=np.exp(-np.logaddexp(0,-data['x']@beta));a=data['count']*data['alpha'];target=data['y']
    for mode in MODES:
        v,g=objective(beta,data,mode,scale=0,penalty=False)
        if mode=='prequential_brier':
            direct=float(np.sum(a*(p-target)**2));grad=np.einsum('ij,ijk->k',a*2*(p-target)*p*(1-p),data['x'])
        else:
            z=data['x']@beta;direct=float(np.sum(a*(target*np.logaddexp(0,-z)+(1-target)*np.logaddexp(0,z))))
            grad=np.einsum('ij,ijk->k',a*(p-target),data['x'])
        assert abs(v-direct)<1e-10 and np.max(np.abs(g-grad))<1e-10
    return {'gradient_checks':cases,'existing_predictor_max_difference':error,'zero_offset_reduction_passed':True,'scope':'Engineering checks, not Agent evidence'}


def prepare():
    assert not (PRIVATE/'contract.json').exists(),'Do not reset an existing objective-comparison contract'
    PRIVATE.mkdir(parents=True,exist_ok=True)
    parent=checked(PUBLIC/'seven-parameter-development-protocol-20261027.json')
    result=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    audit=checked(PUBLIC/'seven-parameter-development-audit-20261027.json')
    rows=read_json(PARENT/'input-events.json');assert digest(rows)==parent['input_rows_sha256']
    assert len(rows)==1553 and sum(r['success'] for r in rows)==420
    (PRIVATE/'input-events.json').write_bytes((PARENT/'input-events.json').read_bytes())
    protected=dict(parent['protected_files'])
    names=[SOURCE,DESIGN,PUBLIC/'seven-parameter-development-protocol-20261027.json',PUBLIC/'seven-parameter-development-results-20261027.json',PUBLIC/'seven-parameter-development-audit-20261027.json']
    names += [PUBLIC/c['prediction_file'] for c in result['configs']]
    for path in names:protected[path.as_posix()]=file_hash(path)
    for name,h in protected.items():assert file_hash(Path(name))==h,name
    reference=next(c for c in result['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    protocol=save(PROTOCOL,{'scope':'Seven-feature estimation-objective development on reused records',
        'parent_protocol_sha256':parent['sha256'],'parent_result_sha256':result['sha256'],'parent_audit_sha256':audit['sha256'],
        'input_rows_sha256':digest(rows),'private_input_file_sha256':file_hash(PRIVATE/'input-events.json'),
        'attempts':1553,'tasks':163,'scenes':27,'trajectories':760,'successes':420,
        'base_model':parent['base_model'],'reference_config':reference['config'],'reference_prediction_file':reference['prediction_file'],
        'reference_prediction_sha256':reference['prediction_sha256'],'modes':MODES,'limits':LIMITS,'folds':parent['folds'],
        'objective':{'data_term':'N times scene-equal(task-equal(trajectory-equal(mean reached-attempt loss)))',
            'posterior':'Reset fixed prior per trajectory, update after own failures; differentiate posterior with respect to beta',
            'penalty':'Original ridge .01 on six nonintercept coefficients plus original symmetric intercept half-count',
            'inference':'Same seven-feature kernel, task scale1.5, offsets/prior and conditional G unchanged',
            'weight_caveat':'Observed trajectory lengths determine weights; score alignment is not a consistency claim for unweighted conditional probabilities'},
        'selection':'Choose lower primary Brier among two new objectives only if reference-minus-candidate scene mean > paired scene SE and both initial_C_N and late_error Brier do not exceed reference; otherwise retain marginal-likelihood reference. No promotion over historical frozen baseline in this development stage.',
        'checks':engineering_checks(rows,parent['base_model']),'protected_files':protected,'new_model_calls':0,'holdout134_calls':0})
    print({'sha256':protocol['sha256'],'checks':protocol['checks'],'protected':len(protected)},flush=True)


def fit(rows,base,mode,deadline,initial):
    data=pack(rows)
    def fun(beta):
        if time.monotonic()>deadline:raise TimeoutError('Whole objective-comparison CPU allowance exhausted')
        value,grad=objective(beta,data,mode)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):raise ValueError('Nonfinite objective')
        return value,grad
    beta,audit=original.bfgs(fun,np.asarray(initial,float))
    return {**base,'p_coefficients':beta.tolist(),'estimation_objective':mode}, {**audit,'objective':fun(beta)[0]},fun


def worker(mode):
    protocol=checked(PROTOCOL);contract=checked(PRIVATE/'contract.json');rows=read_json(PRIVATE/'input-events.json')
    assert digest(rows)==protocol['input_rows_sha256'];base=protocol['base_model'];full=[];fits=0
    for start in [0.,.2,-.2]:
        model,audit,fun=fit(rows,base,mode,contract['deadline_monotonic'],[start]*7);fits+=1
        full.append({'model':model,'audit':audit})
    best=min(full,key=lambda r:r['audit']['objective']);spread=max(r['audit']['objective'] for r in full)-min(r['audit']['objective'] for r in full)
    predictions=[];folds=[]
    for fold in protocol['folds']:
        scene=fold['held_scene'];train=[r for r in rows if r['scene_id']!=scene];test=[r for r in rows if r['scene_id']==scene]
        model,audit,_=fit(train,base,mode,contract['deadline_monotonic'],np.zeros(7));fits+=1
        predictions+=predict(test,model);folds.append({'held_scene':scene,'audit':audit,'coefficients':model['p_coefficients']})
    assert fits==30 and len(predictions)==1553
    out=save(PUBLIC/f'seven-sequential-objectives-predictions-{mode}-20261027.json',{'protocol_sha256':protocol['sha256'],'mode':mode,'predictions':predictions})
    result={'mode':mode,'parameter_fits':fits,'full_model':best['model'],'full_start_audits':[r['audit'] for r in full],
        'full_start_objective_spread':spread,'full_starts_agree':spread<1e-5,'folds':folds,
        'primary':nested_scores(rows,predictions),'groups':groups(rows,predictions),
        'second_attempt':nested_scores(rows,predictions,lambda r:r['trial']==2),
        'prediction_file':f'seven-sequential-objectives-predictions-{mode}-20261027.json','prediction_sha256':out['sha256']}
    save(PRIVATE/f'worker-{mode}.json',result);return result


def run():
    protocol=checked(PROTOCOL);assert protocol['modes']==MODES and protocol['limits']==LIMITS
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    assert not (PRIVATE/'contract.json').exists(),'Do not reset a CPU comparison'
    freeze=(PRIVATE/'freeze-commit.txt').read_text().strip();assert len(freeze)==40
    now=time.monotonic();contract=save(PRIVATE/'contract.json',{'protocol_sha256':protocol['sha256'],'freeze_commit':freeze,
        'started_unix':time.time(),'deadline_monotonic':now+LIMITS['seconds'],'limits':LIMITS})
    with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,MODES))
    parent=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    reference=next(c for c in parent['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    best=min(results,key=lambda r:r['primary']['mean_brier'])
    scenes=sorted(reference['primary']['scene_scores']);delta=np.array([reference['primary']['scene_scores'][s]-best['primary']['scene_scores'][s] for s in scenes])
    gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(scenes)))
    guards={key:best['groups'][key]['mean_brier']<=reference['groups'][key]['mean_brier'] for key in ['initial_C_N','late_error']}
    use=gain>se and all(guards.values()) and best['full_starts_agree']
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    result=save(PUBLIC/'seven-sequential-objectives-results-20261027.json',{'scope':'Development objective comparison, not new frozen Agent validation',
        'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],'reference':reference,'candidates':results,
        'selection':{'best_primary':best['mode'],'reference_minus_candidate_scene_gain':gain,'paired_scene_SE':se,'guards':guards,
            'selected_for_development':best['mode'] if use else 'original_task_marginal_likelihood','replace_frozen_baseline':False},
        'parameter_fits':60,'elapsed_seconds':time.monotonic()-now,'new_model_calls':0,'holdout134_calls':0,'protected_files_unchanged':protocol['protected_files']})
    closeout=save(PRIVATE/'closeout.json',{'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],
        'status':'complete','result_sha256':result['sha256'],'parameter_fits':60,'elapsed_seconds':time.monotonic()-now,'new_model_calls':0})
    save(PUBLIC/'seven-sequential-objectives-closeout-20261027.json',{k:v for k,v in closeout.items() if k!='sha256'})
    print({'sha256':result['sha256'],'selection':result['selection'],'elapsed_seconds':result['elapsed_seconds'],
        'scores':[{k:r[k] for k in ['mode','primary','groups','second_attempt']} for r in results]},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
