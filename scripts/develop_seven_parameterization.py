#!/usr/bin/env python3
"""Bounded seven-feature parameter audit and scene-held-out CPU development."""
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import argparse,time
import numpy as np
from agent_reliability.io import digest,file_hash,read_json,write_json
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import checked
from calibrate_successor_success import features,success_components
from diagnose_success_regularization import objective
import shared_recovery_model as original

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-seven-parameter-development-20261027')
PROTOCOL=PUBLIC/'seven-parameter-development-protocol-20261027.json'
DESIGN=Path('docs/experiments/seven-parameter-development-design-20261027.md')
CONFIGS=[{'name':f'ridge{ridge:g}_scale{scale:g}','ridge':ridge,'scale':scale}
    for ridge in [.01,.1,1.] for scale in [0.,1.5]]
REFERENCE='ridge0.01_scale1.5'
LIMITS={'seconds':900,'max_parameter_fits':180,'cpu_workers':2,'new_agent_calls':0}


def input_rows():
    old_path=Path('runs/reflexion-successor-development-calibration-20261026/input-events.json')
    old_manifest=checked(PUBLIC/'successor-development-inputs-20261026.json')
    old=read_json(old_path)
    assert digest(old)==old_manifest['input_rows_sha256'] and file_hash(old_path)==old_manifest['private_input_file_sha256']
    diagnosis=checked(PUBLIC/'successor-complete-diagnostic-20261027.json')
    old_ids={r['task_id'] for r in old};new_ids={r['task_id'] for r in diagnosis['attempt_rows']}
    assert not old_ids&new_ids
    write={(r['trajectory_id'],r['trial']):r['new_type'] for r in diagnosis['write_rows']}
    rows=[dict(r) for r in old]
    for r in diagnosis['attempt_rows']:
        row={k:r[k] for k in ['trajectory_id','task_id','scene_id','condition','trial','state','success','result_file_sha256']}
        row.update(source='completed_successor_primary',outcome='S' if r['success'] else write.get((r['trajectory_id'],r['trial'])))
        rows.append(row)
    assert len(rows)==1553 and len({(r['trajectory_id'],r['trial']) for r in rows})==1553
    assert len({r['task_id'] for r in rows})==163 and len({r['scene_id'] for r in rows})==27
    assert len({r['trajectory_id'] for r in rows})==760 and sum(r['success'] for r in rows)==420
    assert sum(not r['success'] and r['outcome'] is not None for r in rows)==987
    assert sum(r['outcome'] is None for r in rows)==146
    return rows,old_manifest,diagnosis


def prepare():
    assert not PROTOCOL.exists(),'Preserve prepared protocol'
    rows,old,diagnosis=input_rows();PRIVATE.mkdir(parents=True,exist_ok=True)
    write_json(PRIVATE/'input-events.json',rows)
    params=checked(PUBLIC/'successor-development-parameters-20261026.json')
    parent=checked(PUBLIC/'successor-validation-protocol-20261026.json')
    protected={name:file_hash(Path(name)) for name in parent['protected_files']}
    for name in ['scripts/develop_seven_parameterization.py',str(DESIGN),'scripts/diagnose_success_regularization.py',
        'repro/reflexion-20261025/successor-complete-diagnostic-metrics-20261027.json',
        'repro/reflexion-20261025/successor-late-semantic-labels-20261027.json',
        'repro/reflexion-20261025/successor-remaining-semantic-labels-20261027.json']:
        protected[name]=file_hash(Path(name))
    result=save(PROTOCOL,{'scope':'Development reuse of original controlled records and completed100-task primary validation; seven features only',
        'input_rows_sha256':digest(rows),'private_input_file_sha256':file_hash(PRIVATE/'input-events.json'),
        'old_input_manifest_sha256':old['sha256'],'diagnosis_sha256':diagnosis['sha256'],
        'parent_parameters_sha256':params['sha256'],'base_model':params['models']['seven']['model'],
        'attempts':1553,'tasks':163,'scenes':27,'trajectories':760,'successes':420,
        'known_failed_write_types':987,'unknown_terminal_write_types_success_side_only':146,
        'configs':CONFIGS,'reference_config':REFERENCE,'limits':LIMITS,
        'features':'Unchanged calibrate_successor_success.features(state,seven); original conditional G fixed',
        'folds':[{'held_scene':s} for s in sorted({r['scene_id'] for r in rows})],
        'primary':'Sequential single-attempt Brier: attempts mean within trajectory, trajectories mean within task, tasks mean within scene,27 scenes equally',
        'late_guard':'Mean late(native>=2),error-present(E/B) Brier with the same nesting over available scenes must not exceed reference',
        'selection':'Best mean primary config is selected only if reference-minus-best exceeds paired-scene SE and late_guard holds; otherwise retain reference hyperparameters. No post-hoc substitute if guard fails.',
        'prior_update':'For held trajectories start from fixed prior and update only with their own prior failures; no other branches or future outcomes',
        'optimization':'Fixed seven-column likelihood, symmetric intercept half-count, ridge in grid;27 fold fits plus3 full-data starts per config,6configs=180 fits',
        'protected_files':protected,'new_model_calls':0,'holdout134_calls':0})
    print({k:result[k] for k in ['sha256','attempts','tasks','scenes','trajectories','configs','limits']})


def arrays(rows):
    x=np.asarray([features(r['state'],'seven') for r in rows]);y=np.asarray([r['success'] for r in rows],float)
    tasks=sorted({r['task_id'] for r in rows});groups=[np.asarray([i for i,r in enumerate(rows) if r['task_id']==task]) for task in tasks]
    return x,y,groups


def fit(rows,base,config,deadline,initial):
    x,y,groups=arrays(rows)
    def function(beta):
        if time.monotonic()>deadline:raise TimeoutError('Whole CPU-development allowance exhausted')
        value,gradient=objective(beta,x,y,groups,config['scale'],config['ridge'])
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):raise ValueError('Nonfinite objective')
        return value,gradient
    beta,audit=original.bfgs(function,np.asarray(initial,float))
    model={**base,'p_coefficients':beta.tolist(),'p_ridge':config['ridge'],'task_scale':config['scale']}
    return model,{**audit,'objective':function(beta)[0],'rank':int(np.linalg.matrix_rank(x))},function


def predict(rows,model):
    output=[]
    for tid in sorted({r['trajectory_id'] for r in rows}):
        weights=np.asarray(model['task_prior'],float)
        for row in sorted([r for r in rows if r['trajectory_id']==tid],key=lambda r:r['trial']):
            pp=success_components(model,row['state']);prob=float(weights@pp)
            output.append({'trajectory_id':tid,'trial':row['trial'],'probability':prob})
            if not row['success']:weights*=1-pp;weights/=weights.sum()
    return output


def nested_scores(rows,predictions,predicate=lambda r:True):
    probs={(r['trajectory_id'],r['trial']):r['probability'] for r in predictions}
    trajectories=defaultdict(list);metadata={}
    for r in rows:
        if not predicate(r):continue
        tid=r['trajectory_id'];metadata[tid]=(r['task_id'],r['scene_id'])
        trajectories[tid].append((probs[tid,r['trial']]-float(r['success']))**2)
    tasks=defaultdict(list);scenes=defaultdict(list)
    for tid,losses in trajectories.items():
        task,scene=metadata[tid];tasks[task,scene].append(float(np.mean(losses)))
    for (task,scene),values in tasks.items():scenes[scene].append(float(np.mean(values)))
    scene_scores={s:float(np.mean(values)) for s,values in sorted(scenes.items())}
    return {'mean_brier':float(np.mean(list(scene_scores.values()))),'scenes':len(scene_scores),
        'attempts':sum(len(x) for x in trajectories.values()),'scene_scores':scene_scores}


def groups(rows,predictions):
    probs={(r['trajectory_id'],r['trial']):r['probability'] for r in predictions};out={}
    for name,predicate in [('initial_C_N',lambda r:r['trial']==1 and r['condition'] in ['correct_middle','neutral_full']),
        ('initial_error',lambda r:r['trial']==1 and any(s['type'] in ['E','B'] for s in r['state'])),
        ('late_error',lambda r:sum(s['native'] for s in r['state'])>=2 and any(s['type'] in ['E','B'] for s in r['state'])),
        ('late_no_error',lambda r:sum(s['native'] for s in r['state'])>=2 and not any(s['type'] in ['E','B'] for s in r['state']))]:
        subset=[r for r in rows if predicate(r)]
        out[name]={**nested_scores(rows,predictions,predicate),'observed_successes':sum(r['success'] for r in subset),
            'expected_successes':float(sum(probs[r['trajectory_id'],r['trial']] for r in subset))}
    return out


def worker(config):
    protocol=checked(PROTOCOL);contract=checked(PRIVATE/'contract.json');rows=read_json(PRIVATE/'input-events.json')
    assert digest(rows)==protocol['input_rows_sha256'];base=protocol['base_model'];started=time.monotonic()
    full=[];fits=0
    for value in [0.,.2,-.2]:
        model,audit,function=fit(rows,base,config,contract['deadline_monotonic'],[value]*7);fits+=1
        full.append({'model':model,'audit':audit})
    objectives=[r['audit']['objective'] for r in full]
    if max(objectives)-min(objectives)>1e-5:raise RuntimeError('Full-data starts disagree')
    model=min(full,key=lambda r:r['audit']['objective'])['model'];x,y,task_groups=arrays(rows)
    fun=lambda b:objective(b,x,y,task_groups,config['scale'],config['ridge'])
    beta=np.asarray(model['p_coefficients']);value,grad=fun(beta);numeric=[];columns=[]
    for j in range(7):
        delta=np.eye(7)[j]*1e-5;numeric.append((fun(beta+delta)[0]-fun(beta-delta)[0])/(2e-5))
        columns.append((fun(beta+delta)[1]-fun(beta-delta)[1])/(2e-5))
    err=float(np.max(np.abs(grad-np.asarray(numeric))));assert err<1e-6
    h=np.asarray(columns).T;h=(h+h.T)/2;eig=np.linalg.eigvalsh(h)
    assert eig.min()>0
    inverse=np.linalg.inv(h);corr=inverse/np.sqrt(np.diag(inverse)[:,None]*np.diag(inverse)[None,:])
    predictions=[];folds=[]
    for fold in protocol['folds']:
        scene=fold['held_scene'];train=[r for r in rows if r['scene_id']!=scene];test=[r for r in rows if r['scene_id']==scene]
        fitted,audit,_=fit(train,base,config,contract['deadline_monotonic'],np.zeros(7));fits+=1
        predictions+=predict(test,fitted);folds.append({'scene_id':scene,'training_audit':audit,'coefficients':fitted['p_coefficients']})
    assert fits==30 and len(predictions)==1553
    path=PUBLIC/f'seven-parameter-development-predictions-{config["name"]}-20261027.json'
    output=save(path,{'protocol_sha256':protocol['sha256'],'config':config,'predictions':predictions})
    result={'config':config,'parameter_fits':fits,'full_model':model,'full_start_audits':[r['audit'] for r in full],
        'finite_difference_max_error':err,'hessian_eigenvalues':eig.tolist(),'hessian_condition_number':float(eig.max()/eig.min()),
        'coefficient_local_correlation':corr.tolist(),'folds':folds,
        'primary':nested_scores(rows,predictions),'groups':groups(rows,predictions),
        'prediction_file':path.name,'prediction_sha256':output['sha256'],'elapsed_seconds':time.monotonic()-started}
    save(PRIVATE/f'worker-{config["name"]}.json',result)
    return result


def run():
    protocol=checked(PROTOCOL)
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    assert protocol['configs']==CONFIGS and protocol['limits']==LIMITS
    assert not (PRIVATE/'contract.json').exists(),'Do not reset a CPU campaign'
    freeze=(PRIVATE/'freeze-commit.txt').read_text().strip();assert len(freeze)==40
    now=time.monotonic();contract=save(PRIVATE/'contract.json',{'protocol_sha256':protocol['sha256'],
        'freeze_commit':freeze,'started_unix':time.time(),'deadline_monotonic':now+LIMITS['seconds'],
        'max_fits':180,'cpu_workers':2,'new_agent_calls':0})
    with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,CONFIGS))
    assert sum(r['parameter_fits'] for r in results)==180
    reference=next(r for r in results if r['config']['name']==REFERENCE)
    best=min(results,key=lambda r:r['primary']['mean_brier'])
    scene=sorted(reference['primary']['scene_scores']);delta=np.array([reference['primary']['scene_scores'][s]-best['primary']['scene_scores'][s] for s in scene])
    gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    late_ok=best['groups']['late_error']['mean_brier']<=reference['groups']['late_error']['mean_brier']
    keep=best['config']['name']!=REFERENCE and gain>se and late_ok
    selected=best if keep else reference
    for name,h in protocol['protected_files'].items():assert file_hash(Path(name))==h,name
    result=save(PUBLIC/'seven-parameter-development-results-20261027.json',{
        'scope':'Seven-feature CPU development; all reused validation rows are development for this revision',
        'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],'configs':results,
        'selection':{'reference':REFERENCE,'best_primary':best['config']['name'],'paired_scene_gain':gain,'paired_scene_se':se,
            'late_guard_passed':late_ok,'selected':selected['config']['name'],'hyperparameters_changed':keep},
        'selected_model':selected['full_model'],'parameter_fits':180,'elapsed_seconds':time.monotonic()-now,
        'new_model_calls':0,'holdout134_calls':0,'frozen_files_unchanged':protocol['protected_files']})
    save(PRIVATE/'closeout.json',{'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],
        'status':'complete','result_sha256':result['sha256'],'fits':180,'elapsed_seconds':time.monotonic()-now,'new_model_calls':0})
    print({'sha256':result['sha256'],'selection':result['selection'],'seconds':result['elapsed_seconds'],
        'scores':[{k:r[k] for k in ['config','primary','groups']} for r in results]},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
