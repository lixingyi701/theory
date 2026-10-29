#!/usr/bin/env python3
"""One bounded stage-calibration comparison; no memory-mechanism identification."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, time, math
import numpy as np
from agent_reliability.io import digest, file_hash, read_json, write_json
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import checked
from calibrate_successor_success import features
from diagnose_success_regularization import objective
from compare_seven_reached_risk import score, groups as risk_groups
import shared_recovery_model as original

PUBLIC = Path('repro/reflexion-20261025')
PRIVATE = Path('runs/reflexion-stage-calibration-20261029')
PROTOCOL = PUBLIC/'stage-calibration-protocol-20261029.json'
SOURCE = Path('scripts/compare_stage_calibration.py')
ANALYZER = Path('scripts/analyze_stage_calibration.py')
DESIGN = Path('docs/experiments/stage-calibration-design-20261029.md')
OLD = Path('runs/reflexion-seven-parameter-development-20261027/input-events.json')
LIMITS = dict(cpu_wall_seconds=900, max_parameter_fits=30, cpu_workers=1, new_agent_calls=0)
FORMS = ['linear_count', 'one_native_indicator']


def x(state, form):
    value = np.asarray(features(state, 'seven'), float)
    if form == 'one_native_indicator':
        value[2] = float(sum(s['native'] for s in state) == 1)/3
    elif form != 'linear_count':
        raise ValueError(form)
    return value


def components(model, state, form):
    z = float(x(state, form) @ np.asarray(model['p_coefficients']))
    return 1/(1+np.exp(-np.clip(z+model['task_scale']*np.asarray(model['task_offsets']), -700, 700)))


def predict(rows, model, form):
    out = []; by_id = defaultdict(list)
    for row in rows: by_id[row['trajectory_id']].append(row)
    for tid, entries in sorted(by_id.items()):
        weights = np.asarray(model['task_prior'], float)
        for row in sorted(entries, key=lambda r:r['trial']):
            pp = components(model, row['state'], form)
            out.append(dict(trajectory_id=tid, trial=row['trial'], probability=float(weights@pp)))
            if not row['success']:
                weights *= 1-pp; weights /= weights.sum()
    return out


def inputs():
    parent = checked(PUBLIC/'seven-parameter-development-protocol-20261027.json')
    rows = read_json(OLD)
    assert digest(rows) == parent['input_rows_sha256']
    assert file_hash(OLD) == parent['private_input_file_sha256']
    old = checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    reference = next(c for c in old['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    predictions = checked(PUBLIC/reference['prediction_file'])
    assert predictions['sha256'] == reference['prediction_sha256']
    content = checked(PUBLIC/'content-intervention-protocol-20261028.json')
    observations = []
    for j in range(1,4):
        observations += checked(PUBLIC/f'content-intervention-observed-{j:02d}-20261028.json')['rows']
    assert len(rows)==1553 and len(observations)==276
    return rows, reference, predictions['predictions'], content, observations


def structure(rows, content, observations):
    counts = Counter((r['trial'],sum(s['native'] for s in r['state'])) for r in rows)
    assert counts == {(1,0):760,(2,1):441,(3,2):188,(4,3):164}
    lookup = {(r['trajectory_id'],r['trial']):r for r in rows}
    graph = []
    for c in content['cases']:
        tid, trial = c['parent_attempt_id']; assert trial==2
        parent = lookup[tid,1]; second = lookup[tid,2]
        assert not parent['success'] and parent['task_id']==c['task_id']
        assert parent['result_file_sha256']==c['parent_trial1_result_file_sha256']
        assert second['state']==c['state']
        assert np.array_equal(x(parent['state'],FORMS[0]),x(parent['state'],FORMS[1]))
        assert np.array_equal(x(second['state'],FORMS[0]),x(second['state'],FORMS[1]))
        children = [r['id'] for r in observations if r['task_id']==c['task_id']]
        assert len(children)==6 and len(set(children))==6
        graph.append(dict(task_id=c['task_id'],scene_id=c['scene_id'],
            parent_failure_id=[tid,1],original_second_id=[tid,2],children=sorted(children)))
    assert len(graph)==46 and len({tuple(g['parent_failure_id']) for g in graph})==46
    eligible = [r for r in rows if r['trial']==1 and not r['success'] and r['outcome']=='E']
    selected=[]
    for task in sorted({r['task_id'] for r in eligible}):
        r=min((a for a in eligible if a['task_id']==task),key=lambda a:digest([a['trajectory_id'],a['trial']]))
        selected.append({k:r[k] for k in ['source','trajectory_id','task_id','scene_id','trial','result_file_sha256']})
    return dict(stage_native_table=[dict(trial=k[0],native_count=k[1],records=v) for k,v in sorted(counts.items())],
        violations=0,unique_parent_failures=46,child_branches=276,parent_graph=graph,
        new_parent_failures_added_to_fit=0,content_rows_added_to_fit=0,
        fixed_coefficient_content_prediction_invariance=True,
        mechanism_parent_inventory=dict(rule='First-attempt failure, new reflection E, one identity-only selected parent per task; no later-success eligibility',
            candidate_parent_records=len(eligible),tasks=len(selected),scenes=len({r['scene_id'] for r in selected}),selected=selected,
            status='Eligibility inventory only; full native texts/edits and collection precision not yet prepared'))


def prepare():
    assert not PROTOCOL.exists() and not (PRIVATE/'contract.json').exists(), 'Preserve stage contract'
    rows, reference, _, content, obs = inputs(); PRIVATE.mkdir(parents=True,exist_ok=True)
    audit=structure(rows,content,obs)
    protected=content['protected_files'].copy()
    names=[str(SOURCE),str(ANALYZER),str(DESIGN),str(OLD),
        str(PUBLIC/'seven-parameter-development-results-20261027.json'),
        str(PUBLIC/reference['prediction_file']),str(PUBLIC/'content-intervention-results-20261028.json'),
        str(PUBLIC/'successor-validation-observed-20261026.json')]
    for name in names: protected[name]=file_hash(Path(name))
    p=save(PROTOCOL,dict(scope='One development-only stage calibration test; not identification of native-memory effect',
        input_rows_sha256=digest(rows),input_file_sha256=file_hash(OLD),reference_config=reference['config'],
        reference_prediction_sha256=reference['prediction_sha256'],forms=FORMS,limits=LIMITS,
        attempts=1553,tasks=163,scenes=27,trajectories=760,content_rows=276,
        optimization='Same seven-column task-marginal likelihood, ridge=.01, task scale=1.5; fixed G; candidate27 whole-scene fits+3 full-data starts. Cached linear reference reused, no refit.',
        primary='Existing fixed-start H4 reached Brier risk; equal scenes/tasks/preplanned branches, no realized-length divisor',
        guards=['initial_C_N conditional Brier must not exceed cached linear','late_error conditional Brier must not exceed cached linear'],
        decision='Retain as stage-calibration candidate only if mean primary gain exceeds paired-scene SE and both guards pass. Never automatically promote a causal mechanism or frozen baseline. If failed, close this function branch; do not search another count shape in this stage.',
        content_transfer='Development-only46-parent conditional transfer; each parent failure contributes once to own posterior, no fit on276children. Compare counts, probabilities and two independent seeds only within each condition.',
        recovery_curves='Descriptive reuse of existing25-scene recovery panel, all3arms and budgets1-4, paired scene t95 df24; no new confirmatory claim or refit on those curves.',
        structure=audit,protected_files=protected,new_model_calls=0,holdout134_calls=0))
    print(dict(protocol_sha256=p['sha256'],stage_table=audit['stage_native_table'],parent_inventory={k:v for k,v in audit['mechanism_parent_inventory'].items() if k!='selected'}),flush=True)


def fit(rows, base, deadline, initial):
    xx=np.asarray([x(r['state'],FORMS[1]) for r in rows]); yy=np.asarray([r['success'] for r in rows],float)
    task=sorted({r['task_id'] for r in rows}); group=[np.asarray([i for i,r in enumerate(rows) if r['task_id']==t]) for t in task]
    assert np.linalg.matrix_rank(xx)==7
    def fun(beta):
        if time.monotonic()>deadline: raise TimeoutError('Stage CPU allowance exhausted')
        v,g=objective(beta,xx,yy,group,1.5,.01)
        assert np.isfinite(v) and np.all(np.isfinite(g)); return v,g
    beta,audit=original.bfgs(fun,np.asarray(initial,float))
    model={**base,'p_coefficients':beta.tolist(),'p_count_mapping':FORMS[1],
        'implementation':'scripts/compare_stage_calibration.py:components','status':'stage_calibration_only'}
    return model,dict(audit,objective=fun(beta)[0],rank=7),fun


def transfer(content, obs, models, form):
    values={}; histogram=np.zeros(3)
    for c in content['cases']:
        model=models[c['scene_id']]; b=np.asarray(model['p_coefficients']); offset=model['task_scale']*np.asarray(model['task_offsets'])
        first=np.asarray(c['seven_history'][0],float); later=np.asarray(c['seven_history'][1],float)
        assert first[2]==0 and later[2]==1/3
        p0=1/(1+np.exp(-(first@b+offset))); pp=1/(1+np.exp(-(later@b+offset)))
        posterior=np.asarray(model['task_prior'])*(1-p0); posterior/=posterior.sum()
        values[c['task_id']]=float(posterior@pp)
        histogram += [float(posterior@(math.comb(2,k)*pp**k*(1-pp)**(2-k))) for k in range(3)]
    out={}
    for cond in ['original','neutralized','paraphrase']:
        rr=[r for r in obs if r['condition']==cond]; observed=sum(r['success'] for r in rr)
        mu=observed/len(rr); count=Counter(sum(r['success'] for r in rr if r['task_id']==t) for t in values)
        out[cond]=dict(observed_successes=observed,expected_successes=sum(values[r['task_id']] for r in rr),
            mean_brier=float(np.mean([(values[r['task_id']]-float(r['success']))**2 for r in rr])),
            observed_task_pairs=[count[k] for k in range(3)],model_expected_task_pairs=histogram.tolist(),
            equal_probability_mean_matched_expected_pairs=[46*math.comb(2,k)*mu**k*(1-mu)**(2-k) for k in range(3)])
    return dict(form=form,by_condition=out,probability_by_task=values,
        interpretation='Mean-matched independent reference is descriptive only, not a new fitted mixture or a test of residual dependence.')


def recovery():
    from scipy.stats import t
    data=checked(PUBLIC/'successor-validation-observed-20261026.json')['observed']
    rows=[r for r in data if r['panel']=='recovery']; assert len(rows)==150
    scenes=sorted({r['scene_id'] for r in rows}); assert len(scenes)==25
    arms=['repeated_error','correct_latest','delete_latest']; rates={}; contrasts=[]
    for arm in arms:
        by_scene={s:[np.mean([r['success_curve'][j] for r in rows if r['scene_id']==s and r['condition']==arm]) for j in range(4)] for s in scenes}
        rates[arm]=dict(success_counts=[sum(r['success_curve'][j] for r in rows if r['condition']==arm) for j in range(4)],
            scene_equal_curve=np.mean(list(by_scene.values()),axis=0).tolist(),scene_curves=by_scene)
    critical=float(t.ppf(.975,24))
    for left,right in [('correct_latest','repeated_error'),('delete_latest','repeated_error'),('correct_latest','delete_latest')]:
        for j in range(4):
            dd=np.array([rates[left]['scene_curves'][s][j]-rates[right]['scene_curves'][s][j] for s in scenes]); se=float(dd.std(ddof=1)/5); mean=float(dd.mean())
            contrasts.append(dict(left=left,right=right,budget=j+1,difference=mean,pointwise_t95=[mean-critical*se,mean+critical*se]))
    return dict(scope='Existing controlled-memory recovery panel, not native-error parent experiment; post-collection descriptive curves, no multiplicity-adjusted discoveries',
        tasks=25,scenes=25,trajectories=150,rates=rates,contrasts=contrasts)


def run():
    p=checked(PROTOCOL)
    for name,h in p['protected_files'].items(): assert file_hash(Path(name))==h,name
    assert not (PRIVATE/'contract.json').exists(),'Do not rerun completed/uncertain stage'
    rows,ref,linear_predictions,content,obs=inputs(); assert structure(rows,content,obs)==p['structure']
    freeze=(PRIVATE/'freeze-commit.txt').read_text().strip(); assert len(freeze)==40
    now=time.monotonic(); contract=save(PRIVATE/'contract.json',dict(protocol_sha256=p['sha256'],freeze_commit=freeze,limits=LIMITS,status='running',started_unix=time.time()))
    deadline=now+LIMITS['cpu_wall_seconds']; base=ref['full_model']; full=[]; fits=0
    for a in [0.,.2,-.2]:
        model,audit,fun=fit(rows,base,deadline,[a]*7); full.append(dict(model=model,audit=audit)); fits+=1
    assert max(a['audit']['objective'] for a in full)-min(a['audit']['objective'] for a in full)<1e-5
    best=min(full,key=lambda a:a['audit']['objective']); beta=np.asarray(best['model']['p_coefficients']); _,gg=fun(beta)
    numeric=[]
    for j in range(7):
        d=np.eye(7)[j]*1e-5; numeric.append((fun(beta+d)[0]-fun(beta-d)[0])/(2e-5))
    error=float(np.max(np.abs(gg-numeric))); assert error<1e-6
    folds=[]; predictions=[]; candidate_models={}; reference_models={}
    for fold in ref['folds']:
        scene=fold['scene_id']; train=[r for r in rows if r['scene_id']!=scene]; test=[r for r in rows if r['scene_id']==scene]
        model,audit,_=fit(train,base,deadline,np.zeros(7)); fits+=1
        candidate_models[scene]=model; reference_models[scene]={**base,'p_coefficients':fold['coefficients']}
        predictions+=predict(test,model,FORMS[1]); folds.append(dict(scene_id=scene,coefficients=model['p_coefficients'],audit=audit))
        cached=predict(test,reference_models[scene],FORMS[0]); lookup={(r['trajectory_id'],r['trial']):r['probability'] for r in linear_predictions}
        assert max(abs(r['probability']-lookup[r['trajectory_id'],r['trial']]) for r in cached)<1e-12
    assert fits==30 and len(predictions)==1553
    scores={}; output_parts=[]
    for form,pp in [(FORMS[0],linear_predictions),(FORMS[1],predictions)]:
        scores[form]=dict(primary=score(rows,pp),groups=risk_groups(rows,pp))
    ll=scores[FORMS[0]]; cc=scores[FORMS[1]]
    delta=np.array([ll['primary']['scene_contributions'][s]-cc['primary']['scene_contributions'][s] for s in sorted(reference_models)])
    gain=float(delta.mean()); se=float(delta.std(ddof=1)/np.sqrt(27))
    guards={k:cc['groups'][k]['weighted_conditional_brier']<=ll['groups'][k]['weighted_conditional_brier'] for k in ['initial_C_N','late_error']}
    keep=gain>se and all(guards.values())
    for j in range(2):
        q=save(PUBLIC/f'stage-calibration-predictions-{j+1:02d}-20261029.json',dict(protocol_sha256=p['sha256'],form=FORMS[1],predictions=predictions[j*800:(j+1)*800]))
        output_parts.append(dict(file=f'stage-calibration-predictions-{j+1:02d}-20261029.json',sha256=q['sha256']))
    result=save(PUBLIC/'stage-calibration-results-20261029.json',dict(protocol_sha256=p['sha256'],scope='Reused development records, stage calibration only',
        scores=scores,selection=dict(primary_gain=gain,paired_scene_se=se,guards=guards,retain_stage_candidate=keep,
            frozen_baseline_promoted=False,mechanism_identified=False),
        candidate_full_model=best['model'],full_start_audits=[a['audit'] for a in full],finite_difference_max_error=error,
        folds=folds,prediction_parts=output_parts,
        content_transfer={FORMS[0]:transfer(content,obs,reference_models,FORMS[0]),FORMS[1]:transfer(content,obs,candidate_models,FORMS[1])},
        recovery_curves=recovery(),parameter_fits=fits,elapsed_seconds=time.monotonic()-now,new_model_calls=0,holdout134_calls=0))
    for name,h in p['protected_files'].items(): assert file_hash(Path(name))==h,name
    save(PRIVATE/'closeout.json',dict(status='complete',protocol_sha256=p['sha256'],contract_sha256=contract['sha256'],result_sha256=result['sha256'],fits=fits,elapsed_seconds=time.monotonic()-now,new_model_calls=0,holdout134_calls=0))
    print(dict(result_sha256=result['sha256'],selection=result['selection'],risk={k:v['primary']['risk_per_planned_horizon'] for k,v in scores.items()},seconds=result['elapsed_seconds']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
