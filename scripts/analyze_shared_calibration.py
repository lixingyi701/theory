#!/usr/bin/env python3
"""Join full-text labels by hash, fit frozen shared candidates and retain scene scores."""
from collections import Counter
from pathlib import Path
import argparse
import numpy as np
from agent_reliability.io import digest,file_hash,read_json,write_json
from prepare_shared_calibration import PROTOCOL,STATES,save
from run_shared_calibration import ROOT,verify_protocol
from run_multiround_diagnostic import verify_attempt
import shared_recovery_model as model

PUBLIC = PROTOCOL.parent
LABELS = PUBLIC/'shared-calibration-semantic-labels.json'


def events(protocol,root,labels):
    if labels['sha256']!=digest({k:v for k,v in labels.items() if k!='sha256'}): raise ValueError('Labels integrity')
    if labels['protocol_sha256']!=protocol['sha256']: raise ValueError('Labels protocol mismatch')
    label_map={e['reflection_sha256']:e['category'] for e in labels['entries']}
    if len(label_map)!=len(labels['entries']) or any(v not in model.OUTCOMES[1:] for v in label_map.values()):
        raise ValueError('Duplicate or invalid labels')
    summary=read_json(root/'summary.json'); rows=[];seen=[]
    if summary['status']!='complete' or summary['protocol_sha256']!=protocol['sha256']: raise ValueError('Incomplete/mismatched summary')
    cases={c['task_id']:c for c in protocol['cases']}
    for trajectory in protocol['trajectories']:
        identity=trajectory['id']; records=sorted([r for r in summary['attempts_rows'] if r['trajectory_id']==identity],key=lambda r:r['trial'])
        if not 1<=len(records)<=2: raise ValueError('Invalid attempt count')
        memory=[protocol['entries'][k] for k in trajectory['initial_labels']]
        state=[{'type':k,'native':False} for k in trajectory['initial_labels']]
        if [r['trial'] for r in records]!=list(range(1,len(records)+1)): raise ValueError('Nonconsecutive trials')
        for row in records:
            result_path=Path(row['result_path']);result=read_json(result_path)
            if file_hash(result_path)!=row['result_file_sha256']: raise ValueError('Raw result changed')
            verify_attempt(result,result_path.parent,cases[trajectory['task_id']],trajectory,memory,row['trial'])
            outcome='S';reflection=None
            if not result['success']:
                reflection=digest(result['memory_after'][-1]);outcome=label_map[reflection];seen.append(reflection)
            rows.append({'trajectory_id':identity,'task_id':row['task_id'],'scene_id':result['scene_id'],
                'condition':trajectory['condition'],'seed':trajectory['seed'],'trial':row['trial'],
                'state':list(state[-3:]),'outcome':outcome,'success':result['success'],
                'visible_memory_sha256':digest(memory[-3:]),'new_reflection_sha256':reflection,
                'result_file_sha256':row['result_file_sha256'],'decisions':result['decisions']})
            if result['success']:
                if row!=records[-1]: raise ValueError('Trajectory continued after success')
            else: state.append({'type':outcome,'native':True})
            memory=list(result['memory_after'])
        if len(records)==1 and not records[0]['success']: raise ValueError('Failed trajectory stopped early')
    if len(rows)!=len(summary['attempts_rows']) or len({(r['trajectory_id'],r['trial']) for r in rows})!=len(rows):
        raise ValueError('Raw matrix mismatch')
    if set(seen)!=set(label_map): raise ValueError('Labels do not match generated reflections')
    return rows


def assemble(p_model,g_model):
    return {**p_model,'success_family':p_model['family'],'write_family':g_model['family'],
        'write_coefficients_reference_A':g_model['write_coefficients_reference_A']}


def held_scene_scores(rows,comparison):
    selected=comparison['selected_model'];results=[];forecasts=[]
    for scene in sorted({r['scene_id'] for r in rows}):
        train=[r for r in rows if r['scene_id']!=scene];test=[r for r in rows if r['scene_id']==scene]
        p_model=model.fit(train,selected['success_family'],selected['scale'])
        g_model=model.fit(train,selected['write_family'],0.)
        candidates={'selected':assemble(p_model,g_model),'constant':model.fit(train,'constant',0.)}
        for name,fitted in candidates.items():
            trajectories=[]
            for identity in sorted({r['trajectory_id'] for r in test}):
                sequence=sorted([r for r in test if r['trajectory_id']==identity],key=lambda r:r['trial'])
                prediction=model.predict_sequence(fitted,sequence)
                values=[model.scores(p,r['outcome']) for p,r in zip(prediction,sequence)]
                initial=sequence[0]['state'];two=model.budget_probability(fitted,initial,2)
                two_success=any(r['success'] for r in sequence)
                item={'scene_id':scene,'trajectory_id':identity,'model':name,
                    'condition':sequence[0]['condition'],'first_success_probability':prediction[0]['S'],
                    'two_budget_probability':two,'observed_success':two_success,
                    'two_budget_brier':float((two-two_success)**2),
                    **{k:float(np.mean([v[k] for v in values])) for k in values[0]}}
                trajectories.append(item);forecasts.append(item)
            results.append({'scene_id':scene,'model':name,**{k:float(np.mean([r[k] for r in trajectories]))
                           for k in ['joint_brier','success_brier','log_loss','two_budget_brier']}})
    return {'scene_scores':results,'trajectory_forecasts':forecasts,
        'scope':'Held-scene development scores after candidate selection; independent reserved tasks remain uncollected'}


def bootstrap(values,rng):
    values=np.asarray(values,float)
    selected=rng.integers(0,len(values),size=(20000,len(values)))
    return {'mean':float(values.mean()),'scene_bootstrap_95':np.quantile(values[selected].mean(axis=1),[.025,.975]).tolist(),
            'scene_sd':float(np.std(values,ddof=1)),'scene_values':values.tolist()}


def descriptive(rows,protocol):
    scenes=sorted({r['scene_id'] for r in rows});rng=np.random.default_rng(2026102502);table={};effects=[]
    for condition in STATES:
        trajectories=[]
        for identity in sorted({r['trajectory_id'] for r in rows if r['condition']==condition}):
            sequence=sorted([r for r in rows if r['trajectory_id']==identity],key=lambda r:r['trial'])
            trajectories.append({'trajectory_id':identity,'task_id':sequence[0]['task_id'],'scene_id':sequence[0]['scene_id'],
                'first_success':sequence[0]['success'],'two_success':any(r['success'] for r in sequence)})
        table[condition]={'trajectories':len(trajectories),'first_successes':sum(t['first_success'] for t in trajectories),
            'two_successes':sum(t['two_success'] for t in trajectories),'tasks':trajectories,
            'outcomes_by_trial':{str(t):dict(Counter(r['outcome'] for r in rows if r['condition']==condition and r['trial']==t)) for t in [1,2]}}
    for left,right in [('correct_middle','error_middle'),('correct_latest','delete_latest'),('correct_latest','correct_oldest')]:
        for metric in ['first_success','two_success']:
            values=[]
            for scene in scenes:
                a=[t[metric] for t in table[left]['tasks'] if t['scene_id']==scene]
                b=[t[metric] for t in table[right]['tasks'] if t['scene_id']==scene]
                values.append(float(np.mean(a)-np.mean(b)))
            effects.append({'left':left,'right':right,'metric':metric,**bootstrap(values,rng)})
    return {'condition_table':table,'descriptive_contrasts':effects,'independent_scenes':len(scenes),
            'interval_scope':'Descriptive scene-cluster intervals, no multiplicity-adjusted confirmatory claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['events','fit']);a=p.parse_args();protocol=verify_protocol()
    rows=events(protocol,ROOT,read_json(LABELS))
    event_file=PUBLIC/'shared-calibration-events.json'
    saved=save(event_file,{'protocol_sha256':protocol['sha256'],'labels_file_sha256':file_hash(LABELS),
        'raw_summary_file_sha256':file_hash(ROOT/'summary.json'),'rows':rows,'attempts':len(rows),
        'outcomes':dict(Counter(r['outcome'] for r in rows)),'holdout_model_calls':0,'reserved_validation_model_calls':0})
    if a.action=='events': print({'events_sha256':saved['sha256'],'attempts':len(rows)});return
    comparison=model.compare(rows);held=held_scene_scores(rows,comparison)
    predictions=[]
    for condition,labels in STATES.items():
        state=[{'type':k,'native':False} for k in labels]
        predictions.append({'condition':condition,'state':state,'success_curve':[
            model.budget_probability(comparison['selected_model'],state,b) for b in [1,2]]})
    rng=np.random.default_rng(2026102502);improvements=[]
    for metric in ['joint_brier','success_brier','log_loss','two_budget_brier']:
        values=[]
        for scene in sorted({r['scene_id'] for r in rows}):
            selected=next(r[metric] for r in held['scene_scores'] if r['scene_id']==scene and r['model']=='selected')
            baseline=next(r[metric] for r in held['scene_scores'] if r['scene_id']==scene and r['model']=='constant')
            values.append(baseline-selected)
        improvements.append({'metric':metric,**bootstrap(values,rng)})
    support={}
    for family in model.FAMILIES:
        matrix=np.asarray([model.features(r['state'],family) for r in rows]);support[family]={
            'features':matrix.shape[1],'rank':int(np.linalg.matrix_rank(matrix)),
            'all_zero_columns':[int(i) for i in np.flatnonzero(np.all(matrix==0,axis=0))]}
    parameters=save(PUBLIC/'shared-calibration-parameters.json',{'protocol_sha256':protocol['sha256'],
        'events_sha256':saved['sha256'],'analysis_source_sha256':file_hash(Path(__file__)),
        'model_source_sha256':file_hash(Path('scripts/shared_recovery_model.py')),
        'comparison':comparison,'held_scene_evaluation':held,'prediction_improvements':improvements,
        'frozen_initial_forecasts':predictions,'design_support':support,
        'holdout_model_calls':0,'reserved_validation_model_calls':0})
    report=save(PUBLIC/'shared-calibration-results.json',{'protocol_sha256':protocol['sha256'],
        'events_sha256':saved['sha256'],'parameters_sha256':parameters['sha256'],
        'attempts':len(rows),'new_reflections':sum(r['outcome']!='S' for r in rows),
        'selected_success':comparison['selected_success'],'selected_write':comparison['selected_write'],
        'prediction_improvements':improvements,'design_support':support,**descriptive(rows,protocol),
        'holdout_model_calls':0,'reserved_validation_model_calls':0})
    print({'parameters_sha256':parameters['sha256'],'results_sha256':report['sha256'],
           'selected_success':comparison['selected_success'],'selected_write':comparison['selected_write']})


if __name__=='__main__': main()
