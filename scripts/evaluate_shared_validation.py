#!/usr/bin/env python3
"""Frozen budget-success scoring; evaluator uses native outcomes, no text labels."""
from collections import Counter
import numpy as np
from agent_reliability.io import digest,file_hash,read_json
from prepare_shared_calibration import save,STATES
from prepare_shared_validation import PROTOCOL
from run_shared_validation import ROOT,verify_protocol
from run_multiround_diagnostic import verify_attempt


def interval(values):
    a=np.asarray(values,float)
    if len(a)!=9:raise ValueError('Nine scene means required')
    sd=float(a.std(ddof=1));half=2.306004135*sd/3
    rng=np.random.default_rng(2026102503);idx=rng.integers(0,9,(20000,9))
    return {'mean':float(a.mean()),'scene_sd':sd,'student_t95':[float(a.mean()-half),float(a.mean()+half)],
            'scene_bootstrap95':np.quantile(a[idx].mean(axis=1),[.025,.975]).tolist(),'scene_values':a.tolist()}


def score_rows(protocol,observed):
    scenes=sorted({r['scene_id'] for r in observed});metrics=[]
    if len(observed)!=378 or Counter(r['scene_id'] for r in observed)!=dict.fromkeys(scenes,42):raise ValueError('Complete54x7 matrix required')
    for name in protocol['forecasts']:
        for budget in [1,2]:
            scene_scores=[]
            for scene in scenes:
                selected=[r for r in observed if r['scene_id']==scene]
                losses=[(protocol['forecasts'][name][r['condition']][budget-1]-r['success_curve'][budget-1])**2 for r in selected]
                scene_scores.append({'scene_id':scene,'loss':float(np.mean(losses))})
            metrics.append({'model':name,'budget':budget,'mean_brier':float(np.mean([r['loss'] for r in scene_scores])),'scene_scores':scene_scores})
    contrasts=[]
    for baseline in ['constant_persistent','constant_pooled','error_pooled']:
        for budget in [1,2]:
            a=next(r for r in metrics if r['model']==baseline and r['budget']==budget)
            b=next(r for r in metrics if r['model']=='selected' and r['budget']==budget)
            differences=[x['loss']-y['loss'] for x,y in zip(a['scene_scores'],b['scene_scores'])]
            contrasts.append({'baseline':baseline,'budget':budget,**interval(differences)})
    primary=next(r for r in contrasts if r['baseline']=='constant_persistent' and r['budget']==2)
    lo,hi=primary['student_t95'];decision='prediction_improvement' if lo>0 else ('baseline_favored' if hi<0 else 'inconclusive')
    return {'metrics':metrics,'contrasts':contrasts,'primary':primary,'primary_decision':decision}


def main():
    protocol=verify_protocol();summary=read_json(ROOT/'summary.json')
    if summary['status']!='complete':raise ValueError('No scoring partial validation')
    cases={c['task_id']:c for c in protocol['cases']};rows=[]
    for trajectory in protocol['trajectories']:
        memory=[protocol['entries'][k] for k in trajectory['initial_labels']]
        sequence=sorted([r for r in summary['attempts_rows'] if r['trajectory_id']==trajectory['id']],key=lambda r:r['trial'])
        if [r['trial'] for r in sequence]!=list(range(1,len(sequence)+1)) or not 1<=len(sequence)<=2:raise ValueError('Attempt sequence')
        for row in sequence:
            from pathlib import Path
            path=Path(row['result_path']);result=read_json(path)
            if file_hash(path)!=row['result_file_sha256']:raise ValueError('Changed raw result')
            verify_attempt(result,path.parent,cases[trajectory['task_id']],trajectory,memory,row['trial'])
            memory=result['memory_after']
        if len(sequence)==1 and not sequence[0]['success']:raise ValueError('Early failed stop')
        if any(r['success'] for r in sequence[:-1]):raise ValueError('Continued after success')
        rows.append({'trajectory_id':trajectory['id'],'task_id':trajectory['task_id'],
            'scene_id':cases[trajectory['task_id']]['task']['scene_id'],'condition':trajectory['condition'],
            'success_curve':[bool(sequence[0]['success']),any(r['success'] for r in sequence)],
            'result_hashes':[r['result_file_sha256'] for r in sequence]})
    scores=score_rows(protocol,rows)
    report=save(PROTOCOL.with_name('shared-validation-results.json'),{'scope':'Prospective new-task budget-success prediction',
        'protocol_sha256':protocol['sha256'],'parameters_sha256':protocol['parent_calibration_parameters_sha256'],
        'raw_summary_file_sha256':file_hash(ROOT/'summary.json'),'evaluator_source_sha256':file_hash(__import__('pathlib').Path(__file__)),
        'tasks':54,'scenes':9,'trajectories':378,'observed':rows,**scores,
        'per_condition':{c:{'tasks':54,'first_successes':sum(r['success_curve'][0] for r in rows if r['condition']==c),
            'two_successes':sum(r['success_curve'][1] for r in rows if r['condition']==c)} for c in STATES},
        'parameter_refits':0,'holdout134_model_calls':0,'semantic_annotation_required_for_primary':False})
    print({'sha256':report['sha256'],'primary':report['primary'],'decision':report['primary_decision']})


if __name__=='__main__':main()
