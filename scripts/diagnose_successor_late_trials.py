#!/usr/bin/env python3
"""Read existing fixed-panel failures and diagnose frozen p/G without new fitting."""
from pathlib import Path
from collections import Counter
import re
import argparse, math
import numpy as np
from agent_reliability.io import digest, file_hash, read_json, write_json
from prepare_shared_calibration import save
from prepare_successor_protocol import PROTOCOL, ROOT, checked

PUBLIC=PROTOCOL.parent
PRIVATE=Path('runs/reflexion-successor-late-diagnostic-20261027')
MANIFEST=PUBLIC/'successor-late-review-manifest-20261027.json'
LABELS=PUBLIC/'successor-late-semantic-labels-20261027.json'


def prepare():
    assert not MANIFEST.exists(), 'Preserve prepared panel rather than reset it'
    p=checked(PROTOCOL); summary=checked(ROOT/'summary.json')
    evaluation=checked(PUBLIC/'successor-validation-results-20261026.json')
    tasks=set(p['secondary_recovery_task_ids'])
    trajectories=[t for t in p['trajectories'] if t['panel']=='primary' and t['task_id'] in tasks]
    ids={t['id'] for t in trajectories}
    assert len(ids)==100 and len(tasks)==25
    texts={};selected=[]
    for row in summary['attempts_rows']:
        if row['trajectory_id'] not in ids or row['trial']>=4 or row['success']:continue
        path=Path(row['result_path']);actual=read_json(path)
        assert file_hash(path)==row['result_file_sha256']
        text=actual['memory_after'][-1];h=digest(text)
        assert row['new_reflection_sha256']==[h]
        texts[h]=text
        selected.append({'trajectory_id':row['trajectory_id'],'task_id':row['task_id'],
            'condition':row['condition'],'seed':row['seed'],'trial':row['trial'],
            'reflection_sha256':h,'result_file_sha256':row['result_file_sha256']})
    hashes=sorted(texts,key=lambda h:digest(['successor-late-diagnostic-20261027',h]))
    reading=[{'index':i,'reflection_sha256':h,'reflection':texts[h]} for i,h in enumerate(hashes)]
    index={r['reflection_sha256']:r['index'] for r in reading}
    selected=[{**r,'index':index[r['reflection_sha256']]} for r in selected]
    PRIVATE.mkdir(parents=True,exist_ok=True);write_json(PRIVATE/'reading-view.json',reading)
    d=save(MANIFEST,{'scope':'Postcollection diagnosis on all100 C/N trajectories of the25 precollection-fixed recovery tasks',
        'selection':'All failed writes after trials1..3, no filtering by later success; task panel fixed before original collection',
        'excluded_for_scope':'Remaining75 primary tasks and150 recovery trajectories are not semantically annotated here',
        'protocol_sha256':p['sha256'],'frozen_result_sha256':evaluation['sha256'],
        'selected_task_ids':sorted(tasks),'selected_trajectory_ids':sorted(ids),
        'selected_tasks':25,'selected_trajectories':100,'selected_write_events':len(selected),
        'unique_texts':len(reading),'writes_by_trial':dict(Counter(r['trial'] for r in selected)),
        'private_reading_file_sha256':file_hash(PRIVATE/'reading-view.json'),
        'reading_view':'Full text in stable hash order; task/trial/condition/later outcomes omitted, aggregates already known; not independent blinded annotation',
        'label_schema_source':'shared-calibration-semantic-labels.json','selected':selected,
        'new_model_calls':0,'parameter_refits':0,'source_sha256':file_hash(Path(__file__))})
    print({k:d[k] for k in ['sha256','selected_write_events','unique_texts','writes_by_trial']})


def labels():
    manifest=checked(MANIFEST);reading=read_json(PRIVATE/'reading-view.json')
    assert file_hash(PRIVATE/'reading-view.json')==manifest['private_reading_file_sha256']
    decisions=read_json(PRIVATE/'decisions.json');original=checked(PUBLIC/'shared-calibration-semantic-labels.json')
    assert [r[0] for r in decisions]==list(range(len(reading)))
    entries=[]
    for i,category,evidence,reason in decisions:
        row=reading[i];assert digest(row['reflection'])==row['reflection_sha256']
        assert category in ['E','C','B','U','A'] and evidence and reason
        assert all(q in row['reflection'] for q in evidence),(i,evidence)
        entries.append({'index':i,'reflection_sha256':row['reflection_sha256'],
            'category':category,'evidence':evidence,'reason':reason})
    content={'scope':'Full-text target-prerequisite labels of144 existing reflection texts',
        'manifest_sha256':manifest['sha256'],'reading_file_sha256':manifest['private_reading_file_sha256'],
        'decisions_file_sha256':file_hash(PRIVATE/'decisions.json'),
        'annotator':'Codex full-text reading and self-review before joining later outcomes',
        'human_second_review':False,'independent_blinded_annotation':False,
        'category_definitions':original['category_definitions'],'interpretation_rules':original['interpretation_rules'],
        'entries':entries,'counts':dict(Counter(r['category'] for r in entries)),'new_model_calls':0}
    if LABELS.exists():assert checked(LABELS)['sha256']==digest(content);return
    result=save(LABELS,content);print({'labels_sha256':result['sha256'],'counts':result['counts']})


def analyse():
    from calibrate_successor_success import success_components,component_kernel,features
    from shared_recovery_model import OUTCOMES
    p=checked(PROTOCOL);manifest=checked(MANIFEST);annotations=checked(LABELS)
    summary=checked(ROOT/'summary.json');frozen=checked(PUBLIC/'successor-validation-results-20261026.json')
    before={name:file_hash(Path(name)) for name in p['protected_files']}
    label_map={r['reflection_sha256']:r['category'] for r in annotations['entries']}
    ids=set(manifest['selected_trajectory_ids'])
    trajectories=[t for t in p['trajectories'] if t['id'] in ids]
    models=p['models'];counts=Counter();attempts=[];writes=[];trajectories_out=[]
    for trajectory in trajectories:
        sequence=sorted([r for r in summary['attempts_rows'] if r['trajectory_id']==trajectory['id']],key=lambda r:r['trial'])
        initial=[{'type':k,'native':False} for k in trajectory['initial_labels']]
        state=initial.copy();memory=[p['entries'][k] for k in trajectory['initial_labels']]
        weights={name:np.asarray(model['task_prior'],float) for name,model in models.items()}
        for i,row in enumerate(sequence):
            path=Path(row['result_path']);value=read_json(path)
            assert file_hash(path)==row['result_file_sha256']
            assert value['memory_before']==memory and digest(memory[-3:])==row['visible_memory_sha256']
            assert bool(read_json(path.parent/'checkpoint.json')['transitions'][-1]['won'])==value['success']==row['success']
            predictions={name:float(weights[name]@success_components(model,state)) for name,model in models.items()}
            reset={name:float(np.asarray(model['task_prior'])@success_components(model,state)) for name,model in models.items()}
            entry={'trajectory_id':trajectory['id'],'task_id':trajectory['task_id'],'scene_id':value['scene_id'],
                'condition':trajectory['condition'],'seed':trajectory['seed'],'trial':row['trial'],
                'state':list(state),'native_count':sum(s['native'] for s in state),
                'has_error':any(s['type'] in ['E','B'] for s in state),
                'has_correct':any(s['type'] in ['C','B'] for s in state),
                'success':row['success'],'predictions':predictions,'reset_prior':reset,
                'posterior_before':{name:w.tolist() for name,w in weights.items()},
                'result_file_sha256':row['result_file_sha256']}
            attempts.append(entry)
            if not row['success']:
                for name,model in models.items():
                    weights[name]*=1-success_components(model,state);weights[name]/=weights[name].sum()
                if row['trial']<4:
                    assert i+1<len(sequence)
                    h=digest(value['memory_after'][-1]);kind=label_map[h];counts[kind]+=1
                    assert [h]==row['new_reflection_sha256']
                    ns=(state+[{'type':kind,'native':True}])[-3:]
                    bridges={}
                    for name,model in models.items():
                        kernel=component_kernel(model,state)[0];g=kernel[1:]/(1-kernel[0])
                        integrated=sum(float(g[j]*(weights[name]@success_components(model,(state+[{'type':k,'native':True}])[-3:]))) for j,k in enumerate(OUTCOMES[1:]))
                        actual=float(weights[name]@success_components(model,ns))
                        bridges[name]={'integrated':integrated,'actual_type':actual,'g':g.tolist()}
                    writes.append({**entry,'new_type':kind,'reflection_sha256':h,
                        'bridges':bridges,'next_success':sequence[i+1]['success']})
                    state=ns
            memory=value['memory_after']
        trajectories_out.append({'trajectory_id':trajectory['id'],'scene_id':attempts[-1]['scene_id'],
            'attempts':len(sequence),'success_curve':[any(r['success'] for r in sequence if r['trial']<=b) for b in range(1,5)]})
    assert len(trajectories)==100 and len(writes)==147 and len(attempts)==247
    def sg(rows):
        return {'attempts':len(rows),'observed_successes':sum(r['success'] for r in rows),
            'expected_successes':{name:float(sum(r['predictions'][name] for r in rows)) for name in models},
            'reset_prior_expected':{name:float(sum(r['reset_prior'][name] for r in rows)) for name in models}}
    def wg(rows):
        out={'writes':len(rows),'observed_types':dict(Counter(r['new_type'] for r in rows)),
            'next_observed_successes':sum(r['next_success'] for r in rows),'models':{}}
        for name in models:
            integrated=sum(r['bridges'][name]['integrated'] for r in rows)
            actual=sum(r['bridges'][name]['actual_type'] for r in rows)
            observed=out['next_observed_successes']
            out['models'][name]={'expected_type_counts':dict(zip(OUTCOMES[1:],np.sum([r['bridges'][name]['g'] for r in rows],axis=0).tolist())) if rows else {},
                'integrated_next_successes':float(integrated),'actual_type_next_successes':float(actual),
                'type_substitution_term':float(actual-integrated),'conditional_success_residual':float(observed-actual),
                'total_next_success_residual':float(observed-integrated)}
        return out
    success_groups={str(b):sg([r for r in attempts if r['trial']==b]) for b in range(1,5)}
    by_error={f'trial{b}_error{e}_correct{c}':sg([r for r in attempts if r['trial']==b and r['has_error']==e and r['has_correct']==c])
        for b in range(2,5) for e in [False,True] for c in [False,True]
        if any(r['trial']==b and r['has_error']==e and r['has_correct']==c for r in attempts)}
    write_groups={str(b):wg([r for r in writes if r['trial']==b]) for b in range(1,4)}
    # Keep uncertainty labels unchanged; map them only in named sensitivity calculations.
    sensitivity=[]
    for replacement in ['E','U','C']:
        total={name:{b:0. for b in range(1,5)} for name in models}
        for t in trajectories:
            ww={name:np.asarray(model['task_prior'],float) for name,model in models.items()}
            for row in [r for r in attempts if r['trajectory_id']==t['id']]:
                state=[{**s,'type':replacement if s['type']=='A' else s['type']} for s in row['state']]
                for name,model in models.items():
                    pp=success_components(model,state);total[name][row['trial']]+=float(ww[name]@pp)
                    if not row['success']:ww[name]*=1-pp;ww[name]/=ww[name].sum()
        sensitivity.append({'map_uncertain_A_to':replacement,'expected_successes_by_trial':total})
    # Explicit decomposition of a trajectory failure's posterior shift, not new calibrated predictions.
    primary=frozen['primary'];se=(primary['student_t95'][1]-primary['mean'])/2.063898562
    tstat=primary['mean']/se;df=24;grid=np.linspace(0,tstat,20001)
    density=math.gamma((df+1)/2)/(math.sqrt(df*math.pi)*math.gamma(df/2))*(1+grid*grid/df)**(-(df+1)/2)
    step=grid[1]-grid[0];area=step/3*(density[0]+density[-1]+4*density[1:-1:2].sum()+2*density[2:-1:2].sum())
    two_sided_p=float(1-2*area)
    # An observable history candidate, measured from public action feedback only.
    # It is not a current-environment flag after reset and is not fitted here.
    cases={c['task_id']:c['task'] for c in p['cases']}
    source_by_id={}
    for row in summary['attempts_rows']:source_by_id.setdefault(row['trajectory_id'],[]).append(row)
    history_rows=[]
    for trajectory in [t for t in p['trajectories'] if t['panel']=='primary']:
        target=Path(cases[trajectory['task_id']]['gamefile']).parts[-3].split('-')[1].lower()
        previous=None
        for row in sorted(source_by_id[trajectory['id']],key=lambda r:r['trial']):
            cp=read_json(Path(row['result_path']).parent/'checkpoint.json')
            picked=any(re.match(r'^You pick up the '+re.escape(target)+r' \d+ from ',t['observation'],re.I)
                and re.match(r'^take '+re.escape(target)+r' \d+ from ',t['action'],re.I) for t in cp['transitions'])
            history_rows.append({'trajectory_id':trajectory['id'],'task_id':trajectory['task_id'],
                'trial':row['trial'],'target_type':target,'in_annotation_panel':trajectory['id'] in ids,
                'previous_failed_attempt_had_target_pickup':previous,'success':row['success'],
                'current_attempt_had_target_pickup':picked})
            if not row['success']:previous=picked
    subgroup=[]
    for name,selected in [('annotated25',lambda r:r['task_id'] in set(manifest['selected_task_ids'])),
                          ('remaining75',lambda r:r['task_id'] not in set(manifest['selected_task_ids']))]:
        rows=[r for r in frozen['observed'] if r['panel']=='primary' and selected(r)]
        subgroup.append({'group':name,'trajectories':len(rows),'tasks':len({r['task_id'] for r in rows}),
            'raw_cumulative_success_counts':[sum(r['success_curve'][b] for r in rows) for b in range(4)]})
    pickup_groups={f'trial{b}_previous_pickup{flag}':{
        'attempts':sum(r['trial']==b and r['previous_failed_attempt_had_target_pickup']==flag for r in history_rows),
        'successes':sum(r['trial']==b and r['previous_failed_attempt_had_target_pickup']==flag and r['success'] for r in history_rows)}
        for b in range(2,5) for flag in [False,True]}
    for row in attempts:
        history=next(h for h in history_rows if h['trajectory_id']==row['trajectory_id'] and h['trial']==row['trial'])
        row['previous_failed_attempt_had_target_pickup']=history['previous_failed_attempt_had_target_pickup']
    semantic_pickup={f'error{e}_previous_pickup{flag}':{
        'attempts':sum(r['trial']>1 and r['has_error']==e and r['previous_failed_attempt_had_target_pickup']==flag for r in attempts),
        'successes':sum(r['trial']>1 and r['has_error']==e and r['previous_failed_attempt_had_target_pickup']==flag and r['success'] for r in attempts)}
        for e in [False,True] for flag in [False,True]}
    alias_states={k:[{'type':'N','native':False},{'type':'N','native':False},{'type':k,'native':True}]
        for k in ['E','B','C','U']}
    aliases={k:features(state,'seven') for k,state in alias_states.items()}
    assert aliases['C']==aliases['U'] and aliases['E']==aliases['B']
    assert all(file_hash(Path(name))==h for name,h in before.items())
    out=save(PUBLIC/'successor-late-diagnostic-20261027.json',{
        'scope':'Postcollection, frozen-parameter conditional-state and one-update diagnosis on fixed25-task C/N panel',
        'manifest_sha256':manifest['sha256'],'labels_sha256':annotations['sha256'],
        'protocol_sha256':p['sha256'],'frozen_result_sha256':frozen['sha256'],'source_sha256':file_hash(Path(__file__)),
        'tasks':25,'scenes':25,'trajectories':100,'attempts':len(attempts),'writes':len(writes),
        'cumulative_success_counts':[sum(t['success_curve'][b] for t in trajectories_out) for b in range(4)],
        'write_event_types':dict(counts),'success_by_trial':success_groups,
        'success_by_trial_error_correct':by_error,'writes_by_trial':write_groups,'all_writes':wg(writes),
        'uncertain_label_sensitivity':sensitivity,
        'fixed_panel_vs_remaining_primary':subgroup,
        'history_candidate_definition':'Previous failed attempt contained a successful target pickup, matched by take command and public You pick up feedback; unavailable at trial1. It describes past feedback, not current environment state.',
        'history_candidate_scope':'Postcollection association only. Target type from task identifier, no hidden location/status facts. No eighth-feature model fit or prospective validation.',
        'all_primary_success_by_previous_pickup':pickup_groups,
        'annotated_success_by_error_and_previous_pickup':semantic_pickup,
        'seven_feature_alias_examples':aliases,
        'unfitted_eighth_feature_candidate':'I(any C or B in visible memory), distinguishing C/U and B/E; candidate only, no benefit assumed or fitting performed',
        'primary_original_two_sided_t':{'statistic':tstat,'df':24,'two_sided_p':two_sided_p,
            'standard_error':se,'method':'Numerical Simpson integration of Student t density; interpretation remains original t95 rule'},
        'aggregation':'Descriptive sums over reached attempts; different from full400 scene-equal budget scores. Survivor groups differ. Actual-type substitution and prior reset are postcollection diagnostics, not prospective scores or causal interventions.',
        'attempt_rows':attempts,'write_rows':writes,'trajectory_rows':trajectories_out,
        'frozen_files_unchanged':before,'new_model_calls':0,'parameter_refits':0,'holdout134_model_calls':0})
    save(PUBLIC/'successor-late-diagnostic-metrics-20261027.json',{
        **{k:v for k,v in out.items() if k not in ['attempt_rows','write_rows','trajectory_rows','sha256']},
        'full_analysis_sha256':out['sha256'],
        'full_analysis_file_sha256':file_hash(PUBLIC/'successor-late-diagnostic-20261027.json'),
        'projection':'Aggregate diagnosis; complete bound output remains server-local'})
    save(PUBLIC/'successor-late-diagnostic-attempts-20261027.json',{
        'full_analysis_sha256':out['sha256'],'manifest_sha256':manifest['sha256'],
        'labels_sha256':annotations['sha256'],'attempt_rows':attempts,'trajectory_rows':trajectories_out})
    print({k:out[k] for k in ['sha256','fixed_panel_vs_remaining_primary','all_primary_success_by_previous_pickup','annotated_success_by_error_and_previous_pickup','primary_original_two_sided_t']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','labels','analyse']);args=parser.parse_args()
    {'prepare':prepare,'labels':labels,'analyse':analyse}[args.action]()
