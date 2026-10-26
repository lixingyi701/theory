#!/usr/bin/env python3
"""Join full-text annotations to existing traces and separate success/write residuals."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import numpy as np
from agent_reliability.io import digest, file_hash, read_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
import shared_recovery_model as model

PUBLIC = Path('repro/reflexion-20261025')
PRIVATE = Path('runs/reflexion-successor-support-review-20261026')
ROOT = Path('runs/reflexion-four-attempt-validation-20261025')
LABELS = PUBLIC/'successor-support-semantic-labels-20261026.json'


def annotate():
    manifest = checked(PUBLIC/'successor-support-review-manifest-20261026.json')
    reading = read_json(PRIVATE/'reading-view.json')
    assert file_hash(PRIVATE/'reading-view.json') == manifest['private_reading_file_sha256']
    original = checked(PUBLIC/'shared-calibration-semantic-labels.json')
    decisions = read_json(PRIVATE/'decisions.json')
    assert [r[0] for r in decisions] == list(range(106))
    entries = []
    for index, category, evidence, reason in decisions:
        row = reading[index]
        assert row['index'] == index and digest(row['reflection']) == row['reflection_sha256']
        assert category in model.OUTCOMES[1:] and evidence and reason
        assert all(q in row['reflection'] for q in evidence)
        entries.append({'index':index,'reflection_sha256':row['reflection_sha256'],
                        'category':category,'evidence':evidence,'reason':reason})
    assert {r['reflection_sha256'] for r in entries} == {r['reflection_sha256'] for r in manifest['selected']}
    content = {'scope':'Full-text annotation of selected existing successor reflections',
        'manifest_sha256':manifest['sha256'],'private_reading_file_sha256':manifest['private_reading_file_sha256'],
        'decisions_file_sha256':file_hash(PRIVATE/'decisions.json'),
        'annotator':'Codex full-text reading; self-review before joining per-trajectory later outcomes',
        'human_second_review':False,'independent_blinded_annotation':False,
        'outcome_visibility':'Aggregate results known; reading view omitted task, trial, condition and later outcome',
        'category_definitions':original['category_definitions'],'interpretation_rules':original['interpretation_rules'],
        'entries':entries,'unique_text_counts':dict(Counter(r['category'] for r in entries)),
        'source_sha256':file_hash(Path(__file__)),'new_model_calls':0}
    if LABELS.exists():
        old = checked(LABELS)
        assert old['sha256']==digest(content), 'Existing annotations differ; explicit revision required'
        print({'existing_labels_sha256':old['sha256']}); return
    result = save(LABELS,content)
    print({'labels_sha256':result['sha256'],'unique_text_counts':result['unique_text_counts']})


def has_error(state):
    return any(s['type'] in ['E','B'] for s in state)


def g_probability(fitted, state):
    kernel = model.component_kernel(fitted,state)[0]
    return kernel[1:]/(1-kernel[0])


def summarize_success(rows):
    result = {'attempts':len(rows),'successes':sum(r['success'] for r in rows)}
    for name in ['frozen','weak_ridge']:
        result[name] = {'expected_successes':float(sum(r['predictions'][name] for r in rows)),
                        'mean_brier':float(np.mean([(r['predictions'][name]-r['success'])**2 for r in rows]))}
    return result


def summarize_writes(rows):
    result = {'failed_writes':len(rows),'outcomes':dict(Counter(r['new_type'] for r in rows)),
              'frozen_expected_counts':dict(zip(model.OUTCOMES[1:],np.sum([r['g'] for r in rows],axis=0).tolist())),
              'conditional_type_brier':float(np.mean([sum((np.asarray(r['g'])-np.eye(5)[model.OUTCOMES[1:].index(r['new_type'])])**2) for r in rows])),
              'next_observed_successes':sum(r['next_success'] for r in rows)}
    for name in ['frozen','weak_ridge']:
        result[name] = {'integrated_next_expected_successes':float(sum(r['bridges'][name]['integrated'] for r in rows)),
                        'actual_type_next_expected_successes':float(sum(r['bridges'][name]['actual_type'] for r in rows))}
    return result


def analyze():
    labels = checked(LABELS); manifest = checked(PUBLIC/'successor-support-review-manifest-20261026.json')
    assert labels['manifest_sha256']==manifest['sha256']
    protocol = checked(PUBLIC/'four-attempt-validation-protocol.json')
    params = checked(PUBLIC/'shared-calibration-parameters.json')
    diagnostic = checked(PUBLIC/'success-regularization-diagnostic-20261026.json')
    assert diagnostic['parameters_sha256']==params['sha256']
    assert params['model_source_sha256']==file_hash(Path('scripts/shared_recovery_model.py'))
    before = {p:file_hash(PUBLIC/p) for p in ['shared-calibration-parameters.json','four-attempt-validation-protocol.json',
                 'shared-validation-results.json','four-attempt-validation-results.json','four-attempt-validation-evaluation.json']}
    summary = checked(ROOT/'summary.json')
    assert file_hash(ROOT/'summary.json')==manifest['raw_summary_file_sha256']
    label_map = {r['reflection_sha256']:r['category'] for r in labels['entries']}
    selected_events = {(r['trajectory_id'],r['trial']):r for r in manifest['selected']}
    tasks = set(protocol['secondary_recovery_task_ids'])
    trajectories = [t for t in protocol['trajectories'] if t['panel']=='primary' and t['task_id'] in tasks]
    assert len(trajectories)==108 and len(tasks)==27
    base = params['comparison']['selected_model']
    candidate = next(r for r in diagnostic['calibration_fits'] if r['ridge']==.01)
    models = {'frozen':base,'weak_ridge':{**base,'success_coefficients':candidate['coefficients']}}
    rows = []; writes = []; seen_texts = []; trajectory_results = []
    for trajectory in trajectories:
        records = sorted([r for r in summary['attempts_rows'] if r['trajectory_id']==trajectory['id']], key=lambda r:r['trial'])
        assert [r['trial'] for r in records]==list(range(1,len(records)+1))
        memory = [protocol['entries'][k] for k in trajectory['initial_labels']]
        state = [{'type':k,'native':False} for k in trajectory['initial_labels']]
        weights = {name:model.PRIOR.copy() for name in models}
        sequence = []; curve = []
        for i,record in enumerate(records):
            path = Path(record['result_path']); value = read_json(path)
            assert file_hash(path)==record['result_file_sha256']
            assert value['success']==record['success']==bool(read_json(path.parent/'checkpoint.json')['transitions'][-1]['won'])
            assert value['task_id']==trajectory['task_id'] and value['trial']==record['trial']
            assert value['memory_before']==memory
            assert digest(memory[-3:])==record['visible_memory_sha256']
            if record['success']:
                assert value['memory_after']==memory and i==len(records)-1
            else:
                assert value['memory_after'][:-1]==memory
            predictions = {name:float(weights[name]@model.component_kernel(fitted,state)[:,0]) for name,fitted in models.items()}
            row = {'trajectory_id':trajectory['id'],'task_id':trajectory['task_id'],'scene_id':value['scene_id'],
                   'condition':trajectory['condition'],'seed':trajectory['seed'],'trial':record['trial'],
                   'state':list(state),'native_count':sum(s['native'] for s in state),'has_error':has_error(state),
                   'success':record['success'],'predictions':predictions,'visible_memory_sha256':record['visible_memory_sha256'],
                   'result_file_sha256':record['result_file_sha256']}
            rows.append(row);sequence.append(row);curve.append(record['success'])
            if not record['success']:
                h=digest(value['memory_after'][-1]); assert [h]==record['new_reflection_sha256']
                for name,fitted in models.items():
                    weights[name] *= 1-model.component_kernel(fitted,state)[:,0]
                    weights[name] /= weights[name].sum()
                if record['trial']<4:
                    source = selected_events[trajectory['id'],record['trial']]
                    assert source['reflection_sha256']==h
                    kind = label_map[h];seen_texts.append(h)
                    next_state = (state+[{'type':kind,'native':True}])[-3:]
                    assert i+1<len(records)
                    bridges = {}
                    for name,fitted in models.items():
                        g=g_probability(fitted,state)
                        integrated = sum(float(g[j]*(weights[name]@model.component_kernel(fitted,(state+[{'type':k,'native':True}])[-3:])[:,0]))
                                         for j,k in enumerate(model.OUTCOMES[1:]))
                        bridges[name] = {'integrated':integrated,'actual_type':float(weights[name]@model.component_kernel(fitted,next_state)[:,0])}
                    writes.append({**row,'new_reflection_sha256':h,'new_type':kind,
                                   'g':g_probability(base,state).tolist(),'bridges':bridges,
                                   'next_success':records[i+1]['success']})
                    state = next_state
            memory = value['memory_after']
        assert records[-1]['success'] or len(records)==4
        trajectory_results.append({'trajectory_id':trajectory['id'],'scene_id':sequence[0]['scene_id'],
            'success_curve':[any(curve[:b]) for b in range(1,5)],'attempts':len(records)})
    assert len(writes)==109 and set(seen_texts)==set(label_map)
    assert {(r['trajectory_id'],r['trial']) for r in writes}==set(selected_events)
    success_groups = {str(n):summarize_success([r for r in rows if r['native_count']==n]) for n in range(4)}
    typed_groups = {str(n)+'_'+str(e):summarize_success([r for r in rows if r['native_count']==n and r['has_error']==e])
                    for n in range(4) for e in [False,True] if any(r['native_count']==n and r['has_error']==e for r in rows)}
    write_groups = {str(n):summarize_writes([r for r in writes if r['native_count']==n]) for n in range(3)}
    error_events = {str(e):{'failed_writes':sum(r['has_error']==e for r in writes),
        'new_E_or_B':sum(r['has_error']==e and r['new_type'] in ['E','B'] for r in writes),
        'frozen_expected_new_E_or_B':float(sum(r['g'][0]+r['g'][2] for r in writes if r['has_error']==e))} for e in [False,True]}
    for name,h in before.items(): assert file_hash(PUBLIC/name)==h,name
    output = save(PUBLIC/'successor-support-diagnostic-20261026.json', {
        'scope':'Postcollection development diagnosis on selected108 existing trajectories; conditional state scores, not a new frozen budget validation',
        'manifest_sha256':manifest['sha256'],'labels_sha256':labels['sha256'],
        'parameters_sha256':params['sha256'],'regularization_diagnostic_sha256':diagnostic['sha256'],
        'weak_ridge_candidate_origin':'Positive ridge0.01 chosen on original calibration leave-scene development scores; fixed write kernel; no new fitting here',
        'source_sha256':file_hash(Path(__file__)),'raw_summary_file_sha256':file_hash(ROOT/'summary.json'),
        'trajectories':108,'tasks':27,'scenes':27,'attempts':len(rows),
        'selected_cumulative_success_counts':[sum(t['success_curve'][b-1] for t in trajectory_results) for b in range(1,5)],
        'unique_text_counts':labels['unique_text_counts'],'write_event_counts':dict(Counter(r['new_type'] for r in writes)),
        'success_by_native_count':success_groups,'success_by_native_count_and_error':typed_groups,
        'writes_by_native_count':write_groups,'error_generation_by_starting_error_presence':error_events,
        'aggregation':'Descriptive counts and sums over reached attempts. Survivor populations differ by trial; no independence or causal-effect test. Actual-state scores use post-failure text labels, not initial information.',
        'attempt_rows':rows,'write_rows':writes,'trajectory_rows':trajectory_results,
        'frozen_file_hashes_unchanged':before,'new_model_calls':0,'parameter_refits':0,'holdout134_calls':0})
    print({k:output[k] for k in ['sha256','attempts','unique_text_counts','write_event_counts','success_by_native_count','writes_by_native_count','error_generation_by_starting_error_presence']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['labels','analyze']);args=parser.parse_args()
    annotate() if args.action=='labels' else analyze()
