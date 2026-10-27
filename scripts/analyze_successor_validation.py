#!/usr/bin/env python3
"""Recompute completed successor validation and descriptive curves without fitting."""
from collections import Counter
from pathlib import Path
import math
import statistics as st
from agent_reliability.io import digest, file_hash, read_json
from prepare_shared_calibration import save
from prepare_successor_protocol import ROOT, PROTOCOL
from run_successor_validation import verify_protocol
from run_multiround_diagnostic import verify_attempt
from summarize_reflexion import costs
from continue_reflexion_pilot import ledger

PUBLIC = PROTOCOL.parent


def checked(path):
    d = read_json(path)
    assert d['sha256'] == digest({k: v for k, v in d.items() if k != 'sha256'}), path
    return d


def by_scene(rows, value):
    out = {}
    for scene in sorted({r['scene_id'] for r in rows}):
        subset = [r for r in rows if r['scene_id'] == scene]
        tasks = sorted({r['task_id'] for r in subset})
        out[scene] = st.mean(st.mean(value(r) for r in subset if r['task_id'] == t) for t in tasks)
    assert len(out) == 25
    return out


def interval(values):
    values = list(values)
    assert len(values) == 25
    mean = st.mean(values)
    half = 2.063898562 * st.stdev(values) / math.sqrt(25)
    return {'mean': mean, 'student_t95': [mean-half, mean+half]}


def main():
    p = verify_protocol()
    summary = checked(ROOT/'summary.json')
    result_path = PUBLIC/'successor-validation-results-20261026.json'
    result = checked(result_path)
    contract = checked(ROOT/'contract.json')
    audit = checked(ROOT/'protocol-audit.json')
    shapes = checked(ROOT/'action-shapes.json')
    replicas = checked(ROOT/'replica-audit.json')
    closeout = checked(ROOT/'allowance-closeout.json')
    assert summary['status'] == 'complete' and summary['completed_tasks'] == 100
    assert summary['completed_trajectories'] == 550
    assert summary['worker_exit_codes'] == [0, 0] and summary['worker_statuses'] == ['complete']*2
    assert not (ROOT/'STOP').exists() and audit['passed'] and not audit['errors'] and replicas['passed']
    assert closeout['allowance_status'] == 'closed' and closeout['execution_status'] == 'complete'
    assert closeout['master_exit_code'] == 0 and closeout['contract_sha256'] == contract['sha256']
    assert result['protocol_sha256'] == summary['protocol_sha256'] == p['sha256']
    assert result['parameters_sha256'] == p['parent_calibration_parameters_sha256']
    assert result['raw_summary_file_sha256'] == file_hash(ROOT/'summary.json')
    assert result['evaluator_source_sha256'] == file_hash(Path('scripts/evaluate_successor_validation.py'))
    assert contract['freeze_commit'] == '6911e82bbecc731b2d6d9bcd28fdd698e6732df0'
    assert summary['elapsed_seconds'] + contract['preparation_elapsed_seconds'] < p['limits']['seconds']
    observed = result['observed']
    assert len(observed) == len({r['trajectory_id'] for r in observed}) == 550
    assert len({r['task_id'] for r in observed}) == 100 and len({r['scene_id'] for r in observed}) == 25
    primary = [r for r in observed if r['panel'] == 'primary']
    assert len(primary) == 400 and set(Counter(r['task_id'] for r in primary).values()) == {4}
    case = {c['task_id']: c for c in p['cases']}
    raw = {}
    for row in summary['attempts_rows']:
        raw.setdefault(row['trajectory_id'], []).append(row)
    events = {(r['trajectory_id'], r['trial']): r for r in result['events']}
    assert len(events) == len(summary['attempts_rows']) == 1423
    observed_by_id = {r['trajectory_id']: r for r in observed}
    for trajectory in p['trajectories']:
        memory = [p['entries'][k] for k in trajectory['initial_labels']]
        original_count = len(memory)
        sequence = sorted(raw[trajectory['id']], key=lambda r: r['trial'])
        assert [r['trial'] for r in sequence] == list(range(1, len(sequence)+1))
        assert not any(r['success'] for r in sequence[:-1])
        assert sequence[-1]['success'] or len(sequence) == 4
        for row in sequence:
            path = Path(row['result_path']); actual = read_json(path)
            assert file_hash(path) == row['result_file_sha256']
            verify_attempt(actual, path.parent, case[trajectory['task_id']], trajectory, memory, row['trial'])
            cp = read_json(path.parent/'checkpoint.json')
            assert bool(cp['transitions'][-1]['won']) == actual['success'] == row['success']
            event = events[trajectory['id'], row['trial']]
            assert event['visible_memory_sha256'] == row['visible_memory_sha256'] == digest(memory[-3:])
            assert event['original_entry_indices_visible'] == [i for i in range(original_count) if i >= len(memory)-3]
            assert actual['memory_before'] == memory
            assert actual['memory_after'][:len(memory)] == memory
            if row['trial'] == 4: assert not event['original_entry_indices_visible']
            memory = actual['memory_after']
        rebuilt_curve = [any(r['success'] for r in sequence if r['trial'] <= b) for b in range(1,5)]
        assert rebuilt_curve == observed_by_id[trajectory['id']]['success_curve']
        assert len(sequence) == observed_by_id[trajectory['id']]['attempts_completed']

    scores = {}
    for model, forecasts in p['forecasts'].items():
        for b in range(4):
            scene = by_scene(primary, lambda r: (forecasts[r['condition']][b]-float(r['success_curve'][b]))**2)
            scores[model, b+1] = scene
            original = next(m for m in result['metrics'] if m['model'] == model and m['budget'] == b+1)
            assert abs(st.mean(scene.values())-original['mean_brier']) < 1e-12
    for contrast in result['contrasts']:
        a = scores[contrast['baseline'], contrast['budget']]
        b = scores['full_seven', contrast['budget']]
        independent = interval(a[s]-b[s] for s in sorted(a))
        assert abs(independent['mean']-contrast['mean']) < 1e-12
        assert max(abs(x-y) for x,y in zip(independent['student_t95'], contrast['student_t95'])) < 1e-12

    curves = []
    q = []
    for b in range(4):
        data = by_scene(primary, lambda r: float(r['success_curve'][b]))
        q.append(st.mean(data.values()))
        curves.append({'budget': b+1, 'observed_scene_equal': interval(data.values()),
            'raw_successes': sum(r['success_curve'][b] for r in primary), 'raw_denominator':400,
            'predictions': {model:f['correct_middle'][b] for model,f in p['forecasts'].items()}})
        assert all(f['correct_middle'][b] == f['neutral_full'][b] for f in p['forecasts'].values())
    full = p['forecasts']['full_seven']['correct_middle']
    static = p['forecasts']['unchanged_seven']['correct_middle']
    hazards = []
    for b in range(4):
        prev = q[b-1] if b else 0.
        row = {'attempt':b+1, 'observed_scene_weighted':(q[b]-prev)/(1-prev)}
        for model,f in p['forecasts'].items():
            v=f['correct_middle']; vprev=v[b-1] if b else 0.
            row[model]=(v[b]-vprev)/(1-vprev)
        row['raw_reached']=sum(r['attempts_completed']>=b+1 for r in primary)
        row['raw_successes']=sum(r['success_curve'][b] and (not r['success_curve'][b-1] if b else True) for r in primary)
        hazards.append(row)
    decomposition=[]
    for b in range(1,4):
        rm=(1-full[b])/(1-full[0]); ro=(1-q[b])/(1-q[0])
        first=(q[0]-full[0])*rm; later=(1-q[0])*(rm-ro)
        assert abs(first+later-(q[b]-full[b]))<1e-12
        decomposition.append({'budget':b+1,'total_error':q[b]-full[b],
            'first_attempt_mismatch_term':first,'post_first_conditional_curve_mismatch_term':later})
    identity=(static[3]-full[3])*(static[3]+full[3]-2*q[3])
    assert abs(identity-result['primary']['mean'])<1e-12

    truncations=[]; roles=Counter()
    responses=list(ROOT.glob('**/calls/*/response.json'))
    for path in responses:
        response=read_json(path)['response']
        if response['choices'][0]['finish_reason']=='length':
            parts=path.relative_to(ROOT).parts; assert parts[0]=='trajectories'
            row=observed_by_id[parts[1]];role=path.parent.name.split('-')[0];roles[role]+=1
            truncations.append({'trajectory_id':row['trajectory_id'],'task_id':row['task_id'],
                'scene_id':row['scene_id'],'condition':row['condition'],'role':role,
                'path':path.relative_to(ROOT).as_posix()})
    cost=costs(ROOT)
    assert cost==summary['costs_incremental']==closeout['costs_incremental']
    assert cost['dispatched_calls']==cost['saved_responses']==summary['reserved_calls']==55261
    assert cost['uncertain_or_unrecorded_attempts']==0 and len(responses)==55261
    assert len(truncations)==cost['truncated_responses']==16
    global_ledger=ledger()
    assert sum(v['dispatched'] for v in global_ledger.values())==closeout['global_calls']==200396
    assert sum(v['responses'] for v in global_ledger.values())==closeout['global_responses']==200396

    common={'frozen_result_sha256':result['sha256'],'protocol_sha256':p['sha256']}
    metrics=save(PUBLIC/'successor-validation-metrics-20261026.json',{
        **{k:v for k,v in result.items() if k not in ['events','observed','sha256']},
        **common,'projection':'Frozen output excluding only per-trajectory observations and event records'})
    observations=save(PUBLIC/'successor-validation-observed-20261026.json',{
        **common,'observed':observed})
    parts=[]
    for start in range(0,len(result['events']),400):
        name=f'successor-validation-events-{start//400+1:02d}-20261026.json'
        part=save(PUBLIC/name,{**common,'first_event_index':start,'events':result['events'][start:start+400]})
        parts.append({'file':name,'sha256':part['sha256'],'events':len(part['events'])})
    analysis=save(PUBLIC/'successor-validation-analysis-20261027.json',{
        **common,'scope':'Independent arithmetic and raw-record audit; postcollection descriptive diagnostics, no refit',
        'source_sha256':file_hash(Path(__file__)),'passed':True,'native_attempts_checked':1423,
        'trajectories_checked':550,'metrics_checked':12,'contrasts_checked':8,
        'curves':curves,'conditional_success_diagnostic':hazards,
        'error_decomposition_reference':'Full-seven post-first survival; algebraic allocation, not causal attribution',
        'error_decomposition':decomposition,'equal_expected_brier_success_rate':(full[3]+static[3])/2,
        'brier_difference_identity':identity,
        'main_is_shared_curve_calibration':'C/N forecasts are identical across conditions and tasks at each budget; this primary comparison does not test task ranking',
        'costs':cost,'global_calls':closeout['global_calls'],'collection_seconds':summary['elapsed_seconds'],
        'whole_campaign_seconds':summary['elapsed_seconds']+contract['preparation_elapsed_seconds'],
        'truncations':truncations,'truncations_by_role':dict(roles),
        'truncated_trajectories':len({r['trajectory_id'] for r in truncations}),
        'truncated_scenes':len({r['scene_id'] for r in truncations}),
        'allowance_closeout_sha256':closeout['sha256'],'parameter_refits':0,'new_model_calls':0,'holdout134_model_calls':0})
    shapes_projection=save(PUBLIC/'successor-validation-action-shapes-20261027.json',{
        **{k:v for k,v in shapes.items() if k not in ['review_items','sha256']},
        'source_sha256':shapes['sha256'],'source_file_sha256':file_hash(ROOT/'action-shapes.json'),
        'review_item_count':len(shapes['review_items']),
        'projection':'Aggregate action audit; individual review records remain in the bound server-local source'})
    save(PUBLIC/'successor-validation-collection-integrity-20261027.json',{
        **common,'analysis_sha256':analysis['sha256'],'metrics_projection_sha256':metrics['sha256'],
        'observed_projection_sha256':observations['sha256'],'event_parts':parts,
        'server_local_frozen_result_file_sha256':file_hash(result_path),
        'protocol_audit_sha256':audit['sha256'],'action_shapes_sha256':shapes['sha256'],
        'action_shapes_projection_sha256':shapes_projection['sha256'],
        'closeout_sha256':closeout['sha256'],'contract_sha256':contract['sha256'],
        'passed':True,'new_model_calls':0,'parameter_refits':0})
    for source,name in [(closeout,'successor-validation-closeout-20261027.json'),
                        (audit,'successor-validation-protocol-audit-20261027.json')]:
        # These records contain no local paths; preserve their original internal digests.
        import json
        (PUBLIC/name).write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({'analysis_sha256':analysis['sha256'],'primary':result['primary_decision'],
        'curves':curves,'hazards':hazards,'truncations_by_role':dict(roles),
        'truncated_trajectories':analysis['truncated_trajectories'],'truncated_scenes':analysis['truncated_scenes']})


if __name__=='__main__':main()
