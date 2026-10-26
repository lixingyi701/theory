#!/usr/bin/env python3
"""Audit a completed frozen run and diagnose errors without fitting or generation."""
from collections import Counter
from pathlib import Path
import math
import statistics
import numpy as np

from agent_reliability.io import digest, file_hash, read_json
from continue_reflexion_pilot import ledger
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import PROTOCOL, ROOT
from run_four_attempt_validation import verify_protocol
from run_multiround_diagnostic import verify_attempt

PUBLIC = PROTOCOL.parent


def checked(path):
    a = read_json(path)
    assert a['sha256'] == digest({k: v for k, v in a.items() if k != 'sha256'}), path
    return a


def paired_interval(values):
    assert len(values) == 27
    mean = statistics.mean(values)
    half = 2.055529439 * statistics.stdev(values) / math.sqrt(27)
    return {'mean': mean, 'student_t95': [mean-half, mean+half]}


def weighted_scenes(rows, value):
    result = {}
    for scene in sorted({r['scene_id'] for r in rows}):
        rr = [r for r in rows if r['scene_id'] == scene]
        tasks = sorted({r['task_id'] for r in rr})
        result[scene] = statistics.mean(statistics.mean(value(r) for r in rr if r['task_id'] == t) for t in tasks)
    return result


def curves(protocol, primary):
    observed, scene_curves = [], []
    for b in range(4):
        scene = weighted_scenes(primary, lambda r: float(r['success_curve'][b]))
        scene_curves.append(scene)
        observed.append(paired_interval(list(scene.values())))
    full = protocol['forecasts']['full_update']['correct_middle']
    static = protocol['forecasts']['unchanged_memory']['correct_middle']
    assert full == protocol['forecasts']['full_update']['neutral_full']
    assert static == protocol['forecasts']['unchanged_memory']['neutral_full']
    q = [r['mean'] for r in observed]
    hazards = []
    for b in range(4):
        previous = q[b-1] if b else 0.
        vp = full[b-1] if b else 0.
        ap = static[b-1] if b else 0.
        hazards.append({'attempt': b+1,
            'observed_scene_weighted_conditional_success': (q[b]-previous)/(1-previous),
            'full_conditional_success': (full[b]-vp)/(1-vp),
            'unchanged_conditional_success': (static[b]-ap)/(1-ap)})
    # Exact arithmetic decomposition, NOT a new forecast or a fitted model.
    decomposition = []
    for b in range(1, 4):
        rm = (1-full[b])/(1-full[0])
        ro = (1-q[b])/(1-q[0])
        first = (q[0]-full[0])*rm
        later = (1-q[0])*(rm-ro)
        assert abs(first+later-(q[b]-full[b])) < 1e-12
        decomposition.append({'budget': b+1, 'total_success_probability_error': q[b]-full[b],
            'first_attempt_mismatch_term': first, 'post_first_conditional_curve_mismatch_term': later,
            'first_term_fraction_of_total': first/(q[b]-full[b]),
            'interpretation': 'Post-collection algebraic allocation of aggregate error, not causal attribution or validated recalibration'})
    a, v, truth = static[3], full[3], q[3]
    return {'scope': 'Post-collection descriptive mechanism diagnostics; no coefficients estimated',
        'observed_scene_equal_curves': observed, 'full_update_curve': full, 'unchanged_memory_curve': static,
        'conditional_success': hazards, 'success_error_decomposition': decomposition,
        'equal_expected_brier_success_rate': (a+v)/2,
        'primary_loss_difference_identity': (a-v)*(a+v-2*truth),
        'scene_curves': scene_curves}


def main():
    protocol = verify_protocol()
    summary = checked(ROOT/'summary.json')
    result = checked(PUBLIC/'four-attempt-validation-results.json')
    contract = checked(ROOT/'contract.json')
    audit = checked(ROOT/'protocol-audit.json')
    shapes = checked(ROOT/'action-shapes.json')
    replicas = checked(ROOT/'replica-audit.json')
    assert summary['status'] == 'complete' and summary['completed_tasks'] == 81
    assert summary['completed_trajectories'] == 486
    assert summary['worker_exit_codes'] == [0, 0] and summary['worker_statuses'] == ['complete']*2
    assert not (ROOT/'STOP').exists()
    assert audit['passed'] and not audit['errors'] and replicas['passed'] and len(replicas['rows']) == 4
    assert result['protocol_sha256'] == summary['protocol_sha256'] == protocol['sha256']
    assert result['parameters_sha256'] == protocol['parent_calibration_parameters_sha256']
    assert result['raw_summary_file_sha256'] == file_hash(ROOT/'summary.json')
    assert result['evaluator_source_sha256'] == file_hash(Path('scripts/evaluate_four_attempt_validation.py'))
    assert contract['freeze_commit'] == '60fa71e6927eb884348e6bf3c0ae8677298073bc'
    assert summary['elapsed_seconds'] <= protocol['limits']['seconds']

    rows = result['observed']
    assert len(rows) == len({r['trajectory_id'] for r in rows}) == 486
    assert len({r['task_id'] for r in rows}) == 81 and len({r['scene_id'] for r in rows}) == 27
    primary = [r for r in rows if r['panel'] == 'primary']
    assert len(primary) == 324 and set(Counter(r['task_id'] for r in primary).values()) == {4}
    cases = {c['task_id']: c for c in protocol['cases']}
    raw_by_id = {}
    for r in summary['attempts_rows']:
        raw_by_id.setdefault(r['trajectory_id'], []).append(r)
    event_lookup = {(r['trajectory_id'], r['trial']): r for r in result['events']}
    assert len(event_lookup) == len(summary['attempts_rows']) == 1297
    for trajectory in protocol['trajectories']:
        memory = [protocol['entries'][k] for k in trajectory['initial_labels']]
        original_count = len(memory)
        sequence = sorted(raw_by_id[trajectory['id']], key=lambda r: r['trial'])
        assert [r['trial'] for r in sequence] == list(range(1, len(sequence)+1))
        assert not any(r['success'] for r in sequence[:-1])
        assert sequence[-1]['success'] or len(sequence) == 4
        for row in sequence:
            p = Path(row['result_path']); a = read_json(p)
            assert file_hash(p) == row['result_file_sha256']
            verify_attempt(a, p.parent, cases[trajectory['task_id']], trajectory, memory, row['trial'])
            cp = read_json(p.parent/'checkpoint.json')
            assert bool(cp['transitions'][-1]['won']) == a['success'] == row['success']
            event = event_lookup[trajectory['id'], row['trial']]
            assert event['visible_memory_sha256'] == row['visible_memory_sha256'] == digest(memory[-3:])
            assert event['original_entry_indices_visible'] == [i for i in range(original_count) if i >= len(memory)-3]
            if row['trial'] == 4:
                assert not event['original_entry_indices_visible']
            memory = a['memory_after']

    # Recompute losses independently of the frozen evaluator's aggregation helper.
    recomputed = {}
    for model, forecasts in protocol['forecasts'].items():
        for b in range(4):
            values = weighted_scenes(primary, lambda r: (forecasts[r['condition']][b]-r['success_curve'][b])**2)
            recomputed[model, b+1] = values
            original = next(m for m in result['metrics'] if m['model'] == model and m['budget'] == b+1)
            assert abs(statistics.mean(values.values())-original['mean_brier']) < 1e-12
    delta = [recomputed['unchanged_memory',4][s]-recomputed['full_update',4][s] for s in sorted(recomputed['full_update',4])]
    primary_check = paired_interval(delta)
    assert abs(primary_check['mean']-result['primary']['mean']) < 1e-12
    assert np.allclose(primary_check['student_t95'], result['primary']['student_t95'], atol=1e-12, rtol=0)
    diagnostic = curves(protocol, primary)
    assert abs(diagnostic['primary_loss_difference_identity']-primary_check['mean']) < 1e-12

    # Audit saved response counts and truncations once; never regenerate a response.
    truncations, roles = [], Counter()
    responses = list(ROOT.glob('**/calls/*/response.json'))
    observed_by_id = {r['trajectory_id']: r for r in rows}
    for p in responses:
        a = read_json(p)['response']; role = p.parent.name.split('-')[0]
        if a['choices'][0]['finish_reason'] == 'length':
            parts = p.relative_to(ROOT).parts
            assert parts[0] == 'trajectories'
            t = observed_by_id[parts[1]]
            truncations.append({'path': p.relative_to(ROOT).as_posix(), 'trajectory_id': t['trajectory_id'],
                'task_id': t['task_id'], 'scene_id': t['scene_id'], 'condition': t['condition'], 'role': role})
            roles[role] += 1
    requests = list(ROOT.glob('**/calls/*/attempt-*.request.json'))
    reserved = read_json(ROOT/'budget.json')['reserved']
    assert len(requests) == len(responses) == reserved == summary['costs_incremental']['dispatched_calls'] == 51949
    assert len(truncations) == summary['costs_incremental']['truncated_responses'] == 21
    affected = {r['scene_id'] for r in truncations}
    retained = [s for s in sorted(recomputed['full_update',4]) if s not in affected]
    sensitivity = {'scope': 'Post-collection descriptive affected-scene exclusion; full sample is primary',
        'affected_scenes': sorted(affected), 'retained_scenes': len(retained),
        'primary_gain_point_only': statistics.mean(recomputed['unchanged_memory',4][s]-recomputed['full_update',4][s] for s in retained) if retained else None}

    # Explicit public projections retain the original frozen report unchanged on the server.
    evaluation = save(PUBLIC/'four-attempt-validation-evaluation.json', {
        **{k:v for k,v in result.items() if k not in ['sha256','events','raw_summary_file_sha256','observed']},
        'observed':[{k:v for k,v in r.items() if k != 'result_hashes'} for r in result['observed']],
        'scope': 'Projection of the unchanged frozen evaluator output; events partitioned by trial',
        'source_evaluator_results_sha256': result['sha256'],
        'source_evaluator_file_sha256': file_hash(PUBLIC/'four-attempt-validation-results.json'),
        'raw_summary_file_sha256': file_hash(ROOT/'summary.json'),
        'omitted_fields': ['events','observed[].result_hashes'],
        'result_hashes_location': 'Complete event partitions retain per-attempt result_file_sha256'})
    event_parts = []
    for trial in range(1,5):
        part = save(PUBLIC/f'four-attempt-validation-events-trial-{trial}.json', {
            'scope': 'Complete typed events for this attempt index; no raw model text',
            'protocol_sha256': protocol['sha256'], 'source_evaluator_results_sha256': result['sha256'],
            'trial': trial, 'rows': [r for r in result['events'] if r['trial'] == trial]})
        event_parts.append({'file': f'four-attempt-validation-events-trial-{trial}.json', 'sha256': part['sha256'], 'rows': len(part['rows'])})
    assert sum(p['rows'] for p in event_parts) == 1297
    event_index = save(PUBLIC/'four-attempt-validation-events.json', {'scope':'Index of all1297 typed attempt events',
        'protocol_sha256': protocol['sha256'], 'parts': event_parts})
    collection = save(PUBLIC/'four-attempt-validation-collection-summary.json', {
        **{k:v for k,v in summary.items() if k not in ['sha256','attempts_rows','trajectories']},
        'scope':'Aggregate projection; complete raw summary stays private',
        'raw_summary_file_sha256':file_hash(ROOT/'summary.json'),'events_sha256':event_index['sha256']})
    analysis = save(PUBLIC/'four-attempt-validation-analysis.json', {
        'scope':'Independent arithmetic audit and post-collection diagnostics; frozen primary unchanged',
        'protocol_sha256':protocol['sha256'],'frozen_results_sha256':result['sha256'],
        'primary_recomputed':primary_check,'diagnostic':diagnostic,'truncations':truncations,
        'truncation_roles':dict(roles),'truncation_sensitivity':sensitivity,
        'scenes_positive_primary_gain':sum(v>0 for v in delta),
        'all_panel_success_counts':[sum(r['success_curve'][b] for r in rows) for b in range(4)],
        'primary_panel_success_counts':[sum(r['success_curve'][b] for r in primary) for b in range(4)],
        'annotation_status':'1028 new reflections not semantically annotated in this analysis',
        'analysis_source_sha256':file_hash(Path(__file__)),'parameter_refits':0,'new_model_calls':0})
    all_runs = ledger()
    total = sum(v['dispatched'] for v in all_runs.values())
    assert total == sum(v['responses'] for v in all_runs.values()) == 145135
    holdout = {t['task_id'] for t in read_json(Path('repro/baseline-20261019/holdout134.json'))['tasks']}
    used = {read_json(p).get('task',{}).get('task_id') for p in ROOT.glob('**/identity.json')}
    used.discard(None)
    assert used == set(cases) and not used & holdout
    for name in ['protocol-audit','replica-audit','action-shapes']:
        if name == 'action-shapes':
            save(PUBLIC/f'four-attempt-validation-{name}.json', {
                **{k:v for k,v in shapes.items() if k not in ['sha256','review_items','scope']},
                'scope':'Aggregate projection of the action-shape audit; detailed command review remains private',
                'source_audit_sha256':shapes['sha256'], 'source_audit_file_sha256':file_hash(ROOT/f'{name}.json'),
                'omitted_field':'review_items','review_items_count':len(shapes['review_items'])})
        else:
            (PUBLIC/f'four-attempt-validation-{name}.json').write_bytes((ROOT/f'{name}.json').read_bytes())
    names = [f'four-attempt-validation-{n}.json' for n in ['evaluation','events','collection-summary','analysis','protocol-audit','replica-audit','action-shapes']]
    closeout = save(PUBLIC/'four-attempt-validation-closeout.json', {
        'scope':'Completed frozen four-attempt validation; allowance closed',
        'sampling_freeze':contract['freeze_commit'],'protocol_sha256':protocol['sha256'],
        'source_evaluator_results_sha256':result['sha256'],'parameters_sha256':result['parameters_sha256'],
        'tasks':81,'scenes':27,'trajectories':486,'attempts':1297,'success_counts':[sum(r['success_curve'][b] for r in rows) for b in range(4)],
        'costs_incremental':summary['costs_incremental'],'collection_elapsed_seconds':summary['elapsed_seconds'],
        'created_unix':contract['created_unix'],'collection_end_unix':contract['created_unix']+summary['elapsed_seconds'],
        'all_calls':total,'all_responses':total,'allowance_status':'closed','unused_calls':contract['max_calls']-reserved,
        'contract_sha256':contract['sha256'],'artifacts':{n:file_hash(PUBLIC/n) for n in names},
        'parameter_refits':0,'holdout134_model_calls':0,'new_model_calls':0,'source_sha256':file_hash(Path(__file__))})
    print({'primary':primary_check,'diagnostic':diagnostic,'truncation_roles':dict(roles),
        'affected_trajectories':len({r['trajectory_id'] for r in truncations}),'affected_scenes':len(affected),
        'sensitivity':sensitivity,'all_calls':total,'closeout_sha256':closeout['sha256']})


if __name__ == '__main__':
    main()
