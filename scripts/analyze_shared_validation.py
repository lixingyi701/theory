#!/usr/bin/env python3
"""Offline interpretation and closeout; never fit parameters or call a model."""
from collections import Counter
from pathlib import Path
import statistics
import numpy as np

from agent_reliability.io import digest, file_hash, read_json
from continue_reflexion_pilot import ledger
from evaluate_shared_validation import interval
from prepare_shared_calibration import save, STATES
from prepare_shared_validation import PROTOCOL
from run_shared_validation import ROOT, verify_protocol
from shared_recovery_model import component_kernel, PRIOR
from summarize_reflexion import costs


def checked(path):
    value = read_json(path)
    assert value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}), path
    return value


def main():
    protocol = verify_protocol()
    pub = PROTOCOL.parent
    summary = checked(ROOT / 'summary.json')
    result = checked(pub / 'shared-validation-results.json')
    contract = checked(ROOT / 'contract.json')
    audit = checked(ROOT / 'protocol-audit.json')
    shapes = checked(ROOT / 'action-shapes.json')
    replicas = checked(ROOT / 'replica-audit.json')
    assert summary['status'] == 'complete'
    assert summary['completed_tasks'] == 54 and summary['completed_trajectories'] == 378
    assert summary['worker_exit_codes'] == [0] * 4
    assert summary['worker_statuses'] == ['complete'] * 4
    assert not (ROOT / 'STOP').exists()
    assert result['protocol_sha256'] == summary['protocol_sha256'] == protocol['sha256']
    assert result['parameters_sha256'] == protocol['parent_calibration_parameters_sha256']
    assert result['raw_summary_file_sha256'] == file_hash(ROOT / 'summary.json')
    assert result['evaluator_source_sha256'] == file_hash(Path('scripts/evaluate_shared_validation.py'))
    assert audit['passed'] and not audit['errors'] and replicas['passed']
    assert summary['elapsed_seconds'] <= protocol['limits']['seconds']
    assert contract['freeze_commit'] == 'a50ce8497b1602ee9e8cb3fb45119f07727b488e'

    rows = result['observed']
    scenes = sorted({r['scene_id'] for r in rows})
    by_task = {c['task_id']: c for c in protocol['cases']}
    assert Counter(r['task_id'] for r in rows) == dict.fromkeys(by_task, 7)
    assert Counter(r['scene_id'] for r in rows) == dict.fromkeys(scenes, 42)
    lookup = {(r['task_id'], r['condition']): r for r in rows}
    assert len(lookup) == 378
    assert all(not r['success_curve'][0] or r['success_curve'][1] for r in rows)

    # Recompute the registered score independently of score_rows.
    recomputed = {}
    for name, forecasts in protocol['forecasts'].items():
        for b in (1, 2):
            scene_loss = [statistics.mean(
                (forecasts[r['condition']][b - 1] - r['success_curve'][b - 1]) ** 2
                for r in rows if r['scene_id'] == scene) for scene in scenes]
            original = next(m for m in result['metrics'] if m['model'] == name and m['budget'] == b)
            assert abs(statistics.mean(scene_loss) - original['mean_brier']) < 1e-12
            recomputed[name, b] = scene_loss
    primary_values = [a - b for a, b in zip(recomputed['constant_persistent', 2], recomputed['selected', 2])]
    estimate = statistics.mean(primary_values)
    half = 2.306004135 * statistics.stdev(primary_values) / 3
    assert abs(estimate - result['primary']['mean']) < 1e-12
    assert np.allclose([estimate - half, estimate + half], result['primary']['student_t95'], atol=1e-12)

    per_condition = {}
    for condition in STATES:
        subset = [r for r in rows if r['condition'] == condition]
        first, two = [sum(r['success_curve'][b] for r in subset) for b in (0, 1)]
        p1, p2 = protocol['forecasts']['selected'][condition]
        per_condition[condition] = {
            'tasks': len(subset), 'first_successes': first, 'two_successes': two,
            'observed_success': [first / len(subset), two / len(subset)],
            'frozen_forecast': [p1, p2],
            'second_attempt_successes': two - first, 'second_attempts': len(subset) - first,
            'observed_second_given_first_failure': (two - first) / (len(subset) - first),
            'frozen_second_given_first_failure': (p2 - p1) / (1 - p1),
        }

    comparisons = [
        ('correct_middle', 'error_middle'), ('neutral_full', 'error_middle'),
        ('correct_middle', 'neutral_full'), ('correct_latest', 'delete_latest'),
        ('correct_latest', 'correct_oldest'), ('correct_latest', 'repeated_error'),
        ('delete_latest', 'repeated_error'),
    ]
    contrasts = []
    for a, b in comparisons:
        for budget in (1, 2):
            values = [statistics.mean(
                int(lookup[t, a]['success_curve'][budget - 1]) - int(lookup[t, b]['success_curve'][budget - 1])
                for t in by_task if by_task[t]['task']['scene_id'] == scene) for scene in scenes]
            contrasts.append({'a': a, 'b': b, 'budget': budget, **interval(values)})

    truncated = []
    for path in sorted(ROOT.glob('**/calls/*/validation.json')):
        if read_json(path).get('truncated_output'):
            parts = path.relative_to(ROOT).parts
            trajectory = parts[1]
            row = next(r for r in rows if r['trajectory_id'] == trajectory)
            truncated.append({'path': path.relative_to(ROOT).as_posix(), 'trajectory_id': trajectory,
                              'scene_id': row['scene_id'], 'task_id': row['task_id'],
                              'condition': row['condition'], 'role': parts[-2].split('-')[0]})
    affected_scenes = {r['scene_id'] for r in truncated}
    retained = [s for s in scenes if s not in affected_scenes]
    assert retained
    sensitivity = {
        'scope': 'Prespecified affected-scene exclusion; descriptive point estimates, full sample remains primary',
        'excluded_scenes': sorted(affected_scenes), 'scenes': len(retained), 'tasks': len(retained) * 6,
        'trajectories': len(retained) * 42,
        'two_budget_brier_gain': statistics.mean(primary_values[scenes.index(s)] for s in retained),
    }

    # Added after collection: diagnose the update model without estimating any coefficient.
    # Preserve the SAME persistent task offsets; hold initial memory fixed at the second attempt.
    static_forecasts = {}
    for condition, labels in STATES.items():
        state = [{'type': k, 'native': False} for k in labels]
        p = component_kernel(protocol['models']['selected'], state)[:, 0]
        static_forecasts[condition] = float(PRIOR @ (1 - (1 - p) ** 2))
    static_losses = [statistics.mean(
        (static_forecasts[r['condition']] - r['success_curve'][1]) ** 2
        for r in rows if r['scene_id'] == scene) for scene in scenes]
    update_diagnostic = {
        'scope': 'Post-collection exploratory diagnostic; fixed coefficients, unchanged initial memory on repeat, same persistent task mixture',
        'forecasts_by_condition': static_forecasts,
        'unchanged_memory_brier': statistics.mean(static_losses),
        'full_model_brier': statistics.mean(recomputed['selected', 2]),
        'gain_full_over_unchanged': interval([a - b for a, b in zip(static_losses, recomputed['selected', 2])]),
        'parameter_refits': 0,
    }

    shape_rows = shapes['review_items']
    shape_by_condition = {}
    trajectory_conditions = {r['trajectory_id']: r['condition'] for r in rows}
    for condition in STATES:
        subset = [r for r in shape_rows if trajectory_conditions[Path(r['checkpoint']).parts[1]] == condition]
        shape_by_condition[condition] = {
            'counts': dict(Counter(r['shape'] for r in subset)),
            'trajectories_with_review_items': len({Path(r['checkpoint']).parts[1] for r in subset}),
        }

    report = save(pub / 'shared-validation-analysis.json', {
        'scope': 'Offline interpretation of frozen prospective validation; secondary intervals exploratory and unadjusted',
        'protocol_sha256': protocol['sha256'], 'frozen_results_sha256': result['sha256'],
        'raw_summary_file_sha256': file_hash(ROOT / 'summary.json'),
        'analysis_source_sha256': file_hash(Path(__file__)),
        'primary_recomputed': {'mean': estimate, 'student_t95': [estimate - half, estimate + half]},
        'relative_brier_reduction': estimate / statistics.mean(recomputed['constant_persistent', 2]),
        'scenes_with_positive_primary_gain': sum(v > 0 for v in primary_values),
        'per_condition': per_condition, 'descriptive_contrasts': contrasts,
        'truncations': truncated, 'truncation_sensitivity': sensitivity,
        'exploratory_unchanged_memory': update_diagnostic,
        'action_shape_review_by_condition': shape_by_condition,
        'parameter_refits': 0, 'semantic_labels_used': False,
    })

    # Publish compact typed events and an explicit projection, retaining the complete raw summary privately.
    public_rows = []
    for row in summary['attempts_rows']:
        path = Path(row['result_path'])
        public_rows.append({**row, 'result_path': path.relative_to(ROOT).as_posix()})
    events = save(pub / 'shared-validation-events.json', {
        'scope': 'Complete typed attempt records without raw model texts; result hashes bind private raw records',
        'protocol_sha256': protocol['sha256'], 'raw_summary_file_sha256': file_hash(ROOT / 'summary.json'),
        'rows': public_rows,
    })
    projection = {k: v for k, v in summary.items() if k not in ['sha256', 'attempts_rows', 'trajectories']}
    collection = save(pub / 'shared-validation-collection-summary.json', {
        **projection, 'scope': 'Aggregate projection of the complete private collection summary',
        'raw_summary_file_sha256': file_hash(ROOT / 'summary.json'), 'events_sha256': events['sha256'],
    })

    actual = costs(ROOT)
    reserved = read_json(ROOT / 'budget.json')['reserved']
    assert actual == summary['costs_incremental']
    assert actual['dispatched_calls'] == actual['saved_responses'] == reserved == 29159
    assert reserved <= contract['max_calls'] and not actual['uncertain_or_unrecorded_attempts']
    assert len(truncated) == actual['truncated_responses'] == 6
    all_runs = ledger()
    all_calls = sum(v['dispatched'] for v in all_runs.values())
    all_responses = sum(v['responses'] for v in all_runs.values())
    assert all_calls == all_responses == 93186
    holdout = {t['task_id'] for t in read_json(Path('repro/baseline-20261019/holdout134.json'))['tasks']}
    used = set()
    for path in ROOT.glob('**/identity.json'):
        task_id = read_json(path).get('task', {}).get('task_id')
        if task_id:
            used.add(task_id)
    assert used == set(by_task) and not used & holdout
    names = ['shared-validation-' + n + '.json' for n in [
        'results', 'analysis', 'events', 'collection-summary', 'protocol-audit', 'action-shapes', 'replica-audit']]
    # The three audits are saved by the master under the private run root; mirror their exact bytes.
    for private, public in [('protocol-audit', 'protocol-audit'), ('action-shapes', 'action-shapes'), ('replica-audit', 'replica-audit')]:
        (pub / ('shared-validation-' + public + '.json')).write_bytes((ROOT / (private + '.json')).read_bytes())
    closeout = save(pub / 'shared-validation-closeout.json', {
        'scope': 'Completed prospective frozen-parameter validation',
        'protocol_sha256': protocol['sha256'], 'sampling_freeze': contract['freeze_commit'],
        'parameters_sha256': result['parameters_sha256'],
        'tasks': 54, 'scenes': 9, 'trajectories': 378, 'attempts': 700,
        'first_successes': sum(r['success_curve'][0] for r in rows),
        'successes_by_two': sum(r['success_curve'][1] for r in rows),
        'costs_incremental': actual, 'collection_elapsed_seconds': summary['elapsed_seconds'],
        'all_calls': all_calls, 'all_responses': all_responses, 'allowance_status': 'closed',
        'parameter_refits': 0, 'holdout134_model_calls': 0,
        'validation_semantic_annotation': 'not performed; not required for primary evaluation',
        'contract_sha256': contract['sha256'], 'raw_summary_file_sha256': file_hash(ROOT / 'summary.json'),
        'artifacts': {n: file_hash(pub / n) for n in names},
        'closeout_source_sha256': file_hash(Path(__file__)),
    })
    print({'primary': report['primary_recomputed'], 'per_condition': per_condition,
           'truncation_sensitivity': sensitivity, 'exploratory_unchanged_memory': update_diagnostic,
           'descriptive_contrasts': contrasts, 'closeout_sha256': closeout['sha256']})


if __name__ == '__main__':
    main()
