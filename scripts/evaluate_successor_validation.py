#!/usr/bin/env python3
"""Frozen scene-weighted scoring, with native outcomes and full memory-chain checks."""
from collections import Counter
from pathlib import Path
import numpy as np
from agent_reliability.io import digest, file_hash, read_json
from prepare_shared_calibration import save
from prepare_successor_protocol import PROTOCOL, ROOT, PRIMARY, SECONDARY
from run_successor_validation import verify_protocol
from run_multiround_diagnostic import verify_attempt


def interval(values):
    a = np.asarray(values, float)
    if len(a) != 25:
        raise ValueError('25 paired scene values required')
    half = 2.063898562 * float(a.std(ddof=1)) / np.sqrt(25)
    rng = np.random.default_rng(2026102603)
    indices = rng.integers(0, 25, (20000, 25))
    return {'mean': float(a.mean()), 'student_t95': [float(a.mean()-half), float(a.mean()+half)],
        'scene_bootstrap95': np.quantile(a[indices].mean(axis=1), [.025, .975]).tolist(),
        'scene_values': a.tolist()}


def check_observed(protocol, observed):
    expected = {t['id']: t for t in protocol['trajectories']}
    if len(expected) != 550 or len(observed) != 550 or len({r['trajectory_id'] for r in observed}) != 550:
        raise ValueError('Complete550-trajectory matrix required')
    cases = {c['task_id']: c for c in protocol['cases']}
    seen = set()
    for row in observed:
        trajectory = expected.get(row['trajectory_id'])
        if trajectory is None or any(row[k] != trajectory[k] for k in ['task_id', 'condition', 'seed', 'panel']):
            raise ValueError('Observed matrix identity mismatch')
        if row['scene_id'] != cases[row['task_id']]['task']['scene_id']:
            raise ValueError('Observed scene mismatch')
        curve = row['success_curve']
        if len(curve) != 4 or any(type(v) is not bool for v in curve) or any(a and not b for a, b in zip(curve, curve[1:])):
            raise ValueError('Invalid four-budget success curve')
        seen.add(row['trajectory_id'])
    if seen != set(expected):
        raise ValueError('Incomplete observed panel')


def scene_means(rows, value):
    output = []
    for scene in sorted({r['scene_id'] for r in rows}):
        subset = [r for r in rows if r['scene_id'] == scene]
        task_values = []
        for task in sorted({r['task_id'] for r in subset}):
            task_values.append(float(np.mean([value(r) for r in subset if r['task_id'] == task])))
        output.append({'scene_id': scene, 'value': float(np.mean(task_values))})
    return output


def score_rows(protocol, observed):
    check_observed(protocol, observed)
    primary_rows = [r for r in observed if r['panel'] == 'primary']
    if len(primary_rows) != 400 or set(r['condition'] for r in primary_rows) != set(PRIMARY):
        raise ValueError('Primary400-trajectory panel required')
    metrics = []
    for name, forecasts in protocol['forecasts'].items():
        for budget in range(1, 5):
            means = scene_means(primary_rows, lambda r: (forecasts[r['condition']][budget-1]-r['success_curve'][budget-1])**2)
            metrics.append({'model': name, 'budget': budget, 'mean_brier': float(np.mean([m['value'] for m in means])), 'scene_scores': means})
    contrasts = []
    for baseline in ['unchanged_seven', 'full_six']:
        for budget in range(1, 5):
            a = next(m for m in metrics if m['model'] == baseline and m['budget'] == budget)
            b = next(m for m in metrics if m['model'] == 'full_seven' and m['budget'] == budget)
            contrasts.append({'baseline': baseline, 'budget': budget,
                **interval([x['value']-y['value'] for x, y in zip(a['scene_scores'], b['scene_scores'])])})
    primary = next(c for c in contrasts if c['baseline'] == 'unchanged_seven' and c['budget'] == 4)
    lo, hi = primary['student_t95']
    decision = 'full_seven_favored' if lo > 0 else ('unchanged_seven_favored' if hi < 0 else 'inconclusive')
    recovery = []
    for budget in range(1, 5):
        for a, b in [('correct_latest', 'delete_latest'), ('correct_latest', 'repeated_error'), ('delete_latest', 'repeated_error')]:
            # One fixed task per scene, equal two crossed seeds in each condition.
            means = {}
            for condition in [a, b]:
                means[condition] = scene_means([r for r in observed if r['condition'] == condition], lambda r: float(r['success_curve'][budget-1]))
            values = [x['value']-y['value'] for x, y in zip(means[a], means[b])]
            recovery.append({'contrast': a+' minus '+b, 'budget': budget, **interval(values)})
    return {'metrics': metrics, 'contrasts': contrasts, 'primary': primary, 'primary_decision': decision,
        'recovery_contrasts_secondary': recovery,
        'per_condition': {c: {'trajectories': sum(r['condition'] == c for r in observed),
            'success_counts': [sum(r['success_curve'][b] for r in observed if r['condition'] == c) for b in range(4)],
            'scene_equal_success': [float(np.mean([m['value'] for m in scene_means([r for r in observed if r['condition'] == c], lambda r: float(r['success_curve'][b]))])) for b in range(4)]}
            for c in PRIMARY + SECONDARY}}


def main():
    protocol = verify_protocol()
    summary = read_json(ROOT / 'summary.json')
    if summary['sha256'] != digest({k: v for k, v in summary.items() if k != 'sha256'}) or summary['status'] != 'complete':
        raise ValueError('Complete unchanged summary required')
    if summary['worker_exit_codes'] != [0, 0] or summary['completed_trajectories'] != 550:
        raise ValueError('Incomplete workers')
    cases = {c['task_id']: c for c in protocol['cases']}
    expected_ids = {t['id'] for t in protocol['trajectories']}
    all_rows = summary['attempts_rows']
    if any(r['trajectory_id'] not in expected_ids for r in all_rows):
        raise ValueError('Unexpected raw trajectory')
    rows, events = [], []
    for trajectory in protocol['trajectories']:
        initial = [protocol['entries'][k] for k in trajectory['initial_labels']]
        memory = list(initial)
        sequence = sorted([r for r in all_rows if r['trajectory_id'] == trajectory['id']], key=lambda r: r['trial'])
        if [r['trial'] for r in sequence] != list(range(1, len(sequence)+1)) or not 1 <= len(sequence) <= 4:
            raise ValueError('Attempt sequence')
        for row in sequence:
            path = Path(row['result_path'])
            result = read_json(path)
            if file_hash(path) != row['result_file_sha256']:
                raise ValueError('Raw result file changed')
            verify_attempt(result, path.parent, cases[trajectory['task_id']], trajectory, memory, row['trial'])
            checkpoint = read_json(path.parent / 'checkpoint.json')
            native_won = bool(checkpoint['transitions'] and checkpoint['transitions'][-1]['won'])
            expected_new = [digest(s) for s in result['memory_after'][len(memory):]]
            if (native_won != result['success'] or row['success'] != result['success'] or
                row['memory_before_sha256'] != digest(memory) or row['visible_memory_sha256'] != digest(memory[-3:]) or
                row['new_reflection_sha256'] != expected_new or row['identity_sha256'] != result['identity_sha256'] or
                row['trajectory_sha256'] != result['trajectory_sha256'] or row['seed'] != trajectory['seed']):
                raise ValueError('Native outcome or memory-chain mismatch')
            original_visible = [i for i in range(len(initial)) if i >= len(memory)-3]
            events.append({'trajectory_id': trajectory['id'], 'task_id': trajectory['task_id'],
                'condition': trajectory['condition'], 'seed': trajectory['seed'], 'trial': row['trial'],
                'success': row['success'], 'visible_memory_sha256': row['visible_memory_sha256'],
                'original_entry_indices_visible': original_visible,
                'original_entry_labels_visible': [trajectory['initial_labels'][i] for i in original_visible],
                'new_reflection_sha256': expected_new, 'result_file_sha256': row['result_file_sha256']})
            memory = result['memory_after']
        if len(sequence) < 4 and not sequence[-1]['success']:
            raise ValueError('Early failed stop')
        if any(r['success'] for r in sequence[:-1]):
            raise ValueError('Continued after success')
        rows.append({'trajectory_id': trajectory['id'], 'task_id': trajectory['task_id'],
            'scene_id': cases[trajectory['task_id']]['task']['scene_id'], 'condition': trajectory['condition'],
            'seed': trajectory['seed'], 'panel': trajectory['panel'],
            'success_curve': [any(r['success'] for r in sequence if r['trial'] <= b) for b in range(1, 5)],
            'attempts_completed': len(sequence), 'result_hashes': [r['result_file_sha256'] for r in sequence]})
    if sum(r['attempts_completed'] for r in rows) != len(all_rows):
        raise ValueError('Attempt accounting mismatch')
    scores = score_rows(protocol, rows)
    report = save(PROTOCOL.with_name('successor-validation-results-20261026.json'), {
        'scope': 'Prospective four-attempt native-Reflexion prediction, complete fresh-task matrix',
        'protocol_sha256': protocol['sha256'], 'parameters_sha256': protocol['parent_calibration_parameters_sha256'],
        'raw_summary_file_sha256': file_hash(ROOT / 'summary.json'), 'evaluator_source_sha256': file_hash(Path(__file__)),
        'tasks': 100, 'scenes': 25, 'trajectories': 550, 'observed': rows, 'events': events, **scores,
        'parameter_refits': 0, 'holdout134_model_calls': 0, 'semantic_annotation_required_for_primary': False})
    print({'sha256': report['sha256'], 'primary': report['primary'], 'decision': report['primary_decision']})


if __name__ == '__main__':
    main()
