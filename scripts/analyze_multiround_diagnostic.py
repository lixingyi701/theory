#!/usr/bin/env python3
"""Audit complete native memory chains and compare immutable multi-attempt forecasts."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_reflexion_precondition_forks import procedure_events
from audit_reflexion_action_shapes import action_shape
from prepare_multiround_diagnostic import initial_memory
from run_multiround_diagnostic import verify_attempt, verify_protocol


HISTORIES = ('single_error', 'repeated_error')
OPERATIONS = ('retain', 'correct_latest', 'delete_latest')
BOOTSTRAP_SEED = 2026102402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_digest(value, description):
    require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}),
            description+' integrity')


def known_labels(protocol):
    known = {digest(text): 'U' if key == 'N' else key
             for key, text in protocol['entries'].items() if key in {'E', 'C', 'N'}}
    for directory, prefix in [('reflexion-20261023', 'state-representation'),
                              ('reflexion-20261023', 'middle-context'),
                              ('reflexion-20261024', 'residual-context')]:
        record = read_json(Path('repro')/directory/(prefix+'-semantic-labels.json'))
        check_digest(record, 'Prior semantic labels')
        for entry in record['entries']:
            require(entry['label'] in {'E', 'C', 'U', 'A'}, 'Unknown prior semantic class')
            require(entry['sha256'] not in known or known[entry['sha256']] == entry['label'],
                    'Conflicting frozen semantic labels')
            known[entry['sha256']] = entry['label']
    return known


def expected_trajectories(protocol):
    trajectories = protocol['trajectories']
    tasks = {case['task_id'] for case in protocol['cases']}
    keys = {(t['task_id'], t['history_key'], t['operation']) for t in trajectories}
    require(len(tasks) == 12 and len(trajectories) == len({t['id'] for t in trajectories}) == 72,
            'Expected 72 distinct trajectories on twelve base tasks')
    require(keys == {(task, history, operation) for task in tasks
                     for history in HISTORIES for operation in OPERATIONS}, 'Incomplete trajectory matrix')
    require(all(t['seed'] == 2103 and t['max_attempts'] == 3 for t in trajectories),
            'This campaign fixes seed 2103 and at most three attempts')
    return trajectories


def error_origins(trajectory):
    """Original entry indices remain auditable after one initial intervention."""
    before = list(trajectory['initial_history'])
    origins = list(range(len(before)))
    operation = trajectory['operation']
    if operation != 'retain':
        index = max(i for i, label in enumerate(before) if label == 'E')
        if operation == 'correct_latest':
            before[index] = 'C'
        else:
            del before[index]
            del origins[index]
    require(before == trajectory['initial_labels'], 'Intervened labels differ from starting history')
    return [(position, origin) for position, (origin, label) in enumerate(zip(origins, before)) if label == 'E']


def collect(protocol, root, label_map=None):
    """Validate contiguous committed attempts, reset identity and append-only memory.

    Missing trajectories remain incomplete. A result file alone is never a completed
    trajectory: all identities, checkpoints and preceding memory states are checked.
    """
    trajectories = expected_trajectories(protocol)
    cases = {case['task_id']: case for case in protocol['cases']}
    expected = {t['id'] for t in trajectories}
    for path in root.glob('trajectories/*/trial-*/result.json'):
        require(path.parent.parent.name in expected and path.parent.name in {'trial-01', 'trial-02', 'trial-03'},
                'Unexpected trajectory or attempt')
    frozen = known_labels(protocol)
    labels = dict(frozen)
    if label_map:
        for sha, label in label_map.items():
            require(sha not in labels or labels[sha] == label, 'New labels overwrite frozen evidence')
            labels[sha] = label
    rows, completed, review = [], [], {}
    for trajectory in sorted(trajectories, key=lambda value: value['id']):
        memory = initial_memory(protocol, trajectory)
        require(digest(memory) == trajectory['initial_memory_sha256'], 'Initial memory binding differs')
        require(len(memory) == len(trajectory['initial_labels']), 'Initial label/memory length differs')
        origins = error_origins(trajectory)
        attempts, stopped = [], False
        case = cases[trajectory['task_id']]
        for trial in range(1, 4):
            directory = root/'trajectories'/trajectory['id']/f'trial-{trial:02d}'
            path = directory/'result.json'
            if not path.exists():
                require(not any((root/'trajectories'/trajectory['id']/f'trial-{later:02d}'/'result.json').exists()
                                for later in range(trial+1, 4)), 'Attempt chain has a gap')
                break
            require(not stopped, 'Another attempt was executed after success')
            result = read_json(path)
            verify_attempt(result, directory, case, trajectory, memory, trial)
            identity = read_json(directory/'identity.json')
            check_digest(identity, 'Attempt identity')
            require(identity['memory'] == memory == result['memory_before'], 'Growing memory chain differs')
            require(identity['task'] == case['task'] and identity['seed'] == trajectory['seed'], 'Task/seed mismatch')
            require(identity['trial'] == result['trial'] == trial and
                    result['identity_sha256'] == identity['sha256'], 'Attempt identity binding differs')
            require(result['task_id'] == trajectory['task_id'], 'Result task differs')
            require(result['initial_observation_sha256'] == case['initial_observation_sha256'], 'Environment reset differs')
            require(isinstance(result['success'], bool), 'Success must be a boolean')
            checkpoint = read_json(directory/'checkpoint.json')
            steps = checkpoint['transitions']
            require(digest(steps) == result['trajectory_sha256'] and len(steps) == result['decisions'],
                    'Checkpoint trajectory binding differs')
            after = result['memory_after']
            require(after[:len(memory)] == memory, 'Existing reflection memory was modified')
            fresh = after[len(memory):]
            require(len(fresh) == int(not result['success']) and all(isinstance(text, str) for text in after),
                    'Failure must append exactly one reflection; success must append none')
            for text in fresh:
                sha = digest(text)
                if sha not in frozen:
                    review.setdefault(sha, {'sha256': sha, 'text': text, 'sources': []})['sources'].append(
                        {'trajectory_id': trajectory['id'], 'trial': trial})
            visible = memory[-3:]
            symbols = [labels.get(digest(text), 'A') for text in visible]
            status = ('definite_error' if 'E' in symbols else
                      'ambiguous' if 'A' in symbols else 'definite_absent')
            goal = Path(case['task']['gamefile']).parts[-3].split('-')[1].lower()
            events = procedure_events(steps, goal, 'heat', 'microwave 1')
            row = {'trajectory_id': trajectory['id'], 'task_id': trajectory['task_id'],
                'seed': trajectory['seed'], 'history_key': trajectory['history_key'], 'operation': trajectory['operation'],
                'trial': trial, 'success': result['success'], 'status': result['status'], 'decisions': result['decisions'],
                'result_sha256': file_hash(path), 'identity_sha256': identity['sha256'],
                'trajectory_sha256': result['trajectory_sha256'], 'memory_sha256': digest(memory),
                'visible_memory_sha256': digest(visible), 'visible_reflection_hashes': [digest(text) for text in visible],
                'visible_memory_symbols': symbols, 'errors_visible_status': status,
                'memory_entries_before': len(memory), 'memory_entries_after': len(after),
                'new_reflection_hashes': [digest(text) for text in fresh],
                'original_error_entry_indices_visible_before': [origin for position, origin in origins if position >= max(0, len(memory)-3)],
                'original_error_entry_indices_visible_after': [origin for position, origin in origins if position >= max(0, len(after)-3)],
                'observable_events': {key: value if isinstance(value, bool) else len(value) for key, value in events.items()},
                'action_shapes': dict(Counter(action_shape(step['action'], step['submitted_action']) for step in steps)),
                'no_effect_shapes': dict(Counter(action_shape(step['action'], step['submitted_action']) for step in steps
                                                  if step['observation'] == 'Nothing happens.'))}
            if result['success']:
                row['outcome'] = 'S'
                row['new_E_event'] = None
            else:
                row['outcome'] = labels.get(digest(fresh[0]), 'A')
                row['new_E_event'] = ({'definite_error': 'restatement', 'definite_absent': 'generation',
                                       'ambiguous': 'unresolved'}[status] if row['outcome'] == 'E' else None)
            rows.append(row)
            attempts.append(row)
            stopped = result['success']
            memory = list(after)
        completed.append({'trajectory_id': trajectory['id'], 'task_id': trajectory['task_id'],
            'seed': trajectory['seed'], 'history_key': trajectory['history_key'], 'operation': trajectory['operation'],
            'initial_labels': trajectory['initial_labels'], 'initial_memory_sha256': trajectory['initial_memory_sha256'],
            'complete': bool(stopped or len(attempts) == 3), 'success': stopped,
            'success_trial': next((row['trial'] for row in attempts if row['success']), None),
            'attempts_completed': len(attempts), 'result_file_sha256s': [row['result_sha256'] for row in attempts]})
    return rows, completed, review


def interval(values, resamples):
    array = np.asarray(values, dtype=float)
    return {'mean': float(array.mean()),
            'task_bootstrap_95_interval': np.quantile(array[resamples].mean(axis=1), [.025, .975]).tolist()}


def signed_distance(observed, bounds):
    lower, upper = bounds
    return observed-lower if observed < lower else observed-upper if observed > upper else 0.


def forecast_curves(protocol, trajectories, tasks, resamples):
    forecasts = {row['trajectory_id']: row for row in protocol['forecasts']}
    require(len(forecasts) == 72 and set(forecasts) == {row['trajectory_id'] for row in trajectories}, 'Forecast coverage differs')
    lookup = {(row['task_id'], row['history_key'], row['operation']): row for row in trajectories}
    curves, contrasts = [], []
    for history in HISTORIES:
        for operation in OPERATIONS:
            for budget in range(1, 4):
                successes, probabilities = [], []
                for task in tasks:
                    row = lookup[task, history, operation]
                    successes.append(int(row['success_trial'] is not None and row['success_trial'] <= budget))
                    forecast = forecasts[row['trajectory_id']]
                    require(forecast['task_id'] == task and forecast['history_key'] == history and forecast['operation'] == operation,
                            'Forecast trajectory metadata differs')
                    candidates = [value for value in forecast['budgets'] if value['attempt_budget'] == budget]
                    require(len(candidates) == 1, 'Budget forecast absent or duplicated')
                    bounds = candidates[0]['success_probability_bounds']
                    require(len(bounds) == 2 and 0 <= bounds[0] <= bounds[1] <= 1, 'Invalid candidate probability bounds')
                    probabilities.append(bounds)
                prediction = np.asarray(probabilities).mean(axis=0).tolist()
                observed = interval(successes, resamples)
                curves.append({'history_key': history, 'operation': operation, 'attempt_budget': budget,
                    'successes': sum(successes), 'trajectories': len(tasks), 'observed_success_rate': observed['mean'],
                    'task_bootstrap_95_interval': observed['task_bootstrap_95_interval'],
                    'frozen_success_probability_bounds': prediction,
                    'empirical_minus_nearest_candidate_endpoint': signed_distance(observed['mean'], prediction),
                    'task_values': [{'task_id': task, 'success_by_budget': value} for task, value in zip(tasks, successes)]})
        for budget in range(1, 4):
            correct = next(row for row in curves if row['history_key'] == history and row['operation'] == 'correct_latest' and row['attempt_budget'] == budget)
            delete = next(row for row in curves if row['history_key'] == history and row['operation'] == 'delete_latest' and row['attempt_budget'] == budget)
            differences = [a['success_by_budget']-b['success_by_budget'] for a, b in zip(correct['task_values'], delete['task_values'])]
            cb, db = correct['frozen_success_probability_bounds'], delete['frozen_success_probability_bounds']
            prediction = [cb[0]-db[1], cb[1]-db[0]]
            observed = interval(differences, resamples)
            contrasts.append({'history_key': history, 'attempt_budget': budget,
                'contrast': 'correct_latest_minus_delete_latest', **observed,
                'frozen_contrast_bounds': prediction,
                'empirical_minus_nearest_candidate_endpoint': signed_distance(observed['mean'], prediction),
                'primary_directional_contrast': history == 'repeated_error' and budget == 2,
                'task_differences': [{'task_id': task, 'difference': value} for task, value in zip(tasks, differences)]})
    return {'curves': curves, 'correct_minus_delete': contrasts,
        'primary_directional_contrast': next(row for row in contrasts if row['primary_directional_contrast']),
        'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': len(resamples),
        'statistical_unit': 'Twelve base tasks; histories, operations and budgets from one task stay in its cluster',
        'candidate_bounds_scope': 'Semantic assignment of A under the frozen model; not a confidence interval or a parameter-error bound',
        'interpretation': 'The signed distance is a forecast diagnostic. A rate inside candidate bounds is not a validation pass.'}


def attempt_diagnostics(rows):
    source = read_json(Path('repro/reflexion-20261023/state-representation-presence-baseline.json'))
    kernel = {bool(labels): next(value['joint'] for value in source['one_attempt_forecasts'] if value['labels'] == labels)
              for labels in [[], ['E']]}
    diagnostics = []
    for trial in range(1, 4):
        for status in ('definite_error', 'definite_absent', 'ambiguous'):
            selected = [row for row in rows if row['trial'] == trial and row['errors_visible_status'] == status]
            if not selected:
                continue
            possible = ([kernel[True]['S']] if status == 'definite_error' else
                        [kernel[False]['S']] if status == 'definite_absent' else [kernel[True]['S'], kernel[False]['S']])
            failed = [row for row in selected if not row['success']]
            diagnostics.append({'trial': trial, 'errors_visible_status': status,
                'attempts': len(selected), 'base_tasks': len({row['task_id'] for row in selected}),
                'successes': sum(row['success'] for row in selected), 'observed_success_fraction': sum(row['success'] for row in selected)/len(selected),
                'frozen_single_attempt_success_bounds': [min(possible), max(possible)],
                'failed_attempts': len(failed), 'new_reflection_labels': dict(Counter(row['outcome'] for row in failed)),
                'new_E_events': dict(Counter(row['new_E_event'] for row in failed if row['outcome'] == 'E')),
                'scope': 'Initial assigned states' if trial == 1 else
                         'Failure-selected survivors with a changed task/memory mixture; descriptive diagnostic, not independent random assignment'})
    return {'by_trial_and_visible_error': diagnostics,
        'new_E_events': dict(Counter(row['new_E_event'] for row in rows if row['outcome'] == 'E')),
        'failed_reflection_labels': dict(Counter(row['outcome'] for row in rows if not row['success'])),
        'visible_labels_scope': 'Labels use only text present before that attempt. No later reflection or action becomes a starting prediction input.',
        'event_scope': 'Generation, restatement and ambiguous-background events concern newly written reflection text; actions and task success are distinct.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261024/multiround-diagnostic-protocol.json'))
    parser.add_argument('--export-review', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    rows, trajectories, review = collect(protocol, args.run)
    if args.export_review:
        write_json(args.export_review, {'protocol_sha256': protocol['sha256'], 'items': list(review.values())})
    if not args.labels:
        print({'completed_trajectories': sum(row['complete'] for row in trajectories), 'attempts_completed': len(rows),
               'unique_reflections_requiring_review': len(review)})
        return
    labels = read_json(args.labels)
    check_digest(labels, 'Semantic labels')
    require(labels['protocol_sha256'] == protocol['sha256'], 'Semantic labels belong to another protocol')
    require(len(labels['entries']) == len(review) and {entry['sha256'] for entry in labels['entries']} == set(review),
            'New-text semantic review coverage differs')
    require(all(entry['label'] in {'E', 'C', 'U', 'A'} for entry in labels['entries']), 'Unknown semantic class')
    labels_map = {entry['sha256']: entry['label'] for entry in labels['entries']}
    rows, trajectories, checked_review = collect(protocol, args.run, labels_map)
    require(checked_review == review and all(row['complete'] for row in trajectories), 'Complete analysis requires all 72 completed trajectories')
    require(len(trajectories) == 72, 'Expected 72 trajectories')
    summary = read_json(args.run/'summary.json')
    check_digest(summary, 'Run summary')
    require(summary['protocol_sha256'] == protocol['sha256'], 'Run summary protocol differs')
    tasks = sorted({row['task_id'] for row in trajectories})
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(tasks), size=(20000, len(tasks)))
    value = {'protocol_sha256': protocol['sha256'], 'summary_sha256': summary['sha256'],
        'semantic_labels_sha256': labels['sha256'], 'completed_trajectories': len(trajectories),
        'attempts_completed': len(rows), 'independent_base_tasks': len(tasks), 'seed_replicates_per_task': 1,
        'successes': sum(row['success'] for row in trajectories), 'rows': rows, 'trajectories': trajectories,
        'frozen_multiround_prediction': forecast_curves(protocol, trajectories, tasks, resamples),
        'attempt_diagnostics': attempt_diagnostics(rows), 'parameter_refits': 0,
        'holdout_model_calls': 0, 'semantic_review': {'reviewer': labels.get('reviewer'),
            'human_second_review': labels.get('human_second_review', False), 'ambiguous_class_retained': True},
        'analysis_driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value)
    write_json(args.run/'analysis.json', value)
    print({'completed_trajectories': len(trajectories), 'attempts_completed': len(rows),
        'successes': value['successes'], 'primary_directional_contrast': value['frozen_multiround_prediction']['primary_directional_contrast'],
        'analysis_sha256': value['sha256'], 'parameter_refits': 0})


if __name__ == '__main__':
    main()
