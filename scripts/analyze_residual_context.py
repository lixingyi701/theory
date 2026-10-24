#!/usr/bin/env python3
"""Analyse the frozen residual-context factorial without fitting new parameters."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_reflexion_precondition_forks import procedure_events
from audit_reflexion_action_shapes import action_shape
from prepare_residual_context import memory_for
from run_residual_context import verify_protocol
from state_representation_model import CLASSES, scores


CONDITIONS = ('N_E', 'N_C', 'R_E', 'R_C')
SEEDS = (2101, 2102)
MODELS = ('presence', 'constant', 'mixture_joint')
BOOTSTRAP_SEED = 2026102401


def check_digest(value, description):
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(description+' integrity')


def expected_cells(protocol):
    """Attach block metadata once, retaining the immutable reused-result bindings."""
    cells = []
    for block in protocol['blocks']:
        if block['attempts'] != 1:
            raise ValueError('This analysis requires one attempt per factorial cell')
        for cell in block['cells']:
            row = dict(cell)
            for key in ('task_id', 'seed'):
                if key in row and row[key] != block[key]:
                    raise ValueError('Cell/block '+key+' mismatch')
                row[key] = block[key]
            row['reused'] = False
            cells.append(row)
    for cell in protocol['reused_controls']:
        row = dict(cell)
        row['reused'] = True
        cells.append(row)
    if len(cells) != 96 or len({c['id'] for c in cells}) != 96:
        raise ValueError('Expected 96 distinct factorial cells')
    if sum(c['reused'] for c in cells) != 24:
        raise ValueError('Expected 24 frozen reused controls')
    tasks = {c['task_id'] for c in cells}
    keys = {(c['task_id'], c['seed'], c['condition']) for c in cells}
    if len(tasks) != 12 or keys != {(t, s, c) for t in tasks for s in SEEDS for c in CONDITIONS}:
        raise ValueError('Incomplete or duplicate 12-task, two-seed factorial definition')
    if any(c['reused'] and (c['seed'] != 2101 or c['condition'] not in ('N_E', 'N_C')) for c in cells):
        raise ValueError('Only the preregistered seed-2101 generic controls may be reused')
    return cells


def known_labels(protocol):
    known = {digest(text): ('U' if name == 'N' else name)
             for name, text in protocol['entries'].items() if name in {'E', 'C', 'N'}}
    for name in ('state-representation', 'middle-context'):
        record = read_json(Path('repro/reflexion-20261023')/(name+'-semantic-labels.json'))
        check_digest(record, 'Prior semantic labels')
        for entry in record['entries']:
            if entry['sha256'] in known and known[entry['sha256']] != entry['label']:
                raise ValueError('Conflicting frozen semantic labels')
            known[entry['sha256']] = entry['label']
    return known


def collect(protocol, root):
    """Validate committed attempts; missing new cells remain explicitly incomplete."""
    cells = expected_cells(protocol)
    cases = {c['task_id']: c for c in protocol['cases']}
    if len(cases) != 12 or set(cases) != {c['task_id'] for c in cells}:
        raise ValueError('Case registry does not match factorial tasks')
    new_ids = {c['id'] for c in cells if not c['reused']}
    observed_paths = list(root.glob('cells/*/trial-*/result.json'))
    if any(p.parent.parent.name not in new_ids or p.parent.name != 'trial-01' for p in observed_paths):
        raise ValueError('Unexpected cell or trial in this single-attempt campaign')
    known, rows, review = known_labels(protocol), [], {}
    for cell in sorted(cells, key=lambda c: c['id']):
        directory = ((Path(cell['source_run'])/'cells'/cell['source_cell_id']/'trial-01')
                     if cell['reused'] else root/'cells'/cell['id']/'trial-01')
        path = directory/'result.json'
        if not path.exists():
            if cell['reused']:
                raise FileNotFoundError('Frozen reused control is missing: '+str(path))
            continue
        value, identity = read_json(path), read_json(directory/'identity.json')
        check_digest(identity, 'Attempt identity')
        if cell['reused']:
            if file_hash(path) != cell['result_sha256'] or identity['sha256'] != cell['identity_sha256']:
                raise ValueError('Reused control no longer matches its frozen binding')
            if 'identity_file_sha256' in cell and file_hash(directory/'identity.json') != cell['identity_file_sha256']:
                raise ValueError('Reused identity file changed')
        memory = memory_for(protocol, cell)
        if len(memory) != 3 or digest(memory) != cell['memory_sha256']:
            raise ValueError('Starting memory differs from the frozen three-entry condition')
        case = cases[cell['task_id']]
        if identity['memory'] != memory or value['memory_before'] != memory:
            raise ValueError('Actual starting memory differs from the factorial intervention')
        if identity['task'] != case['task'] or identity['seed'] != cell['seed']:
            raise ValueError('Task or seed mismatch')
        if identity['trial'] != 1 or value['trial'] != 1 or value['identity_sha256'] != identity['sha256']:
            raise ValueError('Attempt identity/trial mismatch')
        if value.get('task_id', cell['task_id']) != cell['task_id']:
            raise ValueError('Result task mismatch')
        if value['initial_observation_sha256'] != case['initial_observation_sha256']:
            raise ValueError('Environment reset observation changed')
        checkpoint = read_json(directory/'checkpoint.json')
        steps = checkpoint['transitions']
        if digest(steps) != value['trajectory_sha256']:
            raise ValueError('Trajectory integrity')
        fresh = value['memory_after'][len(memory):]
        if value['memory_after'][:len(memory)] != memory or len(fresh) != int(not value['success']):
            raise ValueError('Failure must append exactly one native reflection; success must append none')
        for text in fresh:
            sha = digest(text)
            if sha not in known:
                review.setdefault(sha, {'sha256': sha, 'text': text, 'sources': []})['sources'].append(cell['id'])
        goal = Path(case['task']['gamefile']).parts[-3].split('-')[1].lower()
        events = procedure_events(steps, goal, 'heat', 'microwave 1')
        rows.append({'cell_id': cell['id'], 'task_id': cell['task_id'], 'seed': cell['seed'],
            'trial': 1, 'condition': cell['condition'], 'initial_labels': cell['labels'],
            'initial_target_rule': cell['condition'].split('_')[1], 'reused_control': cell['reused'],
            'source_cell_id': cell['source_cell_id'] if cell['reused'] else cell['id'],
            'source_run': cell['source_run'] if cell['reused'] else str(root),
            'success': value['success'], 'decisions': value['decisions'], 'status': value['status'],
            'result_sha256': file_hash(path), 'identity_sha256': identity['sha256'],
            'trajectory_sha256': value['trajectory_sha256'], 'memory_sha256': digest(memory),
            'new_reflection_hashes': [digest(text) for text in fresh],
            'observable_events': {k: v if isinstance(v, bool) else len(v) for k, v in events.items()},
            'action_shapes': dict(Counter(action_shape(s['action'], s['submitted_action']) for s in steps)),
            'no_effect_shapes': dict(Counter(action_shape(s['action'], s['submitted_action'])
                for s in steps if s['observation'] == 'Nothing happens.'))})
    if len({r['cell_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate committed cells')
    return rows, known, review


def interval(values, resamples):
    array = np.asarray(values, dtype=float)
    return {'mean': float(array.mean()),
            'task_bootstrap_95_interval': np.quantile(array[resamples].mean(axis=1), [.025, .975]).tolist()}


def effect_values(success):
    """Signs are fixed: residual minus generic, and corrected minus erroneous."""
    return {'R_vs_N_under_E': success['R_E']-success['N_E'],
        'R_vs_N_under_C': success['R_C']-success['N_C'],
        'correction_under_N': success['N_C']-success['N_E'],
        'correction_under_R': success['R_C']-success['R_E'],
        'interaction': (success['R_C']-success['R_E'])-(success['N_C']-success['N_E'])}


def paired_counts(a, b):
    return dict(Counter('both_success' if x and y else 'first_only_success' if x
                        else 'second_only_success' if y else 'both_failure' for x, y in zip(a, b)))


def factorial_effects(rows, tasks, resamples):
    lookup = {(r['task_id'], r['seed'], r['condition']): int(r['success']) for r in rows}
    if len(lookup) != 96:
        raise ValueError('Paired analysis requires the complete factorial')
    task_effects = []
    for task in tasks:
        success = {c: float(np.mean([lookup[task, seed, c] for seed in SEEDS])) for c in CONDITIONS}
        task_effects.append({'task_id': task, 'success_by_condition': success, **effect_values(success)})
    estimates = {name: interval([r[name] for r in task_effects], resamples)
                 for name in effect_values({c: 0. for c in CONDITIONS})}
    decompositions = []
    comparisons = [('R_vs_N_under_E', 'R_E', 'N_E'), ('R_vs_N_under_C', 'R_C', 'N_C'),
                   ('correction_under_N', 'N_C', 'N_E'), ('correction_under_R', 'R_C', 'R_E')]
    for seed in SEEDS:
        per_task = [effect_values({c: lookup[t, seed, c] for c in CONDITIONS}) for t in tasks]
        decompositions.append({'seed': seed, 'independent_base_tasks': len(tasks),
            'successes_by_condition': {c: sum(lookup[t, seed, c] for t in tasks) for c in CONDITIONS},
            'success_rates_by_condition': {c: float(np.mean([lookup[t, seed, c] for t in tasks])) for c in CONDITIONS},
            'cells_per_condition': len(tasks),
            'effects': {name: float(np.mean([v[name] for v in per_task])) for name in estimates},
            'interaction': interval([v['interaction'] for v in per_task], resamples),
            'pairs': [{'contrast': name, 'first_condition': a, 'second_condition': b,
                       'counts': paired_counts([lookup[t, seed, a] for t in tasks], [lookup[t, seed, b] for t in tasks])}
                      for name, a, b in comparisons]})
    return {'primary_estimand': 'interaction', 'units': 'success probability difference',
        'aggregation': 'Average the two seeds within each base task, then average 12 base tasks',
        'bootstrap_replicates': len(resamples), 'bootstrap_seed': BOOTSTRAP_SEED,
        'interval_scope': 'Task-cluster uncertainty; the five intervals are not five separate pass criteria',
        'estimates': estimates, 'task_effects': task_effects, 'by_seed': decompositions}


def prediction_scores(protocol, rows, tasks, resamples):
    forecasts = {v['cell_id']: v for v in protocol['predictions']}
    if len(forecasts) != 96 or set(forecasts) != {r['cell_id'] for r in rows}:
        raise ValueError('Frozen prediction coverage mismatch')
    task_scores = []
    for task in tasks:
        selected = [r for r in rows if r['task_id'] == task]
        for model in MODELS:
            values = []
            for row in selected:
                prediction = forecasts[row['cell_id']][model]
                if set(prediction) != set(CLASSES) or any(not 0 <= prediction[k] <= 1 for k in CLASSES):
                    raise ValueError('Invalid frozen joint probability')
                if not np.isclose(sum(prediction.values()), 1., atol=1e-12):
                    raise ValueError('Frozen joint probabilities do not sum to one')
                values.append(scores(prediction, row['outcome']))
            task_scores.append({'task_id': task, 'model': model,
                **{metric: float(np.mean([v[metric] for v in values])) for metric in values[0]}})
    metrics = ('brier', 'success_brier', 'log_loss')
    summary = {model: {metric: float(np.mean([r[metric] for r in task_scores if r['model'] == model]))
                       for metric in metrics} for model in MODELS}
    contrasts = []
    for baseline in ('constant', 'mixture_joint'):
        for metric in metrics:
            delta = [next(r[metric] for r in task_scores if r['task_id'] == t and r['model'] == baseline)
                     - next(r[metric] for r in task_scores if r['task_id'] == t and r['model'] == 'presence') for t in tasks]
            contrasts.append({'baseline': baseline, 'candidate': 'presence', 'metric': metric,
                'sign': 'Positive means lower candidate prediction loss', **interval(delta, resamples)})
    return {'models': summary, 'baseline_minus_presence': contrasts, 'task_scores': task_scores,
        'parameter_refits': 0, 'new_gate_defined': False,
        'scope': 'Fixed-forecast diagnostics on these development tasks; success and failed-reflection losses are distinct'}


def condition_summary(rows):
    output = []
    for condition in CONDITIONS:
        selected = [r for r in rows if r['condition'] == condition]
        failures = [r for r in selected if not r['success']]
        counts = Counter(r['outcome'] for r in failures)
        output.append({'condition': condition, 'cells': len(selected), 'successes': sum(r['success'] for r in selected),
            'outcomes': dict(Counter(r['outcome'] for r in selected)),
            'failed_attempts': len(failures), 'failed_reflection_labels': dict(counts),
            'failed_reflection_label_fractions': {label: counts[label]/len(failures) if failures else None for label in ('E', 'C', 'U', 'A')},
            'new_E_event': 'restatement' if condition.endswith('_E') else 'generation',
            'new_E_count': counts['E'],
            'observable_event_totals': {key: sum(r['observable_events'][key] for r in selected)
                for key in selected[0]['observable_events']} if selected else {},
            'action_shapes': dict(sum((Counter(r['action_shapes']) for r in selected), Counter())),
            'no_effect_shapes': dict(sum((Counter(r['no_effect_shapes']) for r in selected), Counter()))})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261024/residual-context-protocol.json'))
    parser.add_argument('--export-review', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    rows, known, review = collect(protocol, args.run)
    if args.export_review:
        write_json(args.export_review, {'protocol_sha256': protocol['sha256'], 'items': list(review.values())})
    if not args.labels:
        print({'completed_cells': len(rows), 'new_cells': sum(not r['reused_control'] for r in rows),
            'reused_controls': sum(r['reused_control'] for r in rows), 'unique_reflections_requiring_review': len(review)})
        return
    labels = read_json(args.labels)
    check_digest(labels, 'Semantic labels')
    if labels['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Semantic labels belong to another protocol')
    if len(labels['entries']) != len(review) or {v['sha256'] for v in labels['entries']} != set(review):
        raise ValueError('New-text semantic review coverage mismatch')
    for entry in labels['entries']:
        if entry['label'] not in {'E', 'C', 'U', 'A'}:
            raise ValueError('Unknown semantic class')
        known[entry['sha256']] = entry['label']
    if len(rows) != 96 or sum(not r['reused_control'] for r in rows) != 72:
        raise ValueError('Incomplete results cannot be reported as the complete 96-cell factorial')
    for row in rows:
        row['outcome'] = 'S' if row['success'] else known[row['new_reflection_hashes'][0]]
    tasks = sorted({r['task_id'] for r in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.integers(0, len(tasks), size=(20000, len(tasks)))
    summary = read_json(args.run/'summary.json')
    check_digest(summary, 'Run summary')
    if summary['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Run summary protocol differs')
    value = {'protocol_sha256': protocol['sha256'], 'summary_sha256': summary['sha256'],
        'semantic_labels_sha256': labels['sha256'], 'completed_cells': len(rows), 'new_cells': 72,
        'reused_controls': 24, 'independent_base_tasks': len(tasks), 'seed_replicates_per_task': len(SEEDS),
        'successes': sum(r['success'] for r in rows), 'new_cell_successes': sum(r['success'] for r in rows if not r['reused_control']),
        'rows': rows, 'primary_effects': factorial_effects(rows, tasks, resamples),
        'frozen_prediction_scores': prediction_scores(protocol, rows, tasks, resamples),
        'by_condition': condition_summary(rows), 'parameter_refits': 0,
        'semantic_review': {'reviewer': labels.get('reviewer'), 'human_second_review': labels.get('human_second_review', False),
                            'ambiguous_class_retained': True},
        'event_scope': 'Observable action chains and newly written reflection content are reported separately; neither is a latent-belief label',
        'analysis_driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value)
    write_json(args.run/'analysis.json', value)
    print({'completed_cells': value['completed_cells'], 'new_cells': value['new_cells'],
           'successes': value['successes'], 'primary_interaction': value['primary_effects']['estimates']['interaction'],
           'analysis_sha256': value['sha256'], 'parameter_refits': 0})


if __name__ == '__main__':
    main()
