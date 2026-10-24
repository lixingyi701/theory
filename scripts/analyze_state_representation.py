#!/usr/bin/env python3
"""Validate frozen cells, export full-text labels privately, and fit shared models."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_reflexion_precondition_forks import procedure_events
from audit_reflexion_action_shapes import action_shape
from run_state_representation import verify_protocol
from state_representation_model import compare


def collect(protocol, root):
    cells = {c['id']: (b, c) for b in protocol['blocks'] for c in b['cells']}
    cases = {c['task_id']: c for c in protocol['cases']}
    known = {digest(v): ('U' if k == 'N' else k) for k, v in protocol['entries'].items()}
    rows, review = [], {}
    for path in sorted(root.glob('cells/*/trial-*/result.json')):
        block, cell = cells[path.parent.parent.name]
        value = read_json(path); identity = read_json(path.parent/'identity.json')
        expected_memory = [protocol['entries'][k] for k in cell['labels']]
        if identity['sha256'] != digest({k: v for k, v in identity.items() if k != 'sha256'}):
            raise ValueError('Attempt identity integrity')
        if identity['memory'] != expected_memory or value['memory_before'] != expected_memory:
            raise ValueError('Wrong constructed memory')
        if identity['task'] != cases[block['task_id']]['task'] or identity['seed'] != block['seed']:
            raise ValueError('Task/seed mismatch')
        if identity['trial'] != 1 or value['trial'] != 1 or value['identity_sha256'] != identity['sha256']:
            raise ValueError('Attempt identity mismatch')
        if value['initial_observation_sha256'] != cases[block['task_id']]['initial_observation_sha256']:
            raise ValueError('Task reset mismatch')
        steps = read_json(path.parent/'checkpoint.json')['transitions']
        if digest(steps) != value['trajectory_sha256']:
            raise ValueError('Trajectory integrity')
        fresh = value['memory_after'][len(expected_memory):]
        if value['memory_after'][:len(expected_memory)] != expected_memory or len(fresh) != int(not value['success']):
            raise ValueError('Native critic append mismatch')
        for text in fresh:
            sha = digest(text)
            if sha not in known:
                review.setdefault(sha, {'sha256': sha, 'text': text, 'sources': []})['sources'].append(cell['id'])
        goal = Path(cases[block['task_id']]['task']['gamefile']).parts[-3].split('-')[1].lower()
        events = procedure_events(steps, goal, 'heat', 'microwave 1')
        rows.append({'cell_id': cell['id'], 'task_id': block['task_id'], 'condition': cell['condition'],
            'seed': block['seed'], 'initial_labels': cell['labels'], 'success': value['success'],
            'decisions': value['decisions'], 'status': value['status'], 'result_sha256': file_hash(path),
            'identity_sha256': identity['sha256'], 'trajectory_sha256': value['trajectory_sha256'],
            'memory_sha256': digest(expected_memory), 'new_reflection_hashes': [digest(v) for v in fresh],
            'observable_events': {k: v if isinstance(v, bool) else len(v) for k, v in events.items()},
            'action_shapes': dict(Counter(action_shape(s['action'], s['submitted_action']) for s in steps)),
            'no_effect_shapes': dict(Counter(action_shape(s['action'], s['submitted_action']) for s in steps if s['observation'] == 'Nothing happens.'))})
    if len({r['cell_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate committed cells')
    return rows, known, review


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261023/state-representation-protocol.json'))
    p.add_argument('--export-review', type=Path)
    p.add_argument('--labels', type=Path)
    args = p.parse_args()
    protocol = verify_protocol(args.protocol)
    rows, known, review = collect(protocol, args.run)
    if args.export_review:
        write_json(args.export_review, {'protocol_sha256': protocol['sha256'], 'items': list(review.values())})
    if not args.labels:
        print({'completed_cells': len(rows), 'unique_reflections_requiring_review': len(review)}); return
    labels = read_json(args.labels)
    if labels['protocol_sha256'] != protocol['sha256'] or labels['sha256'] != digest({k: v for k, v in labels.items() if k != 'sha256'}):
        raise ValueError('Semantic label integrity')
    if set(review) != {r['sha256'] for r in labels['entries']}:
        raise ValueError('Review coverage mismatch')
    for entry in labels['entries']:
        if entry['label'] not in {'E', 'C', 'U', 'A'}:
            raise ValueError('Unknown semantic class')
        known[entry['sha256']] = entry['label']
    if len(rows) != protocol['expected_cells']:
        raise ValueError('Incomplete matrix is not a complete calibration fit')
    for row in rows:
        row['outcome'] = 'S' if row['success'] else known[row['new_reflection_hashes'][0]]
    conditions = []
    for name in protocol['states']:
        selected = [r for r in rows if r['condition'] == name]
        conditions.append({'condition': name, 'initial_labels': protocol['states'][name],
            'attempts': len(selected), 'outcomes': dict(Counter(r['outcome'] for r in selected)),
            'per_task': [{'task_id': r['task_id'], 'outcome': r['outcome']} for r in selected]})
    pairs = [('single_error','single_correct'), ('single_correct','single_generic'),
        ('single_generic','empty'), ('single_generic','full_generic'), ('oldest_error','newest_error'),
        ('oldest_correct','newest_correct'), ('single_error','newest_error')]
    contrasts = []
    for left, right in pairs:
        table = Counter()
        for task in {r['task_id'] for r in rows}:
            a = next(r for r in rows if r['task_id'] == task and r['condition'] == left)
            b = next(r for r in rows if r['task_id'] == task and r['condition'] == right)
            table['both_success' if a['success'] and b['success'] else 'left_only' if a['success']
                  else 'right_only' if b['success'] else 'neither_success'] += 1
        contrasts.append({'left': left, 'right': right, 'task_pairs': dict(table)})
    summary = read_json(args.run/'summary.json')
    value = {'protocol_sha256': protocol['sha256'], 'summary_sha256': summary['sha256'],
        'semantic_labels_sha256': labels['sha256'], 'completed_cells': len(rows), 'independent_base_tasks': 12,
        'successes': sum(r['success'] for r in rows), 'outcomes': dict(Counter(r['outcome'] for r in rows)),
        'conditions': conditions, 'paired_success_contrasts': contrasts, 'rows': rows,
        'analysis_driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value); write_json(args.run/'analysis.json', value)
    fitted = {'protocol_sha256': protocol['sha256'], 'analysis_sha256': value['sha256'],
        'model_source_sha256': file_hash(Path('scripts/state_representation_model.py')),
        **compare(rows)}
    fitted['sha256'] = digest(fitted); write_json(args.run/'model-freeze.json', fitted)
    print({'attempts': len(rows), 'outcomes': value['outcomes'], 'selected': fitted['selected'],
           'scores': fitted['scores'], 'analysis_sha256': value['sha256'], 'model_freeze_sha256': fitted['sha256']})


if __name__ == '__main__':
    main()
