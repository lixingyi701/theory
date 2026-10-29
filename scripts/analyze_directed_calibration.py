#!/usr/bin/env python3
"""Audit completed cells, export semantic review items, then summarize fixed contrasts."""
import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_reflexion_precondition_forks import procedure_events
from audit_reflexion_action_shapes import action_shape
from run_directed_calibration import templates_for


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-radius, center+radius]


def collect(protocol, root):
    cells = {c['id']: (b, c) for b in protocol['blocks'] for c in b['cells']}
    cases = {c['task_id']: c for c in protocol['cases']}
    templates, known, review = {}, {}, {}
    for task_id, case in cases.items():
        _, entries = templates_for(case)
        templates[task_id] = entries
        for symbol, text in entries.items():
            known[digest(text)] = symbol if symbol in {'E', 'C'} else 'U'
    rows = []
    for path in sorted(root.glob('cells/*/trial-*/result.json')):
        result = read_json(path)
        block, cell = cells[path.parent.parent.name]
        identity = read_json(path.parent/'identity.json')
        if identity['sha256'] != digest({k: v for k, v in identity.items() if k != 'sha256'}):
            raise ValueError('Attempt identity integrity')
        if identity['memory'] != result['memory_before'] or identity['task'] != cases[block['task_id']]['task']:
            raise ValueError('Memory/task identity mismatch')
        if identity['trial'] != result['trial'] or identity['seed'] != block['seed']:
            raise ValueError('Seed/trial mismatch')
        if result['identity_sha256'] != identity['sha256']:
            raise ValueError('Result bound to another attempt')
        if result['trial'] == 1 and result['memory_before'] != [templates[block['task_id']][s] for s in cell['labels']]:
            raise ValueError('Starting condition changed')
        if result['trial'] > 1:
            previous = read_json(path.parent.parent/f"trial-{result['trial']-1:02d}"/'result.json')
            if previous['success'] or previous['memory_after'] != result['memory_before']:
                raise ValueError('Native multi-attempt update chain changed')
        steps = read_json(path.parent/'checkpoint.json')['transitions']
        if digest(steps) != result['trajectory_sha256']:
            raise ValueError('Trajectory integrity')
        fresh = result['memory_after'][len(result['memory_before']):]
        if result['memory_after'][:len(result['memory_before'])] != result['memory_before']:
            raise ValueError('Native update unexpectedly edited history')
        if len(fresh) != int(not result['success']):
            raise ValueError('Unexpected critic update count')
        for text in fresh:
            sha = digest(text)
            if sha not in known:
                review.setdefault(sha, {'sha256': sha, 'text': text, 'sources': []})['sources'].append(str(path))
        row = {'cell_id': cell['id'], 'panel': block['panel'], 'condition': cell['condition'],
            'task_id': block['task_id'], 'seed': block['seed'], 'trial': result['trial'],
            'success': result['success'], 'decisions': result['decisions'], 'status': result['status'],
            'result_path': str(path), 'result_file_sha256': file_hash(path),
            'memory_before_hashes': [digest(t) for t in result['memory_before']],
            'visible_memory_hashes': [digest(t) for t in result['memory_before'][-3:]],
            'new_reflection_hashes': [digest(t) for t in fresh], 'maximum_attempts': block['attempts'],
            'initial_labels': cell['labels'], 'trajectory_sha256': result['trajectory_sha256'],
            'constructed_H_entry_visible': templates[block['task_id']]['H'] in result['memory_before'][-3:]}
        obj = {'762a7487ace813d7': 'cup', '64aa51b30cab1f98': 'egg',
               '3e3fa991ee97689e': 'mug', '05f694718d71af12': 'egg'}[block['task_id']]
        events = procedure_events(steps, obj, 'heat', 'microwave 1')
        row['observable_events'] = {k: v if isinstance(v, bool) else len(v) for k, v in events.items()}
        row['action_shapes'] = dict(Counter(action_shape(s['action'], s['submitted_action']) for s in steps))
        row['no_effect_shapes'] = dict(Counter(action_shape(s['action'], s['submitted_action'])
            for s in steps if s['observation'] == 'Nothing happens.'))
        rows.append(row)
    return rows, known, review


def summarize(protocol, root, labels):
    rows, known, review = collect(protocol, root)
    if labels['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Semantic review belongs to another protocol')
    if labels['sha256'] != digest({k: v for k, v in labels.items() if k != 'sha256'}):
        raise ValueError('Semantic review integrity')
    for entry in labels['entries']:
        if entry['label'] not in {'E', 'C', 'U', 'A'} or entry['sha256'] in known:
            raise ValueError('Invalid/overlapping label')
        known[entry['sha256']] = entry['label']
    if set(review)-set(known):
        raise ValueError('New reflections still require semantic review')
    transitions = defaultdict(lambda: {'success': 0, 'attempts': 0, 'new_E': 0,
        'new_C': 0, 'new_U': 0, 'new_A': 0, 'generation_E': 0, 'generation_valid_failures': 0,
        'restatement_E': 0, 'restatement_valid_failures': 0, 'start_ambiguous_failures': 0,
        'generation_ambiguous_new': 0, 'restatement_ambiguous_new': 0})
    for row in rows:
        visible = [known[h] for h in row['visible_memory_hashes']]
        present = True if 'E' in visible else (None if 'A' in visible else False)
        row['visible_target_labels'] = visible
        row['visible_target_error'] = present
        row['error_updates_until_eviction_without_new_errors'] = None if 'A' in visible else (
            3-len(visible)+max(i for i, label in enumerate(visible) if label == 'E')+1 if present else 0)
        new = [known[h] for h in row['new_reflection_hashes']]
        row['new_reflection_labels'] = new
        key = (row['panel'], row['condition'], row['task_id'], row['trial'])
        state = transitions[key]
        state['attempts'] += 1; state['success'] += row['success']
        if new:
            state['new_'+new[0]] += 1
            if present is None:
                state['start_ambiguous_failures'] += 1
            elif new[0] != 'A':
                prefix = 'restatement' if present else 'generation'
                state[prefix+'_valid_failures'] += 1
                state[prefix+'_E'] += new[0] == 'E'
            else:
                prefix = 'restatement' if present else 'generation'
                state[prefix+'_ambiguous_new'] += 1
    conditional = []
    for key, value in sorted(transitions.items()):
        record = dict(zip(['panel', 'condition', 'task_id', 'trial'], key)) | value
        for event in ['generation', 'restatement']:
            record[event+'_wilson_within_task'] = wilson(value[event+'_E'], value[event+'_valid_failures'])
            total = value[event+'_valid_failures']+value[event+'_ambiguous_new']
            record[event+'_ambiguity_bounds'] = [value[event+'_E']/total,
                (value[event+'_E']+value[event+'_ambiguous_new'])/total] if total else None
        conditional.append(record)
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row['cell_id']].append(row)
    cell_rows = []
    for block in protocol['blocks']:
        for cell in block['cells']:
            attempts = sorted(by_cell[cell['id']], key=lambda r: r['trial'])
            complete = bool(attempts) and (any(r['success'] for r in attempts) or len(attempts) == block['attempts'])
            cell_rows.append({'cell_id': cell['id'], 'panel': block['panel'], 'condition': cell['condition'],
                'task_id': block['task_id'], 'seed': block['seed'], 'complete': complete,
                'attempts_observed': len(attempts), 'first_success': attempts[0]['success'] if attempts else None,
                'budget_success': any(r['success'] for r in attempts) if complete else None})
    contrasts = []
    comparisons = {'generation': [('error', 'correct'), ('error', 'empty'), ('correct', 'empty'), ('generic', 'empty')],
        'recovery': [('retain', 'correct'), ('correct', 'replace'), ('replace', 'delete'), ('without_cue', 'replace')],
        'age': [('oldest', 'newest')]}
    for panel, pairs in comparisons.items():
        for left, right in pairs:
            left_rows = {(c['task_id'], c['seed']): c for c in cell_rows if c['panel'] == panel and c['condition'] == left}
            right_rows = {(c['task_id'], c['seed']): c for c in cell_rows if c['panel'] == panel and c['condition'] == right}
            for metric in ['first_success', 'budget_success']:
                paired = [(k, left_rows[k][metric], right_rows[k][metric]) for k in left_rows.keys() & right_rows.keys()
                          if left_rows[k][metric] is not None and right_rows[k][metric] is not None]
                contrasts.append({'panel': panel, 'left': left, 'right': right, 'metric': metric,
                    'pairs': len(paired), 'left_only_success': sum(a and not b for _, a, b in paired),
                    'right_only_success': sum(b and not a for _, a, b in paired),
                    'both_success': sum(a and b for _, a, b in paired),
                    'neither_success': sum(not a and not b for _, a, b in paired),
                    'task_differences': {t: sum(int(b)-int(a) for (task, _), a, b in paired if task == t)
                                         for t in sorted({k[0] for k, _, _ in paired})}})
    summary = read_json(root/'summary.json')
    report = {'scope': protocol['scope'], 'protocol_sha256': protocol['sha256'],
        'summary_sha256': summary['sha256'], 'semantic_labels_sha256': labels['sha256'],
        'driver_sha256': file_hash(Path(__file__)), 'independent_tasks': len(protocol['cases']),
        'attempts_observed': len(rows), 'successes': sum(r['success'] for r in rows),
        'expected_cells': protocol['expected_cells'], 'completed_cells': sum(c['complete'] for c in cell_rows),
        'rows': rows, 'cells': cell_rows, 'conditional_transitions': conditional, 'paired_contrasts': contrasts,
        'costs_incremental': summary['costs_incremental'], 'all_dispatched_calls': summary['all_dispatched_calls'],
        'interpretation': 'Within four known development tasks; repeated seeds and nested attempts are dependent; no universal transition fit'}
    report['sha256'] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ['protocol', 'run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if protocol['sha256'] != digest({k: v for k, v in protocol.items() if k != 'sha256'}):
        raise ValueError('Protocol integrity')
    if args.labels:
        report = summarize(protocol, args.run, read_json(args.labels))
        write_json(args.output, report)
        print({k: v for k, v in report.items() if k not in {'rows', 'cells', 'conditional_transitions', 'paired_contrasts'}})
    else:
        rows, _, review = collect(protocol, args.run)
        write_json(args.output, {'protocol_sha256': protocol['sha256'], 'completed_attempts': len(rows),
                                 'review': list(review.values())})
        print({'completed_attempts': len(rows), 'unique_reflections_to_review': len(review)})


if __name__ == '__main__':
    main()
