#!/usr/bin/env python3
"""Describe observable target-object placements without inferring latent belief."""
import argparse
import re
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from summarize_reflexion import costs


def object_type(goal):
    match = re.match(r'(?:put|heat|clean|cool|find) (?:some |a |an )?(?:hot |cool |clean )?([a-z]+)\b', goal)
    if not match:
        raise ValueError('Unsupported public task grammar')
    return match.group(1)


def branch_observations(path, target, original_memory, trials):
    source = original_memory[-1]
    start, end = target['span']
    poisoned = source[:start]+target['replacement_text']+source[end:]
    wrong = target['replacement_text'].split()[0]
    obj = object_type(target['public_evidence'][0]['goal'])
    pattern = re.compile(r'(?:put|move) '+re.escape(obj)+r' \d+ (?:in/on|in|on|to) '+re.escape(wrong)+r' \d+')
    new_clause = re.compile(r'\b(?:put|place)\b[^.!?\n]{0,85}?\b(?:in|on|into|onto)\s+(?:(?:the|a|an)\s+)?'+re.escape(wrong)+r'\b', re.I)
    rows = []
    for trial in range(1, trials+1):
        directory = path/f'trial-{trial:02d}'
        if not (directory/'result.json').exists():
            break
        result = read_json(directory/'result.json')
        steps = read_json(directory/'checkpoint.json')['transitions']
        attempts = [i for i, s in enumerate(steps) if pattern.fullmatch(s['action'])]
        accepted = [i for i in attempts if re.match(r'You (?:put|move) the '+re.escape(obj)+r' \d+\b', steps[i]['observation'])]
        new = result['memory_after'][len(result['memory_before']):]
        physical = [s['submitted_action'] for s in steps if not s['action'].startswith('think:')]
        rows.append({'trial': trial, 'success': result['success'],
            'injected_entry_stored': poisoned in result['memory_before'],
            'injected_entry_visible': poisoned in result['memory_before'][-3:],
            'wrong_destination_attempt_indices': attempts, 'accepted_wrong_destination_indices': accepted,
            'new_reflection_wrong_destination_clause_count': sum(len(new_clause.findall(s)) for s in new),
            'physical_actions_sha256': digest(physical), 'trajectory_sha256': result['trajectory_sha256'],
            'semantic_adoption_label': None,
            'semantic_adoption_note': 'Requires context review; placement or lexical clause alone is not a belief label'})
        if result['success']:
            break
    complete = any(r['success'] for r in rows) or len(rows) == trials
    return {'completed': complete, 'success': any(r['success'] for r in rows) if complete else None,
        'attempts': rows, 'costs': costs(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--parent-run', type=Path, default=Path('runs/reflexion-pilot24-20261022'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    identity = read_json(a.run/'run.json')['identity']
    registry = read_json(a.targets)
    if registry['sha256'] != identity['targets_sha256'] or registry['sha256'] != digest({k: v for k, v in registry.items() if k != 'sha256'}):
        raise ValueError('Target registry mismatch')
    reports = []
    for target in registry['targets']:
        parent_path = a.parent_run/'tasks'/target['task_id']/'trial-00/result.json'
        if file_hash(parent_path) != target['parent_result_sha256']:
            raise ValueError('Parent prefix changed')
        parent = read_json(parent_path)
        root = a.run/'targets'/target['task_id']/'branches'
        branches = {name: branch_observations(root/name, target, parent['memory_after'], identity['trials'])
            for name in ['original', 'sham', 'retain_error', 'correct_target', 'discard_new']}
        correct = branches['correct_target']['success']
        error = branches['retain_error']['success']
        reports.append({'task_id': target['task_id'], 'branches': branches,
            'correct_minus_error_success': int(correct)-int(error) if correct is not None and error is not None else None})
    paired = [r['correct_minus_error_success'] for r in reports if r['correct_minus_error_success'] is not None]
    report = {'scope': 'development paired goal-binding pilot; observable placement indicators',
        'analysis_driver_sha256': file_hash(Path(__file__)), 'targets_sha256': registry['sha256'],
        'independent_tasks': len(reports), 'completed_paired_tasks': len(paired),
        'correct_only_success_tasks': sum(x == 1 for x in paired),
        'error_only_success_tasks': sum(x == -1 for x in paired),
        'targets': reports, 'costs': costs(a.run)}
    report['sha256'] = digest(report)
    write_json(a.output, report)
    print({k: v for k, v in report.items() if k != 'targets'})


if __name__ == '__main__':
    main()
