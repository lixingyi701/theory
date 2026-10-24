#!/usr/bin/env python3
"""Describe action format independently of the model/tool wiring audit.

This reads completed checkpoints and never changes actions or retries a call.
Matching a public command shape says nothing about its entities, preconditions,
or suitability for the goal. Unmatched shapes remain review items.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json

ENTITY = r'[a-z]+ \d+'
# Public templates in the installed ALFWorld alfred.twl2, after wire translation.
COMMAND_SHAPES = [
    rf'go to {ENTITY}', rf'(?:open|close|examine|use) {ENTITY}',
    rf'take {ENTITY} from {ENTITY}', rf'move {ENTITY} to {ENTITY}',
    rf'put {ENTITY} (?:into|in) {ENTITY}',
    rf'(?:heat|clean|cool|slice) {ENTITY} with {ENTITY}',
    r'(?:look|inventory|help)',
]


def action_shape(raw, submitted):
    if raw.startswith('think:'):
        return 'thought'
    if any(re.fullmatch(pattern, submitted) for pattern in COMMAND_SHAPES):
        return 'public_command_shape'
    if submitted.startswith('You '):
        return 'environment_style_sentence'
    return 'other_unmatched_shape'


def audit(run, grammar):
    if not (run/'summary.json').exists():
        raise ValueError('A run summary is required')
    summary = read_json(run/'summary.json')
    if 'completed_tasks' in summary:
        complete = summary['completed_tasks'] == summary['expected_tasks']
    else:
        complete = (summary['completed_targets'] == summary['expected_targets']
                    and all(t['completed_branches'] == 5 for t in summary['targets']))
    if not complete:
        raise ValueError('Audit final action counts only after the cohort completes')
    grammar_text = grammar.read_text()
    for template in ['open {r}', 'move {o} to {r}', 'take {o} from {r}', 'heat {o} with {r}']:
        if f'template :: "{template}";' not in grammar_text:
            raise ValueError('Public grammar does not match this audit version')
    counts, no_effects, items = Counter(), Counter(), []
    checkpoints = sorted(run.glob('**/checkpoint.json'))
    for path in checkpoints:
        checkpoint_hash = file_hash(path)
        for index, step in enumerate(read_json(path)['transitions']):
            shape = action_shape(step['action'], step['submitted_action'])
            counts[shape] += 1
            if step['observation'] == 'Nothing happens.':
                no_effects[shape] += 1
            if shape in {'environment_style_sentence', 'other_unmatched_shape'}:
                action = step['submitted_action']
                items.append({'checkpoint': path.relative_to(run).as_posix(),
                    'checkpoint_sha256': checkpoint_hash, 'index': index,
                    'shape': shape, 'action_excerpt': action[:160],
                    'action_sha256': digest(action), 'native_won': step['won'],
                    'no_effect': step['observation'] == 'Nothing happens.'})
    report = {
        'scope': 'Executed action shapes under the public numbered-entity grammar; no semantic repairs',
        'run_sha256': file_hash(run/'run.json'),
        'driver_sha256': file_hash(Path(__file__)),
        'public_grammar': 'installed alfworld/data/alfred.twl2',
        'public_grammar_sha256': file_hash(grammar),
        'command_shapes': COMMAND_SHAPES,
        'completed_checkpoints': len(checkpoints), 'counts': dict(counts),
        'no_effect_counts_by_shape': dict(no_effects), 'review_items': items,
        'interpretation': 'Public command shape does not establish valid entities, satisfied preconditions, or correct planning. Environment-style sentences are actor format violations. Other unmatched shapes require review. Thought no-effects are not physical failures.',
    }
    report['sha256'] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--grammar', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.run, args.grammar)
    write_json(args.output, report)
    print({'counts': report['counts'], 'no_effect_counts_by_shape': report['no_effect_counts_by_shape']})


if __name__ == '__main__':
    main()
