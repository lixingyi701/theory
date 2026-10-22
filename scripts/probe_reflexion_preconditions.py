#!/usr/bin/env python3
"""Replay an observed placement-before-macro failure using public actions only."""
import argparse
import re
from pathlib import Path

from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reference import official_environment, submitted_command


def find_candidate(steps):
    for index, step in enumerate(steps):
        match = re.fullmatch(r'(heat|cool) ([a-z]+ \d+) with ([a-z]+ \d+)', step['action'])
        if not match or step['observation'] != 'Nothing happens.':
            continue
        verb, obj, device = match.groups()
        candidates = [j for j in range(index) if steps[j]['submitted_action'] == f'move {obj} to {device}'
                      and steps[j]['observation'].startswith(f'You move the {obj} to the {device}')]
        if not candidates:
            continue
        placement = candidates[-1]
        if any(s['observation'].startswith(f'You pick up the {obj}') for s in steps[placement+1:index]):
            continue
        if not any(s['observation'].startswith(f'You pick up the {obj}') for s in steps[:placement]):
            continue
        return {'verb': verb, 'object': obj, 'device': device,
                'placement_index': placement, 'failed_macro_index': index}
    return None


def replay_probe(task, parent, candidate):
    checkpoint = parent/'checkpoint.json'
    steps = read_json(checkpoint)['transitions']
    prefix = steps[:candidate['placement_index']]
    verb, obj, device = (candidate[k] for k in ['verb', 'object', 'device'])
    macro = f'{verb} {obj} with {device}'
    placement = f'move {obj} to {device}'
    suffixes = {'held_open': [macro], 'held_closed': [f'close {device}', macro],
                'placed_open': [placement, macro],
                'placed_closed': [placement, f'close {device}', macro],
                'retrieve_then_macro': [placement, f'take {obj} from {device}', macro]}
    rows = []
    for name, suffix in suffixes.items():
        env = official_environment(Path('data/alfworld'), task, 42, Path('third_party/react'))
        try:
            initial, _ = env.reset()
            initial = '\n'.join(initial[0].split('\n\n')[1:])
            if digest(initial) != read_json(parent/'result.json')['initial_observation_sha256']:
                raise ValueError('Probe reset changed')
            for saved in prefix:
                obs, reward, done, info = env.step([saved['submitted_action']])
                actual = {'action': saved['action'], 'submitted_action': saved['submitted_action'],
                    'observation': obs[0], 'reward': float(reward[0]), 'done': bool(done[0]), 'won': bool(info['won'][0])}
                if actual != saved:
                    raise ValueError('Probe prefix differs from saved public trajectory')
            future = []
            for action in suffix:
                obs, reward, done, info = env.step([submitted_command(action)])
                future.append({'action': action, 'observation': obs[0], 'reward': float(reward[0]),
                               'done': bool(done[0]), 'won': bool(info['won'][0])})
            rows.append({'condition': name, 'prefix_verified_steps': len(prefix), 'suffix': future,
                         'macro_succeeded': future[-1]['observation'].startswith(f'You {verb} the {obj} using the {device}')})
        finally:
            env.close()
    outcomes = {r['condition']: r['macro_succeeded'] for r in rows}
    report = {'task_id': task['task_id'], 'parent_result_sha256': file_hash(parent/'result.json'),
        'checkpoint_sha256': file_hash(checkpoint), 'candidate': candidate, 'rows': rows,
        'holding_precondition_supported': all(outcomes[k] for k in ['held_open', 'held_closed', 'retrieve_then_macro'])
            and not any(outcomes[k] for k in ['placed_open', 'placed_closed']),
        'environment_steps': sum(r['prefix_verified_steps']+len(r['suffix']) for r in rows), 'model_calls': 0,
        'probe_driver_sha256': file_hash(Path(__file__))}
    report['sha256'] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ['parent-run', 'manifest', 'output']:
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--task-id', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous probe evidence')
    task = next(t for t in load_manifest(args.manifest)['tasks'] if t['task_id'] == args.task_id)
    parent = args.parent_run/'tasks'/args.task_id/'trial-00'
    candidate = find_candidate(read_json(parent/'checkpoint.json')['transitions'])
    if candidate is None:
        raise ValueError('No publicly recorded placement-before-macro candidate')
    report = replay_probe(task, parent, candidate)
    report['driver_sha256'] = file_hash(Path(__file__))
    report.pop('sha256'); report['sha256'] = digest(report)
    write_json(args.output, report)
    print({'task_id': args.task_id, 'supported': report['holding_precondition_supported'],
           'environment_steps': report['environment_steps'], 'model_calls': 0})


if __name__ == '__main__':
    main()
