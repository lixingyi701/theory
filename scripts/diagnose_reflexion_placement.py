#!/usr/bin/env python3
"""Environment-only replay of failed placement spellings; no policy or model call."""
import argparse
import re
from pathlib import Path

from agent_reliability.cli import provenance
from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reference import official_environment, submitted_command


def replay(task, prefix, final_command):
    env = official_environment(Path('data/alfworld'), task, 42, Path('third_party/react'))
    try:
        initial, _ = env.reset()
        for expected in prefix:
            ob, reward, done, info = env.step([submitted_command(expected['action'])])
            actual = {'action': expected['action'], 'submitted_action': submitted_command(expected['action']),
                'observation': ob[0], 'reward': float(reward[0]), 'done': bool(done[0]), 'won': bool(info['won'][0])}
            if actual != expected:
                raise ValueError('Replay public prefix mismatch')
        ob, reward, done, info = env.step([final_command])
        return {'initial_observation_sha256': digest(initial[0]), 'command': final_command,
            'observation': ob[0], 'reward': float(reward[0]), 'done': bool(done[0]), 'won': bool(info['won'][0])}
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=Path('runs/reflexion-pilot24-20261022'))
    p.add_argument('--manifest', type=Path, default=Path('repro/reflexion-20261022/pilot24.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--review-existing', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous replay diagnostic')
    if a.review_existing:
        prior = read_json(a.review_existing)
        if prior['sha256'] != digest({k: v for k, v in prior.items() if k != 'sha256'}):
            raise ValueError('Original replay report integrity')
        report = {k: v for k, v in prior.items() if k != 'sha256'}
        report['previous_report_sha256'] = prior['sha256']
        report['driver_sha256'] = file_hash(Path(__file__))
        report['scoring_correction'] = 'Accept both You move and You put observations; reuse preserved replay outputs without rerunning environments'
        for row in report['rows']:
            row['canonical_placement_accepted'] = bool(re.match(r'You (?:put|move) the ', row['canonical']['observation']))
        report['sha256'] = digest(report)
        write_json(a.output, report)
        print({'diagnosed_tasks': len(report['rows']), 'canonical_accepted': sum(r['canonical_placement_accepted'] for r in report['rows']),
            'single_step_wins': sum(r['canonical']['won'] for r in report['rows']), 'model_dispatches': 0})
        return
    manifest = load_manifest(a.manifest)
    parent = read_json(a.run/'run.json')['identity']
    if parent['source_hashes'] != provenance()['source_hashes'] or parent['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Parent source/manifest mismatch')
    rows = []
    for task in manifest['tasks']:
        directory = a.run/'tasks'/task['task_id']/'trial-00'
        if not (directory/'result.json').exists():
            continue
        result = read_json(directory/'result.json')
        if result['success']:
            continue
        steps = read_json(directory/'checkpoint.json')['transitions']
        for index, step in enumerate(steps):
            match = re.fullmatch(r'put (.+) (?:in|on) (.+)', step['action'])
            if match and step['observation'] == 'Nothing happens.':
                canonical = f'move {match[1]} to {match[2]}'
                raw = replay(task, steps[:index], step['submitted_action'])
                normalized = replay(task, steps[:index], canonical)
                if raw['initial_observation_sha256'] != normalized['initial_observation_sha256']:
                    raise ValueError('Diagnostic initial observation mismatch')
                if raw['observation'] != step['observation'] or raw['won'] != step['won']:
                    raise ValueError('Recorded failed command did not reproduce')
                rows.append({'task_id': task['task_id'], 'transition_index': index,
                    'parent_result_sha256': file_hash(directory/'result.json'),
                    'shared_prefix_sha256': digest(steps[:index]), 'raw': raw, 'canonical': normalized,
                    'canonical_placement_accepted': bool(re.match(r'You (?:put|move) the ', normalized['observation']))})
                break
    report = {'scope': 'environment-only command spelling diagnostic; no agent continuation or model calls',
        'manifest_sha256': manifest['sha256'], 'driver_sha256': file_hash(Path(__file__)),
        'source_hashes': parent['source_hashes'], 'rows': rows, 'model_dispatches': 0}
    report['sha256'] = digest(report)
    write_json(a.output, report)
    print({'diagnosed_tasks': len(rows), 'canonical_accepted': sum(r['canonical_placement_accepted'] for r in rows), 'model_dispatches': 0})


if __name__ == '__main__':
    main()
