#!/usr/bin/env python3
"""Sample fresh thermal tasks for native procedural-premise discovery."""
import argparse
from collections import Counter
from pathlib import Path

from agent_reliability.dataset import enumerate_tasks
from agent_reliability.io import digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Manifest immutable')
    paths = ['repro/baseline-20261019/dev12.json', 'repro/baseline-20261019/holdout134.json',
             'repro/legacy-pilot24.json', 'repro/reflexion-20261022/pilot24.json',
             'repro/reflexion-20261023/pilot-prefix24-v2.json']
    previous = [read_json(Path(path)) for path in paths]
    excluded_ids = {t['task_id'] for m in previous for t in m['tasks']}
    excluded_games = {t['game_sha256'] for m in previous for t in m['tasks']}
    excluded_trajectories = {t['trajectory_sha256'] for m in previous for t in m['tasks']}
    excluded_scenes = {t['scene_id'] for t in previous[1]['tasks']} | {'FloorPlan302', 'FloorPlan319', 'FloorPlan218'}
    tasks, removed = enumerate_tasks(Path('data/alfworld'), 'train')
    kinds = ['pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep']
    pools = {kind: [t for t in tasks if t['task_type'] == kind and t['task_id'] not in excluded_ids
                   and t['game_sha256'] not in excluded_games and t['trajectory_sha256'] not in excluded_trajectories
                   and t['scene_id'] not in excluded_scenes] for kind in kinds}
    selected, scene_counts, used_ids = [], Counter(), set()
    for _ in range(12):
        for kind in kinds:
            candidates = [t for t in pools[kind] if t['task_id'] not in used_ids and scene_counts[t['scene_id']] < 2]
            if not candidates:
                raise ValueError('Insufficient fresh tasks under the declared two-per-room limit')
            task = min(candidates, key=lambda t: (scene_counts[t['scene_id']], digest([2026102302, t['task_id']])))
            selected.append(task); used_ids.add(task['task_id']); scene_counts[task['scene_id']] += 1
    report = {'schema_version': 1, 'split': 'train', 'selection_seed': 2026102302,
        'scope': 'development; prospectively sampled native thermal procedural premises, not a six-class success benchmark',
        'task_exclusion_manifest_sha256': {p: m['sha256'] for p, m in zip(paths, previous)},
        'excluded_scene_ids': sorted(excluded_scenes), 'excluded': removed,
        'task_types': dict(Counter(t['task_type'] for t in selected)), 'scene_counts': dict(scene_counts),
        'selection_rule': 'Twelve heat/cool rounds; least-used eligible room then hash order; at most two tasks per room; exclude previous task/game/trajectory hashes',
        'tasks': selected}
    report['sha256'] = digest(report); write_json(args.output, report)
    print({'tasks': len(selected), 'rooms': len(scene_counts), 'sha256': report['sha256']})


if __name__ == '__main__':
    main()
