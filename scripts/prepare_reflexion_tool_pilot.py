#!/usr/bin/env python3
"""Fresh base tasks in development rooms; keep the four holdout rooms sealed."""
import argparse
from collections import Counter
from pathlib import Path

from agent_reliability.dataset import TASK_TYPES, enumerate_tasks
from agent_reliability.io import digest, read_json, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Manifest immutable')
    paths = ['repro/baseline-20261019/dev12.json', 'repro/baseline-20261019/holdout134.json',
             'repro/legacy-pilot24.json', 'repro/reflexion-20261022/pilot24.json']
    frozen = [read_json(Path(p)) for p in paths]
    excluded_tasks = {t['task_id'] for m in frozen for t in m['tasks']}
    excluded_scenes = {t['scene_id'] for t in frozen[1]['tasks']} | {'FloorPlan302', 'FloorPlan319', 'FloorPlan218'}
    tasks, removed = enumerate_tasks(Path('data/alfworld'), 'train')
    pools = {k: sorted([t for t in tasks if t['task_type'] == k and t['task_id'] not in excluded_tasks and t['scene_id'] not in excluded_scenes],
                      key=lambda t: digest([20261023, t['task_id']])) for k in TASK_TYPES}
    selected, used = [], set()
    for _ in range(4):
        for kind in TASK_TYPES:
            task = next(t for t in pools[kind] if t['scene_id'] not in used)
            selected.append(task); used.add(task['scene_id'])
    report = {'schema_version': 1, 'split': 'train', 'selection_seed': 20261023,
        'scope': 'development native first-attempt prefixes; fresh base tasks, development scenes may recur',
        'task_exclusion_manifest_sha256': {p: m['sha256'] for p, m in zip(paths, frozen)},
        'excluded_scene_ids': sorted(excluded_scenes), 'eligible_count': len(tasks), 'excluded': removed,
        'task_types': dict(Counter(t['task_type'] for t in selected)), 'scene_grouping': 'unique_within_cohort', 'tasks': selected}
    report['sha256'] = digest(report); write_json(a.output, report)
    print({'tasks': len(selected), 'scenes': len(used), 'sha256': report['sha256']})


if __name__ == '__main__':
    main()
