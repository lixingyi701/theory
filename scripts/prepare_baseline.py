#!/usr/bin/env python3
"""Freeze development scenes and a completely separate official unseen holdout.

Development results select/check infrastructure, not estimate generalization.
Public ReAct demonstrations have no published task IDs; this limitation is
recorded rather than claiming proven demonstration-scene independence.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from agent_reliability.dataset import TASK_TYPES, enumerate_tasks, load_manifest
from agent_reliability.io import digest, file_hash, read_json, write_json


def prepare(root, pilot, output):
    if output.exists():
        raise FileExistsError('Manifest bundle is immutable; choose a new path')
    old = load_manifest(pilot)
    train, excluded = enumerate_tasks(root, 'train')
    unseen, unseen_excluded = enumerate_tasks(root, 'valid_unseen')
    old_scenes = {t['scene_id'] for t in old['tasks']}
    # A conservative public-layout match detects some possible demonstration
    # scenes. Nonmatches do not establish independence across data versions.
    demos = read_json(Path('prompts/react_alfworld.json'))
    signatures = {k: Counter(re.findall(r'\b(?:a|an) ([a-z]+) \d+', v.splitlines()[0]))
                  for k, v in demos.items() if k.startswith('react_') and k.endswith(('_0', '_1'))}
    matches = {k: set() for k in signatures}
    for t in train:
        problem = read_json(root / t['gamefile'])['pddl_problem']
        layout = Counter(x.lower() for x in re.findall(r'\(receptacleType\s+\S+\s+(\w+)Type\)', problem))
        for key, signature in signatures.items():
            if layout == signature:
                matches[key].add(t['scene_id'])
    possible_demo_scenes = set().union(*matches.values())
    blocked = old_scenes | possible_demo_scenes
    used, selected = set(), []
    for _ in range(2):
        for kind in TASK_TYPES:
            pool = sorted((t for t in train if t['task_type'] == kind and t['scene_id'] not in blocked | used),
                          key=lambda t: digest([20261019, t['task_id']]))
            if not pool:
                raise ValueError('Insufficient unused scenes; inspect data before relaxing selection')
            selected.append(pool[0]); used.add(pool[0]['scene_id'])
    if used & {t['scene_id'] for t in unseen}:
        raise ValueError('Development/holdout scenes overlap')
    info = {'schema_version': 1, 'purpose': 'baseline development, not benchmark score',
            'selection_seed': 20261019, 'eligible_count': len(train), 'excluded': excluded,
            'source_pilot_sha256': old['sha256'], 'excluded_pilot_scenes': sorted(old_scenes),
            'possible_demonstration_scenes': sorted(possible_demo_scenes),
            'demonstration_layout_matches': {k: sorted(v) for k, v in matches.items()},
            'demonstration_scene_independence': 'unverified; upstream examples lack task/scene IDs',
            'scene_grouping': 'unique_floor_plan', 'split': 'train', 'tasks': selected}
    dev = {**info, 'sha256': digest(info)}
    held = {'schema_version': 1, 'purpose': 'sealed generalization holdout; no LLM calls in baseline development',
            'split': 'valid_unseen', 'eligible_count': len(unseen), 'excluded': unseen_excluded,
            'scene_grouping': 'all tasks of each floor_plan stay in holdout', 'tasks': unseen}
    holdout = {**held, 'sha256': digest(held)}
    write_json(output / 'dev12.json', dev); write_json(output / 'holdout134.json', holdout)
    receipt = {'dev_sha256': dev['sha256'], 'holdout_sha256': holdout['sha256'],
               'dev_tasks': len(selected), 'dev_scenes': len(used), 'holdout_tasks': len(unseen),
               'holdout_scenes': sorted({t['scene_id'] for t in unseen}),
               'demo_file_sha256': file_hash(Path('prompts/react_alfworld.json'))}
    write_json(output / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(a.data_root, a.pilot, a.output)
