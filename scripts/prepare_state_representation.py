#!/usr/bin/env python3
"""Freeze fresh train tasks and matched proposition/occupancy/position probes."""
from collections import Counter
from pathlib import Path

from agent_reliability.cli import provenance
from agent_reliability.dataset import enumerate_tasks
from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reference import official_environment

PROTOCOL = Path('repro/reflexion-20261023/state-representation-protocol.json')
ENTRIES = {
    'E': 'Plan: Before I heat an object with the microwave, I must put it inside the microwave. I should continue working on the household task.',
    'C': 'Plan: Before I heat an object with the microwave, I must keep it in my inventory. I should continue working on the household task.',
    'N': 'Plan: I should continue working on the household task.'}
STATES = {'single_error': ['E'], 'single_correct': ['C'], 'single_generic': ['N'], 'empty': [],
          'full_generic': ['N', 'N', 'N'], 'oldest_error': ['E', 'N', 'N'],
          'newest_error': ['N', 'N', 'E'], 'oldest_correct': ['C', 'N', 'N'],
          'newest_correct': ['N', 'N', 'C']}


def walk_tasks(value):
    if isinstance(value, dict):
        if all(k in value for k in ['task_id', 'gamefile', 'game_sha256', 'trajectory_sha256']):
            yield value
        for child in value.values():
            yield from walk_tasks(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_tasks(child)


def main():
    if PROTOCOL.exists():
        raise FileExistsError('Never overwrite a frozen selection')
    excluded = {k: set() for k in ['task_id', 'game_sha256', 'trajectory_sha256']}
    inputs = {}
    for path in sorted(Path('repro').rglob('*.json')):
        value = read_json(path)
        previous = list(walk_tasks(value))
        if previous:
            inputs[str(path)] = file_hash(path)
            for task in previous:
                for key in excluded:
                    excluded[key].add(task[key])
    # A dispatched but unfinished task is also used, even without a public summary.
    private_identities = []
    for path in sorted(Path('runs').glob('reflexion-*/**/identity.json')):
        value = read_json(path)
        for task in walk_tasks(value):
            for key in excluded:
                excluded[key].add(task[key])
            private_identities.append({'identity_sha256': file_hash(path), 'task_id': task['task_id']})
    sealed = read_json(Path('repro/baseline-20261019/holdout134.json'))
    sealed_scenes = {t['scene_id'] for t in sealed['tasks']}
    tasks, removed = enumerate_tasks(Path('data/alfworld'), 'train')
    pool = [t for t in tasks if t['task_type'] == 'pick_heat_then_place_in_recep'
            and t['scene_id'] not in sealed_scenes
            and all(t[k] not in excluded[k] for k in excluded)]
    selected, scenes = [], set()
    for task in sorted(pool, key=lambda t: digest([2026102304, t['task_id']])):
        if task['scene_id'] in scenes:
            continue
        selected.append(task); scenes.add(task['scene_id'])
        if len(selected) == 12:
            break
    if len(selected) != 12:
        raise ValueError('Insufficient fresh task/scene coverage')
    cases = []
    for task in selected:
        observations = []
        for seed in [2101, 2102]:
            env = official_environment(Path('data/alfworld'), task, seed, Path('third_party/react'))
            try:
                observation, _ = env.reset()
                observations.append(digest('\n'.join(observation[0].split('\n\n')[1:])))
            finally:
                env.close()
        if observations[0] != observations[1]:
            raise ValueError('Seed-dependent reset; inspect task before model sampling')
        cases.append({'task_id': task['task_id'], 'task': task,
                      'initial_observation_sha256': observations[0], 'public_reset_checks': 2})
    blocks = []
    for index, case in enumerate(cases):
        conditions = list(STATES)
        rotation = index % len(conditions)
        conditions = conditions[rotation:]+conditions[:rotation]
        blocks.append({'task_id': case['task_id'], 'seed': 2101, 'panel': 'matched_state',
            'attempts': 1, 'worker': index % 4,
            'cells': [{'id': f"state-{case['task_id']}-2101-{condition}",
                       'condition': condition, 'labels': STATES[condition], 'task_id': case['task_id']}
                      for condition in conditions]})
    dependencies = ['scripts/prepare_state_representation.py', 'scripts/run_state_representation.py',
        'scripts/state_representation_model.py', 'scripts/run_directed_calibration.py',
        'scripts/summarize_reflexion.py', 'scripts/deployment/serve_reflexion_profiles.py',
        'scripts/audit_reflexion_protocol.py', 'configs/reflexion_qwen3_8b_protocol.json',
        'configs/qwen3_8b_non_thinking_wrapper.json',
        'docs/experiments/state-representation-design-20261023.md']
    value = {'schema_version': 1, 'scope': 'Fresh-task development calibration of deliberately constructed memory states',
        'selection_seed': 2026102304, 'split': 'train', 'cases': cases, 'blocks': blocks,
        'entries': ENTRIES, 'entry_hashes': {k: digest(v) for k, v in ENTRIES.items()}, 'states': STATES,
        'ports': [8001, 8002, 8003, 8004], 'gpu_indices': [0, 1, 2, 3], 'visible_memory_entries': 3,
        'limits': {'additional_calls_including_replica_checks': 6000, 'seconds': 10800},
        'expected_cells': 108, 'maximum_attempts': 108, 'independent_base_tasks': 12,
        'question': 'Does remaining target-error lifetime suffice, or do correction content and occupancy/position improve a shared one-attempt joint outcome model?',
        'exclusion': {'public_inputs': inputs, 'used_private_identities_sha256': digest(private_identities),
            'excluded_counts': {k: len(v) for k, v in excluded.items()}, 'sealed_scenes': sorted(sealed_scenes),
            'same_scene_across_future_partitions': 'Keep all tasks from one scene in the same partition',
            'development_scenes_may_recur': True},
        'selection': 'Hash order among fresh heat train tasks; unique scene within cohort; no outcome-based filtering',
        'eligible_tasks': len(pool), 'eligible_scenes': len({t['scene_id'] for t in pool}), 'dataset_removed': removed,
        'object_counts': dict(Counter(Path(c['task']['gamefile']).parts[-3].split('-')[1] for c in cases)),
        'model_comparison': {'outcomes': ['S', 'E', 'C', 'U', 'A'],
            'candidates': ['constant', 'remaining_lifetime', 'content_occupancy_position'],
            'regularization': 1.0, 'intercept_pseudocount_per_class': 0.5,
            'validation': 'leave-one-base-task-out; every task has equal weight; all nine paired cells stay together',
            'primary': 'multiclass Brier; lower is better', 'secondary': ['joint log loss', 'success Brier'],
            'selection': 'Rich over coarse only if its task-bootstrap 95% improvement interval is strictly positive; otherwise coarse over constant under the same criterion; otherwise constant',
            'bootstrap_replicates': 20000, 'bootstrap_seed': 2026102304,
            'freeze': 'Refit selected candidate on complete calibration tasks and freeze all candidates, scores and one-step forecasts; no formal validation score is produced in this batch',
            'native_context_limit': 'Fresh native U/E/C reflections may contain unmodeled advice; later multi-attempt transport must test this separately'},
        'semantic_review': 'Codex full-text E/C/U/A annotation by content hash; no human second review; ambiguous A is an outcome, not silently removed',
        'source_hashes': provenance()['source_hashes'],
        'protected_files': {p: file_hash(Path(p)) for p in dependencies},
        'public_reset_checks': 24, 'holdout_model_calls': 0}
    value['sha256'] = digest(value); write_json(PROTOCOL, value)
    print({k: value[k] for k in ['sha256', 'eligible_tasks', 'eligible_scenes', 'expected_cells', 'public_reset_checks', 'object_counts']})


if __name__ == '__main__':
    main()
