#!/usr/bin/env python3
"""Prepare fresh-task identities and correlation-sensitive planning, with zero LLM calls."""
from pathlib import Path
from collections import Counter
import numpy as np
from agent_reliability.dataset import enumerate_tasks
from agent_reliability.io import read_json, file_hash, digest
from prepare_shared_calibration import save
from prepare_state_representation import walk_tasks
from continue_reflexion_pilot import ledger

PUBLIC = Path('repro/reflexion-20261025')
MANIFEST = PUBLIC / 'four-attempt-validation-manifest.json'
SEED = 2026102504


def planning(counts, forecasts):
    a = forecasts['unchanged_memory']['correct_middle'][3]
    b = forecasts['full_update']['correct_middle'][3]
    delta = a - b
    rng = np.random.default_rng(SEED)
    scenarios = []
    for truth in [b, .65, a]:
        for scene_rho, task_rho in [(.02, .10), (.05, .25), (.10, .50)]:
            values = []
            for n in counts:
                strength = 1 / scene_rho - 1
                scene_p = rng.beta(truth * strength, (1 - truth) * strength, size=(10000, 1))
                scene_p = np.clip(scene_p, 1e-12, 1 - 1e-12)
                strength = 1 / task_rho - 1
                task_p = rng.beta(np.repeat(scene_p * strength, n, axis=1),
                                  np.repeat((1 - scene_p) * strength, n, axis=1))
                y = rng.binomial(4, task_p).sum(axis=1) / (4 * n)
                values.append(delta * (a + b - 2 * y))
            values = np.asarray(values).T
            means = values.mean(axis=1)
            half = 2.055529439 * values.std(axis=1, ddof=1) / np.sqrt(27)
            scenarios.append({'assumed_four_attempt_success': truth,
                'assumed_scene_correlation': scene_rho, 'assumed_conditional_task_correlation': task_rho,
                'analytic_mean_loss_gain': delta * (a + b - 2 * truth),
                'simulated_mean_loss_gain': float(means.mean()),
                'fraction_full_favored': float(np.mean(means - half > 0)),
                'fraction_unchanged_favored': float(np.mean(means + half < 0)),
                'median_interval_halfwidth': float(np.median(half))})
    return {'scope': 'Synthetic planning under assumed success rates and beta-binomial correlations, not Agent measurements or a guaranteed power claim',
            'replicates_per_scenario': 10000, 'seed': SEED, 'primary_forecasts': {'unchanged': a, 'full': b},
            'equal_expected_loss_success_rate': (a + b) / 2, 'scenarios': scenarios}


def main():
    if MANIFEST.exists():
        raise FileExistsError('Preserve selected future tasks; do not select again')
    excluded = {k: set() for k in ['task_id', 'game_sha256', 'trajectory_sha256']}
    for path in list(Path('repro').rglob('*.json')) + list(Path('runs').glob('**/identity.json')):
        for task in walk_tasks(read_json(path)):
            for key in excluded:
                excluded[key].add(task[key])
    old = read_json(PUBLIC / 'shared-calibration-partitions.json')
    tasks, removed = enumerate_tasks(Path('data/alfworld'), 'train')
    pool = [t for t in tasks if t['task_type'] == 'pick_heat_then_place_in_recep' and
            t['scene_id'] not in old['sealed134_scenes'] and all(t[k] not in excluded[k] for k in excluded)]
    scenes = sorted({t['scene_id'] for t in pool}, key=lambda s: digest([SEED, 'scene', s]))
    assert len(scenes) == 27
    grouped = {s: sorted([t for t in pool if t['scene_id'] == s], key=lambda t: digest([SEED, t['task_id']])) for s in scenes}
    assert min(map(len, grouped.values())) >= 2
    counts = dict.fromkeys(scenes, 2)
    while sum(counts.values()) < 81:
        changed = False
        for s in scenes:
            if counts[s] < len(grouped[s]) and sum(counts.values()) < 81:
                counts[s] += 1
                changed = True
        assert changed
    selected = [t for s in scenes for t in grouped[s][:counts[s]]]
    secondary = [grouped[s][0]['task_id'] for s in scenes]
    assert len({t['task_id'] for t in selected}) == 81
    assert len({t['game_sha256'] for t in selected}) == 81
    assert len({t['trajectory_sha256'] for t in selected}) == 81
    runs = ledger()
    assert sum(v['dispatched'] for v in runs.values()) == sum(v['responses'] for v in runs.values()) == 93186
    forecasts = read_json(PUBLIC / 'four-attempt-planning-forecasts.json')
    assert forecasts['parameters_sha256'] == read_json(PUBLIC / 'shared-validation-results.json')['parameters_sha256']
    result = save(MANIFEST, {
        'scope': 'Selected fresh tasks for a planned four-attempt validation; no outcomes sampled',
        'selection_seed': SEED, 'eligible_tasks': len(pool), 'available_by_scene': dict(Counter(t['scene_id'] for t in pool)),
        'tasks': selected, 'counts_by_scene': counts, 'secondary_recovery_task_ids': secondary,
        'selection': 'All 27 nonsealed heat scenes; two tasks per scene then deterministic capacity-capped round-robin allocation to81; hash order, no outcomes',
        'scene_scope': 'Includes18 calibration scenes and9 previous validation scenes; task identities and games are fresh, scenes are not globally unseen',
        'primary_conditions': ['correct_middle', 'neutral_full'],
        'secondary_conditions': ['repeated_error', 'correct_latest', 'delete_latest'],
        'crossed_seeds': [2130, 2131], 'maximum_attempts_per_trajectory': 4,
        'primary_trajectories': 324, 'secondary_trajectories': 162,
        'expected_trajectories': 486, 'maximum_attempts': 1944,
        'proposed_limits': {'additional_calls_including_checks_and_failures': 100000, 'seconds': 43200},
        'prior_global_calls': 93186, 'parameter_refits': 0, 'new_model_calls': 0,
        'precollection_requirements': 'Four-attempt orchestration/evaluator checks, official reset identity checks, current GPU availability, public protocol/source/forecast freeze before any model call',
        'source_sha256': file_hash(Path(__file__)), 'forecasts_sha256': forecasts['sha256'],
        'dataset_removed': removed,
    })
    plan = planning(list(counts.values()), forecasts)
    old_summary = read_json(PUBLIC / 'shared-validation-collection-summary.json')
    calls_per_attempt = (old_summary['costs_incremental']['dispatched_calls'] - 8) / old_summary['attempts_completed']
    expected_attempts = 0.
    for condition in result['primary_conditions'] + result['secondary_conditions']:
        n = 162 if condition in result['primary_conditions'] else 54
        expected_attempts += n * (4 - sum(forecasts['full_update'][condition][:3]))
    estimated_calls = expected_attempts * calls_per_attempt + 8
    throughput = old_summary['costs_incremental']['dispatched_calls'] / old_summary['elapsed_seconds']
    plan.update(manifest_sha256=result['sha256'], proposed_limits=result['proposed_limits'],
                model_implied_expected_attempts=expected_attempts,
                measured_previous_calls_per_attempt=calls_per_attempt,
                estimated_model_calls=estimated_calls,
                estimated_seconds_at_previous_four_service_throughput=estimated_calls / throughput,
                resource_estimate_scope='Planning extrapolation; current availability and speed must be checked before launch',
                source_sha256=file_hash(Path(__file__)), new_model_calls=0)
    report = save(PUBLIC / 'four-attempt-validation-planning.json', plan)
    print({'tasks': len(selected), 'scenes': len(scenes), 'counts': counts, 'trajectories': 486,
           'manifest_sha256': result['sha256'], 'planning_sha256': report['sha256'],
           'estimated_calls': estimated_calls, 'estimated_hours': estimated_calls / throughput / 3600,
           'scenarios': plan['scenarios']})


if __name__ == '__main__':
    main()
