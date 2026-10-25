#!/usr/bin/env python3
"""Bind the selected tasks to reset checks, two replicas and immutable predictions."""
from pathlib import Path
import numpy as np
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json
from agent_reliability.reference import official_environment
from continue_reflexion_pilot import ledger
from prepare_shared_calibration import STATES, ENTRIES, save
from shared_recovery_model import budget_probability, component_kernel, PRIOR

PUBLIC = Path('repro/reflexion-20261025')
MANIFEST = PUBLIC / 'four-attempt-validation-manifest.json'
PROTOCOL = PUBLIC / 'four-attempt-validation-protocol.json'
ROOT = Path('runs/reflexion-four-attempt-validation-20261025')
PRIMARY = ['correct_middle', 'neutral_full']
SECONDARY = ['repeated_error', 'correct_latest', 'delete_latest']
SEEDS = [2130, 2131]
LIMITS = {'additional_calls_including_replica_checks': 100000, 'seconds': 86400}
FILES = [
    'scripts/prepare_four_attempt_protocol.py', 'scripts/run_four_attempt_validation.py',
    'scripts/evaluate_four_attempt_validation.py', 'scripts/run_four_attempt_campaign.sh',
    'tests/test_four_attempt_validation.py', 'scripts/run_shared_calibration.py',
    'scripts/run_multiround_diagnostic.py', 'scripts/run_middle_context_transport.py',
    'scripts/run_residual_context.py', 'scripts/continue_reflexion_pilot.py',
    'scripts/summarize_reflexion.py', 'scripts/shared_recovery_model.py',
    'scripts/prepare_shared_calibration.py', 'scripts/audit_reflexion_protocol.py',
    'scripts/audit_reflexion_action_shapes.py', 'scripts/deployment/serve_reflexion_profiles.py',
    'configs/reflexion_qwen3_8b_protocol.json', 'configs/qwen3_8b_non_thinking_wrapper.json',
    'docs/experiments/four-attempt-validation-design-20261025.md',
    str(MANIFEST), str(PUBLIC / 'four-attempt-planning-forecasts.json'),
    str(PUBLIC / 'four-attempt-validation-planning.json'),
    str(PUBLIC / 'shared-validation-protocol.json'),
    str(PUBLIC / 'shared-calibration-parameters.json'),
    'repro/baseline-20261019/holdout134.json',
]


def checked(path):
    value = read_json(path)
    if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Artifact digest: ' + str(path))
    return value


def make_trajectories(cases, secondary_ids):
    trajectories = []
    for index, case in enumerate(cases):
        conditions = PRIMARY + (SECONDARY if case['task_id'] in secondary_ids else [])
        for seed_index, seed in enumerate(SEEDS):
            shift = (index + seed_index) % len(conditions)
            order = conditions[shift:] + conditions[:shift]
            for condition in order:
                labels = STATES[condition]
                trajectories.append({'id': f'four-{case["task_id"]}-{seed}-{condition}',
                    'task_id': case['task_id'], 'seed': seed, 'condition': condition,
                    'panel': 'primary' if condition in PRIMARY else 'recovery',
                    'history_key': condition, 'operation': 'initial_state',
                    'initial_labels': labels, 'initial_memory_sha256': digest([ENTRIES[k] for k in labels]),
                    'max_attempts': 4, 'worker': seed_index})
    return trajectories


def forecasts(models):
    output = {}
    for name, model in models.items():
        output[name] = {}
        for condition in PRIMARY + SECONDARY:
            state = [{'type': k, 'native': False} for k in STATES[condition]]
            output[name][condition] = [budget_probability(model, state, b) for b in range(1, 5)]
    output['unchanged_memory'] = {}
    for condition in PRIMARY + SECONDARY:
        state = [{'type': k, 'native': False} for k in STATES[condition]]
        p = component_kernel(models['full_update'], state)[:, 0]
        output['unchanged_memory'][condition] = [float(PRIOR @ (1 - (1-p)**b)) for b in range(1, 5)]
    return output


def main():
    if PROTOCOL.exists() or (ROOT / 'contract.json').exists():
        raise FileExistsError('Preserve the frozen protocol and selected task identities')
    manifest = checked(MANIFEST)
    parameters = checked(PUBLIC / 'shared-calibration-parameters.json')
    prior = checked(PUBLIC / 'shared-validation-protocol.json')
    planned = checked(PUBLIC / 'four-attempt-planning-forecasts.json')
    before = ledger()
    if sum(v['dispatched'] for v in before.values()) != 93186 or any(v['dispatched'] != v['responses'] for v in before.values()):
        raise ValueError('Precollection ledger changed')
    selected_ids = {t['task_id'] for t in manifest['tasks']}
    for path in Path('runs').glob('**/identity.json'):
        identity = read_json(path)
        if identity.get('task', {}).get('task_id') in selected_ids:
            raise ValueError('Selected task already executed: ' + str(path))
    cases, resets = [], []
    for task in manifest['tasks']:
        observations = []
        for seed in SEEDS:
            env = official_environment(Path('data/alfworld'), task, seed, Path('third_party/react'))
            try:
                ob, _ = env.reset()
                observations.append(digest('\n'.join(ob[0].split('\n\n')[1:])))
            finally:
                env.close()
            resets.append({'task_id': task['task_id'], 'seed': seed, 'observation_sha256': observations[-1]})
        if len(set(observations)) != 1:
            raise ValueError('Seed-dependent initial observation')
        cases.append({'task_id': task['task_id'], 'task': task, 'initial_observation_sha256': observations[0]})
    reset_report = save(PUBLIC / 'four-attempt-reset-checks.json', {'checks': resets, 'passed': True,
        'scope': '162 official-environment reset identity checks; zero model calls', 'manifest_sha256': manifest['sha256']})
    models = {('full_update' if k == 'selected' else k): v for k, v in prior['models'].items()}
    predictions = forecasts(models)
    for name in ['full_update', 'unchanged_memory']:
        for condition in PRIMARY + SECONDARY:
            if not np.allclose(predictions[name][condition], planned[name][condition], atol=1e-12, rtol=0):
                raise ValueError('Planning forecast changed')
    value = {'scope': 'Prospective four-attempt budget prediction on fresh tasks, with native failure updating',
        'parent_calibration_parameters_sha256': parameters['sha256'], 'manifest_sha256': manifest['sha256'],
        'planning_forecasts_sha256': planned['sha256'], 'reset_checks_sha256': reset_report['sha256'],
        'cases': cases, 'trajectories': make_trajectories(cases, set(manifest['secondary_recovery_task_ids'])),
        'primary_conditions': PRIMARY, 'secondary_conditions': SECONDARY, 'secondary_recovery_task_ids': manifest['secondary_recovery_task_ids'],
        'crossed_seeds': SEEDS, 'states': {c: STATES[c] for c in PRIMARY + SECONDARY}, 'entries': ENTRIES,
        'entry_hashes': {k: digest(v) for k, v in ENTRIES.items()}, 'models': models, 'forecasts': predictions,
        'ports': [8003, 8004], 'gpu_indices': [2, 3], 'tasks': 81, 'scenes': 27,
        'expected_trajectories': 486, 'maximum_attempts': 1944, 'replica_check_calls': 4,
        'limits': LIMITS, 'prior_global_calls': 93186, 'parameter_refits': 0, 'holdout134_model_calls': 0,
        'resource_revision': 'User restricted execution to GPU2/3 on2026-10-25. Before collection, extend proposed wall envelope from43200 to86400 seconds for two replicas; call cap and task matrix unchanged. Preserve all existing processes.',
        'estimated_two_replica_hours': 11.38,
        'schedule': 'Each task has crossed seeds2130/2131 assigned to workers0/1. Within task/seed rotate conditions by(task_index+seed_index) modulo condition_count.',
        'analysis': {'primary': 'Four-budget Brier: unchanged_memory minus full_update, primary two conditions only',
            'aggregation': 'Equal conditions and seeds within task; equal tasks within scene; equal27 scenes',
            'student_t_df': 26, 'student_t95_critical': 2.055529439,
            'bootstrap_replicates': 20000, 'bootstrap_seed': 2026102504,
            'decision': 'Positive lower endpoint favors full update; negative upper endpoint favors unchanged memory; crossing0 is inconclusive',
            'secondary': ['budgets1..4', 'constant_persistent', 'constant_pooled', 'error_pooled', 'recovery contrasts on paired27-task panel'],
            'native_updates': 'All real trajectories update after failure; unchanged_memory is a predictor, not another execution arm',
            'truncation': 'Retain all sampled records; no outcome-based resampling',
            'missing': 'Require complete486-trajectory matrix; do not impute or score a partial matrix',
            'original_entries': 'Record original-entry identity/exit separately from labels assigned later to new reflections'},
        'source_hashes': provenance()['source_hashes'],
        'protected_files': {p: file_hash(Path(p)) for p in FILES + [str(PUBLIC / 'four-attempt-reset-checks.json')]}}
    report = save(PROTOCOL, value)
    print({'protocol_sha256': report['sha256'], 'reset_checks': len(resets), 'trajectories': len(report['trajectories']),
           'new_model_calls': 0, 'gpu_indices': report['gpu_indices'], 'limits': LIMITS})


if __name__ == '__main__':
    main()
