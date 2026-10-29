#!/usr/bin/env python3
"""Predeclare controlled memory dose calibration and held-out order tests."""
import argparse
import time
from pathlib import Path

from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, write_json
from run_reflexion_precondition_forks import repair_text
from reflexion_state_risk import CALIBRATION_STATES, VALIDATION_STATES, FEATURES, RIDGES


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Prediction protocol immutable')
    sources = [
        ('repro/reflexion-20261023/precondition-diagnostic-target.json',
         'repro/reflexion-20261023/pilot-prefix24-v2.json', 'runs/reflexion-normalized-prefix24-20261023',
         'runs/reflexion-precondition-diagnostic-forks-20261023'),
        ('repro/reflexion-20261023/precondition-targets.json',
         'repro/reflexion-20261023/precondition-prefix24-v1.json', 'runs/reflexion-precondition-prefix24-20261023',
         'runs/reflexion-precondition-forks-20261023'),
    ]
    cases = []
    for target_file, manifest_file, parent_run, evidence_run in sources:
        registry = read_json(Path(target_file)); manifest = load_manifest(Path(manifest_file))
        tasks = {t['task_id']: t for t in manifest['tasks']}
        for target in registry['targets']:
            parent = Path(parent_run)/'tasks'/target['task_id']/'trial-00/result.json'
            result = read_json(parent)
            if len(result['memory_after']) != 1 or result['memory_before'] or result['success']:
                raise ValueError('Single native reflection after first failure required')
            if file_hash(parent) != target['parent_result_sha256']:
                raise ValueError('Reviewed parent changed')
            error = result['memory_after'][0]; correct = repair_text(error, target)
            cases.append({'task_id': target['task_id'], 'task': tasks[target['task_id']],
                'target_registry': target_file, 'target_registry_sha256': registry['sha256'],
                'target_sha256': target['sha256'], 'parent_run': parent_run,
                'parent_result_sha256': file_hash(parent), 'evidence_run': evidence_run,
                'evidence_protocol_audit': 'runs/reflexion-precondition-20261023/'+('diagnostic-protocol-audit.json' if 'diagnostic-forks' in evidence_run else 'prospective-forks-protocol-audit.json'),
                'error_entry_sha256': digest(error), 'correct_entry_sha256': digest(correct)})
    if len({c['task_id'] for c in cases}) != len(cases) or len(cases) != 4:
        raise ValueError('Four existing independent development cases required')
    report = {'scope': 'Controlled memory-state development experiment with deliberate entry duplication',
        'endpoint': 'Native won in one post-recovery attempt with 49 author decisions; the native updater still runs after failure',
        'cases': cases, 'calibration_states': CALIBRATION_STATES, 'validation_states': VALIDATION_STATES,
        'reused_states': {'empty': 'discard_new/trial-01', 'error_1': 'original/trial-01', 'correct_1': 'correct_target/trial-01'},
        'state_semantics': 'E/C refers only to the reviewed target precondition. Other native advice is retained in both entries.',
        'fit': {'models': FEATURES, 'ridge_grid': RIDGES, 'selection': 'Leave-one-task-out log loss; ties prefer stronger regularization',
                'penalty': 'Gaussian L2 on all coefficients including intercept', 'constant_baseline': 'Beta(1,1) smoothed calibration mean'},
        'holdout_rule': 'Freeze parameters and both mixed-state predictions before their first dispatch. No mixed-state outcomes used in fitting.',
        'limits': {'calibration_new_calls': 1400, 'validation_new_calls': 800, 'whole_campaign_calls': 6000},
        'seed': 42, 'post_recovery_trial': 1, 'frozen_unix': time.time(),
        'preparer_sha256': file_hash(Path(__file__)), 'state_encoder_sha256': file_hash(Path('scripts/reflexion_state_risk.py'))}
    report['implementation_files'] = {str(Path('scripts')/name): file_hash(Path('scripts')/name) for name in [
        'reflexion_state_risk.py', 'prepare_reflexion_state_prediction.py', 'run_reflexion_state_prediction.py',
        'fit_reflexion_state_prediction.py', 'evaluate_reflexion_state_prediction.py',
        'run_reflexion_precondition_forks.py']}
    report['sha256'] = digest(report); write_json(args.output, report)
    print({'tasks': len(cases), 'calibration_states': len(CALIBRATION_STATES),
           'validation_states': len(VALIDATION_STATES), 'sha256': report['sha256']})


if __name__ == '__main__':
    main()
