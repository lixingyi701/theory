#!/usr/bin/env python3
"""Prospective task-mean diagnostic for the known-task mixed-window study."""
import argparse
import time
from pathlib import Path
import numpy as np
from agent_reliability.io import digest, file_hash, read_json, write_json
from reflexion_state_risk import VALIDATION_STATES, log_loss


def checked(path):
    value = read_json(path)
    if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Artifact integrity: '+str(path))
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['preregister', 'freeze', 'evaluate'], required=True)
    for name in ['protocol', 'calibration', 'validation', 'output']:
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--registration', type=Path)
    p.add_argument('--frozen', type=Path)
    a = p.parse_args(); protocol = checked(a.protocol)
    if a.output.exists():
        raise FileExistsError('Task-baseline artifacts are immutable')
    if a.phase in ['preregister', 'freeze'] and list(a.validation.glob('**/calls/*/attempt-*.request.json')):
        raise ValueError('Task baseline must precede mixed-state requests')
    code = file_hash(Path(__file__))
    report = {'scope': 'Secondary prospective known-task difficulty baseline; not available for unseen tasks',
              'primary_protocol_sha256': protocol['sha256'], 'driver_sha256': code}
    if a.phase == 'preregister':
        report.update(created_unix=time.time(), estimator='Beta(1,1) smoothed mean over each task\'s seven homogeneous calibration states',
                      purpose='Measure predictive information supplied by known task difficulty without memory order features',
                      validation_calls_at_registration=0)
    elif a.phase == 'freeze':
        registration = checked(a.registration); data = checked(a.calibration/'summary.json')
        if registration['driver_sha256'] != code or registration['primary_protocol_sha256'] != protocol['sha256']:
            raise ValueError('Registered task baseline changed')
        if data['completed_cells'] != 28 or data['completed_tasks'] != 4:
            raise ValueError('Four-task calibration incomplete')
        rows = []
        for case in protocol['cases']:
            observed = [r for r in data['rows'] if r['task_id'] == case['task_id']]
            if len(observed) != 7:
                raise ValueError('Seven calibration states required per task')
            for r in observed:
                if file_hash(Path(r['result_path'])) != r['result_file_sha256']:
                    raise ValueError('Source calibration evidence changed')
            successes = sum(r['success'] for r in observed); probability = (successes+1)/9
            rows.extend({'task_id': case['task_id'], 'state': state, 'predicted_success': probability,
                         'calibration_successes': successes, 'calibration_observations': 7}
                        for state in VALIDATION_STATES)
        report.update(registration_sha256=registration['sha256'], calibration_summary_sha256=data['sha256'],
                      frozen_unix=time.time(), validation_calls_at_freeze=0, rows=rows)
    else:
        frozen = checked(a.frozen); data = checked(a.validation/'summary.json')
        run = read_json(a.validation/'run.json')
        if frozen['driver_sha256'] != code or frozen['primary_protocol_sha256'] != protocol['sha256']:
            raise ValueError('Frozen task baseline changed')
        requests = [read_json(path)['dispatched_unix'] for path in a.validation.glob('**/calls/*/attempt-*.request.json')]
        if not requests or min(requests) <= frozen['frozen_unix'] or run['created_unix'] < frozen['frozen_unix']:
            raise ValueError('Validation predates task-baseline freeze')
        if data['completed_cells'] != 8 or data['completed_tasks'] != 4:
            raise ValueError('Mixed-state validation incomplete')
        y, probabilities = [], []
        for row in data['rows']:
            if file_hash(Path(row['result_path'])) != row['result_file_sha256']:
                raise ValueError('Validation source changed')
            prediction = next(r for r in frozen['rows'] if (r['task_id'], r['state']) == (row['task_id'], row['state']))
            y.append(row['success']); probabilities.append(prediction['predicted_success'])
        report.update(frozen_sha256=frozen['sha256'], validation_summary_sha256=data['sha256'],
                      brier=float(np.mean((np.asarray(probabilities)-np.asarray(y))**2)),
                      log_loss=log_loss(y, probabilities), observations=len(y), independent_tasks=4,
                      first_validation_dispatch_unix=min(requests), frozen_unix=frozen['frozen_unix'])
    report['sha256'] = digest(report); write_json(a.output, report)
    print({k: v for k, v in report.items() if k != 'rows'})


if __name__ == '__main__':
    main()
