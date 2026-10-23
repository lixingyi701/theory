#!/usr/bin/env python3
"""Freeze development risk baselines using homogeneous states only."""
import argparse
import time
from pathlib import Path
import numpy as np

from agent_reliability.io import digest, file_hash, read_json, write_json
from reflexion_state_risk import CALIBRATION_STATES, VALIDATION_STATES, encode, sigmoid, train
from run_reflexion_state_prediction import source_entries


def main():
    parser = argparse.ArgumentParser()
    for name in ['protocol', 'calibration', 'validation-run', 'parameters', 'predictions']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.parameters.exists() or args.predictions.exists():
        raise FileExistsError('Parameter and prediction freezes are immutable')
    if list(args.validation_run.glob('**/calls/*/attempt-*.request.json')):
        raise ValueError('Validation requests already exist')
    protocol = read_json(args.protocol); data = read_json(args.calibration/'summary.json')
    if protocol['sha256'] != digest({k: v for k, v in protocol.items() if k != 'sha256'}):
        raise ValueError('Protocol integrity')
    encoder = file_hash(Path('scripts/reflexion_state_risk.py'))
    for name, expected in protocol['implementation_files'].items():
        if file_hash(Path(name)) != expected:
            raise ValueError('Controlled-state implementation changed: '+name)
    if encoder != protocol['state_encoder_sha256']:
        raise ValueError('Prediction encoder changed')
    if data['sha256'] != digest({k: v for k, v in data.items() if k != 'sha256'}):
        raise ValueError('Calibration summary integrity')
    if data['completed_tasks'] != len(protocol['cases']) or data['completed_cells'] != len(protocol['cases'])*7:
        raise ValueError('Complete homogeneous calibration measurements required')
    if {(r['task_id'], r['state']) for r in data['rows']} != {(c['task_id'], s) for c in protocol['cases'] for s in CALIBRATION_STATES}:
        raise ValueError('Calibration cell set differs from the declared design')
    audit = read_json(args.calibration/'protocol-audit.json')
    identity = read_json(args.calibration/'run.json')['identity']
    if identity['protocol_sha256'] != protocol['sha256'] or identity['phase'] != 'calibration':
        raise ValueError('Calibration run belongs to another protocol')
    if not audit['passed'] or audit['run_sha256'] != file_hash(args.calibration/'run.json'):
        raise ValueError('Calibration interface audit required')
    for row in data['rows']:
        if row['labels'] != CALIBRATION_STATES[row['state']] or file_hash(Path(row['result_path'])) != row['result_file_sha256']:
            raise ValueError('Calibration labels or source measurements changed')
    models = {model: train(data['rows'], model) for model in ['count', 'slot']}
    if any(model['design_rank'] != len(model['weights']) for model in models.values()):
        raise ValueError('Declared calibration design is rank deficient')
    probability = (sum(r['success'] for r in data['rows'])+1)/(len(data['rows'])+2)
    now = time.time()
    parameters = {'scope': 'One-attempt development risk baselines; learned on four known development tasks',
        'protocol_sha256': protocol['sha256'], 'calibration_summary_sha256': data['sha256'],
        'calibration_run_sha256': file_hash(args.calibration/'run.json'),
        'state_encoder_sha256': encoder, 'fit_driver_sha256': file_hash(Path(__file__)),
        'numpy_version': np.__version__, 'models': models, 'constant_probability': probability,
        'independent_tasks': len(protocol['cases']), 'observations': len(data['rows']),
        'frozen_unix': now, 'validation_calls_at_freeze': 0}
    parameters['sha256'] = digest(parameters)
    rows = []
    for case in protocol['cases']:
        _, templates = source_entries(case)
        for state, labels in VALIDATION_STATES.items():
            values = {model: float(sigmoid(np.asarray(encode(labels, model))@np.asarray(fitted['weights'])))
                      for model, fitted in models.items()}
            values['constant'] = probability
            rows.append({'task_id': case['task_id'], 'state': state, 'labels': labels,
                         'memory_sha256': digest([templates[label] for label in labels]), 'predicted_success': values})
    predictions = {'scope': 'Prospective mixed-state one-attempt predictions on the same four development tasks',
        'protocol_sha256': protocol['sha256'], 'parameters_sha256': parameters['sha256'],
        'frozen_unix': now, 'rows': rows, 'validation_calls_at_freeze': 0}
    predictions['sha256'] = digest(predictions)
    write_json(args.parameters, parameters); write_json(args.predictions, predictions)
    print({'tasks': len(protocol['cases']), 'calibration_cells': len(data['rows']),
           'design_ranks': {m: f['design_rank'] for m, f in models.items()},
           'parameters_sha256': parameters['sha256'], 'predictions_sha256': predictions['sha256']})


if __name__ == '__main__':
    main()
