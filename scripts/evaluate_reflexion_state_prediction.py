#!/usr/bin/env python3
"""Score immutable mixed-window predictions without changing fitted weights."""
import argparse
from pathlib import Path
import numpy as np

from agent_reliability.io import digest, file_hash, read_json, write_json
from reflexion_state_risk import VALIDATION_STATES, log_loss
from summarize_reflexion import costs


def checked(path):
    value = read_json(path)
    if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Artifact integrity: '+str(path))
    return value


def main():
    parser = argparse.ArgumentParser()
    for name in ['protocol', 'parameters', 'predictions', 'validation', 'output']:
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    protocol = checked(args.protocol); parameters = checked(args.parameters)
    predictions = checked(args.predictions); data = checked(args.validation/'summary.json')
    if predictions['parameters_sha256'] != parameters['sha256']:
        raise ValueError('Parameter freeze mismatch')
    if any(v['protocol_sha256'] != protocol['sha256'] for v in [parameters, predictions]):
        raise ValueError('Protocol mismatch')
    for name, expected in protocol['implementation_files'].items():
        if file_hash(Path(name)) != expected:
            raise ValueError('Frozen implementation changed: '+name)
    run = read_json(args.validation/'run.json'); identity = run['identity']
    if identity['phase'] != 'validation' or identity['predictions_sha256'] != predictions['sha256']:
        raise ValueError('Validation identity mismatch')
    if identity['parameters_sha256'] != parameters['sha256'] or run['created_unix'] < predictions['frozen_unix']:
        raise ValueError('Validation predates the freeze')
    audit = read_json(args.validation/'protocol-audit.json')
    if not audit['passed'] or audit['run_sha256'] != file_hash(args.validation/'run.json'):
        raise ValueError('Completed interface audit required')
    expected = {(case['task_id'], state) for case in protocol['cases'] for state in VALIDATION_STATES}
    actual = {(row['task_id'], row['state']) for row in data['rows']}
    predicted = {(row['task_id'], row['state']) for row in predictions['rows']}
    if actual != expected or predicted != expected or len(data['rows']) != len(expected):
        raise ValueError('Mixed-state measurement set incomplete')
    rows = []
    for row in data['rows']:
        prediction = next(p for p in predictions['rows'] if (p['task_id'], p['state']) == (row['task_id'], row['state']))
        if row['memory_sha256'] != prediction['memory_sha256'] or row['labels'] != VALIDATION_STATES[row['state']]:
            raise ValueError('Prediction and realized memory differ')
        if row['reused'] or file_hash(Path(row['result_path'])) != row['result_file_sha256']:
            raise ValueError('Validation evidence changed or reused')
        result = read_json(Path(row['result_path']))
        if result['success'] != row['success']:
            raise ValueError('Validation outcome differs from native result')
        rows.append({**row, 'predicted_success': prediction['predicted_success']})
    y = np.asarray([r['success'] for r in rows], dtype=float)
    metrics = {}
    for model in ['constant', 'count', 'slot']:
        probability = np.asarray([r['predicted_success'][model] for r in rows])
        metrics[model] = {'brier': float(np.mean((probability-y)**2)), 'log_loss': log_loss(y, probability)}
    pairs = []
    for case in protocol['cases']:
        oldest = next(r for r in rows if r['task_id'] == case['task_id'] and r['state'] == 'error_oldest')
        newest = next(r for r in rows if r['task_id'] == case['task_id'] and r['state'] == 'error_newest')
        pairs.append({'task_id': case['task_id'], 'observed_newest_minus_oldest': int(newest['success'])-int(oldest['success']),
                      'predicted_newest_minus_oldest': {m: newest['predicted_success'][m]-oldest['predicted_success'][m]
                                                        for m in metrics}})
    report = {'scope': 'Prospective mixed-window development check on four known tasks; descriptive results',
        'protocol_sha256': protocol['sha256'], 'parameters_sha256': parameters['sha256'],
        'predictions_sha256': predictions['sha256'], 'driver_sha256': file_hash(Path(__file__)),
        'validation_summary_sha256': data['sha256'], 'independent_tasks': len(protocol['cases']),
        'observations': len(rows), 'successes': int(sum(y)), 'metrics': metrics, 'paired_order_results': pairs,
        'rows': rows, 'costs_incremental': costs(args.validation)}
    report['sha256'] = digest(report); write_json(args.output, report)
    print({k: report[k] for k in ['independent_tasks', 'observations', 'successes', 'metrics', 'sha256']})


if __name__ == '__main__':
    main()
