#!/usr/bin/env python3
"""Collect declared homogeneous calibration or frozen mixed-state tests."""
import argparse
import time
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from continue_reflexion_pilot import ledger
from reflexion_campaign import verify, permit
from reflexion_state_risk import CALIBRATION_STATES, VALIDATION_STATES, NEW_CALIBRATION_STATES, REUSED_BRANCHES
from run_reflexion_precondition_forks import repair_text
from summarize_reflexion import costs


def source_entries(case):
    registry = read_json(Path(case['target_registry']))
    if registry['sha256'] != case['target_registry_sha256']:
        raise ValueError('Source target registry changed')
    target = next(t for t in registry['targets'] if t['task_id'] == case['task_id'])
    if target['sha256'] != case['target_sha256']:
        raise ValueError('Source semantic target changed')
    parent_file = Path(case['parent_run'])/'tasks'/case['task_id']/'trial-00/result.json'
    if file_hash(parent_file) != case['parent_result_sha256']:
        raise ValueError('Source native attempt changed')
    parent = read_json(parent_file); error = parent['memory_after'][-1]
    correct = repair_text(error, target)
    if digest(error) != case['error_entry_sha256'] or digest(correct) != case['correct_entry_sha256']:
        raise ValueError('Controlled memory template changed')
    return parent, {'E': error, 'C': correct}


def observed_row(case, state, labels, path, memory, contract, reused):
    identity = read_json(path/'identity.json'); result = read_json(path/'result.json')
    if identity['memory'] != memory or identity['trial'] != 1 or identity['seed'] != 42 or identity['task'] != case['task']:
        raise ValueError('Evidence does not match the declared state/origin/seed')
    if identity['input_format'] != 'qwen3_non_thinking_actionline' or identity['decoding_profile'] != contract['decoding_profile']:
        raise ValueError('Evidence model input profile mismatch')
    if result['identity_sha256'] != identity['sha256'] or identity['sha256'] != digest({k: v for k, v in identity.items() if k != 'sha256'}):
        raise ValueError('Evidence identity mismatch')
    if result['trajectory_sha256'] != digest(read_json(path/'checkpoint.json')['transitions']):
        raise ValueError('Evidence trajectory changed')
    actual = costs(path)
    if actual['uncertain_or_unrecorded_attempts'] or actual['dispatched_calls'] != result['dispatched_calls']:
        raise ValueError('Evidence requests incomplete')
    return {'task_id': case['task_id'], 'state': state, 'labels': labels, 'success': result['success'],
        'decisions': result['decisions'], 'memory_sha256': digest(memory), 'reused': reused,
        'result_path': str(path/'result.json'), 'result_file_sha256': file_hash(path/'result.json'),
        'source_calls': actual['dispatched_calls']}


def main():
    parser = argparse.ArgumentParser()
    for name in ['campaign', 'protocol', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase', choices=['calibration', 'validation'], required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    with run_lock(args.campaign.resolve()), run_lock(args.output):
        contract = verify(args.campaign); protocol = read_json(args.protocol)
        if protocol['sha256'] != digest({k: v for k, v in protocol.items() if k != 'sha256'}):
            raise ValueError('Controlled-state protocol integrity')
        if protocol['state_encoder_sha256'] != file_hash(Path('scripts/reflexion_state_risk.py')):
            raise ValueError('Feature encoder changed after protocol freeze')
        for name, expected in protocol['implementation_files'].items():
            if file_hash(Path(name)) != expected:
                raise ValueError('Controlled-state implementation changed: '+name)
        if protocol['calibration_states'] != CALIBRATION_STATES or protocol['validation_states'] != VALIDATION_STATES:
            raise ValueError('Declared and implemented memory states differ')
        if any(r['dispatched'] != r['responses'] for r in ledger().values()):
            raise ValueError('Resolve preceding requests first')
        predictions = None
        if args.phase == 'validation':
            if not args.predictions or not args.parameters:
                raise ValueError('Frozen predictions and parameters required')
            predictions = read_json(args.predictions); parameters = read_json(args.parameters)
            for value in [predictions, parameters]:
                if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
                    raise ValueError('Prediction freeze integrity')
                if value['protocol_sha256'] != protocol['sha256']:
                    raise ValueError('Prediction protocol changed')
            if predictions['parameters_sha256'] != parameters['sha256']:
                raise ValueError('Frozen parameter identity mismatch')
            if parameters['state_encoder_sha256'] != file_hash(Path('scripts/reflexion_state_risk.py')):
                raise ValueError('Frozen prediction implementation changed')
        _, upstream = author_namespace()
        maximum = protocol['limits']['calibration_new_calls' if args.phase == 'calibration' else 'validation_new_calls']
        identity = {'phase': args.phase, 'protocol_sha256': protocol['sha256'], 'campaign_sha256': contract['sha256'],
            'source_hashes': contract['source_hashes'], 'driver_sha256': file_hash(Path(__file__)),
            'state_encoder_sha256': protocol['state_encoder_sha256'],
            'memory_repair_driver_sha256': file_hash(Path('scripts/run_reflexion_precondition_forks.py')),
            'cost_helper_sha256': file_hash(Path('scripts/summarize_reflexion.py')),
            'input_format': 'qwen3_non_thinking_actionline', 'decoding_profile': contract['decoding_profile'],
            'serving': contract['serving'], 'upstream': upstream, 'environment_action_alias': ACTION_ALIAS_PROFILE,
            'wrapper_sha256': file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256': file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')),
            'trials': 1, 'seed': 42, 'max_calls': maximum}
        if predictions:
            identity.update(predictions_sha256=predictions['sha256'], parameters_sha256=parameters['sha256'])
        if args.validate_only:
            for case in protocol['cases']:
                _, templates = source_entries(case)
                audit = read_json(Path(case['evidence_protocol_audit']))
                if not audit['passed'] or audit['run_sha256'] != file_hash(Path(case['evidence_run'])/'run.json'):
                    raise ValueError('Source evidence audit incomplete')
                if args.phase == 'calibration':
                    for state, branch in REUSED_BRANCHES.items():
                        path = Path(case['evidence_run'])/'targets'/case['task_id']/'branches'/branch/'trial-01'
                        memory = [templates[label] for label in CALIBRATION_STATES[state]]
                        observed_row(case, state, CALIBRATION_STATES[state], path, memory, contract, True)
            print({'validated_cases': len(protocol['cases']), 'phase': args.phase,
                   'identity_sha256': digest(identity), 'model_calls': 0})
            return
        run_file = args.output/'run.json'
        if run_file.exists():
            if read_json(run_file)['identity'] != identity:
                raise ValueError('Controlled-state run identity changed')
        else:
            remaining = contract['prior_dispatched_calls']+contract['additional_max_calls']-sum(r['dispatched'] for r in ledger().values())
            if maximum > remaining:
                raise ValueError('Controlled-state reservation exceeds campaign allowance')
            write_json(run_file, {'identity': identity, 'created_unix': time.time(), 'scope': protocol['scope']})
        states = CALIBRATION_STATES if args.phase == 'calibration' else VALIDATION_STATES
        rows = []
        transport = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)

        def allowed():
            permit(contract)
            if len(list(args.output.glob('**/calls/*/attempt-*.request.json'))) >= maximum:
                raise RuntimeError('Controlled-state phase call limit')

        for case in protocol['cases']:
            parent, templates = source_entries(case)
            source = read_json(Path(case['evidence_run'])/'run.json')['identity']
            if source['source_hashes'] != contract['source_hashes'] or source['serving'] != contract['serving']:
                raise ValueError('Reused evidence execution profile mismatch')
            audit = read_json(Path(case['evidence_protocol_audit']))
            if not audit['passed'] or audit['run_sha256'] != file_hash(Path(case['evidence_run'])/'run.json'):
                raise ValueError('Reused evidence protocol audit mismatch')
            for state, labels in states.items():
                memory = [templates[label] for label in labels]
                reused = args.phase == 'calibration' and state in REUSED_BRANCHES
                if reused:
                    path = Path(case['evidence_run'])/'targets'/case['task_id']/'branches'/REUSED_BRANCHES[state]/'trial-01'
                else:
                    path = args.output/'cases'/case['task_id']/state/'trial-01'
                    if predictions:
                        prediction = next(row for row in predictions['rows'] if row['task_id'] == case['task_id'] and row['state'] == state)
                        if prediction['memory_sha256'] != digest(memory):
                            raise ValueError('Prediction assigned to another memory state')
                        if read_json(run_file)['created_unix'] < predictions['frozen_unix']:
                            raise ValueError('Validation started before predictions were frozen')
                    result = run_attempt(case['task'], memory, 1, path, transport, Path('data/alfworld'), 42, allowed,
                        input_format=identity['input_format'], decoding_profile=identity['decoding_profile'])
                    if result['initial_observation_sha256'] != parent['initial_observation_sha256']:
                        raise ValueError('Controlled-state environment reset changed')
                rows.append(observed_row(case, state, labels, path, memory, contract, reused))
                complete = sum(sum(r['task_id'] == c['task_id'] for r in rows) == len(states) for c in protocol['cases'])
                report = {'scope': protocol['scope'], 'phase': args.phase, 'expected_tasks': len(protocol['cases']),
                    'completed_tasks': complete, 'expected_cells': len(protocol['cases'])*len(states),
                    'completed_cells': len(rows), 'reused_cells': sum(r['reused'] for r in rows),
                    'rows': rows, 'costs_incremental': costs(args.output)}
                report['sha256'] = digest(report); write_json(args.output/'summary.json', report)
                print({'phase': args.phase, 'task_id': case['task_id'], 'state': state,
                       'success': rows[-1]['success'], 'reused': reused, 'completed_cells': len(rows)}, flush=True)


if __name__ == '__main__':
    main()
