#!/usr/bin/env python3
"""Close the finite multi-attempt diagnostic against its raw chained evidence."""
import argparse
import re
import shutil
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_multiround_diagnostic import check_digest, collect, require
from continue_reflexion_pilot import ledger
from run_multiround_diagnostic import verify_protocol
from summarize_reflexion import costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261024/multiround-diagnostic-protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.freeze_commit) is not None, 'Complete freeze commit SHA is required')
    root = args.run
    protocol = verify_protocol(args.protocol)
    summary, analysis = read_json(root/'summary.json'), read_json(root/'analysis.json')
    contract, audit = read_json(root/'contract.json'), read_json(root/'protocol-audit.json')
    replica, labels = read_json(root/'replica-audit.json'), read_json(root/'semantic-labels.json')
    for record, name in [(protocol, 'Protocol'), (summary, 'Summary'), (analysis, 'Analysis'),
                         (contract, 'Contract'), (audit, 'Protocol audit'), (replica, 'Replica audit'), (labels, 'Semantic labels')]:
        check_digest(record, name)
    require(analysis['protocol_sha256'] == summary['protocol_sha256'] == contract['protocol_sha256']
            == labels['protocol_sha256'] == protocol['sha256'], 'Artifact/protocol identity differs')
    require(analysis['summary_sha256'] == summary['sha256'] and analysis['semantic_labels_sha256'] == labels['sha256'],
            'Analysis binds another summary or label registry')
    require(audit['passed'] and not audit['errors'] and replica['passed'], 'Request or replica audit requires inspection')
    require(summary['worker_exit_codes'] == [0, 0, 0], 'A worker did not complete normally')
    require(analysis['completed_trajectories'] == 72 and analysis['parameter_refits'] == 0,
            'Expected 72 complete trajectories with no parameter refit')
    label_map = {entry['sha256']: entry['label'] for entry in labels['entries']}
    current, trajectories, review = collect(protocol, root, label_map)
    require(len(trajectories) == 72 and all(row['complete'] for row in trajectories), 'Raw trajectories remain incomplete')
    require(len(current) == analysis['attempts_completed'] == len(analysis['rows']), 'Raw attempt coverage differs')
    require(analysis['trajectories'] == trajectories and analysis['rows'] == current, 'Raw evidence changed after analysis')
    require(set(review) == set(label_map) and len(label_map) == len(labels['entries']), 'Semantic coverage differs')
    require(len(summary['trajectories']) == 72 and len(summary['attempts_rows']) == len(current), 'Worker summary coverage differs')
    actual_trajectories = {row['trajectory_id']: row for row in trajectories}
    require(len({row['trajectory_id'] for row in summary['trajectories']}) == 72, 'Duplicate worker trajectory records')
    for row in summary['trajectories']:
        peer = actual_trajectories[row['trajectory_id']]
        for key in ('task_id', 'seed', 'history_key', 'operation', 'initial_labels', 'initial_memory_sha256',
                    'complete', 'success', 'attempts_completed'):
            require(row[key] == peer[key], 'Worker trajectory differs from raw chain: '+row['trajectory_id'])
        require([value['result_file_sha256'] for value in row['attempts']] == peer['result_file_sha256s'],
                'Worker trajectory result bindings differ')
    actual_attempts = {(row['trajectory_id'], row['trial']): row for row in current}
    require(len({(row['trajectory_id'], row['trial']) for row in summary['attempts_rows']}) == len(current),
            'Duplicate worker attempt records')
    for row in summary['attempts_rows']:
        peer = actual_attempts[row['trajectory_id'], row['trial']]
        require(all(row[key] == peer[key] for key in ('task_id', 'seed', 'history_key', 'operation', 'success', 'decisions',
                                                     'identity_sha256', 'trajectory_sha256', 'visible_memory_sha256')),
                'Worker attempt differs from raw evidence')
        require(row['result_file_sha256'] == peer['result_sha256'] and row['memory_before_sha256'] == peer['memory_sha256']
                and row['new_reflection_sha256'] == peer['new_reflection_hashes'], 'Worker attempt binding differs')
    actual = costs(root)
    require(actual == summary['costs_incremental'], 'Journal costs differ from summary')
    reserved = read_json(root/'budget.json')['reserved']
    require(reserved == actual['dispatched_calls'] <= contract['max_calls'] == 11000, 'Call reservations/cap differ')
    require(actual['saved_responses'] == actual['dispatched_calls'] and actual['uncertain_or_unrecorded_attempts'] == 0,
            'Unconfirmed model requests remain')
    require(contract['deadline_unix']-contract['created_unix'] == 10800, 'Declared execution envelope must be three hours')
    timestamps = [read_json(path)['dispatched_unix'] for path in root.glob('**/calls/*/attempt-*.request.json')]
    require(len(timestamps) == actual['dispatched_calls'] and bool(timestamps), 'Dispatch timestamp coverage differs')
    require(min(timestamps) >= contract['created_unix'] and max(timestamps) < contract['deadline_unix'], 'Dispatch outside envelope')
    require(summary['elapsed_seconds'] >= max(timestamps)-contract['created_unix'], 'Elapsed time predates last dispatch')
    prior, current_ledger = contract['prior_ledger'], ledger()
    require(all(value['dispatched'] == value['responses'] for value in prior.values()), 'Prior ledger unresolved')
    before = sum(value['dispatched'] for value in prior.values())
    require(before == 26673, 'Unexpected preceding experiment count')
    all_calls = sum(value['dispatched'] for value in current_ledger.values())
    all_responses = sum(value['responses'] for value in current_ledger.values())
    require(all_calls == all_responses == before+actual['dispatched_calls'], 'Global ledger delta differs from campaign')
    name = root.resolve().name
    for other in set(prior) | set(current_ledger):
        if other != name:
            require(prior.get(other, {'dispatched': 0, 'responses': 0}) ==
                    current_ledger.get(other, {'dispatched': 0, 'responses': 0}), 'Another experiment journal changed: '+other)
    require(len(replica['rows']) == 2*len(protocol['ports']) == 6, 'Replica-check coverage differs')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for source, target_name in [('contract', 'contract'), ('run', 'run'), ('summary', 'raw-summary'),
                                ('analysis', 'results'), ('replica-audit', 'replica-audit'),
                                ('protocol-audit', 'protocol-audit'), ('action-shapes', 'action-shapes'), ('semantic-labels', 'semantic-labels')]:
        origin = root/(source+'.json')
        target = output/('multiround-diagnostic-'+target_name+'.json')
        if target.exists():
            require(target.read_bytes().replace(b'\r\n', b'\n') == origin.read_bytes().replace(b'\r\n', b'\n'),
                    'Existing published evidence differs: '+str(target))
        else:
            shutil.copyfile(origin, target)
        artifacts[target.name] = file_hash(target)
    value = {'protocol_sha256': protocol['sha256'], 'freeze_commit': args.freeze_commit,
        'stop_reason': 'frozen_multiround_diagnostic_complete', 'expected_trajectories': 72, 'completed_trajectories': 72,
        'attempts_completed': len(current), 'successes': analysis['successes'], 'independent_base_tasks': 12,
        'seed_replicates_per_task': 1, 'new_base_tasks': 0, 'max_attempts': 3,
        'costs_incremental': actual, 'replica_check_calls': 6, 'reserved_calls': reserved, 'prior_calls': before,
        'all_calls': all_calls, 'all_responses': all_responses, 'elapsed_seconds': summary['elapsed_seconds'],
        'dispatch_envelope_seconds': 10800, 'holdout_model_calls': 0, 'parameter_refits': 0,
        'ports': protocol['ports'], 'worker_exit_codes': summary['worker_exit_codes'],
        'request_configuration_errors': len(audit['errors']), 'artifacts': artifacts,
        'driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value)
    target = output/'multiround-diagnostic-closeout.json'
    if target.exists():
        require(read_json(target) == value, 'Existing closeout differs')
    else:
        write_json(target, value)
    print({key: item for key, item in value.items() if key != 'artifacts'})


if __name__ == '__main__':
    main()
