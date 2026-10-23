#!/usr/bin/env python3
"""Close the finite residual-context campaign and preserve its evidence bindings."""
import argparse
import re
import shutil
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_residual_context import collect
from continue_reflexion_pilot import ledger
from run_residual_context import verify_protocol
from summarize_reflexion import costs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_digest(value, description):
    require(value.get('sha256') == digest({k: v for k, v in value.items() if k != 'sha256'}), description+' integrity')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261024/residual-context-protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze-commit', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{40}', args.freeze_commit) is not None, 'A complete freeze commit SHA is required')
    root = args.run
    protocol = verify_protocol(args.protocol)
    summary, analysis = read_json(root/'summary.json'), read_json(root/'analysis.json')
    contract, audit = read_json(root/'contract.json'), read_json(root/'protocol-audit.json')
    replica = read_json(root/'replica-audit.json')
    labels = read_json(root/'semantic-labels.json')
    for value, name in [(protocol, 'Protocol'), (summary, 'Summary'), (analysis, 'Analysis'),
                        (contract, 'Contract'), (audit, 'Protocol audit'), (replica, 'Replica audit'), (labels, 'Semantic labels')]:
        check_digest(value, name)
    require(analysis['protocol_sha256'] == summary['protocol_sha256'] == contract['protocol_sha256']
            == labels['protocol_sha256'] == protocol['sha256'], 'Artifact/protocol identity mismatch')
    require(analysis['summary_sha256'] == summary['sha256'], 'Analysis refers to another summary')
    require(analysis['semantic_labels_sha256'] == labels['sha256'], 'Analysis refers to other semantic labels')
    require(audit['passed'] and not audit['errors'] and replica['passed'], 'Request or replica audit requires inspection')
    require(analysis['completed_cells'] == 96 and analysis['new_cells'] == 72 and analysis['reused_controls'] == 24,
            'Complete analysis requires 72 new and 24 reused cells')
    require(summary.get('completed_new_cells', summary.get('completed_cells')) == 72,
            'Run summary does not contain all 72 newly executed cells')
    require(summary['worker_exit_codes'] == [0, 0], 'Worker did not complete normally')
    require(len({r['cell_id'] for r in analysis['rows']}) == 96, 'Duplicate or missing analysis rows')
    require(sum(not r['reused_control'] for r in analysis['rows']) == 72, 'New/reused row mismatch')
    require(len({r['task_id'] for r in analysis['rows']}) == 12, 'Independent task count differs')
    require(analysis['parameter_refits'] == 0, 'This stage must not refit frozen predictions')
    current_rows, _, _ = collect(protocol, root)
    require(len(current_rows) == 96, 'Raw attempts no longer cover the complete factorial')
    analysed = {r['cell_id']: r for r in analysis['rows']}
    for row in current_rows:
        require(all(analysed[row['cell_id']].get(key) == value for key, value in row.items()),
                'Raw attempt changed after analysis: '+row['cell_id'])
    require(len(summary['rows']) == 72 and {r['cell_id'] for r in summary['rows']}
            == {r['cell_id'] for r in current_rows if not r['reused_control']},
            'Worker summary and raw new-cell coverage differ')
    for row in summary['rows']:
        peer = analysed[row['cell_id']]
        require(row['result_file_sha256'] == peer['result_sha256']
                and row['memory_before_sha256'] == peer['memory_sha256']
                and row['seed'] == peer['seed'] and row['success'] == peer['success'],
                'Worker summary differs from raw attempt: '+row['cell_id'])
    actual = costs(root)
    require(actual == summary['costs_incremental'], 'Journal costs differ from summary')
    reserved = read_json(root/'budget.json')['reserved']
    require(reserved == actual['dispatched_calls'] <= contract['max_calls'] == 4500,
            'Call reservations differ from dispatch or exceed the declared 4500-call cap')
    require(actual['saved_responses'] == actual['dispatched_calls'] and actual['uncertain_or_unrecorded_attempts'] == 0,
            'Unconfirmed model requests remain')
    require(contract['deadline_unix']-contract['created_unix'] == 7200, 'Declared execution envelope must be two hours')
    timestamps = [read_json(p)['dispatched_unix'] for p in root.glob('**/calls/*/attempt-*.request.json')]
    require(len(timestamps) == actual['dispatched_calls'] and bool(timestamps), 'Dispatch timestamp coverage mismatch')
    require(min(timestamps) >= contract['created_unix'] and max(timestamps) < contract['deadline_unix'],
            'Model dispatch outside the declared time envelope')
    require(summary['elapsed_seconds'] >= max(timestamps)-contract['created_unix'], 'Elapsed time predates the last dispatch')
    current = ledger()
    prior = contract['prior_ledger']
    require(all(v['dispatched'] == v['responses'] for v in prior.values()), 'Prior ledger was unresolved')
    before = sum(v['dispatched'] for v in prior.values())
    require(before == 23872, 'Unexpected prior experiment count')
    all_calls, all_responses = sum(v['dispatched'] for v in current.values()), sum(v['responses'] for v in current.values())
    require(all_calls == all_responses == before+actual['dispatched_calls'], 'Global ledger delta differs from this campaign')
    campaign_name = root.resolve().name
    for name in set(prior) | set(current):
        previous, present = prior.get(name, {'dispatched': 0, 'responses': 0}), current.get(name, {'dispatched': 0, 'responses': 0})
        if name != campaign_name:
            require(previous == present, 'Another experiment journal changed during this campaign: '+name)
    require(len(replica['rows']) == 2*len(protocol['ports']), 'Replica-check coverage mismatch')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for source, name in [('contract', 'contract'), ('run', 'run'), ('summary', 'raw-summary'), ('analysis', 'results'),
                         ('replica-audit', 'replica-audit'), ('protocol-audit', 'protocol-audit'),
                         ('action-shapes', 'action-shapes'), ('semantic-labels', 'semantic-labels')]:
        origin = root/(source+'.json')
        target = output/('residual-context-'+name+'.json')
        if target.exists():
            require(target.read_bytes().replace(b'\r\n', b'\n') == origin.read_bytes().replace(b'\r\n', b'\n'),
                    'Existing published evidence differs: '+str(target))
        else:
            shutil.copyfile(origin, target)
        artifacts[target.name] = file_hash(target)
    value = {'protocol_sha256': protocol['sha256'], 'freeze_commit': args.freeze_commit,
        'stop_reason': 'frozen_factorial_complete', 'expected_cells': 96, 'completed_cells': 96,
        'new_cells': 72, 'reused_controls': 24, 'successes': analysis['successes'],
        'new_cell_successes': analysis['new_cell_successes'], 'independent_base_tasks': 12,
        'seed_replicates_per_task': 2, 'new_base_tasks': 0, 'costs_incremental': actual,
        'replica_check_calls': len(replica['rows']), 'reserved_calls': reserved, 'prior_calls': before,
        'all_calls': all_calls, 'all_responses': all_responses, 'elapsed_seconds': summary['elapsed_seconds'],
        'dispatch_envelope_seconds': 7200, 'holdout_model_calls': 0, 'parameter_refits': 0,
        'ports': protocol['ports'], 'worker_exit_codes': summary['worker_exit_codes'],
        'request_configuration_errors': len(audit['errors']), 'artifacts': artifacts,
        'source_presence_forecast_sha256': protocol.get('source_presence_freeze_sha256'),
        'driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value)
    target = output/'residual-context-closeout.json'
    if target.exists():
        require(read_json(target) == value, 'Existing closeout differs')
    else:
        write_json(target, value)
    print({k: v for k, v in value.items() if k != 'artifacts'})


if __name__ == '__main__':
    main()
