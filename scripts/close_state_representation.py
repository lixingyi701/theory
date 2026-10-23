#!/usr/bin/env python3
"""Close the fresh-task state calibration without releasing model replies."""
import argparse
import importlib.metadata
import shutil
from pathlib import Path
from agent_reliability.io import digest, file_hash, read_json, write_json
from continue_reflexion_pilot import ledger
from run_state_representation import verify_protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    protocol = verify_protocol(Path('repro/reflexion-20261023/state-representation-protocol.json'))
    root = args.run
    summary = read_json(root/'summary.json'); contract = read_json(root/'contract.json')
    analysis = read_json(root/'analysis.json'); model = read_json(root/'model-freeze.json'); audit = read_json(root/'protocol-audit.json')
    coverage = read_json(root/'state-coverage.json')
    if coverage['model_freeze_sha256'] != model['sha256']:
        raise ValueError('Coverage belongs to another frozen model')
    for value in [summary, contract, analysis, model, coverage, audit]:
        if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
            raise ValueError('Saved artifact integrity')
    if analysis['summary_sha256'] != summary['sha256'] or analysis['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Analysis/protocol identity mismatch')
    if model['analysis_sha256'] != analysis['sha256'] or model['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Frozen model belongs to another calibration')
    if analysis['completed_cells'] != summary['completed_cells']:
        raise ValueError('Independent coverage counts disagree')
    cost = summary['costs_incremental']
    reserved = read_json(root/'budget.json')['reserved']
    if not cost['dispatched_calls'] <= reserved <= contract['max_calls']:
        raise ValueError('Parallel budget exceeded')
    if cost['uncertain_or_unrecorded_attempts']:
        raise ValueError('Resolve unknown model requests before closeout')
    timestamps = [read_json(q)['dispatched_unix'] for q in root.glob('**/calls/*/attempt-*.request.json')]
    if min(timestamps) < contract['created_unix'] or max(timestamps) >= contract['deadline_unix']:
        raise ValueError('Dispatch outside declared time envelope')
    counts = ledger()
    if any(v['dispatched'] != v['responses'] for v in counts.values()):
        raise ValueError('Global ledger has unknown requests')
    prior = sum(v['dispatched'] for v in contract['prior_ledger'].values())
    all_calls = sum(v['dispatched'] for v in counts.values())
    if all_calls != prior+cost['dispatched_calls']:
        raise ValueError('Unexpected concurrent experiment dispatch')
    configuration_errors = [e for e in audit['errors'] if e['rule'] != 'incomplete_task_cohort']
    if configuration_errors:
        raise ValueError('Request/interface audit requires inspection')
    complete = summary['completed_cells'] == protocol['expected_cells']
    reason = 'matrix_complete' if complete else ('declared_call_limit' if reserved == contract['max_calls'] else 'inspect_incomplete')
    if reason == 'inspect_incomplete':
        raise ValueError('Unexpected early stop')
    grammar = Path(importlib.metadata.distribution('alfworld').locate_file('alfworld/data/alfred.twl2'))
    grammar_text = grammar.read_text()
    for template in ['open {r}', 'move {o} to {r}', 'take {o} from {r}', 'heat {o} with {r}']:
        if f'template :: "{template}";' not in grammar_text:
            raise ValueError('Installed command grammar differs')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for source, name in [('contract.json', 'contract'), ('run.json', 'run'), ('summary.json', 'raw-summary'),
                         ('analysis.json', 'results'), ('protocol-audit.json', 'protocol-audit'),
                         ('replica-audit.json', 'replica-audit'), ('model-freeze.json', 'model-freeze'),
                         ('semantic-labels.json', 'semantic-labels'), ('action-shapes.json', 'action-shapes'),
                         ('state-coverage.json', 'state-coverage')]:
        target = output/f'state-representation-{name}.json'
        if target.exists() and file_hash(target) != file_hash(root/source):
            raise FileExistsError('Do not overwrite prior closeout evidence')
        shutil.copyfile(root/source, target)
    value = {'protocol_sha256': protocol['sha256'], 'freeze_commit': '2ce8148d7c4dc3a52533a0b063a6440f1dc7f761',
        'stop_reason': reason, 'expected_cells': protocol['expected_cells'], 'completed_cells': analysis['completed_cells'],
        'completed_attempts': analysis['completed_cells'], 'successes': analysis['successes'],
        'costs_incremental': cost, 'replica_check_calls': 8, 'reserved_calls': reserved,
        'unused_reservations': reserved-cost['dispatched_calls'], 'prior_calls': prior, 'all_calls': all_calls,
        'all_responses': sum(v['responses'] for v in counts.values()), 'elapsed_seconds': summary['elapsed_seconds'],
        'independent_base_tasks': 12, 'holdout_model_calls': 0, 'gpu_indices': [0, 1, 2, 3],
        'request_configuration_errors': len(configuration_errors), 'cohort_audit_passed': audit['passed'],
        'worker_exit_codes': summary['worker_exit_codes'], 'public_grammar_sha256': file_hash(grammar),
        'artifacts': {p.name: file_hash(p) for p in output.glob('state-representation-*.json')
                      if not p.name.endswith('closeout.json')}, 'driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value)
    write_json(output/'state-representation-closeout.json', value)
    print({k: v for k, v in value.items() if k not in {'artifacts', 'coverage_note'}})


if __name__ == '__main__':
    main()
