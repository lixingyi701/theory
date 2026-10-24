#!/usr/bin/env python3
"""Reconcile bounded calibration and publish summaries without model replies."""
import argparse
import importlib.metadata
import shutil
from pathlib import Path
from agent_reliability.io import digest, file_hash, read_json, write_json
from continue_reflexion_pilot import ledger
from run_directed_calibration import verify_protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    protocol = verify_protocol(Path('repro/reflexion-20261023/directed-calibration-protocol.json'))
    root = args.run
    summary = read_json(root/'summary.json'); contract = read_json(root/'contract.json')
    analysis = read_json(root/'analysis.json'); audit = read_json(root/'protocol-audit.json')
    for value in [summary, contract, analysis, audit]:
        if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
            raise ValueError('Saved artifact integrity')
    if analysis['summary_sha256'] != summary['sha256'] or analysis['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Analysis/protocol identity mismatch')
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
                         ('replica-audit.json', 'replica-audit')]:
        target = output/f'directed-calibration-{name}.json'
        if target.exists() and file_hash(target) != file_hash(root/source):
            raise FileExistsError('Do not overwrite prior closeout evidence')
        shutil.copyfile(root/source, target)
    value = {'protocol_sha256': protocol['sha256'], 'freeze_commit': '8d47025e81ee698b648b7067e388a50188222f24',
        'stop_reason': reason, 'expected_cells': protocol['expected_cells'], 'completed_cells': analysis['completed_cells'],
        'completed_attempts': analysis['attempts_observed'], 'successes': analysis['successes'],
        'costs_incremental': cost, 'replica_check_calls': 6, 'reserved_calls': reserved,
        'unused_reservations': reserved-cost['dispatched_calls'], 'prior_calls': prior, 'all_calls': all_calls,
        'all_responses': sum(v['responses'] for v in counts.values()), 'elapsed_seconds': summary['elapsed_seconds'],
        'independent_base_tasks': 4, 'holdout_model_calls': 0, 'gpu_indices': [0, 2, 3],
        'request_configuration_errors': len(configuration_errors), 'cohort_audit_passed': audit['passed'],
        'worker_exit_codes': summary['worker_exit_codes'], 'public_grammar_sha256': file_hash(grammar),
        'artifacts': {p.name: file_hash(p) for p in output.glob('directed-calibration-*.json')
                      if not p.name.endswith('closeout.json')}, 'driver_sha256': file_hash(Path(__file__))}
    if not complete:
        value['coverage_note'] = ('Actor budget exceptions are converted to SystemExit by the pinned upstream llm function; '
            'worker status strings alone do not establish completion. Coverage comes from committed result.json files. '
            'Unfinished attempts are missing, and their calls remain in costs.')
    value['sha256'] = digest(value)
    write_json(output/'directed-calibration-closeout.json', value)
    print({k: v for k, v in value.items() if k not in {'artifacts', 'coverage_note'}})


if __name__ == '__main__':
    main()
