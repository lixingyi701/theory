#!/usr/bin/env python3
"""One-attempt residual-text factorial; 72 new cells and 24 audited controls."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from continue_reflexion_pilot import ledger
from run_middle_context_transport import CalibrationStop, reserve_call, verify_contract, replica_audit
from summarize_reflexion import costs


CONDITIONS = {'N_E': ['N', 'E', 'N'], 'N_C': ['N', 'C', 'N'],
              'R_E': ['R', 'E', 'N'], 'R_C': ['R', 'C', 'N']}


def memory_for(protocol, cell):
    from prepare_residual_context import memory_for as resolve
    return resolve(protocol, cell)


def validate_matrix(protocol, memory_resolver=None):
    """Validate the exact factorial, text positions and all saved memory hashes."""
    resolve = memory_resolver or memory_for
    tasks = [case['task_id'] for case in protocol['cases']]
    if len(tasks) != 12 or len(set(tasks)) != 12:
        raise ValueError('Factorial requires 12 distinct base tasks')
    if protocol['phases'] != ['factorial'] or protocol['phase_cells'] != {'factorial': 72}:
        raise ValueError('Frozen phase coverage changed')
    if protocol['expected_cells'] != 72:
        raise ValueError('Only 72 new cells count as execution coverage')
    ports = protocol['ports']
    if not ports or len(set(ports)) != len(ports):
        raise ValueError('Replica ports must be nonempty and unique')
    entries = protocol['entries']
    seen, new_ids, residuals = set(), set(), {}
    new_count, reused_count = 0, 0

    def check(cell, task, seed, reused):
        nonlocal new_count, reused_count
        if task not in tasks or cell.get('task_id', task) != task:
            raise ValueError('Factorial task identity changed')
        condition = cell['condition']
        if seed not in [2101, 2102] or condition not in CONDITIONS:
            raise ValueError('Unknown factorial seed or condition')
        if cell['labels'] != CONDITIONS[condition]:
            raise ValueError('Target/residual text is in the wrong slot')
        if cell.get('trial', 1) != 1:
            raise ValueError('This factorial permits trial 1 only')
        key = (task, seed, condition)
        if key in seen:
            raise ValueError('Duplicate factorial cell')
        seen.add(key)
        if reused != (seed == 2101 and condition.startswith('N_')):
            raise ValueError('Only seed-2101 N controls may be reused')
        memory = resolve(protocol, cell)
        if len(memory) != 3 or not all(isinstance(text, str) for text in memory):
            raise ValueError('Factorial requires three actual reflection entries')
        if memory[1] != entries[condition[-1]] or memory[2] != entries['N']:
            raise ValueError('Fixed target text or newest generic entry changed')
        if condition.startswith('R_'):
            expected = cell.get('residual_sha256')
            if not expected or digest(memory[0]) != expected:
                raise ValueError('Residual text hash changed')
            if task in residuals and residuals[task] != expected:
                raise ValueError('All four R cells must share the same task residual')
            residuals[task] = expected
        elif memory[0] != entries['N']:
            raise ValueError('Generic control has unexpected residual text')
        if digest(memory) != cell['memory_sha256']:
            raise ValueError('Starting memory hash changed')
        if reused:
            reused_count += 1
        else:
            identity = cell['id']
            if identity in new_ids:
                raise ValueError('Duplicate new execution cell id')
            new_ids.add(identity)
            new_count += 1

    for block in protocol['blocks']:
        if block['panel'] != 'factorial' or block['attempts'] != 1:
            raise ValueError('Unexpected panel or multistep execution')
        if not isinstance(block['worker'], int) or not 0 <= block['worker'] < len(ports):
            raise ValueError('Invalid replica assignment')
        for cell in block['cells']:
            if cell.get('seed', block['seed']) != block['seed']:
                raise ValueError('Block/cell seed mismatch')
            check(cell, block['task_id'], block['seed'], False)
    for cell in protocol['reused_controls']:
        check(cell, cell['task_id'], cell['seed'], True)
    expected = {(task, seed, condition) for task in tasks
                for seed in [2101, 2102] for condition in CONDITIONS}
    if seen != expected or new_count != 72 or reused_count != 24:
        raise ValueError('Incomplete 72-new plus 24-reused factorial matrix')
    return {'new_cells': new_count, 'reused_cells': reused_count, 'factorial_cells': len(seen)}


def verify_dependencies(protocol, visited=None):
    """Follow frozen dependency manifests, including inherited earlier protocols."""
    visited = set() if visited is None else visited
    for name, expected in protocol['protected_files'].items():
        path = Path(name)
        if file_hash(path) != expected:
            raise ValueError('Frozen residual-context dependency changed: '+name)
        if name in visited:
            continue
        visited.add(name)
        if path.suffix == '.json':
            nested = read_json(path)
            if isinstance(nested, dict) and 'protected_files' in nested:
                if nested.get('sha256') != digest({k: v for k, v in nested.items() if k != 'sha256'}):
                    raise ValueError('Inherited frozen protocol integrity: '+name)
                verify_dependencies(nested, visited)


def verify_protocol(path):
    protocol = read_json(path)
    if protocol['sha256'] != digest({k: v for k, v in protocol.items() if k != 'sha256'}):
        raise ValueError('Residual-context protocol integrity')
    verify_dependencies(protocol)
    if provenance()['source_hashes'] != protocol['source_hashes']:
        raise ValueError('Frozen Agent library changed')
    for case in protocol['cases']:
        if case['task_id'] != case['task']['task_id']:
            raise ValueError('Case/task mismatch')
    for name, expected in protocol['entry_hashes'].items():
        if digest(protocol['entries'][name]) != expected:
            raise ValueError('Frozen constructed entry changed')
    limits = protocol['limits']
    if not 0 < limits['additional_calls_including_replica_checks'] <= 4500 or not 0 < limits['seconds'] <= 7200:
        raise ValueError('Residual-context finite allowance changed')
    validate_matrix(protocol)
    return protocol


def worker(root, protocol, index):
    contract = verify_contract(root, protocol)
    port = protocol['ports'][index]
    transport = CompletionTransport(f'http://127.0.0.1:{port}/v1', 'local-development', 180)
    if transport._request('/models') != contract['serving']:
        raise ValueError('Worker serving identity changed')
    cases = {case['task_id']: case for case in protocol['cases']}
    rows, status = [], 'complete'
    try:
        for block in protocol['blocks']:
            if block['worker'] != index:
                continue
            case = cases[block['task_id']]
            for cell in block['cells']:
                memory = memory_for(protocol, cell)
                if digest(memory) != cell['memory_sha256']:
                    raise ValueError('Starting text changed after validation')
                path = root/'cells'/cell['id']/'trial-01'
                result = run_attempt(case['task'], memory, 1, path, transport,
                    Path('data/alfworld'), block['seed'], lambda: reserve_call(root, contract),
                    input_format='qwen3_non_thinking_actionline',
                    decoding_profile='qwen3_non_thinking_recommended_v1')
                if result['initial_observation_sha256'] != case['initial_observation_sha256']:
                    raise ValueError('Task reset observation changed')
                if not result['success'] and len(result['memory_after']) != len(memory)+1:
                    raise ValueError('Native critic did not append exactly one reflection')
                if result['success'] and result['memory_after'] != memory:
                    raise ValueError('Successful attempt unexpectedly updated reflection memory')
                row = {'cell_id': cell['id'], 'panel': 'factorial', 'condition': cell['condition'],
                    'initial_labels': cell['labels'], 'task_id': case['task_id'], 'seed': block['seed'],
                    'port': port, 'trial': 1, 'success': result['success'], 'decisions': result['decisions'],
                    'result_path': str(path/'result.json'), 'result_file_sha256': file_hash(path/'result.json'),
                    'memory_before_sha256': digest(memory), 'visible_memory_sha256': digest(memory),
                    'new_reflection_sha256': [digest(e) for e in result['memory_after'][len(memory):]]}
                if 'residual_sha256' in cell:
                    row['residual_sha256'] = cell['residual_sha256']
                rows.append(row)
                write_json(root/f'worker-factorial-{index}.json', {'status': 'running', 'rows': rows})
                print(json.dumps({k: row[k] for k in ['cell_id', 'trial', 'success', 'decisions']}), flush=True)
    except CalibrationStop as exc:
        status = 'allowance_stop'
        (root/'STOP').write_text(str(exc), encoding='utf-8')
    except BaseException:
        status = 'infrastructure_or_integrity_stop'
        (root/'STOP').write_text(f'Worker {index} requires inspection; no automatic retry', encoding='utf-8')
        raise
    finally:
        write_json(root/f'worker-factorial-{index}.json', {'status': status, 'rows': rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--phase', choices=['factorial'], default='factorial')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    if args.validate_only:
        print({'validated_cases': len(protocol['cases']), **validate_matrix(protocol), 'model_calls': 0})
        return
    root = args.output
    if args.worker is not None:
        worker(root, protocol, args.worker)
        return
    with run_lock(root):
        if (root/'contract.json').exists():
            raise FileExistsError('Existing campaign cannot be reset; inspect saved workers')
        before = ledger()
        if any(row['dispatched'] != row['responses'] for row in before.values()):
            raise ValueError('Unresolved preceding request')
        health = CompletionTransport(f"http://127.0.0.1:{protocol['ports'][0]}/v1", 'local-development', 180)._request('/models')
        _, upstream = author_namespace()
        current, now = provenance(), time.time()
        contract = {'protocol_sha256': protocol['sha256'], 'created_unix': now,
            'deadline_unix': now+protocol['limits']['seconds'],
            'max_calls': protocol['limits']['additional_calls_including_replica_checks'], 'prior_ledger': before,
            'source_hashes': current['source_hashes'], 'native_hashes': current['native_hashes'], 'serving': health,
            'authorization': 'User requested residual-context diagnosis and continued execution on 2026-10-24; idle GPUs authorized'}
        contract['sha256'] = digest(contract)
        write_json(root/'contract.json', contract)
        identity = {'protocol_sha256': protocol['sha256'], 'campaign_sha256': contract['sha256'],
            'source_hashes': current['source_hashes'], 'serving': health, 'upstream': upstream,
            'input_format': 'qwen3_non_thinking_actionline',
            'decoding_profile': 'qwen3_non_thinking_recommended_v1',
            'environment_action_alias': ACTION_ALIAS_PROFILE,
            'wrapper_sha256': file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256': file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))}
        write_json(root/'run.json', {'identity': identity, 'created_unix': now, 'scope': protocol['scope']})
        replica_audit(root, protocol, contract)
        children = []
        for index in range(len(protocol['ports'])):
            log = (root/f'worker-factorial-{index}.log').open('a', encoding='utf-8')
            children.append((subprocess.Popen([sys.executable, __file__, '--protocol', str(args.protocol),
                '--output', str(root), '--worker', str(index), '--phase', 'factorial'],
                stdout=log, stderr=subprocess.STDOUT), log))
        codes = []
        for child, log in children:
            codes.append(child.wait())
            log.close()
        rows, statuses = [], []
        for index in range(len(protocol['ports'])):
            path = root/f'worker-factorial-{index}.json'
            saved = read_json(path) if path.exists() else {'status': 'missing_worker_record', 'rows': []}
            rows.extend(saved['rows'])
            statuses.append({'phase': 'factorial', 'worker': index, 'status': saved['status']})
        expected_ids = {cell['id'] for block in protocol['blocks'] for cell in block['cells']}
        actual_ids = [row['cell_id'] for row in rows]
        complete = (len(actual_ids) == len(set(actual_ids)) == 72 and set(actual_ids) == expected_ids
                    and not any(codes) and all(row['status'] == 'complete' for row in statuses))
        actual = costs(root)
        reserved = read_json(root/'budget.json')['reserved']
        phase_record = {'phase': 'factorial', 'completed_cells': len(rows), 'expected_cells': 72,
                        'exit_codes': codes, 'reserved_calls': reserved}
        write_json(root/'phase-factorial-summary.json', phase_record)
        report = {'protocol_sha256': protocol['sha256'], 'rows': sorted(rows, key=lambda row: row['cell_id']),
            'status': 'complete' if complete else 'partial', 'expected_cells': 72, 'completed_cells': len(rows),
            'reused_controls': len(protocol['reused_controls']), 'expected_factorial_cells_in_analysis': 96,
            'worker_statuses': statuses, 'worker_exit_codes': codes, 'phases': [phase_record],
            'costs_incremental': actual, 'reserved_calls': reserved,
            'unused_reservations': reserved-actual['dispatched_calls'],
            'all_dispatched_calls': sum(value['dispatched'] for value in ledger().values()),
            'completed_tasks': len(protocol['cases']) if complete else 0,
            'expected_tasks': len(protocol['cases']), 'elapsed_seconds': time.time()-now}
        report['sha256'] = digest(report)
        write_json(root/'summary.json', report)
        print({key: value for key, value in report.items() if key != 'rows'}, flush=True)
        if not complete or actual['uncertain_or_unrecorded_attempts'] or reserved < actual['dispatched_calls']:
            raise RuntimeError('Factorial incomplete; inspect saved status and journals before execution')


if __name__ == '__main__':
    main()
