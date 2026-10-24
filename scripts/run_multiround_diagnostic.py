#!/usr/bin/env python3
"""Frozen one-time interventions followed by up to three native Reflexion attempts."""
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
from run_middle_context_transport import CalibrationStop, replica_audit, reserve_call, verify_contract
from run_residual_context import verify_dependencies
from summarize_reflexion import costs


HISTORIES = {'single_error': ['E'], 'repeated_error': ['E', 'N', 'E']}
OPERATIONS = ('retain', 'correct_latest', 'delete_latest')
PRIOR_CALLS = 26673


def initial_memory(protocol, trajectory):
    from prepare_multiround_diagnostic import initial_memory as resolve
    return resolve(protocol, trajectory)


def recovered_labels(history, operation):
    """Apply the sole intervention before the first attempt, with no older entries."""
    labels = list(history)
    latest = max(i for i, label in enumerate(labels) if label == 'E')
    if operation == 'correct_latest':
        labels[latest] = 'C'
    elif operation == 'delete_latest':
        del labels[latest]
    elif operation != 'retain':
        raise ValueError('Unknown initial intervention')
    return labels


def validate_matrix(protocol, memory_resolver=None):
    resolve = memory_resolver or initial_memory
    tasks = [case['task_id'] for case in protocol['cases']]
    if len(tasks) != len(set(tasks)) or len(tasks) != 12:
        raise ValueError('Diagnostic requires 12 distinct development tasks')
    if protocol['ports'] != [8001, 8002, 8003] or protocol['gpu_indices'] != [0, 1, 3]:
        raise ValueError('Frozen three-replica assignment changed')
    trajectories = protocol['trajectories']
    if len(trajectories) != 72 or len({item['id'] for item in trajectories}) != 72:
        raise ValueError('Exactly 72 unique trajectories required')
    seen = set()
    for trajectory in trajectories:
        task, history_key, operation = (trajectory['task_id'], trajectory['history_key'], trajectory['operation'])
        if task not in tasks or history_key not in HISTORIES or operation not in OPERATIONS:
            raise ValueError('Unknown task, history or intervention')
        key = (task, history_key, operation)
        if key in seen:
            raise ValueError('Duplicate task/history/intervention')
        seen.add(key)
        if trajectory['initial_history'] != HISTORIES[history_key]:
            raise ValueError('Pre-intervention history changed')
        labels = recovered_labels(HISTORIES[history_key], operation)
        if trajectory['initial_labels'] != labels:
            raise ValueError('Initial intervention mapping changed')
        if trajectory['seed'] != 2103 or trajectory['max_attempts'] != 3:
            raise ValueError('Frozen seed or three-attempt horizon changed')
        if trajectory['worker'] != tasks.index(task) % 3:
            raise ValueError('Frozen worker assignment changed')
        memory = resolve(protocol, trajectory)
        if memory != [protocol['entries'][label] for label in labels]:
            raise ValueError('Unexpected initial or hidden memory')
        if digest(memory) != trajectory['initial_memory_sha256']:
            raise ValueError('Initial memory hash changed')
    expected = {(task, history, operation) for task in tasks
                for history in HISTORIES for operation in OPERATIONS}
    if seen != expected:
        raise ValueError('Incomplete task/history/intervention matrix')
    return {'trajectories': len(trajectories), 'base_tasks': len(tasks), 'maximum_attempts': 216}


def verify_protocol(path):
    protocol = read_json(path)
    if protocol['sha256'] != digest({k: v for k, v in protocol.items() if k != 'sha256'}):
        raise ValueError('Multiround protocol integrity')
    verify_dependencies(protocol)
    if provenance()['source_hashes'] != protocol['source_hashes']:
        raise ValueError('Frozen Agent library changed')
    for case in protocol['cases']:
        if case['task_id'] != case['task']['task_id']:
            raise ValueError('Case/task mismatch')
    if set(protocol['entries']) != {'E', 'C', 'N'}:
        raise ValueError('Only frozen E/C/N initial entries are permitted')
    if set(protocol['entry_hashes']) != set(protocol['entries']):
        raise ValueError('Missing initial entry hash')
    for name, expected in protocol['entry_hashes'].items():
        if digest(protocol['entries'][name]) != expected:
            raise ValueError('Frozen initial entry changed')
    limits = protocol['limits']
    if limits['additional_calls_including_replica_checks'] != 11000 or limits['seconds'] != 10800:
        raise ValueError('Frozen finite execution allowance changed')
    validate_matrix(protocol)
    return protocol


def verify_attempt(result, path, case, trajectory, memory, trial):
    """Check a cached or fresh attempt before its memory can affect another attempt."""
    identity = read_json(path/'identity.json')
    if identity['sha256'] != digest({k: v for k, v in identity.items() if k != 'sha256'}):
        raise ValueError('Attempt identity integrity')
    expected = {'task': case['task'], 'memory': memory, 'trial': trial,
                'seed': trajectory['seed'], 'input_format': 'qwen3_non_thinking_actionline',
                'decoding_profile': 'qwen3_non_thinking_recommended_v1'}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError('Attempt task, seed, trial, memory or model profile changed')
    if result['identity_sha256'] != identity['sha256'] or read_json(path/'result.json') != result:
        raise ValueError('Saved result/identity mismatch')
    for key in ['task_id', 'scene_id', 'task_type']:
        if result[key] != case['task'][key]:
            raise ValueError('Attempt task identity changed')
    if result['trial'] != trial or result['memory_before'] != memory:
        raise ValueError('Attempt input memory or trial changed')
    if result['initial_observation_sha256'] != case['initial_observation_sha256']:
        raise ValueError('Task reset observation changed')
    if not isinstance(result['success'], bool) or not 0 < result['decisions'] <= 49:
        raise ValueError('Attempt outcome or decision count invalid')
    after = result['memory_after']
    if not isinstance(after, list) or after[:len(memory)] != memory:
        raise ValueError('Native reflection update changed the preceding history')
    if result['success']:
        if after != memory:
            raise ValueError('Successful attempt unexpectedly updated reflection memory')
    elif len(after) != len(memory)+1 or not isinstance(after[-1], str) or not after[-1].strip():
        raise ValueError('Failed attempt did not append one nonempty native reflection')


def trajectory_summary(trajectory, rows):
    success = any(row['success'] for row in rows)
    return {'trajectory_id': trajectory['id'], 'task_id': trajectory['task_id'],
            'seed': trajectory['seed'], 'history_key': trajectory['history_key'],
            'operation': trajectory['operation'], 'initial_labels': trajectory['initial_labels'],
            'initial_memory_sha256': trajectory['initial_memory_sha256'],
            'max_attempts': trajectory['max_attempts'], 'attempts_completed': len(rows),
            'complete': success or len(rows) == trajectory['max_attempts'],
            'success': success, 'attempts': list(rows)}


def execute_trajectory(root, protocol, case, trajectory, transport, allowance,
                       on_attempt=None, attempt_runner=None, memory_resolver=None):
    """Keep full native history; the pinned actor and critic expose its last three entries."""
    resolve, runner = memory_resolver or initial_memory, attempt_runner or run_attempt
    memory = resolve(protocol, trajectory)
    if digest(memory) != trajectory['initial_memory_sha256']:
        raise ValueError('Initial memory changed immediately before execution')
    rows = []
    for trial in range(1, trajectory['max_attempts']+1):
        before = list(memory)
        path = root/'trajectories'/trajectory['id']/f'trial-{trial:02d}'
        result = runner(case['task'], before, trial, path, transport,
                        Path('data/alfworld'), trajectory['seed'], allowance,
                        input_format='qwen3_non_thinking_actionline',
                        decoding_profile='qwen3_non_thinking_recommended_v1')
        verify_attempt(result, path, case, trajectory, before, trial)
        row = {'trajectory_id': trajectory['id'], 'task_id': case['task_id'],
               'seed': trajectory['seed'], 'history_key': trajectory['history_key'],
               'operation': trajectory['operation'], 'initial_labels': trajectory['initial_labels'],
               'trial': trial, 'success': result['success'], 'decisions': result['decisions'],
               'memory_before_sha256': digest(before), 'visible_memory_sha256': digest(before[-3:]),
               'new_reflection_sha256': [digest(text) for text in result['memory_after'][len(before):]],
               'result_path': str(path/'result.json'), 'result_file_sha256': file_hash(path/'result.json'),
               'identity_sha256': result['identity_sha256'], 'trajectory_sha256': result['trajectory_sha256']}
        rows.append(row)
        if on_attempt:
            on_attempt(row, trajectory_summary(trajectory, rows))
        memory = list(result['memory_after'])
        if result['success']:
            break
    return trajectory_summary(trajectory, rows)


def worker(root, protocol, index):
    if not 0 <= index < len(protocol['ports']):
        raise ValueError('Invalid worker index')
    contract = verify_contract(root, protocol)
    port = protocol['ports'][index]
    transport = CompletionTransport(f'http://127.0.0.1:{port}/v1', 'local-development', 180)
    if transport._request('/models') != contract['serving']:
        raise ValueError('Worker serving identity changed')
    cases = {case['task_id']: case for case in protocol['cases']}
    rows, trajectories, status = [], [], 'complete'
    record_path = root/f'worker-{index}.json'

    def save(current_status):
        write_json(record_path, {'status': current_status, 'rows': rows, 'trajectories': trajectories})

    try:
        for trajectory in protocol['trajectories']:
            if trajectory['worker'] != index:
                continue
            trajectories.append(trajectory_summary(trajectory, []))
            save('running')

            def observed(row, summary):
                row['port'] = port
                rows.append(row)
                trajectories[-1] = summary
                save('running')
                print(json.dumps({key: row[key] for key in ['trajectory_id', 'trial', 'success', 'decisions']}), flush=True)

            trajectories[-1] = execute_trajectory(root, protocol, cases[trajectory['task_id']],
                trajectory, transport, lambda: reserve_call(root, contract), observed)
            save('running')
    except CalibrationStop as exc:
        status = 'allowance_stop'
        (root/'STOP').write_text(str(exc), encoding='utf-8')
    except BaseException:
        status = 'infrastructure_or_integrity_stop'
        (root/'STOP').write_text(f'Worker {index} requires inspection; no automatic retry', encoding='utf-8')
        raise
    finally:
        save(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    if args.validate_only:
        print({**validate_matrix(protocol), 'model_calls': 0})
        return
    root = args.output
    if args.worker is not None:
        worker(root, protocol, args.worker)
        return
    with run_lock(root):
        if (root/'contract.json').exists():
            raise FileExistsError('Existing campaign cannot be reset; inspect saved workers')
        before = ledger()
        if any(item['dispatched'] != item['responses'] for item in before.values()):
            raise ValueError('Unresolved preceding request')
        if sum(item['dispatched'] for item in before.values()) != PRIOR_CALLS:
            raise ValueError('Prior campaign ledger differs from frozen 26673 calls')
        health = CompletionTransport(f"http://127.0.0.1:{protocol['ports'][0]}/v1", 'local-development', 180)._request('/models')
        _, upstream = author_namespace()
        current, now = provenance(), time.time()
        contract = {'protocol_sha256': protocol['sha256'], 'created_unix': now,
            'deadline_unix': now+protocol['limits']['seconds'],
            'max_calls': protocol['limits']['additional_calls_including_replica_checks'], 'prior_ledger': before,
            'source_hashes': current['source_hashes'], 'native_hashes': current['native_hashes'], 'serving': health,
            'authorization': 'User requested background-effect analysis and the next multiround diagnostic on 2026-10-24; idle GPUs authorized'}
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
            log = (root/f'worker-{index}.log').open('a', encoding='utf-8')
            child = subprocess.Popen([sys.executable, __file__, '--protocol', str(args.protocol),
                '--output', str(root), '--worker', str(index)], stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log))
        codes = []
        for child, log in children:
            codes.append(child.wait())
            log.close()
        rows, trajectories, statuses = [], [], []
        for index in range(len(protocol['ports'])):
            path = root/f'worker-{index}.json'
            saved = read_json(path) if path.exists() else {'status': 'missing_worker_record', 'rows': [], 'trajectories': []}
            rows.extend(saved['rows'])
            trajectories.extend(saved['trajectories'])
            statuses.append({'worker': index, 'status': saved['status']})
        expected_ids = {item['id'] for item in protocol['trajectories']}
        actual_ids = [item['trajectory_id'] for item in trajectories]
        row_ids = [(row['trajectory_id'], row['trial']) for row in rows]
        complete = (len(actual_ids) == len(set(actual_ids)) == 72 and set(actual_ids) == expected_ids
                    and len(row_ids) == len(set(row_ids))
                    and all(item['complete'] for item in trajectories)
                    and all(item['status'] == 'complete' for item in statuses) and not any(codes))
        for trajectory in trajectories:
            selected = sorted((row for row in rows if row['trajectory_id'] == trajectory['trajectory_id']), key=lambda row: row['trial'])
            if selected != trajectory['attempts'] or [row['trial'] for row in selected] != list(range(1, len(selected)+1)):
                raise ValueError('Attempt rows and trajectory summaries diverged')
        actual = costs(root)
        reserved = read_json(root/'budget.json')['reserved']
        after = ledger()
        report = {'protocol_sha256': protocol['sha256'], 'status': 'complete' if complete else 'partial',
            'expected_trajectories': 72, 'completed_trajectories': sum(item['complete'] for item in trajectories),
            'attempts_completed': len(rows), 'attempts_rows': sorted(rows, key=lambda row: (row['trajectory_id'], row['trial'])),
            'trajectories': sorted(trajectories, key=lambda item: item['trajectory_id']),
            'worker_statuses': statuses, 'worker_exit_codes': codes,
            'costs_incremental': actual, 'reserved_calls': reserved,
            'unused_reservations': reserved-actual['dispatched_calls'], 'prior_ledger': before,
            'all_dispatched_calls': sum(item['dispatched'] for item in after.values()),
            'all_saved_responses': sum(item['responses'] for item in after.values()),
            'completed_tasks': len(protocol['cases']) if complete else 0,
            'expected_tasks': len(protocol['cases']), 'elapsed_seconds': time.time()-now}
        report['sha256'] = digest(report)
        write_json(root/'summary.json', report)
        print({key: value for key, value in report.items() if key not in ['attempts_rows', 'trajectories', 'prior_ledger']}, flush=True)
        if not complete or actual['uncertain_or_unrecorded_attempts'] or reserved < actual['dispatched_calls']:
            raise RuntimeError('Diagnostic incomplete; inspect saved status and journals before execution')


if __name__ == '__main__':
    main()
