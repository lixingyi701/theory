#!/usr/bin/env python3
"""Two-stage prospective condition transport; one shared allowance and frozen forecasts."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.llm import CallJournal
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from continue_reflexion_pilot import ledger
from summarize_reflexion import costs
from prepare_middle_context_transport import memory_for


class CalibrationStop(BaseException):
    pass


def reserve_call(root, contract):
    """Serialize reservations across processes before any fresh request dispatch."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root/'budget.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(root/'budget.json') if (root/'budget.json').exists() else {'reserved': 0}
        if (root/'STOP').exists():
            raise CalibrationStop('Another worker stopped; inspect its saved status')
        if time.time() >= contract['deadline_unix'] or state['reserved'] >= contract['max_calls']:
            raise CalibrationStop('Declared time/call allowance reached')
        state['reserved'] += 1
        write_json(root/'budget.json', state)
        return state['reserved']


def verify_protocol(path):
    protocol = read_json(path)
    if protocol['sha256'] != digest({k: v for k, v in protocol.items() if k != 'sha256'}):
        raise ValueError('Calibration protocol integrity')
    for name, expected in protocol['protected_files'].items():
        if file_hash(Path(name)) != expected:
            raise ValueError('Frozen calibration dependency changed: '+name)
    if provenance()['source_hashes'] != protocol['source_hashes']:
        raise ValueError('Frozen Agent library changed')
    for case in protocol['cases']:
        if case['task_id'] != case['task']['task_id']:
            raise ValueError('Case/task mismatch')
    for name, expected in protocol['entry_hashes'].items():
        if digest(protocol['entries'][name]) != expected:
            raise ValueError('Frozen constructed entry changed')
    for block in protocol['blocks']:
        for cell in block['cells']:
            memory_for(protocol, cell)
    return protocol


def verify_contract(root, protocol):
    contract = read_json(root/'contract.json')
    if contract['sha256'] != digest({k: v for k, v in contract.items() if k != 'sha256'}):
        raise ValueError('Calibration contract integrity')
    if contract['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Contract/protocol mismatch')
    current = provenance()
    if current['source_hashes'] != contract['source_hashes'] or current['native_hashes'] != contract['native_hashes']:
        raise ValueError('Source or environment changed')
    return contract


def replica_audit(root, protocol, contract):
    """Replay one recorded actor and critic payload on each replica, count all calls."""
    prior = read_json(Path('repro/reflexion-20261023/directed-calibration-protocol.json'))
    case = prior['cases'][0]
    source = Path(case['parent_run'])/'tasks'/case['task_id']/'trial-00'/'calls'
    files = []
    for role in ['actor', 'reflector']:
        files.append(sorted(source.glob(role+'-*/attempt-000.request.json'))[0])
    rows = []
    for port in protocol['ports']:
        transport = CompletionTransport(f'http://127.0.0.1:{port}/v1', 'local-development', 180)
        if transport._request('/models') != contract['serving']:
            raise ValueError('Replica serving identity mismatch')
        journal = CallJournal(root/'replica-audit'/str(port)/'calls', transport, 2)
        for role, source_path in zip(['actor', 'reflector'], files):
            logical = role+'-0000'
            payload = read_json(source_path)['payload']
            if not (journal.directory/logical/'response.json').exists():
                reserve_call(root, contract)
            saved = journal.complete(logical, payload)
            text = saved['response']['choices'][0]['text']
            write_json(journal.directory/logical/'validation.json', {'valid': bool(text.strip()),
                'truncated_output': saved['response']['choices'][0]['finish_reason'] == 'length'})
            rows.append({'port': port, 'role': role, 'payload_sha256': digest(payload),
                         'text_sha256': digest(text), 'source_request_sha256': file_hash(source_path)})
    for role in ['actor', 'reflector']:
        if len({r['text_sha256'] for r in rows if r['role'] == role}) != 1:
            raise ValueError('Matched seeded replica outputs differ; inspect before sampling')
    report = {'rows': rows, 'passed': True, 'claim': 'Two seeded payload checks; not distributional equivalence'}
    report['sha256'] = digest(report)
    write_json(root/'replica-audit.json', report)


def worker(root, protocol, index, phase):
    contract = verify_contract(root, protocol)
    port = protocol['ports'][index]
    transport = CompletionTransport(f'http://127.0.0.1:{port}/v1', 'local-development', 180)
    if transport._request('/models') != contract['serving']:
        raise ValueError('Worker serving identity changed')
    cases = {c['task_id']: c for c in protocol['cases']}
    rows = []
    status = 'complete'
    try:
        for block in protocol['blocks']:
            if block['worker'] != index or block['panel'] != phase:
                continue
            case = cases[block['task_id']]
            parent, entries = case, protocol['entries']
            for cell in block['cells']:
                memory = memory_for(protocol, cell)
                if 'memory_sha256' in cell and digest(memory) != cell['memory_sha256']:
                    raise ValueError('Native starting text changed')
                for trial in range(1, block['attempts']+1):
                    path = root/'cells'/cell['id']/f'trial-{trial:02d}'
                    result = run_attempt(case['task'], memory, trial, path, transport,
                        Path('data/alfworld'), block['seed'], lambda: reserve_call(root, contract),
                        input_format='qwen3_non_thinking_actionline',
                        decoding_profile='qwen3_non_thinking_recommended_v1')
                    if result['initial_observation_sha256'] != parent['initial_observation_sha256']:
                        raise ValueError('Task reset observation changed')
                    if not result['success'] and len(result['memory_after']) != len(memory)+1:
                        raise ValueError('Native critic did not append exactly one reflection')
                    row = {'cell_id': cell['id'], 'panel': block['panel'], 'condition': cell['condition'],
                        'initial_labels': cell['labels'], 'task_id': case['task_id'], 'seed': block['seed'],
                        'port': port, 'trial': trial, 'success': result['success'], 'decisions': result['decisions'],
                        'result_path': str(path/'result.json'), 'result_file_sha256': file_hash(path/'result.json'),
                        'memory_before_sha256': digest(memory), 'visible_memory_sha256': digest(memory[-3:]),
                        'new_reflection_sha256': [digest(e) for e in result['memory_after'][len(memory):]]}
                    rows.append(row)
                    write_json(root/f'worker-{phase}-{index}.json', {'status': 'running', 'rows': rows})
                    print(json.dumps({k: row[k] for k in ['cell_id', 'trial', 'success', 'decisions']}), flush=True)
                    memory = result['memory_after']
                    if result['success']:
                        break
    except CalibrationStop as exc:
        status = 'allowance_stop'
        (root/'STOP').write_text(str(exc))
    except BaseException:
        status = 'infrastructure_or_integrity_stop'
        (root/'STOP').write_text(f'Worker {index} requires inspection; no automatic retry')
        raise
    finally:
        write_json(root/f'worker-{phase}-{index}.json', {'status': status, 'rows': rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--phase', choices=['middle','native'])
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    if args.validate_only:
        print({'validated_cases': len(protocol['cases']), 'blocks': len(protocol['blocks']), 'model_calls': 0})
        return
    root = args.output
    if args.worker is not None:
        worker(root, protocol, args.worker, args.phase)
        return
    with run_lock(root):
        if (root/'contract.json').exists():
            raise FileExistsError('Existing campaign cannot be reset; inspect saved workers')
        before = ledger()
        if any(r['dispatched'] != r['responses'] for r in before.values()):
            raise ValueError('Unresolved preceding request')
        health = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)._request('/models')
        _, upstream = author_namespace()
        current = provenance()
        now = time.time()
        contract = {'protocol_sha256': protocol['sha256'], 'created_unix': now, 'deadline_unix': now+protocol['limits']['seconds'],
            'max_calls': protocol['limits']['additional_calls_including_replica_checks'], 'prior_ledger': before,
            'source_hashes': current['source_hashes'], 'native_hashes': current['native_hashes'], 'serving': health,
            'authorization': 'User continued middle-position and native-context checks on 2026-10-23; idle GPUs authorized'}
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
        rows, codes, statuses, phases = [], [], [], []
        for phase in protocol['phases']:
            children = []
            for index in range(len(protocol['ports'])):
                log = (root/f'worker-{phase}-{index}.log').open('a')
                children.append((subprocess.Popen([sys.executable, __file__, '--protocol', str(args.protocol),
                    '--output', str(root), '--worker', str(index), '--phase', phase],
                    stdout=log, stderr=subprocess.STDOUT), log))
            phase_codes = []
            for child, log in children:
                phase_codes.append(child.wait()); log.close()
            phase_rows = []
            for index in range(len(protocol['ports'])):
                path = root/f'worker-{phase}-{index}.json'
                data = read_json(path) if path.exists() else {'status':'missing_worker_record','rows':[]}
                phase_rows.extend(data['rows']); statuses.append({'phase':phase,'worker':index,'status':data['status']})
            rows.extend(phase_rows); codes.extend(phase_codes)
            record = {'phase':phase, 'completed_cells':len(phase_rows), 'expected_cells':protocol['phase_cells'][phase],
                      'exit_codes':phase_codes, 'reserved_calls':read_json(root/'budget.json')['reserved']}
            write_json(root/f'phase-{phase}-summary.json', record); phases.append(record)
            if any(phase_codes) or len(phase_rows) != protocol['phase_cells'][phase]:
                break
            if costs(root)['uncertain_or_unrecorded_attempts']:
                raise RuntimeError('Unconfirmed request before next phase')
        actual = costs(root); reserved = read_json(root/'budget.json')['reserved']
        report = {'protocol_sha256':protocol['sha256'], 'rows':sorted(rows,key=lambda r:r['cell_id']),
            'expected_cells':protocol['expected_cells'], 'completed_cells':len(rows),
            'worker_statuses':statuses, 'worker_exit_codes':codes, 'phases':phases,
            'costs_incremental':actual, 'reserved_calls':reserved, 'unused_reservations':reserved-actual['dispatched_calls'],
            'all_dispatched_calls':sum(v['dispatched'] for v in ledger().values()),
            'completed_tasks':len(protocol['cases']) if len(rows)==protocol['expected_cells'] else 0,
            'expected_tasks':len(protocol['cases']), 'elapsed_seconds':time.time()-now}
        report['sha256']=digest(report); write_json(root/'summary.json',report)
        print({k:v for k,v in report.items() if k!='rows'},flush=True)
        if any(codes) or actual['uncertain_or_unrecorded_attempts'] or reserved<actual['dispatched_calls']:
            raise RuntimeError('Transport incomplete; inspect journals before execution')



if __name__ == '__main__':
    main()
