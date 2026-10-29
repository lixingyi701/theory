#!/usr/bin/env python3
"""Resume the frozen pilot under an explicitly recorded continuation allowance.

The original run identity and request history remain immutable. A new user
continuation authorizes an additional finite envelope, not a new task sample.
"""
import argparse
import time
from pathlib import Path

from agent_reliability.audit import verify_audit
from agent_reliability.cli import provenance
from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from summarize_reflexion import summarize


PHASE = Path('runs/reflexion-continuation-20261023')
PILOT = Path('runs/reflexion-pilot24-20261022')
MANIFEST = Path('repro/reflexion-20261022/pilot24.json')


def roots():
    return sorted({p.resolve() for p in Path('runs').glob('reflexion-*') if p.is_dir()})


def ledger():
    return {p.name: {'dispatched': len(list(p.glob('**/calls/*/attempt-*.request.json'))),
                     'responses': len(list(p.glob('**/calls/*/response.json')))} for p in roots()}


def allowance(contract):
    count = sum(v['dispatched'] for v in ledger().values())
    if count >= contract['prior_dispatched_calls'] + contract['additional_max_calls']:
        raise RuntimeError('Continuation call allowance exhausted')
    if time.time() >= contract['deadline_unix']:
        raise RuntimeError('Continuation wall-clock allowance exhausted')


def verify_contract(contract):
    if contract['sha256'] != digest({k: v for k, v in contract.items() if k != 'sha256'}):
        raise ValueError('Continuation contract integrity')
    if contract['pilot_run_sha256'] != file_hash(PILOT/'run.json'):
        raise ValueError('Original pilot identity changed')
    if contract['source_hashes'] != provenance()['source_hashes']:
        raise ValueError('Frozen execution source changed')
    if contract['pilot_driver_sha256'] != file_hash(Path(__file__)):
        raise ValueError('Continuation driver changed')
    return contract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initialize', action='store_true')
    p.add_argument('--additional-calls', type=int, default=6000)
    p.add_argument('--seconds', type=int, default=10800)
    a = p.parse_args()
    with run_lock(PHASE), run_lock(PILOT):
        manifest = load_manifest(MANIFEST)
        identity = read_json(PILOT/'run.json')['identity']
        current = provenance()
        if identity['source_hashes'] != current['source_hashes']:
            raise ValueError('Parent execution source changed')
        if identity['manifest_sha256'] != manifest['sha256']:
            raise ValueError('Pilot manifest changed')
        if identity['wrapper_sha256'] != file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')):
            raise ValueError('Qwen input wrapper changed')
        if identity['input_format'] != 'qwen3_non_thinking_actionline':
            raise ValueError('Expected frozen action-line input profile')
        _, receipt = author_namespace()
        if identity['upstream'] != receipt:
            raise ValueError('Upstream source changed')
        audit = read_json(Path('runs/audit/reflexion-pilot24-direct-20261022.json'))
        verify_audit(audit, manifest)
        equivalent = read_json(Path('runs/audit/reflexion-pilot24-official-20261022.json'))
        if equivalent['sha256'] != identity['environment_audit_sha256'] or not equivalent['passed']:
            raise ValueError('Official environment audit mismatch')
        if equivalent['native_hashes'] != current['native_hashes']:
            raise ValueError('Native environment changed')
        transport = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)
        if transport._request('/models') != identity['serving']:
            raise ValueError('Model service changed')
        path = PHASE/'contract.json'
        if a.initialize:
            if path.exists():
                raise ValueError('Continuation already initialized; resume without --initialize')
            if a.additional_calls <= 0 or a.seconds <= 0:
                raise ValueError('Finite positive allowance required')
            before = ledger()
            if any(v['dispatched'] != v['responses'] for v in before.values()):
                raise ValueError('Resolve uncertain requests before continuation')
            now = time.time()
            contract = {'created_unix': now, 'deadline_unix': now+a.seconds,
                'additional_max_calls': a.additional_calls,
                'prior_dispatched_calls': sum(v['dispatched'] for v in before.values()),
                'prior_ledger': before, 'prior_pilot_summary': read_json(PILOT/'summary.json'),
                'pilot_run_sha256': file_hash(PILOT/'run.json'),
                'source_hashes': current['source_hashes'],
                'pilot_driver_sha256': file_hash(Path(__file__)),
                'scope': 'Complete frozen development pilot, then evidence-qualified paired forks; holdout remains sealed',
                'authorization': 'User continuation on 2026-10-23; prior 5200-call envelope preserved'}
            contract['sha256'] = digest(contract)
            write_json(path, contract)
        contract = verify_contract(read_json(path))
        for task in manifest['tasks']:
            memory = []
            for trial in range(identity['trials']):
                result = run_attempt(task, memory, trial,
                    PILOT/'tasks'/task['task_id']/f'trial-{trial:02d}', transport,
                    Path('data/alfworld'), identity['seed'], lambda: allowance(contract),
                    input_format=identity['input_format'])
                memory = result['memory_after']
                print({k: result[k] for k in ['task_id', 'trial', 'success', 'decisions', 'dispatched_calls']}, flush=True)
                if result['success']:
                    break
            report = summarize(PILOT, manifest_path=MANIFEST)
            write_json(PHASE/'pilot-summary.json', report)
            write_json(PHASE/'status.json', {'stage': 'pilot_running', 'completed_tasks': report['completed_tasks'],
                'successes': report['successes'], 'ledger': ledger(), 'updated_unix': time.time()})
        write_json(PHASE/'status.json', {'stage': 'pilot_completed', 'completed_tasks': report['completed_tasks'],
            'successes': report['successes'], 'ledger': ledger(), 'updated_unix': time.time()})


if __name__ == '__main__':
    main()
