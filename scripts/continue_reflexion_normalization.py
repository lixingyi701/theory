#!/usr/bin/env python3
"""Finish profile comparison, freeze the engineering interface, collect prefixes.

Uses the existing finite continuation envelope. It stops for target review and
does not fit mechanisms or run the sealed holdout.
"""
import subprocess
import sys
import time
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.llm import CallJournal
from agent_reliability.reference import CompletionTransport
from continue_reflexion_pilot import PHASE, allowance, ledger
from run_reflexion_tool_stage import verify_stage_revision


AUTHOR = Path('runs/reflexion-tools-v2-dev12-20261023')
NATIVE = Path('runs/reflexion-qwen-native-dev12-20261023')
PREFIX = Path('runs/reflexion-normalized-prefix24-20261023')
MANIFEST = Path('repro/reflexion-20261023/pilot-prefix24-v2.json')


def run(command, contract, log):
    with log.open('a') as stream:
        result = subprocess.run([sys.executable, *command], stdout=stream, stderr=subprocess.STDOUT,
            timeout=max(1, contract['deadline_unix']-time.time()+200))
    if result.returncode:
        raise RuntimeError(f'Stage stopped: {command[0]} ({result.returncode})')


def serving_equivalence(contract):
    root = Path('runs/reflexion-serving-equivalence-20261023')
    transport = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)
    actor = AUTHOR/'tasks/650aae4563b0cee5/trial-00/calls/actor-0000'
    reflector = next(iter(sorted((AUTHOR/'tasks/1bcfccd955e18c46/trial-00/calls').glob('reflector-*'))))
    identity = {'parent_run_sha256': file_hash(AUTHOR/'run.json'), 'serving': transport._request('/models'),
        'driver_sha256': file_hash(Path(__file__)), 'continuation_sha256': contract['sha256']}
    if (root/'run.json').exists():
        if read_json(root/'run.json')['identity'] != identity:
            raise ValueError('Serving equivalence identity changed')
    else:
        write_json(root/'run.json', {'identity': identity, 'created_unix': time.time()})
    journal = CallJournal(root/'calls', transport, 2)
    rows = []
    with run_lock(PHASE.resolve()):
        for index, source in enumerate([actor, reflector]):
            payload = read_json(source/'attempt-000.request.json')['payload']
            logical = f'probe-{index:04d}'
            if not (root/'calls'/logical/'response.json').exists(): allowance(contract)
            saved = journal.complete(logical, payload)
            expected = read_json(source/'response.json')['response']['choices'][0]
            actual = saved['response']['choices'][0]
            write_json(root/'calls'/logical/'validation.json', {'valid': bool(actual['text'].strip()),
                'truncated_output': actual['finish_reason']=='length'})
            rows.append({'role': 'actor' if index==0 else 'reflector', 'payload_sha256': digest(payload),
                'same_raw_text': actual['text']==expected['text'],
                'same_finish_reason': actual['finish_reason']==expected['finish_reason']})
    report = {'scope': 'Fresh same-payload compatibility check after serving upgrade', 'rows': rows,
        'passed': all(r['same_raw_text'] and r['same_finish_reason'] for r in rows), 'costs': journal.totals()}
    report['sha256'] = digest(report); write_json(root/'summary.json', report)
    if not report['passed']:
        raise ValueError('Serving compatibility needs inspection')
    return report


def main():
    with run_lock(PHASE.resolve()/'normalization-controller'):
        contract, revision = verify_stage_revision(kind='model_profile')
        write_json(PHASE/'status.json', {'stage': 'waiting_for_sampling_comparison', 'updated_unix': time.time()})
        while time.time() < contract['deadline_unix']:
            summary = read_json(NATIVE/'summary.json')
            live = subprocess.run(['tmux','has-session','-t','theory-reflexion-qwen-dev12'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
            if summary['completed_tasks']==12 and not live:
                break
            if not live and summary['completed_tasks']!=12:
                raise RuntimeError('Sampling run stopped before all twelve tasks completed')
            time.sleep(5)
        else:
            raise RuntimeError('Continuation deadline reached')
        if any(v['dispatched'] != v['responses'] for v in ledger().values()):
            raise ValueError('Resolve requests before profile freeze')
        log = Path('runs/setup/normalization-controller.log')
        native_audit = PHASE/'qwen-native-dev12-protocol-audit.json'
        run(['scripts/audit_reflexion_protocol.py','--run',str(NATIVE),'--output',str(native_audit)],contract,log)
        native = read_json(NATIVE/'summary.json'); author = read_json(AUTHOR/'summary.json')
        if not read_json(PHASE/'tools-v2-dev12-protocol-audit.json')['passed']:
            raise ValueError('Author profile interface audit')
        compatibility = serving_equivalence(contract)
        chosen = 'qwen3_non_thinking_recommended_v1' if native['successes'] >= author['successes'] else 'author_reflexion'
        capability = NATIVE if chosen!='author_reflexion' else AUTHOR
        freeze = {'kind': 'engineering_interface_freeze', 'selected_decoding_profile': chosen,
            'rule': 'Passing interface contracts, most dev12 successes within three attempts; ties use official Qwen sampling',
            'author_summary_sha256': author['sha256'], 'native_summary_sha256': native['sha256'],
            'source_hashes': revision['source_hashes'], 'model_revision_record_sha256': revision['sha256'],
            'model_protocol_sha256': file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')),
            'serving_compatibility_sha256': compatibility['sha256'], 'holdout_model_calls': 0}
        freeze['sha256'] = digest(freeze)
        path = PHASE/'interface-freeze.json'
        if path.exists() and read_json(path)!=freeze:
            raise ValueError('Engineering interface freeze changed')
        write_json(path,freeze)
        remaining = contract['prior_dispatched_calls']+contract['additional_max_calls']-sum(v['dispatched'] for v in ledger().values())
        cap = read_json(PREFIX/'run.json')['identity']['max_calls'] if (PREFIX/'run.json').exists() else min(1400,remaining)
        if cap<=0:
            raise RuntimeError('No continuation allowance for new prefixes')
        run(['scripts/run_reflexion_tool_stage.py','--manifest',str(MANIFEST),'--output',str(PREFIX),
            '--audit','runs/audit/tools-v2-prefix24-direct.json','--environment-audit','runs/audit/tools-v2-prefix24-official.json',
            '--trials','1','--max-calls',str(cap),'--capability-run',str(capability),
            '--revision-kind','model_profile','--decoding-profile',chosen],contract,Path('runs/setup/normalized-prefix24.log'))
        run(['scripts/audit_reflexion_protocol.py','--run',str(PREFIX),'--output',str(PHASE/'prefix24-protocol-audit.json')],contract,log)
        preview = PHASE/'normalized-target-preview.json'
        run(['scripts/prepare_reflexion_goal_targets.py','--parent-run',str(PREFIX),'--manifest',str(MANIFEST),'--output',str(preview)],contract,log)
        write_json(PHASE/'status.json', {'stage':'ready_for_prefix_target_review','selected_decoding_profile':chosen,
            'prefix_summary':str(PREFIX/'summary.json'),'target_preview':str(preview),
            'ledger':ledger(),'updated_unix':time.time()})


if __name__=='__main__':
    main()
