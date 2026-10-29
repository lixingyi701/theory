#!/usr/bin/env python3
"""Capability/prefix runs after the evidence-backed placement grammar revision."""
import argparse
import time
from pathlib import Path

from agent_reliability.audit import verify_audit
from agent_reliability.cli import provenance
from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from continue_reflexion_pilot import PHASE, allowance, ledger
from summarize_reflexion import summarize


def verify_stage_revision(initialize=False,kind='tool'):
    contract = read_json(PHASE/'contract.json')
    if contract['sha256'] != digest({k: v for k, v in contract.items() if k != 'sha256'}):
        raise ValueError('Continuation contract integrity')
    if contract['pilot_driver_sha256']!=file_hash(Path('scripts/continue_reflexion_pilot.py')):
        raise ValueError('Shared allowance/ledger code changed')
    current = provenance()
    path = PHASE/('tool-revision.json' if kind=='tool' else 'model-profile-revision.json')
    if initialize:
        if path.exists():
            raise FileExistsError('Tool revision already recorded')
        if read_json(PHASE/'pilot-summary.json')['completed_tasks'] != 24:
            raise ValueError('Complete preceding pilot before tool revision')
        diagnostic = read_json(PHASE/'placement-diagnostic-reviewed.json')
        if diagnostic['sha256'] != digest({k: v for k, v in diagnostic.items() if k != 'sha256'}):
            raise ValueError('Placement replay evidence integrity')
        predecessor = contract if kind=='tool' else read_json(PHASE/'tool-revision.json')
        if kind!='tool' and read_json(Path('runs/reflexion-tools-v2-dev12-20261023/summary.json'))['completed_tasks']!=12:
            raise ValueError('Complete controlled grammar check before sampling comparison')
        changed = sorted(k for k, v in current['source_hashes'].items() if predecessor['source_hashes'].get(k) != v)
        if changed != (['reference.py', 'reflexion.py'] if kind=='tool' else ['reflexion.py']):
            raise ValueError(f'Unexpected source changes: {changed}')
        revision = {'continuation_sha256': contract['sha256'], 'source_hashes': current['source_hashes'],
            'environment_action_alias': ACTION_ALIAS_PROFILE, 'created_unix': time.time(),
            'diagnostic_sha256': diagnostic['sha256'],
            'reason': 'Six public failed placement spellings accepted after equivalent grammar translation; five single-step wins. Preserve actor decisions and count all new calls in the original continuation envelope.'}
        if kind!='tool':
            revision.update(predecessor_sha256=predecessor['sha256'],
                reason='User requested normalized model settings; compare the official Qwen non-thinking sampling profile with author decoding after the controlled grammar check. Same campaign allowance and sealed holdout.',
                model_protocol_sha256=file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')))
        revision['sha256'] = digest(revision)
        write_json(path, revision)
    revision = read_json(path)
    if revision['sha256'] != digest({k: v for k, v in revision.items() if k != 'sha256'}):
        raise ValueError('Tool revision integrity')
    if revision['source_hashes'] != current['source_hashes'] or revision['continuation_sha256'] != contract['sha256']:
        raise ValueError('Tool revision source/contract mismatch')
    if 'model_protocol_sha256' in revision and revision['model_protocol_sha256']!=file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')):
        raise ValueError('Model protocol changed')
    return contract, revision


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--environment-audit', type=Path, required=True)
    p.add_argument('--trials', type=int, choices=[1, 3], required=True)
    p.add_argument('--max-calls', type=int, required=True)
    p.add_argument('--capability-run', type=Path)
    p.add_argument('--initialize-transition', action='store_true')
    p.add_argument('--revision-kind', choices=['tool','model_profile'],default='tool')
    p.add_argument('--decoding-profile',choices=['author_reflexion','qwen3_non_thinking_recommended_v1'],default='author_reflexion')
    p.add_argument('--stop-after-tasks',type=int)
    a = p.parse_args()
    with run_lock(PHASE.resolve()), run_lock(a.output):
        if any(v['dispatched'] != v['responses'] for v in ledger().values()):
            raise ValueError('Resolve preceding requests before new stage')
        contract, revision = verify_stage_revision(a.initialize_transition,a.revision_kind)
        manifest = load_manifest(a.manifest)
        if a.manifest.name not in {'dev12.json', 'pilot-prefix24-v2.json'}:
            raise ValueError('Development manifests only; holdout remains sealed')
        if a.manifest.name == 'pilot-prefix24-v2.json':
            if a.trials != 1 or not a.capability_run:
                raise ValueError('Pilot prefix requires one attempt and a capability run')
            capability = read_json(a.capability_run/'summary.json')
            kinds = {r['task_type'] for r in capability['tasks'] if r['success']}
            if capability['completed_tasks'] != 12 or capability['successes'] < 6 or len(kinds) < 4:
                raise ValueError('Revised interface capability gate')
        audit = read_json(a.audit); verify_audit(audit, manifest)
        equivalent = read_json(a.environment_audit); current = provenance()
        if equivalent['sha256'] != digest({k: v for k, v in equivalent.items() if k != 'sha256'}) or not equivalent['passed']:
            raise ValueError('Official environment audit integrity')
        if equivalent['manifest_sha256'] != manifest['sha256'] or equivalent['native_hashes'] != current['native_hashes']:
            raise ValueError('Environment identity mismatch')
        for name in ['alfworld.py', 'env.py', 'audit.py', 'reference.py']:
            if equivalent['source_hashes'][name] != current['source_hashes'][name]:
                raise ValueError('Audited adapter changed')
        transport = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)
        health = transport._request('/models'); _, upstream = author_namespace()
        if health['model_revision'] != 'b968826d9c46dd6066d109eabc6255188de91218' or health['serving_profile'] not in {'upstream_reflexion_raw_serial_v1','reflexion_raw_serial_profiles_v2'}:
            raise ValueError('Model serving identity')
        if a.decoding_profile!='author_reflexion' and a.decoding_profile not in health.get('sampling_profiles',[]):
            raise ValueError('Service does not implement chosen model profile')
        if health['serving_profile']=='reflexion_raw_serial_profiles_v2' and (health['model_protocol_sha256']!=file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')) or health['dtype']!='bfloat16'):
            raise ValueError('Client/server model protocol mismatch')
        identity = {'manifest_sha256': manifest['sha256'], 'trials': a.trials, 'max_calls': a.max_calls,
            'source_hashes': current['source_hashes'], 'driver_sha256': file_hash(Path(__file__)),
            'tool_revision_sha256': revision['sha256'], 'continuation_sha256': contract['sha256'],
            'input_format': 'qwen3_non_thinking_actionline', 'environment_action_alias': ACTION_ALIAS_PROFILE,
            'decoding_profile':a.decoding_profile,'model_protocol_sha256':file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')),
            'wrapper_sha256': file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'seed': 42, 'serving': health, 'upstream': upstream,
            'audit_sha256': audit['sha256'], 'environment_audit_sha256': equivalent['sha256']}
        if (a.output/'run.json').exists():
            if read_json(a.output/'run.json')['identity'] != identity:
                raise ValueError('Run identity changed')
        else:
            remaining = contract['prior_dispatched_calls']+contract['additional_max_calls']-sum(v['dispatched'] for v in ledger().values())
            if not 0 < a.max_calls <= remaining:
                raise ValueError('Stage reservation exceeds remaining continuation allowance')
            write_json(a.output/'run.json', {'identity': identity, 'created_unix': time.time(),
                'scope': 'development; revised public tool grammar; native actor and reflector'})

        def permitted():
            allowance(contract)
            if len(list(a.output.glob('**/calls/*/attempt-*.request.json'))) >= a.max_calls:
                raise RuntimeError('Stage call allowance exhausted')

        for index,task in enumerate(manifest['tasks']):
            memory = []
            for trial in range(a.trials):
                result = run_attempt(task, memory, trial, a.output/'tasks'/task['task_id']/f'trial-{trial:02d}',
                    transport, Path('data/alfworld'), 42, permitted, input_format=identity['input_format'],decoding_profile=a.decoding_profile)
                memory = result['memory_after']
                print({k: result[k] for k in ['task_id', 'trial', 'success', 'decisions', 'dispatched_calls']}, flush=True)
                if result['success']:
                    break
            report = summarize(a.output, manifest_path=a.manifest)
            report = {k: v for k, v in report.items() if k != 'sha256'}
            report['input_format'] = identity['input_format']; report['environment_action_alias'] = ACTION_ALIAS_PROFILE
            report['decoding_profile']=a.decoding_profile
            report['sha256'] = digest(report)
            write_json(a.output/'summary.json', report)
            if a.stop_after_tasks and index+1>=a.stop_after_tasks: break
        write_json(PHASE/'status.json', {'stage': 'tool_stage_completed' if report['completed_tasks']==report['expected_tasks'] else 'tool_stage_paused', 'run': str(a.output),
            'completed_tasks': report['completed_tasks'], 'successes': report['successes'],
            'ledger': ledger(), 'updated_unix': time.time()})


if __name__ == '__main__':
    main()
