#!/usr/bin/env python3
"""Five-arm repair of an evidence-qualified native procedural error."""
import argparse
import time
from pathlib import Path

from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import ROOT, author_namespace, run_attempt
from continue_reflexion_pilot import ledger
from reflexion_campaign import verify, permit
from run_reflexion_pilot_forks import payloads
from summarize_reflexion import costs


def memories(parent, target):
    native = list(parent['memory_after']); text = native[-1]
    start, end = target['span']
    if text[start:end] != target['original_text'] or target['original_text'] == target['corrected_text']:
        raise ValueError('Native procedural span mismatch')
    repaired = text[:start]+target['corrected_text']+text[end:]
    return {'original': native, 'sham': list(native), 'retain_error': list(native),
            'correct_target': native[:-1]+[repaired], 'discard_new': native[:-1]}


def check_target(target, parent_dir):
    if target['sha256'] != digest({k: v for k, v in target.items() if k != 'sha256'}):
        raise ValueError('Target integrity')
    parent = read_json(parent_dir/'result.json')
    if parent['success'] or target['parent_trial'] != 0 or not target['public_evidence_reviewed']:
        raise ValueError('Reviewed first native failure required')
    if file_hash(parent_dir/'result.json') != target['parent_result_sha256']:
        raise ValueError('Parent result changed')
    if len(parent['memory_after']) != len(parent['memory_before'])+1:
        raise ValueError('One new native reflection required')
    probe = read_json(Path(target['probe_file']))
    if probe['sha256'] != digest({k: v for k, v in probe.items() if k != 'sha256'}):
        raise ValueError('Public replay evidence integrity')
    if probe['sha256'] != target['probe_sha256'] or not probe['holding_precondition_supported']:
        raise ValueError('Holding premise not supported by replay')
    if probe['parent_result_sha256'] != target['parent_result_sha256'] or probe['checkpoint_sha256'] != file_hash(parent_dir/'checkpoint.json'):
        raise ValueError('Public replay used another prefix')
    demos = read_json(ROOT/'prompts/alfworld_3prompts.json')
    if digest(demos[target['shared_demo_key']]) != target['shared_demo_sha256']:
        raise ValueError('Shared demonstration changed')
    return parent, memories(parent, target)


def summarize_target(root, target, trials):
    rows = []
    for branch in ['original', 'sham', 'retain_error', 'correct_target', 'discard_new']:
        path = root/'branches'/branch
        results = [read_json(p) for p in sorted(path.glob('trial-*/result.json'))]
        complete = any(r['success'] for r in results) or len(results) == trials
        rows.append({'branch': branch, 'completed': complete, 'attempts': len(results),
            'success': any(r['success'] for r in results) if complete else None,
            'decisions': sum(r['decisions'] for r in results), 'costs': costs(path)})
    controls = {}
    for branch in ['sham', 'retain_error']:
        pairs = []
        for trial in [1, 2]:
            a = root/'branches/original'/f'trial-{trial:02d}'
            b = root/'branches'/branch/f'trial-{trial:02d}'
            if (a/'result.json').exists() and (b/'result.json').exists():
                pairs.append({'trial': trial, 'fresh_payloads_equal': payloads(a) == payloads(b),
                              'trajectory_equal': read_json(a/'result.json')['trajectory_sha256'] == read_json(b/'result.json')['trajectory_sha256']})
        controls[branch] = pairs
    report = {'task_id': target['task_id'], 'target_sha256': target['sha256'],
              'source_truth': 'native_error', 'rows': rows, 'completed_branches': sum(r['completed'] for r in rows),
              'fresh_input_controls': controls, 'costs': costs(root)}
    report['sha256'] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ['campaign', 'parent-run', 'manifest', 'targets', 'parent-protocol-audit', 'output']:
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--max-calls', type=int, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    with run_lock(args.campaign.resolve()), run_lock(args.output):
        contract = verify(args.campaign)
        if any(row['dispatched'] != row['responses'] for row in ledger().values()):
            raise ValueError('Unresolved preceding requests')
        manifest = load_manifest(args.manifest)
        summary = read_json(args.parent_run/'summary.json')
        if summary['completed_tasks'] != summary['expected_tasks']:
            raise ValueError('Complete parent cohort first')
        parent_identity = read_json(args.parent_run/'run.json')['identity']
        audit = read_json(args.parent_protocol_audit)
        if audit['sha256'] != digest({k: v for k, v in audit.items() if k != 'sha256'}) or not audit['passed']:
            raise ValueError('Parent interface audit failed')
        if audit['run_sha256'] != file_hash(args.parent_run/'run.json'):
            raise ValueError('Parent audit identity changed')
        if parent_identity['source_hashes'] != contract['source_hashes'] or parent_identity['serving'] != contract['serving']:
            raise ValueError('Parent source or service differs from frozen campaign')
        if parent_identity['manifest_sha256'] != manifest['sha256']:
            raise ValueError('Parent manifest mismatch')
        registry = read_json(args.targets)
        if registry['sha256'] != digest({k: v for k, v in registry.items() if k != 'sha256'}):
            raise ValueError('Target registry integrity')
        if registry['manifest_sha256'] != manifest['sha256'] or not 1 <= len(registry['targets']) <= 4:
            raise ValueError('Expected up to four qualified targets')
        if len({t['task_id'] for t in registry['targets']}) != len(registry['targets']):
            raise ValueError('One target per base task')
        tasks = {t['task_id']: t for t in manifest['tasks']}
        prepared = [(target, *check_target(target, args.parent_run/'tasks'/target['task_id']/'trial-00'))
                    for target in registry['targets']]
        _, upstream = author_namespace()
        identity = {'targets_sha256': registry['sha256'], 'parent_run_sha256': file_hash(args.parent_run/'run.json'),
            'campaign_sha256': contract['sha256'], 'source_hashes': contract['source_hashes'],
            'driver_sha256': file_hash(Path(__file__)), 'summary_payload_helper_sha256': file_hash(Path('scripts/run_reflexion_pilot_forks.py')),
            'cost_helper_sha256': file_hash(Path('scripts/summarize_reflexion.py')),
            'trials': 2, 'max_calls': args.max_calls, 'seed': 42,
            'input_format': parent_identity['input_format'], 'decoding_profile': parent_identity['decoding_profile'],
            'wrapper_sha256': parent_identity['wrapper_sha256'], 'serving': contract['serving'], 'upstream': upstream,
            'environment_action_alias': ACTION_ALIAS_PROFILE, 'manifest_sha256': manifest['sha256'],
            'model_protocol_sha256': file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))}
        if args.validate_only:
            print({'validated_targets': len(prepared), 'identity_sha256': digest(identity), 'model_calls': 0})
            return
        if (args.output/'run.json').exists():
            if read_json(args.output/'run.json')['identity'] != identity:
                raise ValueError('Fork identity changed')
        else:
            remaining = contract['prior_dispatched_calls']+contract['additional_max_calls']-sum(x['dispatched'] for x in ledger().values())
            if not 0 < args.max_calls <= remaining or registry['frozen_unix'] > time.time():
                raise ValueError('Fork budget or freeze time invalid')
            write_json(args.output/'run.json', {'identity': identity, 'created_unix': time.time(), 'scope': registry['scope']})

        def allowed():
            permit(contract)
            if len(list(args.output.glob('**/calls/*/attempt-*.request.json'))) >= args.max_calls:
                raise RuntimeError('Fork stage call limit')

        transport = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)
        reports = []
        for target, parent, branches in prepared:
            root = args.output/'targets'/target['task_id']
            for branch, initial in branches.items():
                memory = list(initial)
                for trial in [1, 2]:
                    result = run_attempt(tasks[target['task_id']], memory, trial, root/'branches'/branch/f'trial-{trial:02d}',
                        transport, Path('data/alfworld'), 42, allowed, input_format=identity['input_format'],
                        decoding_profile=identity['decoding_profile'])
                    if result['initial_observation_sha256'] != parent['initial_observation_sha256']:
                        raise ValueError('Fork reset changed')
                    memory = result['memory_after']
                    print({'task_id': target['task_id'], 'branch': branch, 'trial': trial,
                           'success': result['success'], 'decisions': result['decisions']}, flush=True)
                    if result['success']:
                        break
                report = summarize_target(root, target, 2)
                write_json(root/'summary.json', report)
            reports.append(report)
            write_json(args.output/'summary.json', {'completed_targets': len(reports), 'expected_targets': len(prepared),
                'targets': reports, 'costs': costs(args.output)})
        write_json(args.campaign/'status.json', {'stage': 'precondition_forks_completed',
                    'output': str(args.output), 'ledger': ledger(), 'updated_unix': time.time()})


if __name__ == '__main__':
    main()
