#!/usr/bin/env python3
"""Paired five-arm pilot using the frozen parent input profile and public facts."""
import argparse
import time
from pathlib import Path

from agent_reliability.cli import provenance
from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from continue_reflexion_pilot import PHASE, PILOT, MANIFEST, allowance, ledger
from run_reflexion_tool_stage import verify_stage_revision
from run_reflexion_forks import memories
from summarize_reflexion import costs


def check_target(target, parent_dir):
    if target['sha256'] != digest({k: v for k, v in target.items() if k != 'sha256'}):
        raise ValueError('Target integrity')
    parent = read_json(parent_dir/'result.json')
    if parent['success'] or file_hash(parent_dir/'result.json') != target['parent_result_sha256']:
        raise ValueError('Parent mismatch')
    if not target['public_evidence_reviewed'] or target['parent_trial'] != 0:
        raise ValueError('Require pre-reviewed first failed attempt')
    if len(parent['memory_after']) != len(parent['memory_before'])+1:
        raise ValueError('One native reflection required')
    transitions = read_json(parent_dir/'checkpoint.json')['transitions']
    if not target['public_evidence']:
        raise ValueError('Missing public evidence')
    for evidence in target['public_evidence']:
        if evidence.get('source') == 'initial_public_task':
            request = parent_dir/'calls/actor-0000/attempt-000.request.json'
            if file_hash(request) != evidence['request_sha256'] or parent['initial_observation_sha256'] != evidence['initial_observation_sha256']:
                raise ValueError('Public task evidence changed')
            if target['stratum'] != 'native_plan_goal_destination' or target['original_text'] != evidence['correct_destination'] or target['replacement_text'] != evidence['wrong_destination']:
                raise ValueError('Goal binding target differs from reviewed evidence')
            continue
        step = transitions[evidence['transition_index']]
        if step['action'] != evidence['action'] or digest(step['observation']) != evidence['observation_sha256']:
            raise ValueError('Shared-prefix evidence changed')
    return parent, memories(parent, target)


def payloads(path):
    return {p.parent.name: read_json(p)['payload_sha256']
            for p in sorted((path/'calls').glob('*/attempt-000.request.json'))}


def summarize_target(root, target, trials,parent_run):
    rows = []
    for branch in ['original', 'sham', 'retain_error', 'correct_target', 'discard_new']:
        path = root/'branches'/branch
        results = [read_json(p) for p in sorted(path.glob('trial-*/result.json'))]
        complete = any(r['success'] for r in results) or len(results) == trials
        rows.append({'branch': branch, 'completed': complete,
            'success': any(r['success'] for r in results) if complete else None,
            'attempts': len(results), 'decisions': sum(r['decisions'] for r in results), 'costs': costs(path)})
    controls = {}
    for name in ['sham', 'correct_target']:
        pairs = []
        for offset in range(trials):
            trial = target['parent_trial']+1+offset
            a = root/'branches/original'/f'trial-{trial:02d}'
            b = root/'branches'/name/f'trial-{trial:02d}'
            if (a/'result.json').exists() and (b/'result.json').exists():
                pairs.append({'trial': trial, 'fresh_payloads_equal': payloads(a) == payloads(b),
                    'trajectory_equal': read_json(a/'result.json')['trajectory_sha256'] == read_json(b/'result.json')['trajectory_sha256']})
        controls[name] = pairs
    report = {'task_id': target['task_id'], 'target_sha256': target['sha256'],
        'rows': rows, 'completed_branches': sum(r['completed'] for r in rows),
        'fresh_input_controls': controls, 'costs': costs(root)}
    native_controls = []
    for offset in range(trials):
        trial = target['parent_trial']+1+offset
        a = root/'branches/original'/f'trial-{trial:02d}'
        b = parent_run/'tasks'/target['task_id']/f'trial-{trial:02d}'
        if (a/'result.json').exists() and (b/'result.json').exists():
            native_controls.append({'trial': trial, 'fresh_payloads_equal': payloads(a) == payloads(b),
                'trajectory_equal': read_json(a/'result.json')['trajectory_sha256'] == read_json(b/'result.json')['trajectory_sha256']})
    report['native_prefix_controls'] = native_controls
    report['sha256'] = digest(report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('runs/reflexion-chat-forks-20261023'))
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--max-calls', type=int, default=3500)
    p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--revision-kind',choices=['tool','model_profile'],default='model_profile')
    p.add_argument('--validate-only',action='store_true')
    a = p.parse_args()
    if a.trials != 2 or a.max_calls <= 0:
        raise ValueError('Frozen two-attempt pilot budget required')
    with run_lock(PHASE.resolve()), run_lock(a.output):
        contract,revision = verify_stage_revision(kind=a.revision_kind)
        preceding = ledger()
        if any(v['dispatched'] != v['responses'] for v in preceding.values()):
            raise ValueError('Resolve uncertain requests before forks')
        pilot = read_json(a.parent_run/'summary.json')
        if pilot['completed_tasks'] != 24:
            raise ValueError('Complete parent pilot before its paired forks')
        manifest = load_manifest(a.manifest)
        tasks = {t['task_id']: t for t in manifest['tasks']}
        registry = read_json(a.targets)
        if registry['sha256'] != digest({k: v for k, v in registry.items() if k != 'sha256'}):
            raise ValueError('Target registry integrity')
        if registry['manifest_sha256'] != manifest['sha256'] or not 1 <= len(registry['targets']) <= 4:
            raise ValueError('Expected up to four evidence-qualified pilot tasks')
        if len({t['task_id'] for t in registry['targets']}) != len(registry['targets']):
            raise ValueError('One target per independent task')
        upstream = read_json(a.parent_run/'run.json')['identity']
        _, receipt = author_namespace()
        if upstream['source_hashes'] != provenance()['source_hashes'] or upstream['upstream'] != receipt:
            raise ValueError('Parent execution source changed')
        if upstream['wrapper_sha256'] != file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')):
            raise ValueError('Input wrapper changed')
        transport = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)
        if transport._request('/models') != upstream['serving']:
            raise ValueError('Parent serving identity changed')
        prepared = []
        for target in registry['targets']:
            directory = a.parent_run/'tasks'/target['task_id']/f"trial-{target['parent_trial']:02d}"
            parent, branches = check_target(target, directory)
            prepared.append((target, parent, branches))
        identity = {'targets_sha256': registry['sha256'], 'parent_run_sha256': file_hash(a.parent_run/'run.json'),
            'continuation_sha256': contract['sha256'], 'source_hashes': upstream['source_hashes'],
            'driver_sha256': file_hash(Path(__file__)), 'trials': a.trials, 'max_calls': a.max_calls,
            'memory_intervention_driver_sha256': file_hash(Path('scripts/run_reflexion_forks.py')),
            'summary_driver_sha256': file_hash(Path('scripts/summarize_reflexion.py')),
            'input_format': upstream['input_format'], 'wrapper_sha256': upstream['wrapper_sha256'], 'seed': upstream['seed']}
        identity.update(tool_revision_sha256=revision['sha256'],decoding_profile=upstream['decoding_profile'])
        identity.update(serving=upstream['serving'],upstream=upstream['upstream'],
            environment_action_alias=upstream['environment_action_alias'],manifest_sha256=manifest['sha256'],
            model_protocol_sha256=file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')))
        if a.validate_only:
            print({'validated_targets':len(prepared),'identity_sha256':digest(identity),'model_dispatches':0})
            return
        if (a.output/'run.json').exists():
            if read_json(a.output/'run.json')['identity'] != identity:
                raise ValueError('Fork identity changed')
        else:
            if registry['frozen_unix'] > time.time():
                raise ValueError('Future-dated target registry')
            remaining=contract['prior_dispatched_calls']+contract['additional_max_calls']-sum(v['dispatched'] for v in ledger().values())
            if a.max_calls>remaining:
                raise ValueError('Fork reservation exceeds remaining continuation allowance')
            write_json(a.output/'run.json', {'identity': identity, 'created_unix': time.time(),
                'scope': 'development paired forks; target eligibility reviewed before branch outcomes'})

        def permitted():
            allowance(contract)
            if len(list(a.output.glob('**/calls/*/attempt-*.request.json'))) >= a.max_calls:
                raise RuntimeError('Fork call allowance exhausted')

        reports = []
        for target, parent, branches in prepared:
            root = a.output/'targets'/target['task_id']
            for branch, initial_memory in branches.items():
                memory = list(initial_memory)
                for offset in range(a.trials):
                    trial = target['parent_trial']+1+offset
                    result = run_attempt(tasks[target['task_id']], memory, trial,
                        root/'branches'/branch/f'trial-{trial:02d}', transport,
                        Path('data/alfworld'), upstream['seed'], permitted, input_format=upstream['input_format'],decoding_profile=upstream['decoding_profile'])
                    if result['initial_observation_sha256'] != parent['initial_observation_sha256']:
                        raise ValueError('Fork initial observation differs from parent')
                    memory = result['memory_after']
                    print({'task_id': target['task_id'], 'branch': branch, 'trial': trial,
                        'success': result['success'], 'decisions': result['decisions']}, flush=True)
                    if result['success']:
                        break
                report = summarize_target(root, target, a.trials,a.parent_run)
                write_json(root/'summary.json', report)
            reports.append(report)
            write_json(a.output/'summary.json', {'completed_targets': len(reports),
                'expected_targets': len(prepared), 'targets': reports, 'costs': costs(a.output)})
        write_json(PHASE/'status.json', {'stage': 'forks_completed', 'completed_targets': len(reports),
            'ledger': ledger(), 'updated_unix': time.time()})


if __name__ == '__main__':
    main()
