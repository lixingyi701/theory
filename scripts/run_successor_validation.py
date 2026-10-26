#!/usr/bin/env python3
"""Execute a separate four-attempt campaign using the unchanged native worker."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import CompletionTransport, ACTION_ALIAS_PROFILE
from agent_reliability.reflexion import author_namespace
from continue_reflexion_pilot import ledger
from prepare_shared_calibration import save
from prepare_successor_protocol import PROTOCOL, MANIFEST, ROOT, PRIMARY, SECONDARY, SEEDS, LIMITS, make_trajectories, checked, forecasts, PARAMETERS, PLAN
from run_shared_calibration import worker
from run_residual_context import verify_dependencies
from run_middle_context_transport import replica_audit
from summarize_reflexion import costs


def validate_matrix(protocol, manifest):
    cases = protocol['cases']
    ids = [c['task_id'] for c in cases]
    if len(ids) != len(set(ids)) or len(ids) != 100:
        raise ValueError('100 distinct tasks required')
    if [c['task'] for c in cases] != manifest['tasks'] or protocol['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Selected task manifest changed')
    if Counter(c['task']['scene_id'] for c in cases) != manifest['counts_by_scene'] or len(manifest['counts_by_scene']) != 25:
        raise ValueError('Scene coverage changed')
    if protocol['primary_conditions'] != PRIMARY or protocol['secondary_conditions'] != SECONDARY or protocol['crossed_seeds'] != SEEDS:
        raise ValueError('Panels or crossed seeds changed')
    if protocol['secondary_recovery_task_ids'] != manifest['secondary_recovery_task_ids']:
        raise ValueError('Recovery task panel changed')
    expected = make_trajectories(cases, set(manifest['secondary_recovery_task_ids']))
    if protocol['trajectories'] != expected or len({r['id'] for r in expected}) != 550:
        raise ValueError('Incomplete or altered four-attempt matrix/order')
    if any(c['task_id'] != c['task']['task_id'] for c in cases):
        raise ValueError('Case/task identity mismatch')
    secondary_scenes = [c['task']['scene_id'] for c in cases if c['task_id'] in manifest['secondary_recovery_task_ids']]
    if len(secondary_scenes) != 25 or len(set(secondary_scenes)) != 25:
        raise ValueError('Recovery panel requires one task per scene')
    return {'tasks': 100, 'scenes': 25, 'trajectories': 550, 'maximum_attempts': 2200}


def verify_protocol():
    protocol = checked(PROTOCOL)
    verify_dependencies(protocol)
    if provenance()['source_hashes'] != protocol['source_hashes']:
        raise ValueError('Agent source changed')
    if protocol['limits'] != LIMITS or protocol['gpu_indices'] != [2, 3] or protocol['ports'] != [8003, 8004]:
        raise ValueError('Resource assignment or bounded allowance changed')
    parameters = checked(PARAMETERS)
    plan = checked(PLAN)
    expected_models = {'full_seven': parameters['models']['seven']['model'],
                       'full_six': parameters['models']['six']['model']}
    parent = checked(PROTOCOL.with_name('shared-validation-protocol.json'))
    if protocol['models'] != expected_models or protocol['entries'] != parent['entries']:
        raise ValueError('Frozen predictor or initial entries changed')
    if protocol['parent_calibration_parameters_sha256'] != parameters['sha256'] or protocol['plan_sha256'] != plan['sha256']:
        raise ValueError('New parameter/design lineage changed')
    for k, v in protocol['entries'].items():
        if digest(v) != protocol['entry_hashes'][k]:
            raise ValueError('Entry hash changed')
    for name, predictions in forecasts(protocol['models']).items():
        for condition, curve in predictions.items():
            if not np.allclose(protocol['forecasts'][name][condition], curve, atol=1e-12, rtol=0):
                raise ValueError('Frozen prediction mismatch')
    validate_matrix(protocol, checked(MANIFEST))
    return protocol


def serving_health(protocol):
    replicas = [CompletionTransport(f'http://127.0.0.1:{port}/v1', 'local-development', 180)._request('/models')
                for port in protocol['ports']]
    if any(h != replicas[0] for h in replicas):
        raise ValueError('Replica serving identities differ')
    health = replicas[0]
    if (health['service_sha256'] != file_hash(Path('scripts/deployment/serve_reflexion_profiles.py')) or
        health['serving_profile'] != 'reflexion_raw_serial_profiles_v2' or health['dtype'] != 'bfloat16' or
        health['model_revision'] != 'b968826d9c46dd6066d109eabc6255188de91218' or
        health['gpu_memory_utilization'] != 0.70 or
        health['model_protocol_sha256'] != file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))):
        raise ValueError('Serving implementation changed')
    return health


def run(protocol):
    with run_lock(ROOT):
        if (ROOT / 'contract.json').exists():
            raise FileExistsError('Do not restart/reset a campaign')
        before = ledger()
        if sum(v['dispatched'] for v in before.values()) != protocol['prior_global_calls'] or any(v['dispatched'] != v['responses'] for v in before.values()):
            raise ValueError('Prior ledger changed or has an uncertain request')
        freeze = Path('runs/setup/successor-validation-freeze-commit.txt').read_text().strip()
        if len(freeze) != 40 or any(c not in '0123456789abcdef' for c in freeze):
            raise ValueError('Public source/protocol freeze missing')
        health = serving_health(protocol)
        current = provenance()
        _, upstream = author_namespace()
        now = time.time()
        ownership_path = Path('runs/setup/successor-service-ownership.json')
        preparation_started = read_json(ownership_path)['created_unix']
        if not 0 <= now-preparation_started < LIMITS['seconds']:
            raise ValueError('Whole-campaign wall allowance exhausted before sampling')
        contract = save(ROOT / 'contract.json', {'protocol_sha256': protocol['sha256'], 'freeze_commit': freeze,
            'created_unix': now, 'preparation_started_unix': preparation_started,
            'service_ownership_file_sha256': file_hash(ownership_path),
            'preparation_elapsed_seconds': now-preparation_started,
            'deadline_unix': preparation_started + LIMITS['seconds'],
            'max_calls': LIMITS['additional_calls_including_replica_checks'], 'prior_ledger': before,
            'serving': health, 'source_hashes': current['source_hashes'], 'native_hashes': current['native_hashes'],
            'authorization': 'User authorized the100-task successor validation on GPU2/3, preserving other processes,2026-10-26'})
        write_json(ROOT / 'run.json', {'created_unix': now, 'identity': {'protocol_sha256': protocol['sha256'],
            'campaign_sha256': contract['sha256'], 'serving': health, 'upstream': upstream, 'source_hashes': current['source_hashes'],
            'input_format': 'qwen3_non_thinking_actionline', 'decoding_profile': 'qwen3_non_thinking_recommended_v1',
            'environment_action_alias': ACTION_ALIAS_PROFILE,
            'wrapper_sha256': file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256': file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))}})
        replica_audit(ROOT, protocol, contract)
        children = []
        for index in range(2):
            log = (ROOT / f'worker-{index}.log').open('a')
            children.append((subprocess.Popen([sys.executable, __file__, 'run', '--worker', str(index)], stdout=log, stderr=subprocess.STDOUT), log))
        codes = []
        for process, log in children:
            codes.append(process.wait())
            log.close()
        workers = [read_json(ROOT / f'worker-{i}.json') for i in range(2)]
        rows = [r for w in workers for r in w['rows']]
        trajectories = [t for w in workers for t in w['trajectories']]
        expected = {r['id'] for r in protocol['trajectories']}
        observed = [r['trajectory_id'] for r in trajectories]
        complete = (codes == [0, 0] and len(observed) == len(set(observed)) == 550 and set(observed) == expected
            and all(t['complete'] for t in trajectories) and all(w['status'] == 'complete' for w in workers))
        actual = costs(ROOT)
        reserved = read_json(ROOT / 'budget.json')['reserved']
        summary = save(ROOT / 'summary.json', {'protocol_sha256': protocol['sha256'], 'status': 'complete' if complete else 'partial',
            'expected_tasks': 100, 'completed_tasks': 100 if complete else 0, 'expected_trajectories': 550,
            'completed_trajectories': sum(t['complete'] for t in trajectories), 'attempts_completed': len(rows),
            'attempts_rows': sorted(rows, key=lambda r: (r['trajectory_id'], r['trial'])),
            'trajectories': sorted(trajectories, key=lambda t: t['trajectory_id']),
            'worker_exit_codes': codes, 'worker_statuses': [w['status'] for w in workers],
            'costs_incremental': actual, 'reserved_calls': reserved, 'elapsed_seconds': time.time() - now,
            'parameter_refits': 0, 'holdout134_model_calls': 0})
        print({k: v for k, v in summary.items() if k not in ['attempts_rows', 'trajectories']}, flush=True)
        if not complete or reserved != actual['dispatched_calls'] or reserved != actual['saved_responses'] or actual['uncertain_or_unrecorded_attempts']:
            raise RuntimeError('Incomplete/unconfirmed validation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate', 'run'])
    parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    protocol = verify_protocol()
    if args.action == 'validate':
        health = serving_health(protocol)
        print({**validate_matrix(protocol, checked(MANIFEST)), 'serving_profile': health['serving_profile'], 'new_model_calls': 0})
    elif args.worker is None:
        run(protocol)
    else:
        worker(ROOT, protocol, args.worker)


if __name__ == '__main__':
    main()
