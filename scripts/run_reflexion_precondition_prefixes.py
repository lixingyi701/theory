#!/usr/bin/env python3
"""One native attempt per frozen thermal task under a finite campaign."""
import argparse
import time
from pathlib import Path

from agent_reliability.audit import verify_audit
from agent_reliability.cli import provenance
from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from continue_reflexion_pilot import ledger
from reflexion_campaign import verify, permit
from summarize_reflexion import summarize


def main():
    parser = argparse.ArgumentParser()
    for name in ['campaign', 'manifest', 'output', 'audit', 'environment-audit']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    with run_lock(args.campaign.resolve()), run_lock(args.output):
        contract = verify(args.campaign)
        if any(row['dispatched'] != row['responses'] for row in ledger().values()):
            raise ValueError('Unresolved preceding requests')
        manifest = load_manifest(args.manifest)
        if manifest['sha256'] != contract['manifest_sha256'] or len(manifest['tasks']) != 24:
            raise ValueError('Frozen prospective manifest required')
        audit = read_json(args.audit); verify_audit(audit, manifest)
        official = read_json(args.environment_audit); current = provenance()
        if official['sha256'] != digest({k: v for k, v in official.items() if k != 'sha256'}) or not official['passed']:
            raise ValueError('Official environment audit failed')
        if official['manifest_sha256'] != manifest['sha256'] or official['native_hashes'] != current['native_hashes']:
            raise ValueError('Environment identity mismatch')
        for name in ['alfworld.py', 'env.py', 'audit.py', 'reference.py']:
            if official['source_hashes'][name] != current['source_hashes'][name]:
                raise ValueError('Environment adapter changed')
        _, receipt = author_namespace()
        identity = {'manifest_sha256': manifest['sha256'], 'trials': 1, 'max_calls': 1600,
            'source_hashes': current['source_hashes'], 'driver_sha256': file_hash(Path(__file__)),
            'campaign_sha256': contract['sha256'], 'input_format': 'qwen3_non_thinking_actionline',
            'environment_action_alias': ACTION_ALIAS_PROFILE, 'decoding_profile': contract['decoding_profile'],
            'model_protocol_sha256': file_hash(Path('configs/reflexion_qwen3_8b_protocol.json')),
            'wrapper_sha256': file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'seed': 42, 'serving': contract['serving'], 'upstream': receipt,
            'audit_sha256': audit['sha256'], 'environment_audit_sha256': official['sha256']}
        path = args.output/'run.json'
        if path.exists():
            if read_json(path)['identity'] != identity:
                raise ValueError('Prefix identity changed')
        else:
            write_json(path, {'identity': identity, 'created_unix': time.time(),
                             'scope': 'prospective thermal development prefixes; native first attempts'})
        transport = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)

        def allowed():
            permit(contract)
            if len(list(args.output.glob('**/calls/*/attempt-*.request.json'))) >= 1600:
                raise RuntimeError('Prefix stage call limit')

        for task in manifest['tasks']:
            result = run_attempt(task, [], 0, args.output/'tasks'/task['task_id']/'trial-00', transport,
                Path('data/alfworld'), 42, allowed, input_format=identity['input_format'],
                decoding_profile=identity['decoding_profile'])
            summary = summarize(args.output, manifest_path=args.manifest)
            write_json(args.output/'summary.json', summary)
            write_json(args.campaign/'prefix-summary.json', summary)
            print({'task_id': task['task_id'], 'success': result['success'], 'calls': result['dispatched_calls'],
                   'completed_tasks': summary['completed_tasks']}, flush=True)


if __name__ == '__main__':
    main()
