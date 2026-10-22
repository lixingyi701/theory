"""Finite continuation accounting with the already frozen model/tool interface."""
import argparse
import time
from pathlib import Path

from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import CompletionTransport
from continue_reflexion_pilot import ledger

FREEZE = Path('repro/reflexion-20261023/engineering-interface-freeze.json')
PREVIOUS = Path('repro/reflexion-20261023/normalization-closeout.json')
PROTOCOL = Path('configs/reflexion_qwen3_8b_protocol.json')


def verify(path):
    contract = read_json(path/'contract.json')
    if contract['sha256'] != digest({k: v for k, v in contract.items() if k != 'sha256'}):
        raise ValueError('Campaign integrity')
    current = provenance()
    if current['source_hashes'] != contract['source_hashes'] or current['native_hashes'] != contract['native_hashes']:
        raise ValueError('Frozen library or environment changed')
    for name, expected in contract['protected_files'].items():
        if file_hash(Path(name)) != expected:
            raise ValueError('Campaign dependency changed: '+name)
    health = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)._request('/models')
    if health != contract['serving']:
        raise ValueError('Serving identity changed')
    return contract


def permit(contract):
    count = sum(row['dispatched'] for row in ledger().values())
    if count >= contract['prior_dispatched_calls']+contract['additional_max_calls']:
        raise RuntimeError('Campaign call limit reached')
    if time.time() >= contract['deadline_unix']:
        raise RuntimeError('Campaign deadline reached')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    with run_lock(args.campaign):
        if (args.campaign/'contract.json').exists():
            raise FileExistsError('An existing campaign cannot be reset')
        old = read_json(PREVIOUS); current = provenance(); freeze = read_json(FREEZE)
        if freeze['source_hashes'] != current['source_hashes']:
            raise ValueError('Engineering interface source mismatch')
        entries = ledger()
        if any(row['dispatched'] != row['responses'] for row in entries.values()):
            raise ValueError('Resolve previous requests first')
        before = sum(row['dispatched'] for row in entries.values())
        if before < old['all_dispatched_calls']:
            raise ValueError('Earlier request history is missing')
        health = CompletionTransport('http://127.0.0.1:8001/v1', 'local-development', 180)._request('/models')
        if health['model_protocol_sha256'] != file_hash(PROTOCOL):
            raise ValueError('Model protocol mismatch')
        protected = [str(FREEZE), str(PREVIOUS), str(PROTOCOL),
                     'configs/qwen3_8b_non_thinking_wrapper.json', str(args.manifest),
                     'scripts/reflexion_campaign.py', 'scripts/continue_reflexion_pilot.py']
        now = time.time()
        contract = {'authorization': 'User explicitly continued the next research stage on 2026-10-23',
            'scope': 'Prospective thermal-task prefixes, public precondition replay, evidence-qualified paired forks; sealed holdout excluded',
            'created_unix': now, 'deadline_unix': now+10800, 'additional_max_calls': 6000,
            'prior_dispatched_calls': before, 'prior_ledger': entries,
            'source_commit_before_this_stage': '22d570ecff779cb0115a681d6fbd0db2c6528dbe',
            'source_hashes': current['source_hashes'], 'native_hashes': current['native_hashes'],
            'serving': health, 'decoding_profile': freeze['selected_decoding_profile'],
            'manifest': str(args.manifest), 'manifest_sha256': read_json(args.manifest)['sha256'],
            'protected_files': {name: file_hash(Path(name)) for name in protected}}
        contract['sha256'] = digest(contract)
        write_json(args.campaign/'contract.json', contract)
        print({'prior_calls': before, 'additional_limit': 6000, 'seconds': 10800, 'sha256': contract['sha256']})


if __name__ == '__main__':
    main()
