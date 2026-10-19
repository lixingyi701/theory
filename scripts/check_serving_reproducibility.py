#!/usr/bin/env python3
"""Regenerate 12 fixed observed payloads in three orders, with durable journal."""
import argparse
import urllib.request
from pathlib import Path
from agent_reliability.io import digest, read_json, write_json, run_lock
from agent_reliability.llm import CallJournal, endpoint_from_config, response_content


def check(source, output, config):
    manifest = read_json(Path('repro/baseline-20261019/dev12.json'))
    selected = []
    for i, t in enumerate(manifest['tasks']):
        root = source / 'episodes' / f"{t['task_id']}-s42-memory" / 'calls'
        if t['task_id'] == 'ec35b3ffe3eb225f':
            p = root / 'decision-0012-policy' / 'attempt-000.request.json'
        else:
            pattern = 'step-*-memory-0' if i % 2 == 0 else 'decision-*-policy'
            p = sorted(root.glob(pattern + '/attempt-000.request.json'))[0]
        selected.append({'task_id': t['task_id'], 'payload': read_json(p)['payload']})
    transport = endpoint_from_config(config)
    with urllib.request.urlopen(transport.base_url + '/models', timeout=10) as r:
        import json
        health = json.load(r)
    if health.get('serving_profile') != 'offline_serial_eager_nocache_v1':
        raise ValueError('Expected serial offline serving profile')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new probe directory; no implicit retry')
    write_json(output / 'selection.json', selected)
    write_json(output / 'health.json', health)
    journal = CallJournal(output / 'calls', transport, 36)
    hashes = {i: [] for i in range(12)}
    for n, order in enumerate([range(12), range(11, -1, -1), range(12)]):
        for i in order:
            saved = journal.complete(f'pass-{n}-payload-{i:02d}', selected[i]['payload'])
            text, finish = response_content(saved)
            hashes[i].append(digest({'text': text, 'finish': finish}))
    result = {'profile': health['serving_profile'], 'tested_payloads': 12, 'passes': 3,
              'all_equal': all(len(set(v)) == 1 for v in hashes.values()),
              'payloads': [{'task_id': x['task_id'], 'payload_sha256': digest(x['payload']),
                            'output_hashes': hashes[i], 'equal': len(set(hashes[i])) == 1} for i, x in enumerate(selected)],
              'scope': 'finite same-process/order perturbation check; not cross-hardware determinism or task accuracy',
              **journal.totals()}
    write_json(output / 'result.json', result)
    print({k: result[k] for k in ['tested_payloads', 'all_equal', 'dispatched_calls']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/react_base.json'))
    a = p.parse_args()
    with run_lock(a.output):
        check(a.source, a.output, read_json(a.config))
