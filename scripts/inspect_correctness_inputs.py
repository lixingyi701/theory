#!/usr/bin/env python3
"""Check saved first-actor prompts without making model or environment calls."""
import argparse
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from run_correctness_diagnostic import verify_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261024/correctness-diagnostic-protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    entries = protocol['entries']
    header = '\n\nYour memory for the task below:'

    def block(labels):
        return header + ''.join(f'\nTrial {i}:\n{entries[label].strip()}' for i, label in enumerate(labels))

    replacements = [('N', 'empty', ['N'], []), ('C', 'N', ['C'], ['N']),
                    ('EC', 'EN', ['E', 'N', 'C'], ['E', 'N', 'N'])]
    rows = []
    for case in protocol['cases']:
        task_id = case['task_id']
        for seed in protocol['seeds']:
            requests = {}
            for condition in ['C', 'N', 'empty', 'EC', 'EN']:
                path = args.run / 'cells' / f'correctness-{task_id}-{seed}-{condition}' / 'trial-01' / 'calls' / 'actor-0000' / 'attempt-000.request.json'
                requests[condition] = (path, read_json(path)['payload'])
            checks = {}
            for left, right, before, after in replacements:
                original = block(before)
                replacement = block(after) if after else ''
                a = requests[left][1]
                b = requests[right][1]
                if a['prompt'].count(original) != 1:
                    raise ValueError(f'Unexpected memory rendering: {task_id}/{seed}/{left}')
                replaced = {**a, 'prompt': a['prompt'].replace(original, replacement, 1)}
                if replaced != b:
                    raise ValueError(f'Additional payload differences: {task_id}/{seed}/{left}/{right}')
                checks[f'{left}_to_{right}'] = True
            rows.append({'task_id': task_id, 'seed': seed, 'checks': checks,
                         'first_request_file_sha256': {c: file_hash(p) for c, (p, _) in requests.items()}})
    report = {'scope': 'Postcollection comparison of saved first-actor payloads; no new calls',
              'protocol_sha256': protocol['sha256'], 'driver_sha256': file_hash(Path(__file__)),
              'renderer_sha256': file_hash(Path('third_party/reflexion/env_history.py')),
              'task_seed_pairs': len(rows), 'replacement_checks': len(rows) * 3,
              'memory_header': header, 'entry_prefix': '\nTrial {i}:\n',
              'N_to_empty_removes_header_and_entry': True,
              'C_to_N_preserves_header_and_replaces_only_entry_text': True,
              'EC_to_EN_preserves_three_entry_layout': True, 'rows': rows}
    report['sha256'] = digest(report)
    write_json(args.output, report)
    print({'task_seed_pairs': len(rows), 'replacement_checks': len(rows) * 3, 'sha256': report['sha256']})


if __name__ == '__main__':
    main()
