#!/usr/bin/env python3
"""Publish aggregate engineering gates; never publish complete model responses."""
import argparse
from collections import Counter
from pathlib import Path
from agent_reliability.io import digest, read_json, write_json


def calls(root):
    requests = list(root.glob('episodes/*/calls/*/attempt-*.request.json'))
    responses = [read_json(p)['response'] for p in root.glob('episodes/*/calls/*/response.json')]
    validation = [read_json(p) for p in root.glob('episodes/*/calls/*/validation.json')]
    return {'dispatches': len(requests), 'saved_responses': len(responses),
            'uncertain_or_unrecorded_attempts': len(requests) - len(responses),
            'input_tokens_known': sum(x.get('usage', {}).get('prompt_tokens', 0) for x in responses),
            'output_tokens_known': sum(x.get('usage', {}).get('completion_tokens', 0) for x in responses),
            'invalid_outputs': sum(not x['valid'] for x in validation),
            'truncated_outputs': sum(x['truncated_output'] for x in validation)}


def analyze(base, instrumented):
    runs = [read_json(p / 'run.json') for p in [base, instrumented]]
    models_match = runs[0]['model'] == runs[1]['model']
    source_match = runs[0]['source_hashes'] == runs[1]['source_hashes']
    config_match = ({k: v for k, v in runs[0]['config'].items() if k != 'conditions'} ==
                    {k: v for k, v in runs[1]['config'].items() if k != 'conditions'})
    manifest_match = runs[0]['manifest_sha256'] == runs[1]['manifest_sha256']
    manifest = read_json(Path('repro/baseline-20261019/dev12.json'))
    expected = len(manifest['tasks']) * len(runs[0]['config']['seeds'])
    rows = [read_json(p) for root in [base, instrumented] for p in root.glob('episodes/*/result.json')]
    indexed = {(x['task_id'], x['seed'], x['condition']): x for x in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate episode identity')
    conditions = {}
    for c in ['base', 'memory', 'sham']:
        subset = [x for x in rows if x['condition'] == c]
        conditions[c] = {'completed': len(subset), 'expected': expected, 'missing': expected - len(subset),
                         'successes': sum(x['success'] for x in subset),
                         'successful_types': sorted({x['task_type'] for x in subset if x['success']}),
                         'status_counts': dict(Counter(x['status'] for x in subset)),
                         'env_steps': sum(x['env_steps'] for x in subset),
                         'decisions': sum(x['decisions'] for x in subset),
                         'sham_applied': sum(x['sham_applied'] for x in subset),
                         'by_type': {kind: {'completed': sum(x['task_type'] == kind for x in subset),
                                            'successes': sum(x['success'] for x in subset if x['task_type'] == kind)}
                                     for kind in sorted({t['task_type'] for t in manifest['tasks']})}}
    pairs = []
    for task in manifest['tasks']:
        for seed in runs[0]['config']['seeds']:
            group = {c: indexed.get((task['task_id'], seed, c)) for c in conditions}
            pair = {'task_id': task['task_id'], 'scene_id': task['scene_id'], 'task_type': task['task_type'],
                    'seed': seed, 'success': {c: r['success'] if r else None for c, r in group.items()}}
            if group['memory'] and group['sham']:
                prompt_hashes = []
                for c in ['memory', 'sham']:
                    d = instrumented / 'episodes' / f"{task['task_id']}-s{seed}-{c}"
                    # Include both memory and policy calls, their payloads and actual attempts.
                    prompt_hashes.append([(p.parent.name, p.name, read_json(p)['payload_sha256'])
                                          for p in sorted((d / 'calls').glob('*/attempt-*.request.json'))])
                pair['memory_sham_payloads_equal'] = prompt_hashes[0] == prompt_hashes[1]
                pair['memory_sham_trajectory_equal'] = group['memory']['trajectory_sha256'] == group['sham']['trajectory_sha256']
                pair['sham_applied'] = group['sham']['sham_applied']
            pairs.append(pair)
    complete = all(x['completed'] == expected for x in conditions.values())
    base_gate = (conditions['base']['completed'] == expected and conditions['base']['successes'] >= 6
                 and len(conditions['base']['successful_types']) >= 4)
    lost = sum(p['success']['base'] is True and p['success']['memory'] is False for p in pairs)
    gained = sum(p['success']['base'] is False and p['success']['memory'] is True for p in pairs)
    instrumentation_gate = (complete and lost <= 1 and conditions['memory']['successes'] >= 6
                            and len(conditions['memory']['successful_types']) >= 4)
    sham_gate = (complete and conditions['sham']['sham_applied'] >= 6
                 and all(p.get('memory_sham_payloads_equal') and p.get('memory_sham_trajectory_equal') for p in pairs))
    integrity = models_match and source_match and config_match and manifest_match
    report = {'schema_version': 1, 'scope': 'development capability and instrumentation gates; not a paper score or causal result',
              'complete': complete, 'conditions': conditions, 'pairs': pairs,
              'base_to_memory': {'success_lost': lost, 'success_gained': gained},
              'integrity': {'same_model': models_match, 'same_source_hashes': source_match,
                            'same_config_except_conditions': config_match, 'same_manifest': manifest_match},
              'gates': {'base_capability': base_gate, 'instrumentation': instrumentation_gate,
                        'sham': sham_gate, 'all_passed': integrity and base_gate and instrumentation_gate and sham_gate},
              'costs': {'base': calls(base), 'instrumented': calls(instrumented)},
              'manifest_sha256': runs[0]['manifest_sha256'], 'model': runs[0]['model'],
              'source_hashes': runs[0]['source_hashes'], 'audit_sha256': runs[0]['audit_sha256']}
    return {**report, 'sha256': digest(report)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--instrumented', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); report = analyze(a.base, a.instrumented); write_json(a.output, report)
    print({'complete': report['complete'], 'gates': report['gates'], 'base_to_memory': report['base_to_memory']})
