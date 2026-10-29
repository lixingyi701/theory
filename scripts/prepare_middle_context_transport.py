#!/usr/bin/env python3
"""Freeze condition transport on known tasks: middle position, then native text."""
from pathlib import Path
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, write_json
from run_state_representation import verify_protocol as verify_source

OUTPUT = Path('repro/reflexion-20261023/middle-context-protocol.json')
SOURCE_RUN = 'runs/reflexion-state-representation-20261023'


def native_text(source):
    path = Path(SOURCE_RUN)/'cells'/source['cell_id']/'trial-01'/'result.json'
    result = read_json(path)
    if file_hash(path) != source['result_sha256'] or result['success']:
        raise ValueError('Native source identity mismatch')
    fresh = result['memory_after'][len(result['memory_before']):]
    if len(fresh) != 1 or digest(fresh[0]) != source['reflection_sha256']:
        raise ValueError('Native source reflection changed')
    return fresh[0]


def memory_for(protocol, cell):
    entries = protocol['entries']
    if 'native_source' in cell:
        return [entries['N'], native_text(cell['native_source']), entries['N']]
    return [entries[v] for v in cell['labels']]


def main():
    if OUTPUT.exists():
        raise FileExistsError('Never overwrite frozen predictions or selection')
    source = verify_source(Path('repro/reflexion-20261023/state-representation-protocol.json'))
    analysis = read_json(Path('repro/reflexion-20261023/state-representation-results.json'))
    primary = read_json(Path('repro/reflexion-20261023/state-representation-model-freeze.json'))
    presence = read_json(Path('repro/reflexion-20261023/state-representation-presence-baseline.json'))
    from state_representation_model import predict
    models = {'presence': {tuple(v['labels']): v['joint'] for v in presence['one_attempt_forecasts']},
              'remaining_lifetime': {tuple(v['labels']): v['joint'] for v in primary['one_attempt_forecasts']}}
    blocks = []
    cases = sorted(source['cases'], key=lambda c: c['task_id'])
    for index, case in enumerate(cases):
        task = case['task_id']
        symbols = ['E','C'] if index % 2 == 0 else ['C','E']
        middle = [{'id': f'middle-{task}-{label}', 'task_id': task, 'condition': 'middle_'+label,
                   'labels': ['N',label,'N']} for label in symbols]
        blocks.append({'task_id': task, 'seed': 2101, 'panel': 'middle', 'attempts': 1,
                       'worker': index % 2, 'cells': middle})
        long = []
        for label in ['E','C','U']:
            pool = [r for r in analysis['rows'] if r['task_id'] == task and r['outcome'] == label]
            if not pool:
                continue
            selected = min(pool, key=lambda r: r['new_reflection_hashes'][0])
            native = {'cell_id': selected['cell_id'], 'result_sha256': selected['result_sha256'],
                'reflection_sha256': selected['new_reflection_hashes'][0], 'label': label}
            text = native_text(native)
            compressed = ['N', 'N' if label == 'U' else label, 'N']
            control = ({'source_run': SOURCE_RUN, 'cell_id': f'state-{task}-2101-full_generic'} if label == 'U'
                       else {'source_run': 'runs/reflexion-middle-context-20261023', 'cell_id': f'middle-{task}-{label}'})
            long.append({'id': f'native-{task}-{label}', 'task_id': task, 'condition': 'native_'+label,
                'labels': compressed, 'native_source': native, 'compressed_control': control,
                'memory_sha256': digest([source['entries']['N'], text, source['entries']['N']])})
        if index % 2:
            long.reverse()
        blocks.append({'task_id': task, 'seed': 2101, 'panel': 'native', 'attempts': 1,
                       'worker': index % 2, 'cells': long})
    predictions = []
    for block in blocks:
        for cell in block['cells']:
            key = tuple(cell['labels'])
            predictions.append({'cell_id': cell['id'], 'task_id': block['task_id'], 'panel': block['panel'],
                'initial_labels': cell['labels'], 'presence': models['presence'][key],
                'remaining_lifetime': models['remaining_lifetime'][key],
                'constant': predict(primary['all_fitted_models']['constant'], cell['labels']),
                'remaining_lifetime_middle_feature': 'unestimated ridge-zero extrapolation; diagnostic comparator'})
    dependencies = ['scripts/prepare_middle_context_transport.py', 'scripts/run_middle_context_transport.py',
        'scripts/analyze_middle_context_transport.py', 'scripts/state_representation_model.py',
        'scripts/run_state_representation.py', 'scripts/prepare_state_representation.py',
        'scripts/analyze_state_representation.py', 'scripts/deployment/serve_reflexion_profiles.py',
        'configs/reflexion_qwen3_8b_protocol.json', 'configs/qwen3_8b_non_thinking_wrapper.json',
        'repro/reflexion-20261023/state-representation-protocol.json',
        'repro/reflexion-20261023/state-representation-results.json',
        'repro/reflexion-20261023/state-representation-semantic-labels.json',
        'repro/reflexion-20261023/state-representation-model-freeze.json',
        'repro/reflexion-20261023/state-representation-presence-baseline.json',
        'docs/experiments/middle-context-design-20261023.md']
    protocol = {'schema_version': 1, 'scope': 'Prospective memory-condition transport within 12 known development tasks',
        'scientific_assumptions': ['A shared one-attempt mechanism mainly depends on visible target-error presence',
            'A target semantic label plus slot position retains enough information when replacing native long text'],
        'source_commit': 'c59dfa39b1a6781f5f560e74dc4d8535a8149314',
        'source_protocol_sha256': source['sha256'], 'source_analysis_sha256': analysis['sha256'],
        'source_primary_freeze_sha256': primary['sha256'], 'source_presence_freeze_sha256': presence['sha256'],
        'cases': cases, 'entries': source['entries'], 'entry_hashes': source['entry_hashes'],
        'blocks': blocks, 'predictions': predictions, 'ports': [8001,8002], 'gpu_indices': [0,1],
        'phases': ['middle','native'], 'expected_cells': 48, 'maximum_attempts': 48,
        'phase_cells': {'middle': 24, 'native': 24}, 'native_start_labels': {'E':12,'C':2,'U':10},
        'native_selection': 'Within each task and each unambiguous E/C/U source label, choose the minimum reflection hash from all prior failed cells; exclude A sources; no selection using new outcomes',
        'reuse': 'Native E/C controls are the new middle cells; native U controls are the prior full-generic cells at the same task/seed/trial, source identities and inputs audited; controls are not new independent tasks',
        'limits': {'additional_calls_including_replica_checks': 3000, 'seconds': 7200},
        'analysis': {'primary': 'Fixed presence-model joint five-class Brier versus fixed constant; task-balanced means in each phase; task-bootstrap differences with 20000 resamples, seed 2026102305',
            'decision': 'A strictly positive 95% bootstrap improvement interval supports advantage in these tested conditions; otherwise unresolved or negative; no parameter refit in this batch',
            'secondary': 'Success Brier, joint log loss, paired middle-versus-edge and native-versus-compressed success tables; report native E/U/C strata with C limited to two source cases',
            'unit': '12 known base tasks with nested conditions/source texts; condition transport, not independent-task generalization',
            'semantic': 'Full-text Codex E/C/U/A annotation by hash; keep A as an outcome; prior source labels remain unchanged',
            'content_limit': 'Full versus compressed changes all discarded text, including length and other advice; a difference does not isolate one specific advice type',
            'probability_limit': 'One paired seed per task; a single trajectory discordance is not proof of unequal stochastic kernels'},
        'protected_files': {p: file_hash(Path(p)) for p in dependencies},
        'source_hashes': provenance()['source_hashes'], 'holdout_model_calls': 0}
    assert len(predictions) == 48
    for block in blocks:
        for cell in block['cells']:
            if 'memory_sha256' in cell:
                assert cell['memory_sha256'] == digest(memory_for(protocol, cell))
    protocol['sha256'] = digest(protocol); write_json(OUTPUT, protocol)
    print({'sha256': protocol['sha256'], 'phases': protocol['phase_cells'], 'new_model_calls': 0})


if __name__ == '__main__':
    main()
