#!/usr/bin/env python3
"""Freeze targeted calibration cells after public-observation-only cue checks."""
from pathlib import Path
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reference import official_environment
from run_reflexion_state_prediction import source_entries

PROTOCOL = Path('repro/reflexion-20261023/directed-calibration-protocol.json')
STATES = {
    'generation': {'error': ['E'], 'correct': ['C'], 'generic': ['N'], 'empty': []},
    'recovery': {'retain': ['H', 'E', 'N'], 'correct': ['H', 'C', 'N'],
                 'replace': ['H', 'N', 'N'], 'delete': ['H', 'N'], 'without_cue': ['N', 'N', 'N']},
    'age': {'oldest': ['E', 'N', 'N'], 'newest': ['N', 'N', 'E']}}


def main():
    if PROTOCOL.exists():
        raise FileExistsError('Never overwrite a frozen protocol')
    source = read_json(Path('repro/reflexion-20261023/state-prediction-protocol.json'))
    placements = [('cup 1', 'countertop 2'), ('egg 2', 'countertop 3'),
                  ('mug 2', 'countertop 1'), ('egg 1', 'countertop 1')]
    cases = []
    for case, (obj, place) in zip(source['cases'], placements):
        parent, entries = source_entries(case)
        checkpoint = read_json(Path(case['parent_run'])/'tasks'/case['task_id']/'trial-00/checkpoint.json')
        evidence = [(i, t) for i, t in enumerate(checkpoint['transitions'])
                    if t['action'] == 'go to '+place and obj in t['observation']]
        if not evidence:
            raise ValueError('Missing public location observation')
        probes = []
        for seed in [1701, 1702, 1703]:
            env = official_environment(Path('data/alfworld'), case['task'], seed, Path('third_party/react'))
            try:
                obs, _ = env.reset()
                if digest('\n'.join(obs[0].split('\n\n')[1:])) != parent['initial_observation_sha256']:
                    raise ValueError('Seed changed task reset')
                observed, _, _, _ = env.step(['go to '+place])
                if obj not in observed[0]:
                    raise ValueError('Location cue not supported after public reset')
                probes.append({'seed': seed, 'action': 'go to '+place, 'observation': observed[0]})
            finally:
                env.close()
        cue = f'Plan: In the previous attempt, I found {obj} on {place}.'
        entries.update(N='Plan: I should continue working on the household task.', H=cue)
        cases.append({**case, 'location_cue': cue, 'cue_public_evidence': {
            'source_transition_index': evidence[0][0], 'source_transition_sha256': digest(evidence[0][1]),
            'source_observation': evidence[0][1]['observation'], 'reset_probes': probes},
            'entry_hashes': {k: digest(v) for k, v in entries.items()}})
    blocks = []
    # Each task/seed/panel block stays on one replica; rotate block assignment.
    for repetition, seed in enumerate([1701, 1702, 1703]):
        for panel in STATES:
            if panel != 'generation' and repetition == 2:
                continue
            for case_index, case in enumerate(cases):
                conditions = list(STATES[panel])
                rotation = (case_index+repetition) % len(conditions)
                conditions = conditions[rotation:]+conditions[:rotation]
                block = {'task_id': case['task_id'], 'seed': seed, 'panel': panel,
                         'attempts': 1 if panel == 'generation' else 2, 'worker': len(blocks) % 3, 'cells': []}
                for name in conditions:
                    cell_id = f"{panel}-{case['task_id']}-{seed}-{name}"
                    block['cells'].append({'id': cell_id, 'condition': name, 'labels': STATES[panel][name],
                                          'task_id': case['task_id']})
                blocks.append(block)
    protected = ['scripts/run_directed_calibration.py', 'scripts/prepare_directed_calibration.py',
        'scripts/run_reflexion_state_prediction.py', 'scripts/run_reflexion_precondition_forks.py',
        'scripts/summarize_reflexion.py', 'scripts/deployment/serve_reflexion_profiles.py',
        'configs/reflexion_qwen3_8b_protocol.json', 'configs/qwen3_8b_non_thinking_wrapper.json',
        'repro/reflexion-20261023/state-prediction-protocol.json']
    protocol = {'scope': 'First targeted mechanism calibration on four known development tasks; not independent-task validation',
        'source_prediction_protocol_sha256': source['sha256'], 'cases': cases, 'blocks': blocks,
        'ports': [8001, 8002, 8003], 'seeds': [1701, 1702, 1703], 'visible_memory_entries': 3,
        'limits': {'additional_calls_including_replica_checks': 6000, 'seconds': 10800},
        'states': STATES, 'expected_cells': 104, 'maximum_attempts': 160,
        'questions': ['Generation versus restatement conditional on failure and starting context',
            'Success and reflection transitions under correction, replacement and deletion',
            'Dependence on error entry age and a constructed public location cue'],
        'terminology': {'E': 'native entry containing the reviewed target error',
            'C': 'same entry with only the reviewed target proposition corrected; other advice retained',
            'N': 'real generic reflection entry without the target proposition or specific location cue; effect unknown',
            'H': 'constructed entry containing a previously observed location fact; utility to be measured'},
        'analysis': {'unit': 'task x seed x starting condition; attempts nested; four independent base tasks only',
            'generation': 'new reflection contains target error after visible error-absent start and failed attempt',
            'restatement': 'new reflection contains target error after visible error-present start and failed attempt',
            'labels': 'manual semantic E/C/U/A (error/correct/unmentioned/ambiguous), linked by content hash',
            'missing': 'success has no critic update; ambiguous labels and incomplete attempts are not negative events',
            'intervals': 'Wilson intervals describe within-task seed variability; no population interval from pooled dependent cells',
            'contrasts': 'paired task/seed first-attempt success and two-attempt success; no universal lambda/rho fit',
            'gate': 'Assess task, context and position dependence before fitting a shared transition mechanism'},
        'source_hashes': provenance()['source_hashes'],
        'protected_files': {p: file_hash(Path(p)) for p in protected}}
    assert sum(len(b['cells']) for b in blocks) == protocol['expected_cells']
    assert sum(len(b['cells'])*b['attempts'] for b in blocks) == protocol['maximum_attempts']
    protocol['sha256'] = digest(protocol)
    write_json(PROTOCOL, protocol)
    print({'protocol_sha256': protocol['sha256'], 'cells': 104, 'maximum_attempts': 160, 'model_calls': 0})


if __name__ == '__main__':
    main()
