#!/usr/bin/env python3
"""Diagnose already collected updates and compute fixed longer-budget forecasts."""
from collections import Counter
from pathlib import Path
import numpy as np
from agent_reliability.io import read_json, file_hash, digest
from prepare_shared_calibration import save, STATES
from prepare_shared_validation import PROTOCOL
from run_shared_validation import ROOT, verify_protocol
from shared_recovery_model import component_kernel, budget_probability, PRIOR, OUTCOMES
from continue_reflexion_pilot import ledger


def checked(path):
    value = read_json(path)
    assert digest({k: v for k, v in value.items() if k != 'sha256'}) == value['sha256']
    return value


def main():
    protocol = verify_protocol()
    pub = PROTOCOL.parent
    manifest = checked(pub / 'update-diagnostic-manifest.json')
    labels = checked(pub / 'update-diagnostic-semantic-labels.json')
    frozen = checked(pub / 'shared-validation-results.json')
    assert labels['manifest_sha256'] == manifest['sha256']
    private = Path('runs/reflexion-update-diagnostic-20261025')
    assert file_hash(private / 'reading-view.json') == labels['private_reading_file_sha256']
    lm = {r['reflection_sha256']: r['category'] for r in labels['entries']}
    assert set(lm) == {r['reflection_sha256'] for r in manifest['selected']}
    outcomes = {r['trajectory_id']: r for r in frozen['observed']}
    model = protocol['models']['selected']
    rows = []
    for item in manifest['selected']:
        kind = lm[item['reflection_sha256']]
        state = [{'type': k, 'native': False} for k in STATES[item['condition']]]
        kernel = component_kernel(model, state)
        outcome_index = OUTCOMES.index(kind)
        posterior = PRIOR * kernel[:, outcome_index]
        posterior /= posterior.sum()
        updated = (state + [{'type': kind, 'native': True}])[-3:]
        p2 = float(posterior @ component_kernel(model, updated)[:, 0])
        write = PRIOR @ kernel[:, 1:]
        write /= write.sum()
        rows.append({**item, 'category': kind, 'state_before_second': updated,
                     'second_success': outcomes[item['trajectory_id']]['success_curve'][1],
                     'conditional_second_probability': p2,
                     'frozen_first_failure_write_probabilities': dict(zip(OUTCOMES[1:], map(float, write)))})
    tables = {}
    for condition in STATES:
        selected = [r for r in rows if r['condition'] == condition]
        if not selected:
            continue
        tables[condition] = {
            'first_failure_texts': len(selected), 'category_counts': dict(Counter(r['category'] for r in selected)),
            'second_successes': sum(r['second_success'] for r in selected),
            'conditional_expected_second_successes': sum(r['conditional_second_probability'] for r in selected),
            'successes_by_new_category': {k: {'n': sum(r['category'] == k for r in selected),
                'successes': sum(r['category'] == k and r['second_success'] for r in selected)} for k in OUTCOMES[1:]},
            'frozen_write_probabilities': selected[0]['frozen_first_failure_write_probabilities'],
            'selection_scope': 'All first failures in this condition' if condition in ['correct_latest', 'delete_latest'] else 'FloorPlan26 first failures only',
        }
    scene = []
    for condition in STATES:
        rr = [r for r in frozen['observed'] if r['scene_id'] == 'FloorPlan26' and r['condition'] == condition]
        a = protocol['forecasts']['constant_persistent'][condition][1]
        b = protocol['forecasts']['selected'][condition][1]
        scene.append({'condition': condition, 'n': len(rr), 'two_successes': sum(r['success_curve'][1] for r in rr),
                      'two_budget_brier_gain': float(np.mean([(a-r['success_curve'][1])**2-(b-r['success_curve'][1])**2 for r in rr]))})
    cases = read_json(private / 'successful-deletion-cases.json')
    case_evidence = []
    row_map = {r['trajectory_id']: r for r in rows}
    for case in cases:
        transitions = case['transitions']
        heat_index = next(i for i, t in enumerate(transitions) if t['action'].startswith('heat ') and t['observation'].startswith('You heat '))
        heating = transitions[heat_index]
        target = heating['action'][5:].split(' with ')[0]
        prior = transitions[:heat_index]
        put_target = [t for t in prior if t['action'].startswith('put ' + target + ' ') and 'microwave' in t['action']]
        assert not put_target and transitions[-1]['won']
        unsuccessful_other_take = [t for t in prior if t['action'].startswith('take ') and
                                  not t['action'].startswith('take ' + target + ' ') and t['observation'] == 'Nothing happens.']
        r = row_map[case['trajectory_id']]
        assert r['state_before_second'][0] == {'type': 'E', 'native': False}
        case_evidence.append({k: v for k, v in case.items() if k != 'transitions'} | {
            'new_reflection_category': r['category'], 'reflection_sha256': r['reflection_sha256'],
            'old_E_visible': True, 'target_put_in_microwave_before_heat': False,
            'successful_heating_action': heating['action'], 'successful_heating_observation': heating['observation'],
            'unsuccessful_other_take_count': len(unsuccessful_other_take),
        })
    full, unchanged = {}, {}
    for condition, initial in STATES.items():
        state = [{'type': k, 'native': False} for k in initial]
        p = component_kernel(model, state)[:, 0]
        full[condition] = [budget_probability(model, state, b) for b in range(1, 5)]
        unchanged[condition] = [float(PRIOR @ (1 - (1 - p) ** b)) for b in range(1, 5)]
        assert np.allclose(full[condition][:2], protocol['forecasts']['selected'][condition], atol=1e-12)
    all_runs = ledger()
    assert sum(v['dispatched'] for v in all_runs.values()) == sum(v['responses'] for v in all_runs.values()) == 93186
    report = save(pub / 'update-diagnostic-results.json', {
        'scope': 'Post-collection targeted diagnosis; no causal claim from the success-conditioned cases',
        'manifest_sha256': manifest['sha256'], 'labels_sha256': labels['sha256'],
        'frozen_results_sha256': frozen['sha256'], 'parameters_sha256': frozen['parameters_sha256'],
        'source_sha256': file_hash(Path(__file__)), 'rows': rows, 'per_condition': tables,
        'FloorPlan26_per_condition': scene, 'successful_deletion_cases': case_evidence,
        'case_interpretation': 'Every selected successful deletion trace retains old E, omits putting its target in the microwave, then executes successful heat; text presence is not deterministic behavioral adoption',
        'semantic_counts': dict(Counter(lm.values())), 'human_second_review': False,
        'new_model_calls': 0, 'parameter_refits': 0,
    })
    forecasts = save(pub / 'four-attempt-planning-forecasts.json', {
        'scope': 'Exact frozen-model planning forecasts, not sampled Agent results',
        'parameters_sha256': frozen['parameters_sha256'], 'source_sha256': file_hash(Path(__file__)),
        'budgets': [1, 2, 3, 4], 'full_update': full, 'unchanged_memory': unchanged,
        'new_model_calls': 0, 'parameter_refits': 0,
    })
    print({'tables': tables, 'FloorPlan26': scene, 'successful_deletion_cases': case_evidence,
           'full_update': full, 'unchanged_memory': unchanged,
           'diagnostic_sha256': report['sha256'], 'forecasts_sha256': forecasts['sha256']})


if __name__ == '__main__':
    main()
