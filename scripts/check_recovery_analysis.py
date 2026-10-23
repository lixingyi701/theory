#!/usr/bin/env python3
"""Numerical algebra checks of a stated submodel, never Agent measurements."""
import argparse
from pathlib import Path
from agent_reliability.io import digest, file_hash, read_json, write_json


def dynamic(p, q, regeneration, budget):
    value0 = value1 = 0.0
    for _ in range(budget):
        value0, value1 = (p+(1-p)*((1-regeneration)*value0+regeneration*value1),
                          q+(1-q)*value1)
    return value0, value1


def closed(p, q, regeneration, budget):
    a = (1-p)*(1-regeneration); d = 1-q
    delta = p-q
    gain = delta/(delta+(1-p)*regeneration)*(d**budget-a**budget)
    return 1-d**budget+gain, 1-d**budget


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', required=True, type=Path)
    a = parser.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ['analytic-checks.json', 'evidence-mapping.json']:
        if (a.output_dir/name).exists():
            raise FileExistsError('Keep the earlier numerical verification: '+name)
    horizons = [1, 2, 3, 5, 10, 20]; rates = [0.0, 0.1, 0.5, 0.9, 1.0]
    errors = []; monotonic_checks = 0
    for p in [.2, .5, .8, 1.0]:
        for q in [0.0, .05, .15]:
            for budget in horizons:
                gains = []
                for rate in rates:
                    expected = closed(p, q, rate, budget); actual = dynamic(p, q, rate, budget)
                    error = max(abs(x-y) for x, y in zip(expected, actual)); errors.append(error)
                    assert error < 1e-12
                    gains.append(actual[0]-actual[1])
                assert all(x+1e-12 >= y for x, y in zip(gains, gains[1:]))
                monotonic_checks += 1
    examples = []
    for q in [0.0, .02]:
        rows = [{'budget': budget, 'local_correction': dynamic(.6, q, .9, budget)[0],
                 'discard': dynamic(.4, q, .1, budget)[0]} for budget in [1, 2, 3, 10]]
        assert rows[0]['local_correction'] > rows[0]['discard']
        assert rows[2]['local_correction'] < rows[2]['discard']
        examples.append({'p_correction': .6, 'lambda_correction': .9, 'p_discard': .4,
                         'lambda_discard': .1, 'q': q, 'rows': rows})
    report = {'scope': 'Algebraic numerical verification of an assumed analytical submodel; no LLM calls or Monte Carlo',
        'parameter_source': 'Explicitly constructed grid, not fitted to ALFWorld',
        'closed_form_dp_comparisons': len(errors), 'max_absolute_difference': max(errors),
        'regeneration_monotonicity_sequences_checked': monotonic_checks,
        'constructed_rank_reversals': examples, 'driver_sha256': file_hash(Path(__file__))}
    report['sha256'] = digest(report); write_json(a.output_dir/'analytic-checks.json', report)

    summary_path = Path('repro/reflexion-20261023/state-calibration-summary.json')
    review_path = Path('repro/reflexion-20261023/state-calibration-reflection-review.json')
    summary = read_json(summary_path); review = read_json(review_path)
    for value in [summary, review]:
        assert value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'})
    measured = {(r['task_id'], r['state']): r for r in summary['rows']}
    groups = {'target_error_visible': [], 'target_error_absent': []}
    for row in review['rows']:
        source = measured[row['task_id'], row['state']]; assert not source['success']
        group = 'target_error_visible' if 'E' in source['labels'] else 'target_error_absent'
        groups[group].append({'task_id': row['task_id'], 'state': row['state'], 'new_label': row['label'],
                              'new_reflection_sha256': row['new_reflection_sha256']})
    mapping = {'scope': 'Reanalysis of existing failed calibration cells, not new Agent execution or a universal rate estimate',
        'summary_sha256': summary['sha256'], 'review_sha256': review['sha256'],
        'driver_sha256': file_hash(Path(__file__)), 'groups': {}}
    for group, rows in groups.items():
        mapping['groups'][group] = {'failed_cells': len(rows), 'base_tasks': len({r['task_id'] for r in rows}),
            'new_labels': {label: sum(r['new_label'] == label for r in rows) for label in ['E', 'C', 'U', 'A']}, 'rows': rows}
    mapping['sha256'] = digest(mapping); write_json(a.output_dir/'evidence-mapping.json', mapping)
    print({'algebra_comparisons': len(errors), 'max_absolute_difference': max(errors),
           'rank_reversal_examples': examples,
           'evidence_groups': {k: {n: v for n, v in group.items() if n != 'rows'} for k, group in mapping['groups'].items()}})


if __name__ == '__main__':
    main()
