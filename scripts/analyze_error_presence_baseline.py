#!/usr/bin/env python3
"""Post-primary exploratory baseline: distinguish error presence from entry age."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from agent_reliability.io import digest, file_hash, read_json, write_json
from state_representation_model import fit, predict, scores


def binary(labels):
    return ['E'] if 'E' in labels else []


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); args = p.parse_args()
    analysis = read_json(args.run/'analysis.json'); primary = read_json(args.run/'model-freeze.json')
    rows = analysis['rows']; tasks = sorted({r['task_id'] for r in rows}); folds = []
    for task in tasks:
        train = [{**r, 'initial_labels': binary(r['initial_labels'])} for r in rows if r['task_id'] != task]
        fitted = fit(train, 'remaining_lifetime')
        values = [scores(predict(fitted, binary(r['initial_labels'])), r['outcome']) for r in rows if r['task_id'] == task]
        folds.append({'task_id': task, **{k: float(np.mean([v[k] for v in values])) for k in values[0]}})
    rng = np.random.default_rng(2026102304); sampled = rng.integers(0, len(tasks), size=(20000, len(tasks)))
    contrasts = []
    for candidate in ['constant', 'remaining_lifetime']:
        differences = np.asarray([next(v['brier'] for v in primary['folds'] if v['task_id'] == task and v['model'] == candidate)
            -next(v['brier'] for v in folds if v['task_id'] == task) for task in tasks])
        contrasts.append({'reference': candidate, 'positive_means_presence_better': True,
            'mean_brier_improvement': float(differences.mean()),
            'task_bootstrap_95_interval': np.quantile(differences[sampled].mean(axis=1), [.025,.975]).tolist()})
    fitted = fit([{**r, 'initial_labels': binary(r['initial_labels'])} for r in rows], 'remaining_lifetime')
    forecast = [{'labels': list(state), 'joint': predict(fitted, binary(list(state)))}
        for n in range(4) for state in itertools.product(['E','C','N'], repeat=n)]
    value = {'scope': 'Exploratory post-primary analysis, not a predeclared fourth candidate or independent validation',
        'question': 'Is the improvement over a constant mostly error presence rather than an identified entry-age effect?',
        'analysis_sha256': analysis['sha256'], 'primary_model_freeze_sha256': primary['sha256'],
        'representation': 'error presence only',
        'implementation': 'Map every error-present input to [E], all others to []; reuse the frozen ridge fit. Only intercept and r=3 coefficients are active, exactly a binary-presence model.',
        'folds': folds, 'scores': {k: float(np.mean([v[k] for v in folds])) for k in ['brier','log_loss','success_brier']},
        'contrasts': contrasts, 'fitted_binary_mapping': fitted, 'one_attempt_forecasts': forecast,
        'prospective_question': 'Test whether an error in the unobserved middle position behaves like other error-present states; this tests a compression assumption, not an estimated middle-position coefficient.',
        'driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value); write_json(args.run/'presence-baseline.json', value)
    print({k: value[k] for k in ['scores', 'contrasts', 'sha256']})


if __name__ == '__main__':
    main()
