#!/usr/bin/env python3
"""Inspect frozen success-model calibration on its original fitting data; never refit."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from agent_reliability.io import file_hash, read_json, digest
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from shared_recovery_model import component_kernel, PRIOR, softmax


def main():
    root = Path('repro/reflexion-20261025')
    data = checked(root/'shared-calibration-events.json')
    parameters = checked(root/'shared-calibration-parameters.json')
    assert parameters['events_sha256'] == data['sha256']
    model = parameters['comparison']['selected_model']
    rows = data['rows']
    assert all(r['success'] == (r['outcome'] == 'S') for r in rows)
    # Compare native event counts with deduplicated semantic text labels.
    from collections import Counter
    labels = checked(root/'shared-calibration-semantic-labels.json')
    label_map = {r['reflection_sha256']:r['category'] for r in labels['entries']}
    summary_path = Path('runs/reflexion-shared-calibration-20261025/summary.json')
    summary = checked(summary_path)
    raw_events = {(r['trajectory_id'],r['trial']):r for r in summary['attempts_rows']}
    uses = Counter()
    for row in rows:
        raw = raw_events[row['trajectory_id'],row['trial']]
        p = Path(raw['result_path']); result = read_json(p)
        assert file_hash(p) == raw['result_file_sha256'] == row['result_file_sha256']
        assert result['success'] == row['success'] == bool(read_json(p.parent/'checkpoint.json')['transitions'][-1]['won'])
        if not result['success']:
            h = digest(result['memory_after'][-1]);uses[h] += 1
            assert row['outcome'] == label_map[h]
    assert set(uses) == set(label_map)
    old_closeout = checked(root/'shared-calibration-closeout.json')
    correction = save(root/'shared-calibration-count-audit-20261026.json', {
        'scope':'Native-outcome and event-versus-unique-text count correction; frozen sources unchanged',
        'attempts':len(rows),'first_successes':sum(r['success'] for r in rows if r['trial']==1),
        'successes_by_two':sum(r['success'] for r in rows),'failed_write_events':sum(uses.values()),
        'unique_reflection_texts':len(uses),'unique_text_labels':dict(Counter(label_map.values())),
        'failure_event_labels':dict(Counter(r['outcome'] for r in rows if not r['success'])),
        'repeated_texts':[{'reflection_sha256':h,'event_uses':n,'category':label_map[h]} for h,n in uses.items() if n>1],
        'historical_closeout_successes_by_two':old_closeout['successes_by_two'],
        'raw_summary_file_sha256':file_hash(summary_path),'events_sha256':data['sha256'],
        'labels_sha256':labels['sha256'],'parameters_sha256':parameters['sha256'],
        'historical_closeout_file_sha256':file_hash(root/'shared-calibration-closeout.json'),
        'model_fitting_events_use_correct_counts':True,'parameter_refits':0,'new_model_calls':0,
        'source_sha256':file_hash(Path(__file__))})
    posterior = {}
    for task in sorted({r['task_id'] for r in rows}):
        log_weights = np.log(PRIOR)
        for row in rows:
            if row['task_id'] == task:
                p = component_kernel(model, row['state'])[:,0]
                log_weights += np.log(p if row['success'] else 1-p)
        posterior[task] = softmax(log_weights)
    groups = defaultdict(list)
    for row in rows:
        key = ('initial_C_N' if row['condition'] in ['correct_middle','neutral_full'] else 'other_initial') if row['trial'] == 1 else 'after_first_failure'
        p = component_kernel(model,row['state'])[:,0]
        groups[key].append((float(row['success']),float(PRIOR@p),float(posterior[row['task_id']]@p)))
    output = {}
    for key, values in groups.items():
        a = np.asarray(values)
        output[key] = {'attempts':len(values),'observed_successes':int(a[:,0].sum()),
            'prior_marginal_expected_successes':float(a[:,1].sum()),
            'posterior_fitted_expected_successes':float(a[:,2].sum()),
            'observed_rate':float(a[:,0].mean()),'prior_marginal_rate':float(a[:,1].mean()),
            'posterior_fitted_rate':float(a[:,2].mean())}
    a = save(root/'success-calibration-audit-20261026.json', {
        'scope':'Post-collection, calibration-data-only fitted-value audit; no coefficient estimation and no new predictor',
        'parameters_sha256':parameters['sha256'],'calibration_events_sha256':data['sha256'],
        'count_correction_sha256':correction['sha256'],
        'frozen_task_prior':PRIOR.tolist(),'mean_calibration_task_posterior':np.mean(list(posterior.values()),axis=0).tolist(),
        'groups':output,'observed_successes':sum(r['success'] for r in rows),
        'posterior_fitted_total_expected_successes':sum(v['posterior_fitted_expected_successes'] for v in output.values()),
        'interpretation':'The C/N initial residual already exists in the calibration data, including task-posterior fitted values. Inspect shared success-feature/regularization fit; do not attribute the entire error to prior mismatch or reflection generation.',
        'no_validation_predictions_rescored':True,'parameter_refits':0,'new_model_calls':0,
        'source_sha256':file_hash(Path(__file__))})
    print({'sha256':a['sha256'],'count_correction_sha256':correction['sha256'],'groups':output,'mean_task_posterior':a['mean_calibration_task_posterior']})


if __name__ == '__main__':
    main()
