#!/usr/bin/env python3
"""Calibration-only diagnosis; preserve all frozen models and validation scores."""
from collections import defaultdict, Counter
from pathlib import Path
import time
import numpy as np
from agent_reliability.io import digest, file_hash
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
import shared_recovery_model as original

PUBLIC = Path('repro/reflexion-20261025')
LAMBDAS = [1., .1, .01, 0.]
DEADLINE_SECONDS = 900


def objective(beta, x, y, groups, scale, ridge):
    logits = x@beta[:, None]+scale*original.OFFSETS[None, :]
    p = 1/(1+np.exp(-logits))
    logs = y[:, None]*(-np.logaddexp(0, -logits))+(1-y[:, None])*(-np.logaddexp(0, logits))
    loss = 0.; grad = np.zeros_like(beta)
    for idx in groups:
        joint = np.log(original.PRIOR)+logs[idx].sum(axis=0)
        weights = original.softmax(joint)
        m = joint.max(); loss -= m+np.log(np.exp(joint-m).sum())
        grad += x[idx].T@((p[idx]-y[idx, None])@weights)
    penalty = beta.copy(); penalty[0] = 0.
    loss += .5*ridge*(penalty@penalty)+.5*np.logaddexp(0, -beta[0])+.5*np.logaddexp(0, beta[0])
    grad += ridge*penalty; grad[0] += 1/(1+np.exp(-beta[0]))-.5
    return float(loss), grad


def arrays(rows, family):
    x = np.asarray([original.features(r['state'], family) for r in rows])
    y = np.asarray([r['success'] for r in rows], float)
    tasks = sorted({r['task_id'] for r in rows})
    groups = [np.asarray([i for i, r in enumerate(rows) if r['task_id'] == t]) for t in tasks]
    return x, y, groups, tasks


def fitted(rows, base, ridge):
    x, y, groups, _ = arrays(rows, base['success_family'])
    function = lambda b: objective(b, x, y, groups, base['scale'], ridge)
    beta, audit = original.bfgs(function, np.zeros(x.shape[1]))
    return {**base, 'success_coefficients': beta.tolist(), 'diagnostic_ridge': ridge}, {
        **audit, 'objective': function(beta)[0], 'feature_rank': int(np.linalg.matrix_rank(x))}


def training_groups(rows, model):
    x, y, groups, tasks = arrays(rows, model['success_family'])
    b = np.asarray(model['success_coefficients'])
    p = 1/(1+np.exp(-(x@b[:, None]+model['scale']*original.OFFSETS[None, :])))
    post = {}
    for task, idx in zip(tasks, groups):
        logs = y[idx, None]*np.log(p[idx])+(1-y[idx, None])*np.log1p(-p[idx])
        post[task] = original.softmax(np.log(original.PRIOR)+logs.sum(axis=0))
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        key = ('initial_C_N' if row['condition'] in ['correct_middle', 'neutral_full'] else 'other_initial') if row['trial'] == 1 else 'after_first_failure'
        buckets[key].append((y[i], float(original.PRIOR@p[i]), float(post[row['task_id']]@p[i])))
    return {key: {'attempts': len(v), 'observed_successes': int(sum(a[0] for a in v)),
                  'prior_mean': float(np.mean([a[1] for a in v])),
                  'posterior_fitted_mean': float(np.mean([a[2] for a in v]))}
            for key, v in buckets.items()}


def numerical_audit(rows, base):
    x, y, groups, _ = arrays(rows, base['success_family'])
    beta = np.asarray(base['success_coefficients'])
    fun = lambda b: objective(b, x, y, groups, base['scale'], 1.)
    loss, grad = fun(beta); old_loss, old_grad = original.success_objective(beta, x, y, groups, base['scale'])
    assert abs(loss-old_loss)<1e-10 and np.max(np.abs(grad-old_grad))<1e-10
    step = 1e-5
    numeric = []; columns = []
    for j in range(len(beta)):
        d = np.eye(len(beta))[j]*step
        numeric.append((fun(beta+d)[0]-fun(beta-d)[0])/(2*step))
        columns.append((fun(beta+d)[1]-fun(beta-d)[1])/(2*step))
    error = float(np.max(np.abs(grad-np.asarray(numeric))))
    assert error<1e-6
    hessian = np.asarray(columns).T; hessian = (hessian+hessian.T)/2
    restarts = []
    for initial in [np.zeros(len(beta)), np.full(len(beta), .5), np.full(len(beta), -.5)]:
        b, audit = original.bfgs(fun, initial)
        restarts.append({'initial': initial.tolist(), 'loss': fun(b)[0],
                         'max_coefficient_difference': float(np.max(np.abs(b-beta))), **audit})
    return {'objective': loss, 'max_gradient': float(np.max(np.abs(grad))),
            'max_finite_difference_error': error, 'hessian_eigenvalues': np.linalg.eigvalsh(hessian).tolist(),
            'score_data_gradient': (grad-np.r_[1/(1+np.exp(-beta[0]))-.5, beta[1:]]).tolist(),
            'restarts': restarts, 'scope': 'Stationarity, finite differences, local curvature and three starts; no global optimality guarantee'}


def separation_audit(rows, base):
    x, y, _, _ = arrays(rows, base['success_family'])
    names = ['intercept','length_over3','native_count_over3','A_count_over3','has_E_or_B','latest_E_or_B_position_over3']
    directions = []
    for j in range(1, x.shape[1]):
        positive = x[:, j]>0
        if not positive.any() or np.any(x[:, j]<0): continue
        successes = int(y[positive].sum()); n = int(positive.sum())
        if successes in [0,n]:
            directions.append({'feature_index': j, 'feature': names[j], 'positive_rows': n,
                'successes_in_positive_rows': successes, 'coefficient_direction': '-infinity' if successes==0 else '+infinity',
                'scenes': len({r['scene_id'] for i,r in enumerate(rows) if positive[i]}),
                'tasks': len({r['task_id'] for i,r in enumerate(rows) if positive[i]})})
    return {'one_sided_nonnegative_feature_directions': directions,
            'zero_ridge_has_no_finite_interior_optimum_along_detected_directions': bool(directions),
            'interpretation': 'A small optimizer gradient need not mean a finite maximum-likelihood solution when a coefficient can diverge.'}


def scene_scores(rows, model):
    trajectories = []; initial = []; clean = []; later = []
    for identity in sorted({r['trajectory_id'] for r in rows}):
        seq = sorted([r for r in rows if r['trajectory_id'] == identity], key=lambda r:r['trial'])
        predictions = original.predict_sequence(model, seq)
        losses = []
        for row, pred in zip(seq, predictions):
            p = pred['S']; loss = (p-float(row['success']))**2; losses.append(loss)
            item = [loss, p-float(row['success'])]
            if row['trial'] == 1:
                initial.append(item)
                if row['condition'] in ['correct_middle', 'neutral_full']:
                    clean.append(item)
            else:
                later.append(item)
        trajectories.append(float(np.mean(losses)))
    return {'trajectory_mean_success_brier': float(np.mean(trajectories)),
            **{key: {'attempts': len(v), 'brier': float(np.mean([a[0] for a in v])),
                      'mean_prediction_minus_observation': float(np.mean([a[1] for a in v]))}
               for key, v in [('first_attempt', initial), ('initial_C_N', clean), ('after_first_failure', later)]}}


def write_calibration(rows, base):
    groups = defaultdict(list)
    for row in rows:
        if row['success']: continue
        key = ('initial_C_N' if row['condition'] in ['correct_middle', 'neutral_full'] else 'other_initial') if row['trial'] == 1 else 'after_first_failure'
        groups[key].append(row)
    result = {}
    for key, group in groups.items():
        probabilities = []
        for row in group:
            k = original.component_kernel(base, row['state'])[0]
            probabilities.append(k[1:]/(1-k[0]))
        result[key] = {'events': len(group), 'observed_counts': dict(Counter(r['outcome'] for r in group)),
                       'fitted_expected_counts': dict(zip(original.OUTCOMES[1:], np.asarray(probabilities).sum(axis=0).tolist()))}
    return result


def one_update_bridge(rows, base):
    paired = {(r['trajectory_id'],r['trial']):r for r in rows}
    groups = defaultdict(list)
    for row in rows:
        if row['trial']!=1 or row['success']: continue
        initial = row['state']; next_row = paired[row['trajectory_id'],2]
        kernel = original.component_kernel(base, initial)
        weights = original.PRIOR*(1-kernel[:,0]); weights /= weights.sum()
        g = kernel[0,1:]/(1-kernel[0,0])
        integrated = sum(g[j]*float(weights@original.component_kernel(base,(initial+[{'type':kind,'native':True}])[-3:])[:,0])
                         for j,kind in enumerate(original.OUTCOMES[1:]))
        actual_text = float(weights@original.component_kernel(base,next_row['state'])[:,0])
        key = 'initial_C_N' if row['condition'] in ['correct_middle','neutral_full'] else 'other_initial'
        groups[key].append([float(next_row['success']),integrated,actual_text])
    return {key:{'first_failures':len(v), 'observed_next_successes':int(sum(a[0] for a in v)),
                 'model_integrated_next_expected_successes':float(sum(a[1] for a in v)),
                 'actual_next_state_conditioned_expected_successes':float(sum(a[2] for a in v)),
                 'scope':'Calibration-only descriptive bridge using frozen coefficients and failure-updated task weights. Actual next-state labels are post-failure information, not a prospective baseline.'}
            for key,v in groups.items()}


def state_support(rows, base):
    train = np.asarray([original.features(r['state'], base['success_family']) for r in rows])
    counts = Counter(sum(s['native'] for s in r['state']) for r in rows)
    all_cases = {}
    for labels in [('N','C','N'), ('N','N','N')]:
        # Distribution retains original surviving mass and persistent task class.
        initial = tuple((k, False) for k in labels)
        mass = {(initial, z): float(w) for z, w in enumerate(original.PRIOR)}
        curve = []
        for trial in range(1, 5):
            total = sum(mass.values()); outside = 0.; native = defaultdict(float); next_mass = defaultdict(float)
            for (state, z), weight in mass.items():
                s = [{'type': k, 'native': n} for k, n in state]
                x = np.asarray(original.features(s, base['success_family']))
                if np.any(x<train.min(axis=0)-1e-12) or np.any(x>train.max(axis=0)+1e-12): outside += weight
                native[sum(n for k,n in state)] += weight
                kernel = original.component_kernel(base, s)[z]
                for j, kind in enumerate(original.OUTCOMES[1:]):
                    next_state = (state+((kind, True),))[-3:]
                    next_mass[(next_state, z)] += weight*kernel[j+1]
            curve.append({'trial': trial, 'survival_mass': total,
                          'outside_training_coordinate_range_given_survival': outside/total,
                          'native_count_given_survival': {str(k): v/total for k,v in native.items()}})
            mass = next_mass
        all_cases[''.join(labels)] = curve
    return {'observed_native_counts': dict(counts), 'feature_min': train.min(axis=0).tolist(),
            'feature_max': train.max(axis=0).tolist(), 'frozen_model_prospective_survivor_support': all_cases,
            'scope': 'Model-implied support check; no new reflection labels or observed successor outcomes used'}


def main():
    started = time.monotonic()
    data = checked(PUBLIC/'shared-calibration-events.json')
    params = checked(PUBLIC/'shared-calibration-parameters.json')
    assert params['events_sha256'] == data['sha256']
    assert params['model_source_sha256'] == file_hash(Path('scripts/shared_recovery_model.py'))
    rows = data['rows']; base = params['comparison']['selected_model']
    assert len(rows)==463 and sum(r['success'] for r in rows)==58
    before = {p: file_hash(PUBLIC/p) for p in ['shared-calibration-events.json', 'shared-calibration-parameters.json',
               'shared-validation-results.json', 'four-attempt-validation-results.json',
               'four-attempt-validation-evaluation.json'] if (PUBLIC/p).exists()}
    plan = save(PUBLIC/'success-regularization-diagnostic-plan-20261026.json', {
        'scope': 'Development diagnosis after validation feedback; original36 calibration tasks only',
        'events_sha256': data['sha256'], 'parameters_sha256': params['sha256'],
        'diagnostic_ridge_strengths': LAMBDAS, 'success_family': base['success_family'], 'latent_scale': base['scale'],
        'unchanged': ['feature definitions and normalization', 'latent prior and scale', 'intercept half-count', 'conditional write kernel'],
        'questions': ['Optimization error versus regularization-induced underfit', 'In-sample versus held-scene response to reduced ridge', 'Successor-state support and conditional write residuals'],
        'evaluation': 'Leave one of18 calibration scenes out; refit successes only, fixed write kernel. Trajectory-mean sequential success Brier and first-attempt/C-N/later diagnostics. Report every ridge strength.',
        'selection': 'No automatic promotion and no new validation scores; diagnose and design the next bounded step',
        'max_success_fits': 79, 'cpu_wall_seconds': DEADLINE_SECONDS, 'new_model_calls': 0,
        'frozen_parameters_overwritten': False, 'source_sha256': file_hash(Path(__file__))})
    audit = numerical_audit(rows, base)
    full_fits = []; folds = []; failures = []
    scenes = sorted({r['scene_id'] for r in rows})
    assert len(scenes)==18
    for ridge in LAMBDAS:
        try:
            model, fit_audit = fitted(rows, base, ridge)
            full_fits.append({'ridge': ridge, 'coefficients': model['success_coefficients'], 'optimizer': fit_audit,
                              'training_groups': training_groups(rows, model)})
        except (RuntimeError, FloatingPointError) as e:
            failures.append({'ridge': ridge, 'scope': 'full calibration fit', 'error': str(e)})
        for scene in scenes:
            if time.monotonic()-started>DEADLINE_SECONDS: raise RuntimeError('CPU diagnostic wall allowance exhausted')
            train = [r for r in rows if r['scene_id'] != scene]; test = [r for r in rows if r['scene_id'] == scene]
            try:
                model, fit_audit = fitted(train, base, ridge)
                folds.append({'ridge': ridge, 'scene_id': scene, 'optimizer': fit_audit,
                              'coefficients': model['success_coefficients'], **scene_scores(test, model)})
            except (RuntimeError, FloatingPointError) as e:
                failures.append({'ridge': ridge, 'scene_id': scene, 'error': str(e)})
        print({'ridge': ridge, 'folds_complete': sum(r['ridge']==ridge for r in folds), 'failures': len(failures)}, flush=True)
    summaries = []
    references = {r['scene_id']: r for r in folds if r['ridge']==1.}
    for ridge in LAMBDAS:
        group = [r for r in folds if r['ridge']==ridge]
        if len(group)!=18:
            summaries.append({'ridge': ridge, 'complete': False, 'completed_scenes': len(group)}); continue
        delta = [references[r['scene_id']]['trajectory_mean_success_brier']-r['trajectory_mean_success_brier'] for r in group]
        summaries.append({'ridge': ridge, 'complete': True,
            'trajectory_mean_success_brier': float(np.mean([r['trajectory_mean_success_brier'] for r in group])),
            'baseline_minus_candidate_brier': float(np.mean(delta)), 'scenes_with_lower_loss': sum(d>1e-12 for d in delta),
            'paired_scene_values': delta,
            **{key: {metric: float(np.mean([r[key][metric] for r in group])) for metric in ['brier', 'mean_prediction_minus_observation']}
               for key in ['first_attempt','initial_C_N','after_first_failure']}})
    for name,h in before.items(): assert file_hash(PUBLIC/name)==h, name
    result = save(PUBLIC/'success-regularization-diagnostic-20261026.json', {
        'scope': 'Post-validation development diagnosis; no independent validation claim',
        'plan_sha256': plan['sha256'], 'source_sha256': file_hash(Path(__file__)),
        'parameters_sha256': params['sha256'], 'events_sha256': data['sha256'],
        'optimization_audit': audit, 'separation_audit': separation_audit(rows, base),
        'calibration_fits': full_fits, 'held_scene_summaries': summaries,
        'folds': folds, 'fit_failures': failures, 'write_kernel_training_calibration': write_calibration(rows, base),
        'one_update_success_bridge': one_update_bridge(rows, base),
        'state_support': state_support(rows, base), 'frozen_file_hashes_unchanged': before,
        'new_model_calls': 0, 'original_parameter_refits': 0, 'development_success_fits': 4+18*4+3,
        'elapsed_seconds': time.monotonic()-started})
    print({'sha256': result['sha256'], 'summaries': summaries, 'full_training_groups': full_fits, 'optimization': audit}, flush=True)


if __name__ == '__main__':
    main()
