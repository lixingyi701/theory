#!/usr/bin/env python3
"""Predeclared joint success/critic models; NumPy only, task-grouped scoring."""
import itertools
import numpy as np

CLASSES = ['S', 'E', 'C', 'U', 'A']
MODELS = ['constant', 'remaining_lifetime', 'content_occupancy_position']


def features(labels, model):
    if len(labels) > 3 or any(v not in {'E', 'C', 'N'} for v in labels):
        raise ValueError('Model supports constructed E/C/N entries and at most three slots')
    if model == 'constant':
        return [1.]
    remaining = 3-len(labels)+max(i for i, v in enumerate(labels) if v == 'E')+1 if 'E' in labels else 0
    if model == 'remaining_lifetime':
        return [1.]+[float(remaining == i) for i in [1, 2, 3]]
    if model != 'content_occupancy_position':
        raise ValueError('Unknown candidate')
    aligned = ['_']*(3-len(labels))+labels
    return [1., len(labels)/3.]+[float(v == symbol) for symbol in ['E', 'C'] for v in aligned]


def softmax(z):
    z = z-z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(axis=-1, keepdims=True)


def fit(rows, model):
    x = np.asarray([features(r['initial_labels'], model) for r in rows])
    y = np.eye(5)[[CLASSES.index(r['outcome']) for r in rows]]
    # Reference category A has coefficient zero, removing the softmax gauge.
    d = x.shape[1]; b = np.zeros((d, 4)); penalty = np.ones_like(b); penalty[0] = 0
    pseudo_x = np.zeros((5, d)); pseudo_x[:, 0] = 1
    xx = np.concatenate([x, pseudo_x]); yy = np.concatenate([y, np.eye(5)])
    weights = np.r_[np.ones(len(rows)), np.full(5, .5)]

    def evaluate(value, derivatives=False):
        logits = np.c_[xx@value, np.zeros(len(xx))]
        probabilities = softmax(logits)
        loss = float(-np.sum(weights[:, None]*yy*np.log(np.clip(probabilities, 1e-300, 1)))+.5*np.sum(penalty*value*value))
        if not derivatives:
            return loss
        gradient = xx.T@(weights[:, None]*(probabilities[:, :4]-yy[:, :4]))+penalty*value
        pp = probabilities[:, :4]
        covariance = np.asarray([np.diag(p)-np.outer(p, p) for p in pp])
        hessian = np.einsum('n,ni,nj,nkl->ikjl', weights, xx, xx, covariance).reshape(d*4, d*4)
        hessian += np.diag(penalty.ravel()+1e-9)
        return loss, gradient, hessian

    converged = False
    for iteration in range(100):
        loss, gradient, hessian = evaluate(b, True)
        if np.max(np.abs(gradient)) < 1e-7:
            converged = True; break
        direction = np.linalg.solve(hessian, gradient.ravel()).reshape(d, 4)
        scale = 1.
        while scale > 2**-30 and evaluate(b-scale*direction) > loss-1e-4*scale*np.sum(gradient*direction):
            scale /= 2
        if scale <= 2**-30:
            raise RuntimeError('Joint model line search did not converge')
        b -= scale*direction
    if not converged:
        raise RuntimeError('Joint model optimization did not converge')
    return {'model': model, 'classes': CLASSES, 'coefficients_reference_A': b.tolist(),
            'penalty_nonintercept': 1.0, 'intercept_pseudocount_per_class': .5,
            'iterations': iteration, 'max_absolute_gradient': float(np.max(np.abs(gradient)))}


def predict(fitted, labels):
    z = np.asarray(features(labels, fitted['model']))@np.asarray(fitted['coefficients_reference_A'])
    p = softmax(np.r_[z, 0.])
    return {k: float(v) for k, v in zip(CLASSES, p)}


def scores(prediction, outcome):
    p = np.asarray([prediction[k] for k in CLASSES]); y = np.eye(5)[CLASSES.index(outcome)]
    return {'brier': float(np.sum((p-y)**2)), 'log_loss': float(-np.log(max(prediction[outcome], 1e-300))),
            'success_brier': float((prediction['S']-float(outcome == 'S'))**2)}


def compare(rows):
    tasks = sorted({r['task_id'] for r in rows})
    if len(tasks) != 12 or any(sum(r['task_id'] == t for r in rows) != 9 for t in tasks):
        raise ValueError('Predeclared 12 complete paired task blocks required')
    folds = []
    for task in tasks:
        train = [r for r in rows if r['task_id'] != task]
        test = [r for r in rows if r['task_id'] == task]
        for name in MODELS:
            fitted = fit(train, name)
            values = [scores(predict(fitted, r['initial_labels']), r['outcome']) for r in test]
            folds.append({'task_id': task, 'model': name,
                **{k: float(np.mean([v[k] for v in values])) for k in values[0]}})
    summary = {name: {metric: float(np.mean([f[metric] for f in folds if f['model'] == name]))
        for metric in ['brier', 'log_loss', 'success_brier']} for name in MODELS}
    rng = np.random.default_rng(2026102304)
    resamples = rng.integers(0, len(tasks), size=(20000, len(tasks)))
    contrasts = []
    for simpler, richer in [('constant', 'remaining_lifetime'), ('remaining_lifetime', 'content_occupancy_position'),
                            ('constant', 'content_occupancy_position')]:
        differences = np.asarray([next(f['brier'] for f in folds if f['task_id'] == t and f['model'] == simpler)
            -next(f['brier'] for f in folds if f['task_id'] == t and f['model'] == richer) for t in tasks])
        interval = np.quantile(differences[resamples].mean(axis=1), [.025, .975]).tolist()
        contrasts.append({'simpler': simpler, 'richer': richer, 'mean_brier_improvement': float(differences.mean()),
                          'task_bootstrap_95_interval': interval, 'per_task_improvements': differences.tolist()})
    def positive(simple, rich):
        return next(c['task_bootstrap_95_interval'][0] for c in contrasts if c['simpler'] == simple and c['richer'] == rich) > 0
    selected = ('content_occupancy_position' if positive('remaining_lifetime', 'content_occupancy_position')
                and positive('constant', 'content_occupancy_position') else
                'remaining_lifetime' if positive('constant', 'remaining_lifetime') else 'constant')
    fitted = {name: fit(rows, name) for name in MODELS}
    forecasts = []
    for n in range(4):
        for state in itertools.product(['E', 'C', 'N'], repeat=n):
            probabilities = predict(fitted[selected], list(state))
            failure = 1-probabilities['S']
            forecasts.append({'labels': list(state), 'joint': probabilities,
                'failure_conditional_reflection': {k: probabilities[k]/failure for k in CLASSES[1:]}})
    return {'folds': folds, 'scores': summary, 'contrasts': contrasts, 'selected': selected,
            'all_fitted_models': fitted, 'one_attempt_forecasts': forecasts,
            'scope': 'Task-grouped development model selection; prospective transport and native-context sufficiency remain to be tested'}
