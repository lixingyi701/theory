"""Declared controlled-memory states and regularized Bernoulli risk baselines."""
import numpy as np

CALIBRATION_STATES = {'empty': [], 'error_1': ['E'], 'error_2': ['E', 'E'],
                      'error_3': ['E', 'E', 'E'], 'correct_1': ['C'],
                      'correct_2': ['C', 'C'], 'correct_3': ['C', 'C', 'C']}
VALIDATION_STATES = {'error_oldest': ['E', 'C', 'C'], 'error_newest': ['C', 'C', 'E']}
NEW_CALIBRATION_STATES = ['error_2', 'error_3', 'correct_2', 'correct_3']
REUSED_BRANCHES = {'empty': 'discard_new', 'error_1': 'original', 'correct_1': 'correct_target'}
FEATURES = {'count': ['intercept', 'error_count', 'correct_count'],
            'slot': ['intercept', 'error_oldest', 'error_middle', 'error_newest',
                     'correct_oldest', 'correct_middle', 'correct_newest']}
RIDGES = [0.1, 1.0, 10.0]


def encode(labels, model):
    if len(labels) > 3 or any(label not in {'E', 'C'} for label in labels):
        raise ValueError('Declared window labels only')
    if model == 'count':
        return [1.0, float(labels.count('E')), float(labels.count('C'))]
    if model == 'slot':
        slots = [None]*(3-len(labels))+list(labels)
        return [1.0]+[float(label == 'E') for label in slots]+[float(label == 'C') for label in slots]
    raise ValueError('Unknown state risk baseline')


def sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z, -40, 40)))


def objective(x, y, weights, ridge):
    z = x@weights
    return float(np.sum(np.logaddexp(0.0, z)-y*z)+0.5*ridge*(weights@weights))


def fit(x, y, ridge):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) == 0 or not np.isin(y, [0, 1]).all() or ridge <= 0:
        raise ValueError('Observed binary outcomes and positive regularization required')
    weights = np.zeros(x.shape[1])
    for iteration in range(100):
        p = sigmoid(x@weights)
        gradient = x.T@(p-y)+ridge*weights
        hessian = x.T@((p*(1-p))[:, None]*x)+ridge*np.eye(x.shape[1])
        if float(np.max(np.abs(gradient))) < 1e-8:
            break
        delta = np.linalg.solve(hessian, gradient)
        old = objective(x, y, weights, ridge)
        fraction = 1.0
        while fraction > 2**-30:
            candidate = weights-fraction*delta
            if objective(x, y, candidate, ridge) <= old+1e-12:
                weights = candidate
                break
            fraction *= .5
        else:
            raise RuntimeError('Risk solver line search failed')
    else:
        raise RuntimeError('Risk solver did not converge')
    gradient = x.T@(sigmoid(x@weights)-y)+ridge*weights
    if not np.isfinite(weights).all() or float(np.max(np.abs(gradient))) > 1e-6:
        raise RuntimeError('Risk solver failed its numerical check')
    return {'weights': weights.tolist(), 'ridge': ridge, 'iterations': iteration,
            'gradient_max': float(np.max(np.abs(gradient))), 'objective': objective(x, y, weights, ridge)}


def log_loss(y, probability):
    y = np.asarray(y, dtype=float); probability = np.clip(np.asarray(probability), 1e-12, 1-1e-12)
    return float(np.mean(-y*np.log(probability)-(1-y)*np.log(1-probability)))


def train(rows, model):
    x = np.asarray([encode(row['labels'], model) for row in rows]); y = np.asarray([row['success'] for row in rows], dtype=float)
    groups = sorted({row['task_id'] for row in rows})
    if len(groups) < 3:
        raise ValueError('At least three independent development tasks required for grouped model selection')
    scores = []
    for ridge in RIDGES:
        predictions = np.empty(len(rows))
        for group in groups:
            held = np.asarray([row['task_id'] == group for row in rows])
            fitted = fit(x[~held], y[~held], ridge)
            predictions[held] = sigmoid(x[held]@np.asarray(fitted['weights']))
        scores.append({'ridge': ridge, 'grouped_log_loss': log_loss(y, predictions)})
    chosen = min(scores, key=lambda row: (round(row['grouped_log_loss'], 12), -row['ridge']))['ridge']
    result = fit(x, y, chosen)
    result.update(features=FEATURES[model], design_rank=int(np.linalg.matrix_rank(x)),
                  cross_validation=scores, independent_tasks=len(groups), observations=len(rows))
    return result
