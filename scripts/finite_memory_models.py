"""Specified finite-memory mechanisms and exact references, without Agent calls."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction as F

import numpy as np

E, C, H, N = 1, 2, 4, 0
BUDGETS = [1, 2, 3, 4, 6, 10, 20, 40, 80, 160]
SEEDS = list(range(21023, 21033))


def specification():
    reference = dict(p0=.4, dE=.3, dC=.05, dH=.1, dX=0., lam=.1,
                     rho=.5, sC=.1, kappa=.2, chi=0., nu=[.2] * 4)
    variants = [dict(id="reference", parameters=reference)]
    for key, values in (("dH", [0., .3]), ("sC", [0., .4]),
                        ("dX", [.1, .3]), ("chi", [.25, .5])):
        for value in values:
            variants.append(dict(id=f"{key}-{value:g}", parameters={**reference, key: value}))
    return {
        "protocol_version": 1,
        "scientific_scope": "finite_FIFO_theory_controlled_simulation",
        "design_commit": "a7f1ef0d4247abc3da081f84715c7240d8665e90",
        "budgets": BUDGETS, "max_attempts": 160, "seeds": SEEDS,
        "trajectories_per_seed": 1000, "random_generator": "numpy.random.PCG64",
        "random_draw_order": "per_attempt_then_success_array_then_reflection_array",
        "numpy_version": "1.26.4", "numeric_dtype": "float64",
        "binary_grid": {"W": [1, 2, 3, 5, 8], "p": [.2, .5, .8],
            "q": [0., .02, .1], "lam": [0., .25, .5, .75, 1.],
            "rho": [0., .25, .5, .75, .9]},
        "binary_mc": [dict(id=f"binary-{i}", W=w, p=p, q=q, lam=lam, rho=rho)
            for i, (w, p, q, lam, rho) in enumerate([
                (3, .5, 0., 0., 0.), (3, .5, 0., 1., 0.),
                (3, .5, .02, 0., 0.), (3, .5, .02, 1., 0.),
                (3, .5, 0., .5, .5), (3, .5, 0., .5, .9),
                (1, .5, .02, .5, .5), (5, .5, .02, .5, .5)])],
        "information_variants": variants,
        "information_starts": [[H, E, N], [N, E | H, N]],
        "operations": ["retain", "correct", "remove_target", "neutralize", "delete"],
        "information_counterexample": {"W": 3, "p0": .02, "dH": .38,
            "dC": .05, "dE": .1, "new_reflection": N, "initial": [H, E, N]},
        "sensitivity": {"W": 3, "success_intercept": .8,
            "alpha": [.1, .1, .1], "emission_intercept": .1,
            "beta": [.1, .1, .2], "p_min": .5},
        "marginal_checks": {"family_size": 260, "family_error_probability": .01,
            "tolerance_formula": "sqrt(log(2*m/delta)/(2*n))",
            "interval": "pointwise_95_percent_Wilson"},
        "coupling_checks": {"family_size": 1920, "family_error_probability": .01},
        "deterministic_tolerances": {"full_vs_compressed_abs": 2e-12,
            "linear_system_residual_abs": 2e-10, "bound_abs": 2e-12},
        "execution": {"workers": 3, "threads_per_worker": 1,
            "wall_seconds": 1800, "gpu_devices": [], "new_llm_calls": 0,
            "expected_monte_carlo_trajectories": 260000},
        "raw_evidence": "first_success_and_phase_exit_per_seed; coupling_counts; exact_curves",
    }


def binary_compressed(w, p, q, lam, rho, max_b):
    history = np.zeros((max_b + 1, w + 1), dtype=np.float64)
    for b in range(1, max_b + 1):
        v = history[b - 1]
        history[b, 0] = p + (1 - p) * ((1 - lam) * v[0] + lam * v[w])
        history[b, 1:] = q + (1 - q) * ((1 - rho) * v[:-1] + rho * v[w])
    return history


def binary_full(w, p, q, lam, rho, max_b):
    states = np.arange(2 ** w, dtype=np.int64)
    without = (states << 1) & (2 ** w - 1)
    with_error = without | 1
    success = np.where(states != 0, q, p)
    emission = np.where(states != 0, rho, lam)
    history = np.zeros((max_b + 1, len(states)), dtype=np.float64)
    for b in range(1, max_b + 1):
        v = history[b - 1]
        history[b] = success + (1 - success) * (
            (1 - emission) * v[without] + emission * v[with_error])
    mapping = np.array([0 if x == 0 else w - ((x & -x).bit_length() - 1)
                        for x in range(2 ** w)], dtype=np.int64)
    return history, mapping


def rational_solve(matrix, rhs):
    a = [list(row) + [v] for row, v in zip(matrix, rhs)]
    n = len(rhs)
    for col in range(n):
        pivot = next(row for row in range(col, n) if a[row][col])
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [x / scale for x in a[col]]
        for row in range(n):
            factor = a[row][col]
            if row != col and factor:
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]


def binary_exact_expectation(w, p, q, lam, rho):
    p, q, lam, rho = [F(str(x)) for x in (p, q, lam, rho)]
    size = w + 1
    matrix = [[F(i == j) for j in range(size)] for i in range(size)]
    matrix[0][0] -= (1 - p) * (1 - lam)
    matrix[0][w] -= (1 - p) * lam
    for r in range(1, size):
        matrix[r][r - 1] -= (1 - q) * (1 - rho)
        matrix[r][w] -= (1 - q) * rho
    independent = rational_solve(matrix, [F(1)] * size)
    a, omega = (1 - q) * (1 - rho), (1 - q) * rho
    s = [sum((a ** j for j in range(r)), F(0)) for r in range(size)]
    lengths = [x / (1 - omega * s[w]) for x in s]
    length = lengths[w]
    t0 = (1 + (1 - p) * lam * length) / (p + (1 - p) * lam * q * length)
    formula = [t0 + x * (1 - q * t0) for x in lengths]
    assert independent == formula
    return [float(x) for x in formula], [str(x) for x in formula], float(length)


def binary_phase_restricted(w, q, rho, max_b):
    # Killed at success or at r=0. Sum survival at t=0..max_b.
    survival = np.ones(w, dtype=np.float64)
    total = np.ones(w, dtype=np.float64)
    for _ in range(max_b):
        lower = np.concatenate(([0.], survival[:-1]))
        survival = (1 - q) * ((1 - rho) * lower + rho * survival[-1])
        total += survival
    return float(total[-1])


def uniform_draws(seed, max_b, n):
    return np.random.Generator(np.random.PCG64(seed)).random((max_b, 2, n))


def sample_binary(spec, start, draws):
    max_b, _, n = draws.shape
    memory = np.full(n, 0 if start == 0 else 1, dtype=np.int64)
    first = np.full(n, max_b + 1, dtype=np.int16)
    phase = np.full(n, max_b + 1, dtype=np.int16)
    if start == 0:
        phase[:] = 0
    for t in range(max_b):
        has_error = memory != 0
        won = draws[t, 0] < np.where(has_error, spec['q'], spec['p'])
        alive = first == max_b + 1
        first[alive & won] = t + 1
        emitted_error = draws[t, 1] < np.where(has_error, spec['rho'], spec['lam'])
        updated = ((memory << 1) & (2 ** spec['W'] - 1)) | emitted_error
        phase[(phase == max_b + 1) & (won | (updated == 0))] = t + 1
        memory = updated
    return first, phase


def information_states(w=3):
    states = [m for length in range(w + 1) for m in itertools.product(range(8), repeat=length)]
    index = {m: i for i, m in enumerate(states)}
    successor = np.array([[index[(m + (z,))[-w:]] for z in range(8)] for m in states], dtype=np.int64)
    return states, index, successor


def flags(states):
    return np.array([[any(z & bit for z in m) for bit in (E, C, H)] for m in states], dtype=np.float64)


def information_kernel(states, params):
    e, c, h = flags(states).T
    p = np.clip(params['p0'] - params['dE'] * e + params['dC'] * c
                + params['dH'] * h - params['dX'] * e * c, 0., 1.)
    g = np.clip(params['lam'] + (params['rho'] - params['lam']) * e - params['sC'] * c, 0., 1.)
    kernel = np.empty((len(states), 8), dtype=np.float64)
    for z in range(8):
        ej, cj, hj = bool(z & E), bool(z & C), bool(z & H)
        pc = params['chi'] if ej else params['kappa']
        ph = params['nu'][2 * ej + cj]
        kernel[:, z] = (g if ej else 1 - g) * (pc if cj else 1 - pc) * (ph if hj else 1 - ph)
    assert np.allclose(kernel.sum(axis=1), 1., atol=1e-14, rtol=0.)
    return p, kernel


def information_counterexample_kernel(states, params):
    e, c, h = flags(states).T
    p = np.clip(params['p0'] + params['dH'] * h + params['dC'] * c - params['dE'] * e, 0., 1.)
    kernel = np.zeros((len(states), 8), dtype=np.float64)
    kernel[:, params['new_reflection']] = 1.
    return p, kernel


def intervene(memory, operation):
    out = list(memory)
    targets = [i for i, z in enumerate(out) if z & E]
    if len(targets) != 1:
        raise ValueError('This protocol requires exactly one target-error entry')
    i = targets[0]
    if operation == 'correct':
        out[i] = (out[i] & ~E) | C
    elif operation == 'remove_target':
        out[i] &= ~E
    elif operation == 'neutralize':
        out[i] = N
    elif operation == 'delete':
        out.pop(i)
    elif operation != 'retain':
        raise ValueError(operation)
    return tuple(out)


def information_exact(p, kernel, successor, max_b):
    history = np.zeros((max_b + 1, len(p)), dtype=np.float64)
    for b in range(1, max_b + 1):
        history[b] = p + (1 - p) * (kernel * history[b - 1][successor]).sum(axis=1)
    q = np.zeros((len(p), len(p)), dtype=np.float64)
    for j in range(kernel.shape[1]):
        np.add.at(q, (np.arange(len(p)), successor[:, j]), (1 - p) * kernel[:, j])
    expected = np.linalg.solve(np.eye(len(p)) - q, np.ones(len(p)))
    residual = float(np.max(np.abs(expected - 1 - q @ expected)))
    return history, expected, residual


def sample_information(initial, p, kernel, successor, draws):
    max_b, _, n = draws.shape
    memory = np.full(n, initial, dtype=np.int64)
    first = np.full(n, max_b + 1, dtype=np.int16)
    cdf = np.cumsum(kernel, axis=1)
    cdf[:, -1] = 1.
    for t in range(max_b):
        won = draws[t, 0] < p[memory]
        first[(first == max_b + 1) & won] = t + 1
        emitted = (draws[t, 1, :, None] >= cdf[memory]).sum(axis=1)
        memory = successor[memory, emitted]
    return first


def sensitivity_arrays(spec):
    bits = np.array(list(itertools.product((0, 1), repeat=spec['W'])), dtype=np.int64)
    p = spec['success_intercept'] - bits @ np.array(spec['alpha'])
    g = spec['emission_intercept'] + bits @ np.array(spec['beta'])
    return bits, p, g


def binary_general_exact(p, g, max_b):
    states = np.arange(len(p), dtype=np.int64)
    without = (states << 1) & (len(states) - 1)
    history = np.zeros((max_b + 1, len(p)), dtype=np.float64)
    for b in range(1, max_b + 1):
        v = history[b - 1]
        history[b] = p + (1 - p) * ((1 - g) * v[without] + g * v[without | 1])
    return history


def sensitivity_bound(spec, max_b):
    bits, _, _ = sensitivity_arrays(spec)
    initial = (bits[:, None, :] != bits[None, :, :]).astype(np.float64)
    matrix = np.zeros((spec['W'], spec['W']))
    for i in range(spec['W'] - 1):
        matrix[i, i + 1] = 1.
    matrix[-1] = spec['beta']
    d = initial.copy()
    d_bound = np.empty((max_b + 1,) + initial.shape)
    accumulated = np.zeros(initial.shape[:2])
    bound = np.zeros((max_b + 1,) + initial.shape[:2])
    f = 1 - spec['p_min']
    for t in range(max_b + 1):
        d_bound[t] = d
        if t < max_b:
            accumulated += d @ np.array(spec['alpha'])
            bound[t + 1] = np.minimum(f ** (t + 1), f ** t * accumulated)
            d = d @ matrix.T
    return bound, d_bound


def sample_sensitivity(spec, draws):
    bits, p, g = sensitivity_arrays(spec)
    max_b, _, n = draws.shape
    memory = np.repeat(np.arange(len(bits))[:, None], n, axis=1)
    first = np.full(memory.shape, max_b + 1, dtype=np.int16)
    counts = np.empty((max_b + 1, len(bits), len(bits), spec['W']), dtype=np.int32)
    for t in range(max_b + 1):
        current = bits[memory]
        counts[t] = (current[:, None, :, :] != current[None, :, :, :]).sum(axis=2)
        if t < max_b:
            won = draws[t, 0][None, :] < p[memory]
            first[(first == max_b + 1) & won] = t + 1
            error = draws[t, 1][None, :] < g[memory]
            memory = ((memory << 1) & (len(bits) - 1)) | error
    return first, counts


def coupled_memory_exact(g, w, max_b):
    """Exact law of two auxiliary chains using one shared Bernoulli uniform."""
    size = 2 ** w
    pairs = list(itertools.product(range(size), repeat=2))
    transition = np.zeros((size * size, size * size), dtype=np.float64)
    for i, (left, right) in enumerate(pairs):
        joint = [(0, 0, 1 - max(g[left], g[right])),
                 (1, 1, min(g[left], g[right])),
                 (1, 0, max(g[left] - g[right], 0.)),
                 (0, 1, max(g[right] - g[left], 0.))]
        for le, re, probability in joint:
            nl, nr = ((left << 1) & (size - 1)) | le, ((right << 1) & (size - 1)) | re
            transition[i, nl * size + nr] += probability
    assert np.allclose(transition.sum(axis=1), 1., atol=1e-14, rtol=0.)
    bits = np.array(list(itertools.product((0, 1), repeat=w)))
    mismatch = np.array([bits[l] != bits[r] for l, r in pairs], dtype=np.float64)
    distribution = np.eye(size * size)
    result = np.empty((max_b + 1, size, size, w))
    for t in range(max_b + 1):
        result[t] = (distribution @ mismatch).reshape(size, size, w)
        distribution = distribution @ transition
    return result


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0., center - radius), min(1., center + radius)]


def mc_summary(first, exact, protocol):
    n = first.size
    family = protocol['marginal_checks']
    tolerance = math.sqrt(math.log(2 * family['family_size'] / family['family_error_probability']) / (2 * n))
    rows = []
    for b in protocol['budgets']:
        count = int(np.count_nonzero(first <= b))
        actual = count / n
        error = abs(actual - float(exact[b]))
        rows.append(dict(budget=b, successes=count, n=n, estimate=actual,
            exact=float(exact[b]), absolute_error=error, wilson95=wilson(count, n),
            hoeffding_tolerance=tolerance, within_tolerance=error <= tolerance))
    return dict(curve=rows, censored_at_max=int(np.count_nonzero(first > protocol['max_attempts'])),
        restricted_mean_observed=float(first.mean()),
        restricted_mean_exact=float(np.sum(1 - exact)),
        restricted_mean_cap=protocol['max_attempts'] + 1,
        all_marginal_checks_passed=all(x['within_tolerance'] for x in rows))
