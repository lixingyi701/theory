"""Run the frozen controlled FIFO simulation; no LLM or benchmark execution."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from agent_reliability.io import digest, file_hash, read_json, write_json
from finite_memory_models import (BUDGETS, C, E, H, N, binary_compressed,
    binary_exact_expectation, binary_full, binary_general_exact,
    binary_phase_restricted, coupled_memory_exact, information_counterexample_kernel,
    information_exact, information_kernel, information_states, intervene, mc_summary,
    sample_binary, sample_information, sample_sensitivity, sensitivity_arrays,
    sensitivity_bound, specification, uniform_draws)

SOURCES = ['scripts/finite_memory_models.py', 'scripts/run_finite_memory_simulation.py',
           'tests/test_finite_memory_simulation.py']
DESIGN = 'docs/experiments/finite-memory-simulation-design.md'


def save(path, value):
    value = {**value, 'sha256': digest(value)}
    write_json(path, value)
    return value


def deadline_check(deadline):
    if time.time() > deadline:
        raise TimeoutError('Campaign wall-time allowance exhausted')


def checkpoint(root, group, phase, done):
    write_json(root / 'checkpoint.json', dict(group=group, phase=phase, completed=done, time_unix=time.time()))


def verify_sources(protocol):
    assert protocol['sha256'] == digest({k: v for k, v in protocol.items() if k != 'sha256'})
    assert np.__version__ == protocol['numpy_version']
    for name, expected in protocol['source_hashes'].items():
        if file_hash(Path(name)) != expected:
            raise RuntimeError('Frozen simulation source changed: ' + name)
    assert file_hash(Path(DESIGN)) == protocol['design_sha256']
    for name, expected in protocol['protected_files'].items():
        assert file_hash(Path(name)) == expected, name
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(name) == '1', name


def prepare(path):
    if path.exists():
        raise FileExistsError(path)
    p = specification()
    p['source_hashes'] = {name: file_hash(Path(name)) for name in SOURCES}
    p['design_sha256'] = file_hash(Path(DESIGN))
    previous = read_json(Path('repro/reflexion-20261023/precondition-campaign-contract.json'))
    p['protected_files'] = {**previous['protected_files'], **{
        'src/agent_reliability/' + name: value for name, value in previous['source_hashes'].items()}}
    for name, expected in p['protected_files'].items():
        assert file_hash(Path(name)) == expected, name
    p['expected_counts'] = dict(binary_grid_conditions=1125, information_conditions=90,
        binary_mc_trajectories=160000, information_mc_trajectories=20000,
        sensitivity_mc_trajectories=80000, marginal_curve_comparisons=260,
        coupled_position_comparisons=1920)
    p['created_unix'] = time.time()
    save(path, p)
    print(json.dumps(dict(protocol=str(path), sha256=read_json(path)['sha256']), sort_keys=True))


def group_binary(protocol, root, deadline):
    max_b, budgets = protocol['max_attempts'], protocol['budgets']
    grid = protocol['binary_grid']
    raw = []
    maximum_error = 0.
    exact_expectation_comparisons = 0
    for w, p, q, lam, rho in itertools.product(*(grid[k] for k in ('W', 'p', 'q', 'lam', 'rho'))):
        deadline_check(deadline)
        compressed = binary_compressed(w, p, q, lam, rho, max_b)
        full, mapping = binary_full(w, p, q, lam, rho, max_b)
        error = float(np.max(np.abs(full - compressed[:, mapping])))
        maximum_error = max(maximum_error, error)
        assert error <= protocol['deterministic_tolerances']['full_vs_compressed_abs']
        expected, rational, length = binary_exact_expectation(w, p, q, lam, rho)
        exact_expectation_comparisons += w + 1
        raw.append(dict(W=w, p=p, q=q, lam=lam, rho=rho,
            success_absent=compressed[:, 0].tolist(), success_newest_error=compressed[:, w].tolist(),
            expected_attempts=expected, expected_attempts_exact=rational, L=length))
        if len(raw) % 50 == 0:
            checkpoint(root, 'binary', 'exact_grid', len(raw))
    save(root / 'exact-binary-grid.json', dict(conditions=raw))

    # Assess every adjacent lambda pair over every positive budget, not selected cases.
    families = {}
    for row in raw:
        key = tuple(row[k] for k in ('W', 'p', 'q', 'rho'))
        families.setdefault(key, []).append(row)
    increases = comparisons = family_increases = 0
    largest = None
    for key, rows in families.items():
        rows.sort(key=lambda x: x['lam'])
        affected = False
        for left, right in zip(rows, rows[1:]):
            gl = np.array(left['success_absent']) - left['success_newest_error']
            gr = np.array(right['success_absent']) - right['success_newest_error']
            change = gr - gl
            count = int(np.count_nonzero(change[1:] > 1e-12))
            increases += count; comparisons += max_b; affected |= bool(count)
            b = int(np.argmax(change[1:])) + 1
            if largest is None or change[b] > largest['increase']:
                largest = dict(W=key[0], p=key[1], q=key[2], rho=key[3],
                    lambda_from=left['lam'], lambda_to=right['lam'], budget=b,
                    gain_before=float(gl[b]), gain_after=float(gr[b]), increase=float(change[b]))
        family_increases += int(affected)

    mc = []
    first_raw = []; phase_raw = []
    by_key = {tuple(row[k] for k in ('W', 'p', 'q', 'lam', 'rho')): row for row in raw}
    for case in protocol['binary_mc']:
        key = tuple(case[k] for k in ('W', 'p', 'q', 'lam', 'rho'))
        exact = by_key[key]
        for start in (0, case['W']):
            first_seeds, phase_seeds = [], []
            for seed in protocol['seeds']:
                deadline_check(deadline)
                draws = uniform_draws(seed, max_b, protocol['trajectories_per_seed'])
                first, phase = sample_binary(case, start, draws)
                first_seeds.append(first); phase_seeds.append(phase)
            first = np.stack(first_seeds); phase = np.stack(phase_seeds)
            first_raw.append(first); phase_raw.append(phase)
            curve = np.array(exact['success_absent' if start == 0 else 'success_newest_error'])
            result = mc_summary(first, curve, protocol)
            row = dict(case=case, start_r=start, **result,
                full_expected_attempts=exact['expected_attempts'][start])
            if start:
                row.update(phase_restricted_mean_observed=float(phase.mean()),
                    phase_restricted_mean_exact=binary_phase_restricted(case['W'], case['q'], case['rho'], max_b),
                    full_phase_expected=exact['L'])
            mc.append(row)
            checkpoint(root, 'binary', 'monte_carlo_starts', len(mc))
    np.savez_compressed(root / 'binary-first-success.npz', first_success=np.stack(first_raw),
        phase_exit=np.stack(phase_raw), seeds=np.array(protocol['seeds']))
    boundary = binary_compressed(3, .5, 0., 0., 1., max_b)
    assert np.all(boundary[:, 3] == 0.)
    result = dict(group='binary', exact_conditions=len(raw), max_full_vs_compressed_error=maximum_error,
        rational_expectation_comparisons=exact_expectation_comparisons,
        lambda_increase_scan=dict(comparisons=comparisons, increases=increases,
            families=len(families), families_with_increase=family_increases, largest=largest),
        mc_trajectories=sum(row['curve'][0]['n'] for row in mc), mc=mc,
        persistent_error_boundary=dict(W=3, p=.5, q=0., lam=0., rho=1.,
            budgets=budgets, error_absent_success=boundary[budgets, 0].tolist(),
            newest_error_success=boundary[budgets, 3].tolist(), expected_error_start='infinite'),
        all_checks_passed=all(x['all_marginal_checks_passed'] for x in mc))
    save(root / 'summary.json', result)
    checkpoint(root, 'binary', 'complete', len(raw))
    return result


def group_information(protocol, root, deadline):
    max_b, budgets = protocol['max_attempts'], protocol['budgets']
    states, index, successor = information_states()
    raw = []; contrasts = []; maximum_residual = 0.
    for variant in protocol['information_variants']:
        deadline_check(deadline)
        p, kernel = information_kernel(states, variant['parameters'])
        history, expected, residual = information_exact(p, kernel, successor, max_b)
        maximum_residual = max(maximum_residual, residual)
        assert residual <= protocol['deterministic_tolerances']['linear_system_residual_abs']
        for start_id, memory in enumerate(protocol['information_starts']):
            curves = {}; means = {}
            for operation in protocol['operations']:
                changed = intervene(memory, operation)
                position = index[changed]
                curves[operation] = history[:, position]
                means[operation] = float(expected[position])
                raw.append(dict(variant=variant['id'], initial_id=start_id, initial=memory,
                    operation=operation, after=list(changed), success=history[:, position].tolist(),
                    expected_attempts=float(expected[position])))
            for left, right, interpretation in (
                ('correct', 'remove_target', 'correct_advice'),
                ('remove_target', 'neutralize', 'within_entry_useful_information'),
                ('delete', 'neutralize', 'occupancy_and_position'),
                ('correct', 'delete', 'recovery_comparison')):
                gap = curves[left] - curves[right]
                sign = np.sign(gap[1]) if abs(gap[1]) > 1e-12 else 0
                reverse = np.flatnonzero(gap[1:] * sign < -1e-12) + 1 if sign else []
                contrasts.append(dict(variant=variant['id'], initial_id=start_id,
                    left=left, right=right, mechanism=interpretation,
                    budgets=budgets, success_difference=gap[budgets].tolist(),
                    first_rank_reversal_budget=int(reverse[0]) if len(reverse) else None,
                    expected_attempts_left=means[left], expected_attempts_right=means[right]))
        checkpoint(root, 'information', 'exact_variants', len(raw))
    save(root / 'exact-information-grid.json', dict(states=len(states), conditions=raw, contrasts=contrasts))

    params = protocol['information_counterexample']
    p, kernel = information_counterexample_kernel(states, params)
    history, expected, residual = information_exact(p, kernel, successor, max_b)
    assert residual <= protocol['deterministic_tolerances']['linear_system_residual_abs']
    counterexample = []; first_raw = []
    for operation in ('correct', 'delete'):
        changed = intervene(params['initial'], operation); position = index[changed]
        first_seeds = []
        for seed in protocol['seeds']:
            deadline_check(deadline)
            first_seeds.append(sample_information(position, p, kernel, successor,
                uniform_draws(seed, max_b, protocol['trajectories_per_seed'])))
        first = np.stack(first_seeds); first_raw.append(first)
        counterexample.append(dict(operation=operation, after=list(changed),
            **mc_summary(first, history[:, position], protocol),
            full_expected_attempts=float(expected[position])))
    assert abs(history[2, index[(H, C, N)]] - .4885) < 1e-14
    assert abs(history[2, index[(H, N)]] - .64) < 1e-14
    assert abs(expected[index[(H, C, N)]] - 27.125) < 1e-10
    assert abs(expected[index[(H, N)]] - 19.6) < 1e-10
    np.savez_compressed(root / 'information-first-success.npz', first_success=np.stack(first_raw), seeds=np.array(protocol['seeds']))
    result = dict(group='information', states=len(states), exact_conditions=len(raw),
        max_linear_system_residual=maximum_residual, contrasts=contrasts,
        curves=[{**{k: v for k, v in row.items() if k != 'success'}, 'budgets': budgets,
            'success': [row['success'][b] for b in budgets]} for row in raw],
        counterexample=counterexample, mc_trajectories=20000,
        all_checks_passed=all(row['all_marginal_checks_passed'] for row in counterexample))
    save(root / 'summary.json', result)
    checkpoint(root, 'information', 'complete', len(raw))
    return result


def group_sensitivity(protocol, root, deadline):
    spec = protocol['sensitivity']; max_b, budgets = protocol['max_attempts'], protocol['budgets']
    bits, p, g = sensitivity_arrays(spec)
    history = binary_general_exact(p, g, max_b)
    bound, d_bound = sensitivity_bound(spec, max_b)
    memory_exact = coupled_memory_exact(g, spec['W'], max_b)
    success_gap = np.abs(history[:, :, None] - history[:, None, :])
    exact_success_excess = float(np.max(success_gap - bound))
    exact_memory_excess = float(np.max(memory_exact - d_bound))
    assert exact_success_excess <= protocol['deterministic_tolerances']['bound_abs']
    assert exact_memory_excess <= protocol['deterministic_tolerances']['bound_abs']
    first_seeds = []; count_seeds = []
    for seed in protocol['seeds']:
        deadline_check(deadline)
        first, counts = sample_sensitivity(spec, uniform_draws(seed, max_b, protocol['trajectories_per_seed']))
        first_seeds.append(first); count_seeds.append(counts)
        checkpoint(root, 'sensitivity', 'monte_carlo_seeds', len(first_seeds))
    first = np.stack(first_seeds); counts = np.stack(count_seeds)
    np.savez_compressed(root / 'sensitivity-first-success.npz', first_success=first, seeds=np.array(protocol['seeds']))
    np.savez_compressed(root / 'auxiliary-position-counts.npz', counts=counts, seeds=np.array(protocol['seeds']))
    np.savez_compressed(root / 'exact-sensitivity-curves.npz', success=history,
        success_gap=success_gap, success_bound=bound, memory_difference=memory_exact, memory_bound=d_bound)
    mc = [dict(initial=bits[i].tolist(), **mc_summary(first[:, i], history[:, i], protocol)) for i in range(len(bits))]
    n = len(protocol['seeds']) * protocol['trajectories_per_seed']
    observed_memory = counts.sum(axis=0) / n
    family = protocol['coupling_checks']
    tolerance = math.sqrt(math.log(2 * family['family_size'] / family['family_error_probability']) / (2 * n))
    discrepancies = np.abs(observed_memory[budgets] - memory_exact[budgets])
    assert discrepancies.size == family['family_size']
    boundaries = []
    for name, other in (
        ('memory_independent_update', {**spec, 'beta': [0., 0., 0.]}),
        ('copies_newest_error', {**spec, 'success_intercept': .5, 'alpha': [0., 0., .5],
            'emission_intercept': 0., 'beta': [0., 0., 1.], 'p_min': 0.})):
        _, op, og = sensitivity_arrays(other)
        oh = binary_general_exact(op, og, max_b)
        od = coupled_memory_exact(og, other['W'], max_b)
        if name == 'memory_independent_update':
            assert float(np.max(od[other['W']:])) < 1e-12
        else:
            assert np.all(od[:, 0, -1] == 1.)
        boundaries.append(dict(name=name, parameters=other, budgets=budgets,
            success_all_absent=oh[budgets, 0].tolist(), success_all_error=oh[budgets, -1].tolist(),
            auxiliary_difference_all_absent_vs_all_error=od[budgets, 0, -1].tolist()))
    result = dict(group='sensitivity', exact_success_bound_comparisons=max_b * 64,
        exact_position_bound_comparisons=(max_b + 1) * 64 * 3,
        max_exact_success_bound_excess=exact_success_excess,
        max_exact_position_bound_excess=exact_memory_excess,
        coupled_position_mc_comparisons=int(discrepancies.size),
        max_coupled_position_mc_error=float(discrepancies.max()),
        coupled_position_hoeffding_tolerance=tolerance,
        all_coupled_position_checks_passed=bool(np.all(discrepancies <= tolerance)),
        mc=mc, mc_trajectories=80000, boundaries=boundaries,
        example=dict(initial_left=bits[0].tolist(), initial_right=bits[-1].tolist(),
            budgets=budgets, exact_success_gap=success_gap[budgets, 0, -1].tolist(),
            success_bound=bound[budgets, 0, -1].tolist(),
            auxiliary_difference_exact=memory_exact[budgets, 0, -1].tolist(),
            auxiliary_difference_observed=observed_memory[budgets, 0, -1].tolist(),
            auxiliary_difference_bound=d_bound[budgets, 0, -1].tolist()),
        all_checks_passed=all(x['all_marginal_checks_passed'] for x in mc) and bool(np.all(discrepancies <= tolerance)))
    save(root / 'summary.json', result)
    checkpoint(root, 'sensitivity', 'complete', len(mc))
    return result


def worker(group, protocol, root, deadline):
    started = time.time()
    root.mkdir(parents=True, exist_ok=False)
    try:
        result = {'binary': group_binary, 'information': group_information, 'sensitivity': group_sensitivity}[group](protocol, root, deadline)
        save(root / 'completion.json', dict(status='complete', group=group,
            elapsed_seconds=time.time() - started, all_checks_passed=result['all_checks_passed']))
        return dict(group=group, status='complete', elapsed_seconds=time.time() - started,
                    all_checks_passed=result['all_checks_passed'])
    except BaseException:
        save(root / 'failure.json', dict(status='failed', group=group,
            elapsed_seconds=time.time() - started, traceback=traceback.format_exc()))
        raise


def run(protocol_path, output, source_commit):
    protocol = read_json(protocol_path); verify_sources(protocol)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.time(); deadline = started + protocol['execution']['wall_seconds']
    resource = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'], text=True)
    save(output / 'run.json', dict(protocol_sha256=protocol['sha256'], source_commit=source_commit,
        actual_argv=sys.argv, started_unix=started, deadline_unix=deadline,
        python=platform.python_version(), numpy=np.__version__, workers=protocol['execution']['workers'],
        threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        initial_gpu_snapshot=resource, gpu_devices_used=[], llm_calls=0,
        simulation_sources=protocol['source_hashes']))
    completed = []; errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=protocol['execution']['workers']) as pool:
        futures = {pool.submit(worker, group, protocol, output / group, deadline): group
                   for group in ('binary', 'information', 'sensitivity')}
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result(); completed.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            except BaseException as exc:
                errors.append(dict(group=futures[future], error=repr(exc)))
                print(json.dumps(errors[-1]), flush=True)
            save(output / 'checkpoint.json', dict(completed=completed, errors=errors))
    results = {row['group']: read_json(output / row['group'] / 'summary.json') for row in completed}
    marginal = [r for group in results.values() for sample in group.get('mc', group.get('counterexample', [])) for r in sample['curve']]
    count = sum(row['mc_trajectories'] for row in results.values())
    okay = len(completed) == 3 and not errors and all(x['all_checks_passed'] for x in completed)
    if okay:
        assert count == protocol['execution']['expected_monte_carlo_trajectories']
        assert len(marginal) == protocol['marginal_checks']['family_size']
        verify_sources(protocol)
    artifacts = {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}
    summary = save(output / 'completion.json', dict(status='complete' if okay else 'failed',
        protocol_sha256=protocol['sha256'], source_commit=source_commit, elapsed_seconds=time.time() - started,
        monte_carlo_trajectories=count, marginal_curve_comparisons=len(marginal),
        max_marginal_mc_error=max((x['absolute_error'] for x in marginal), default=None),
        marginal_hoeffding_tolerance=marginal[0]['hoeffding_tolerance'] if marginal else None,
        groups=completed, errors=errors, all_checks_passed=okay, new_llm_calls=0,
        raw_artifacts=artifacts))
    print(json.dumps({k: v for k, v in summary.items() if k != 'raw_artifacts'}, sort_keys=True), flush=True)
    if not okay:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-protocol', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    if args.prepare_protocol:
        prepare(args.prepare_protocol)
    elif args.protocol and args.output and args.source_commit:
        run(args.protocol, args.output, args.source_commit)
    else:
        parser.error('Use --prepare-protocol, or --protocol --output --source-commit')


if __name__ == '__main__':
    main()
