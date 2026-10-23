"""Audit saved simulation records, export summaries and render measured curves.

The audit reaggregates stored first-success records without running trajectories.
Rendering can use a separate plotting dependency directory; it does not refit or
change the frozen simulation implementation or its acceptance conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from agent_reliability.io import digest, file_hash, read_json, write_json


def sealed(path):
    value = read_json(path)
    assert value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}), path
    return value


def save(path, value):
    write_json(path, {**value, 'sha256': digest(value)})


def check_first(first, rows, protocol):
    n = len(protocol['seeds']) * protocol['trajectories_per_seed']
    assert first.shape == (len(rows), len(protocol['seeds']), protocol['trajectories_per_seed'])
    assert np.all((first >= 1) & (first <= protocol['max_attempts'] + 1))
    count = 0
    for stored, row in zip(first, rows):
        assert stored.size == n
        for curve in row['curve']:
            actual = int(np.count_nonzero(stored <= curve['budget']))
            assert actual == curve['successes']
            assert actual / n == curve['estimate']
            assert curve['n'] == n
            count += 1
        assert float(stored.mean()) == row['restricted_mean_observed']
        assert int(np.count_nonzero(stored > protocol['max_attempts'])) == row['censored_at_max']
    return count


def audit(run, protocol_path, output):
    protocol = sealed(protocol_path); completion = sealed(run / 'completion.json')
    assert completion['status'] == 'complete' and completion['all_checks_passed']
    assert completion['protocol_sha256'] == protocol['sha256']
    for name, expected in completion['raw_artifacts'].items():
        assert file_hash(run / name) == expected, name
    for name, expected in protocol['source_hashes'].items():
        assert file_hash(Path(name)) == expected, name
    for name, expected in protocol['protected_files'].items():
        assert file_hash(Path(name)) == expected, name
    metadata = sealed(run / 'run.json')
    assert protocol['created_unix'] < metadata['started_unix']
    assert metadata['numpy'] == protocol['numpy_version']
    groups = {name: sealed(run / name / 'summary.json') for name in ('binary', 'information', 'sensitivity')}
    with np.load(run / 'binary/binary-first-success.npz') as raw:
        assert raw['seeds'].tolist() == protocol['seeds']
        bfirst = raw['first_success']; phase = raw['phase_exit']
        comparisons = check_first(bfirst, groups['binary']['mc'], protocol)
        for i, row in enumerate(groups['binary']['mc']):
            if row['start_r']:
                assert float(phase[i].mean()) == row['phase_restricted_mean_observed']
    with np.load(run / 'information/information-first-success.npz') as raw:
        assert raw['seeds'].tolist() == protocol['seeds']
        ifirst = raw['first_success']
        comparisons += check_first(ifirst, groups['information']['counterexample'], protocol)
    with np.load(run / 'sensitivity/sensitivity-first-success.npz') as raw:
        assert raw['seeds'].tolist() == protocol['seeds']
        sfirst = raw['first_success'].transpose(1, 0, 2)
        comparisons += check_first(sfirst, groups['sensitivity']['mc'], protocol)
    total = bfirst.size + ifirst.size + sfirst.size
    assert comparisons == protocol['marginal_checks']['family_size']
    assert total == completion['monte_carlo_trajectories'] == 260000
    with np.load(run / 'sensitivity/auxiliary-position-counts.npz') as raw:
        assert raw['seeds'].tolist() == protocol['seeds']
        counts = raw['counts']
        assert counts.shape == (10, 161, 8, 8, 3)
        assert np.all(counts == counts.transpose(0, 1, 3, 2, 4))
        assert np.all((counts >= 0) & (counts <= protocol['trajectories_per_seed']))
        assert np.all(counts[:, :, np.arange(8), np.arange(8)] == 0)
        with np.load(run / 'sensitivity/exact-sensitivity-curves.npz') as exact:
            discrepancy = np.abs(counts.sum(axis=0)[protocol['budgets']] / 10000
                                 - exact['memory_difference'][protocol['budgets']])
            assert discrepancy.size == protocol['coupling_checks']['family_size']
            assert float(discrepancy.max()) == groups['sensitivity']['max_coupled_position_mc_error']
    output.mkdir(parents=True, exist_ok=True)
    for group, value in groups.items():
        target = output / f'finite-memory-{group}-results.json'
        if target.exists():
            raise FileExistsError(target)
        write_json(target, value)
    target = output / 'finite-memory-simulation-completion.json'
    if target.exists():
        raise FileExistsError(target)
    write_json(target, completion)
    from continue_reflexion_pilot import ledger
    call_rows = ledger()
    calls = sum(row['dispatched'] for row in call_rows.values())
    responses = sum(row['responses'] for row in call_rows.values())
    report = dict(status='passed', source_freeze_commit=completion['source_commit'],
        protocol_sha256=protocol['sha256'], simulation_completion_sha256=completion['sha256'],
        post_run_audit_source_sha256=file_hash(Path(__file__)),
        protected_files_checked=len(protocol['protected_files']),
        immutable_raw_artifacts_checked=len(completion['raw_artifacts']),
        saved_trajectories_reaggregated=total, marginal_comparisons_reaggregated=comparisons,
        coupled_position_comparisons_reaggregated=int(discrepancy.size),
        protocol_created_before_run=True, all_seed_ids_verified=True,
        all_reflexion_calls=calls, all_reflexion_responses=responses, new_llm_calls=0,
        numpy=metadata['numpy'], gpu_devices_used=metadata['gpu_devices_used'])
    save(output / 'finite-memory-simulation-audit.json', report)
    print(json.dumps(report, sort_keys=True))


def render(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    try:
        font_manager.findfont('Times New Roman', fallback_to_default=False)
        font = 'Times New Roman'
    except ValueError:
        font = 'DejaVu Serif'
    matplotlib.rcParams.update({'font.family': font, 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 9,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'lines.linewidth': 1.6, 'savefig.dpi': 180,
        'mathtext.fontset': 'dejavuserif'})
    blue, orange, green, gray = '#0072B2', '#D55E00', '#009E73', '#666666'
    groups = {name: sealed(run / name / 'summary.json') for name in ('binary', 'information', 'sensitivity')}
    grid = sealed(run / 'binary/exact-binary-grid.json')['conditions']
    info = sealed(run / 'information/exact-information-grid.json')
    max_b = 160
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.65))
    ax = axes[0, 0]
    for lam, color in ((0., blue), (1., orange)):
        row = next(r for r in grid if (r['W'], r['p'], r['q'], r['lam'], r['rho']) == (3, .5, .02, lam, 0.))
        gain = np.array(row['success_absent']) - row['success_newest_error']
        ax.plot(range(1, 11), gain[1:11], color=color, marker='o', markersize=3,
                label=rf'$\lambda={lam:g}$')
    ax.set(title='(a) Budget changes the generation effect', xlabel='Attempt budget', ylabel='Success probability gain',
           xlim=(1, 10), ylim=(0, .9), xticks=[1, 3, 6, 10])
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for operation, color, label in (('correct', blue, 'Local correction'), ('delete', orange, 'Delete reflection')):
        row = next(r for r in groups['information']['counterexample'] if r['operation'] == operation)
        chosen = [r for r in row['curve'] if r['budget'] <= 10]
        ax.plot([r['budget'] for r in chosen], [r['exact'] for r in chosen], color=color, label=label)
        x = [r['budget'] for r in chosen]; y = np.array([r['estimate'] for r in chosen])
        ci = np.array([r['wilson95'] for r in chosen])
        ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt='o', color=color,
                    markersize=3, capsize=2, elinewidth=.7)
    ax.set(title='(b) Recovery ranking can reverse', xlabel='Attempt budget', ylabel='Success probability',
           xlim=(1, 10), ylim=(0, 1), xticks=[1, 2, 4, 6, 10])
    ax.legend(frameon=False, loc='lower right')
    ax = axes[1, 0]
    for rho, color in ((0., blue), (.5, green), (.9, orange)):
        rows = sorted([r for r in grid if (r['p'], r['q'], r['lam'], r['rho']) == (.5, 0., .5, rho)], key=lambda r: r['W'])
        ax.plot([r['W'] for r in rows], [r['L'] for r in rows], marker='o', markersize=3,
                color=color, label=rf'$\rho={rho:g}$')
    ax.set(title='(c) Restatement prolongs error visibility', xlabel='Window capacity',
           ylabel='Mean phase length (attempts)', yscale='log', xticks=[1, 2, 3, 5, 8], xlim=(1, 8))
    ax.legend(frameon=False)
    ax = axes[1, 1]
    with np.load(run / 'sensitivity/exact-sensitivity-curves.npz') as raw:
        exact_d = raw['memory_difference'][:, 0, -1].mean(axis=1)
        bound = raw['memory_bound'][:, 0, -1].mean(axis=1)
    ax.plot(range(13), exact_d[:13], color=blue, label='Exact auxiliary-chain difference')
    ax.plot(range(13), bound[:13], color=gray, linestyle='--', label='Position sensitivity bound')
    example = groups['sensitivity']['example']
    chosen = [i for i, b in enumerate(example['budgets']) if b <= 12]
    ax.scatter([example['budgets'][i] for i in chosen],
        [np.mean(example['auxiliary_difference_observed'][i]) for i in chosen],
        color=blue, s=13, zorder=4, label='Monte Carlo')
    ax.set(title='(d) Start differences enter new reflections', xlabel='Auxiliary memory updates',
           ylabel='Mean slot disagreement', xlim=(0, 12), ylim=(0, 1.03), xticks=[0, 3, 6, 9, 12])
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    for ax in axes.flat:
        ax.grid(axis='y', color='#dddddd', linewidth=.5, zorder=0)
    fig.subplots_adjust(left=.095, right=.985, bottom=.08, top=.95, wspace=.34, hspace=.46)
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(output / f'finite-memory-mechanisms.{ext}')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05))
    for name, color, label, key in (('binary', blue, 'Binary FIFO', 'mc'),
        ('information', orange, 'Information example', 'counterexample'),
        ('sensitivity', green, 'Sensitivity model', 'mc')):
        rows = [point for item in groups[name][key] for point in item['curve']]
        x = [r['exact'] for r in rows]; y = np.array([r['estimate'] for r in rows])
        ci = np.array([r['wilson95'] for r in rows])
        axes[0].errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt='o', color=color,
            markersize=2.7, alpha=.5, elinewidth=.4, label=label, linestyle='none')
    axes[0].plot([0, 1], [0, 1], color=gray, linestyle='--', linewidth=.8)
    axes[0].set(title='(a) Trajectories match exact recursion', xlabel='Exact success probability',
        ylabel='Monte Carlo estimate', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend(frameon=False, loc='lower right', fontsize=9)
    long = next(x for x in groups['binary']['mc'] if x['case']['id'] == 'binary-5' and x['start_r'] == 3)
    exact = next(r for r in grid if (r['W'], r['p'], r['q'], r['lam'], r['rho']) == (3, .5, 0., .5, .9))
    axes[1].plot(range(161), exact['success_newest_error'], color=blue, label='Exact')
    x = [r['budget'] for r in long['curve']]; y = np.array([r['estimate'] for r in long['curve']])
    ci = np.array([r['wilson95'] for r in long['curve']])
    axes[1].errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt='o', markersize=3,
        color=orange, capsize=2, elinewidth=.7, label='Monte Carlo')
    axes[1].set(title='(b) Long-tail censoring', xlabel='Attempt budget',
        ylabel='Success probability', xlim=(0, 160), ylim=(0, .11), xticks=[0, 40, 80, 120, 160])
    axes[1].text(.035, .95, f"{long['censored_at_max'] / 100:.2f}% censored at 160\n"
        f"Full mean: {long['full_expected_attempts']:.0f} attempts\n"
        f"Capped mean: {long['restricted_mean_exact']:.2f} attempts (exact)",
        transform=axes[1].transAxes, va='top', fontsize=9)
    axes[1].legend(frameon=False, loc='lower right')
    for ax in axes:
        ax.grid(axis='y', color='#dddddd', linewidth=.5, zorder=0)
    fig.subplots_adjust(left=.08, right=.985, bottom=.18, top=.88, wspace=.38)
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(output / f'finite-memory-validation.{ext}')
    plt.close(fig)
    receipt = dict(render_only=True, matplotlib=matplotlib.__version__, numpy=np.__version__,
        python=platform.python_version(), font=font,
        font_note='Explicit serif fallback when Times New Roman is unavailable',
        sources={str(p.relative_to(run)): file_hash(p) for p in [
            run / 'binary/summary.json', run / 'information/summary.json',
            run / 'sensitivity/summary.json', run / 'binary/exact-binary-grid.json',
            run / 'sensitivity/exact-sensitivity-curves.npz']},
        figure_hashes={p.name: file_hash(p) for p in output.iterdir() if p.suffix in ('.svg', '.png', '.pdf')})
    save(output / 'render-receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        render(args.run, args.output_dir)
    elif args.protocol:
        audit(args.run, args.protocol, args.output_dir)
    else:
        parser.error('Audit requires --protocol; drawing requires --render-only')


if __name__ == '__main__':
    main()
