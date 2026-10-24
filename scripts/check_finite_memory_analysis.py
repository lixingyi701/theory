"""Exact algebra checks for the finite-window recovery analysis; no LLM calls.

The binary FIFO recursion is implemented independently of the compressed
remaining-updates recursion and closed-form expectation. Fractions keep the
small counterexamples and comparisons exact. This is not Monte Carlo evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from fractions import Fraction as F
from pathlib import Path

from agent_reliability.io import digest, write_json


def full_values(w, p, q, lam, rho, budget):
    states = list(itertools.product((0, 1), repeat=w))
    values = {m: F(0) for m in states}
    history = [values]
    for _ in range(budget):
        updated = {}
        for m in states:
            success, emission = (q, rho) if any(m) else (p, lam)
            updated[m] = success + (1 - success) * (
                (1 - emission) * values[m[1:] + (0,)]
                + emission * values[m[1:] + (1,)]
            )
        values = updated
        history.append(values)
    return history


def remaining(m):
    return max((i + 1 for i, item in enumerate(m) if item), default=0)


def compressed_values(w, p, q, lam, rho, budget):
    values = [F(0)] * (w + 1)
    history = [values]
    for _ in range(budget):
        values = [
            p + (1 - p) * ((1 - lam) * values[0] + lam * values[w])
        ] + [
            q + (1 - q) * ((1 - rho) * values[r - 1] + rho * values[w])
            for r in range(1, w + 1)
        ]
        history.append(values)
    return history


def linear_solve(matrix, rhs):
    augmented = [list(row) + [y] for row, y in zip(matrix, rhs)]
    size = len(rhs)
    for col in range(size):
        pivot = next(i for i in range(col, size) if augmented[i][col])
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [x / scale for x in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [
                    x - factor * y for x, y in zip(augmented[row], augmented[col])
                ]
    return [row[-1] for row in augmented]


def full_expectations(w, p, q, lam, rho):
    states = list(itertools.product((0, 1), repeat=w))
    indices = {m: i for i, m in enumerate(states)}
    matrix = [[F(i == j) for j in range(len(states))] for i in range(len(states))]
    for m, i in indices.items():
        success, emission = (q, rho) if any(m) else (p, lam)
        matrix[i][indices[m[1:] + (0,)]] -= (1 - success) * (1 - emission)
        matrix[i][indices[m[1:] + (1,)]] -= (1 - success) * emission
    return states, linear_solve(matrix, [F(1)] * len(states))


def closed_expectations(w, p, q, lam, rho):
    a, beta = (1 - q) * (1 - rho), (1 - q) * rho
    s = [sum((a ** j for j in range(r)), F(0)) for r in range(w + 1)]
    lengths = [x / (1 - beta * s[w]) for x in s]
    length = lengths[w]
    t0 = (1 + (1 - p) * lam * length) / (p + (1 - p) * lam * q * length)
    return [t0 + x * (1 - q * t0) for x in lengths]


def as_number(value):
    return {"exact": str(value), "decimal": float(value)}


def sensitivity_checks():
    """Bounded-difference example, all pairs of full three-bit windows."""
    w, budget = 3, 12
    alpha = [F(1, 10)] * w
    beta = [F(1, 10), F(1, 10), F(1, 5)]
    states = list(itertools.product((0, 1), repeat=w))
    success = {m: F(4, 5) - sum(a * x for a, x in zip(alpha, m)) for m in states}
    emission = {m: F(1, 10) + sum(b * x for b, x in zip(beta, m)) for m in states}
    values = {m: F(0) for m in states}
    history = [values]
    for _ in range(budget):
        values = {m: success[m] + (1 - success[m]) * (
            (1 - emission[m]) * values[m[1:] + (0,)]
            + emission[m] * values[m[1:] + (1,)]
        ) for m in states}
        history.append(values)
    count = 0
    for left, right in itertools.product(states, repeat=2):
        d = [F(x != y) for x, y in zip(left, right)]
        accumulated = F(0)
        for b in range(1, budget + 1):
            accumulated += sum(a * x for a, x in zip(alpha, d))
            rmax = 1 - min(success.values())
            bound = min(rmax ** b, rmax ** (b - 1) * accumulated)
            assert abs(history[b][left] - history[b][right]) <= bound
            d = d[1:] + [sum(x * y for x, y in zip(beta, d))]
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir / "finite-memory-analytic-checks.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")

    point_count = value_count = expectation_count = 0
    for w, p, q, lam, rho in itertools.product(
        (1, 2, 3, 4), (F(1, 5), F(1, 2), F(4, 5)),
        (F(0), F(1, 50), F(1, 10)), (F(0), F(1, 4), F(1)),
        (F(0), F(1, 2), F(9, 10))
    ):
        full = full_values(w, p, q, lam, rho, 12)
        compressed = compressed_values(w, p, q, lam, rho, 12)
        for b in range(13):
            for m, value in full[b].items():
                assert value == compressed[b][remaining(m)]
                value_count += 1
        states, numeric = full_expectations(w, p, q, lam, rho)
        closed = closed_expectations(w, p, q, lam, rho)
        for m, actual in zip(states, numeric):
            assert actual == closed[remaining(m)]
            expectation_count += 1
        point_count += 1

    counterexample = []
    for q in (F(0), F(1, 50)):
        for lam in (F(0), F(1)):
            history = compressed_values(3, F(1, 2), q, lam, F(0), 10)
            expected = closed_expectations(3, F(1, 2), q, lam, F(0))
            counterexample.append({
                "p": as_number(F(1, 2)), "q": as_number(q),
                "lambda": as_number(lam), "rho": as_number(F(0)), "window": 3,
                "budget_results": [{"budget": b,
                    "error_absent_start": as_number(history[b][0]),
                    "newest_error_start": as_number(history[b][3]),
                    "probability_gain": as_number(history[b][0] - history[b][3])
                } for b in (1, 3, 6, 10)],
                "expected_attempts_absent": as_number(expected[0]),
                "expected_attempts_newest_error": as_number(expected[3]),
                "expected_attempts_saved": as_number(expected[3] - expected[0]),
            })
    for q in (F(0), F(1, 50)):
        rows = [row for row in counterexample if F(row["q"]["exact"]) == q]
        gains = [[F(x["probability_gain"]["exact"]) for x in row["budget_results"]] for row in rows]
        assert gains[0][1] > gains[1][1] and gains[0][2] < gains[1][2]

    # A separate example keeps useful information and correct advice explicit.
    # N is a real neutral reflection; None is an unoccupied slot.
    corrected, deleted = ("H", "C", "N"), (None, "H", "N")
    def chance(m):
        return F(1, 50) + F(19, 50) * ("H" in m) + F(1, 20) * ("C" in m)
    def path(m):
        result = []
        for _ in range(3):
            result.append(chance(m)); m = m[1:] + ("N",)
        return result
    info_rows = []
    for name, m in (("local_correction", corrected), ("delete_error_reflection", deleted)):
        chances = path(m)
        expected = 1 + (1 - chances[0]) + (1 - chances[0]) * (1 - chances[1]) / chances[2]
        info_rows.append({"operation": name, "initial_window": list(m),
            "success_probabilities_first_three_attempts": [as_number(x) for x in chances],
            "success_by_one_attempt": as_number(chances[0]),
            "success_by_two_attempts": as_number(1 - (1 - chances[0]) * (1 - chances[1])),
            "expected_attempts_to_success": as_number(expected)})
    assert F(info_rows[0]["success_by_one_attempt"]["exact"]) > F(info_rows[1]["success_by_one_attempt"]["exact"])
    assert F(info_rows[0]["success_by_two_attempts"]["exact"]) < F(info_rows[1]["success_by_two_attempts"]["exact"])
    assert F(info_rows[0]["expected_attempts_to_success"]["exact"]) == F(217, 8)
    assert F(info_rows[1]["expected_attempts_to_success"]["exact"]) == F(98, 5)

    payload = {
        "kind": "exact_algebra_and_finite_state_recursion_check",
        "model": "binary_FIFO_and_explicit_information_counterexample",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arithmetic": "fractions.Fraction; all comparisons exact",
        "parameter_points": point_count,
        "full_window_vs_remaining_state_comparisons": value_count,
        "full_window_linear_system_vs_closed_expectation_comparisons": expectation_count,
        "sensitivity_bound_comparisons": sensitivity_checks(),
        "parameter_grid": {"window": [1, 2, 3, 4], "p": ["1/5", "1/2", "4/5"],
            "q": ["0", "1/50", "1/10"], "lambda": ["0", "1/4", "1"],
            "rho": ["0", "1/2", "9/10"], "budgets": list(range(13))},
        "generation_probability_counterexamples": counterexample,
        "information_position_counterexample": info_rows,
        "new_llm_calls": 0, "monte_carlo_trajectories": 0,
        "parameter_provenance": "constructed; not fitted to ALFWorld",
        "status": "all_exact_checks_passed",
    }
    payload["sha256"] = digest(payload)
    write_json(output, payload)
    print({key: payload[key] for key in ("parameter_points", "full_window_vs_remaining_state_comparisons",
        "full_window_linear_system_vs_closed_expectation_comparisons", "sensitivity_bound_comparisons", "new_llm_calls")})


if __name__ == "__main__":
    main()
