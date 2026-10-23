"""Changed-path checks; these outcomes are not scientific simulation results."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finite_memory_models import (C, E, H, N, information_states, intervene,
    sample_binary, sample_sensitivity, specification, uniform_draws)


class FiniteMemoryPaths(unittest.TestCase):
    def test_termination_and_censoring(self):
        draws = np.full((6, 2, 4), .5)
        first, phase = sample_binary(dict(W=3, p=1., q=0., lam=1., rho=0.), 3, draws)
        self.assertEqual(first.tolist(), [4] * 4)
        self.assertEqual(phase.tolist(), [3] * 4)
        first, _ = sample_binary(dict(W=3, p=0., q=0., lam=1., rho=1.), 0, draws)
        self.assertEqual(first.tolist(), [7] * 4)

    def test_deletion_is_not_neutral_replacement(self):
        _, index, successor = information_states()
        deleted = intervene((H, E, N), 'delete')
        neutral = intervene((H, E, N), 'neutralize')
        self.assertEqual(deleted, (H, N))
        self.assertEqual(neutral, (H, N, N))
        self.assertNotEqual(index[deleted], index[neutral])
        self.assertEqual(successor[index[deleted], N], index[(H, N, N)])
        self.assertEqual(successor[index[neutral], N], index[(N, N, N)])
        self.assertEqual(intervene((N, E | H, N), 'correct'), (N, C | H, N))

    def test_shared_auxiliary_update_erases_start_within_capacity(self):
        spec = {**specification()['sensitivity'], 'beta': [0., 0., 0.]}
        _, counts = sample_sensitivity(spec, uniform_draws(21023, 6, 10))
        self.assertTrue(np.all(counts[3:] == 0))


if __name__ == '__main__':
    unittest.main()
