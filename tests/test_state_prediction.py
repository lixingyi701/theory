"""Feature identifiability and numerical fitting checks; no benchmark outcomes."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reflexion_state_risk import CALIBRATION_STATES, VALIDATION_STATES, encode, fit, objective


class StatePredictionTests(unittest.TestCase):
    def test_homogeneous_design_identifies_slots_and_mixed_order_differs(self):
        x = np.asarray([encode(labels, 'slot') for labels in CALIBRATION_STATES.values()])
        self.assertEqual(np.linalg.matrix_rank(x), 7)
        oldest, newest = VALIDATION_STATES.values()
        self.assertEqual(encode(oldest, 'count'), encode(newest, 'count'))
        self.assertNotEqual(encode(oldest, 'slot'), encode(newest, 'slot'))
        self.assertEqual(encode(['E'], 'slot'), [1, 0, 0, 1, 0, 0, 0])

    def test_regularized_fit_matches_numerical_objective_stationarity(self):
        x = np.asarray([encode(labels, 'slot') for labels in CALIBRATION_STATES.values()])
        y = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=float)
        result = fit(x, y, 1.0); weights = np.asarray(result['weights'])
        self.assertLess(objective(x, y, weights, 1.0), objective(x, y, np.zeros(7), 1.0))
        eps = 1e-5
        gradients = []
        for index in range(7):
            step = np.eye(7)[index]*eps
            gradients.append((objective(x, y, weights+step, 1.0)-objective(x, y, weights-step, 1.0))/(2*eps))
        self.assertLess(max(abs(v) for v in gradients), 1e-5)


if __name__ == '__main__':
    unittest.main()
