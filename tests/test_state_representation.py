"""Checks for the changed stop path and the actual statistical design."""
import sys
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_reliability.reflexion import author_namespace
from run_state_representation import CalibrationStop, reserve_call
from state_representation_model import features, fit, predict


class StateRepresentationTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Server fcntl accounting')
    def test_budget_stop_bypasses_upstream_exception_conversion(self):
        namespace, _ = author_namespace()
        def exhausted(*args, **kwargs):
            raise CalibrationStop('finite allowance')
        namespace['get_completion'] = exhausted
        with self.assertRaises(CalibrationStop):
            namespace['llm']('test', 'text-davinci-003')
        with tempfile.TemporaryDirectory() as directory:
            contract = {'deadline_unix': time.time()+60, 'max_calls': 1}
            reserve_call(Path(directory), contract)
            with self.assertRaises(CalibrationStop):
                reserve_call(Path(directory), contract)

    def test_state_aliases_are_the_intended_compression_only(self):
        self.assertEqual(features(['E'], 'remaining_lifetime'), features(['N','N','E'], 'remaining_lifetime'))
        self.assertEqual(features(['C'], 'remaining_lifetime'), features([], 'remaining_lifetime'))
        self.assertNotEqual(features(['C'], 'content_occupancy_position'), features([], 'content_occupancy_position'))
        self.assertNotEqual(features(['E'], 'content_occupancy_position'), features(['N','N','E'], 'content_occupancy_position'))

    def test_joint_constant_fit_matches_dirichlet_prediction(self):
        rows = [{'initial_labels': [], 'outcome': value} for value in ['S','E','E','U']]
        fitted = fit(rows, 'constant')
        probabilities = predict(fitted, [])
        np.testing.assert_allclose(list(probabilities.values()), np.asarray([1.5,2.5,.5,1.5,.5])/6.5, atol=1e-7)
        self.assertGreater(probabilities['A'], 0)


if __name__ == '__main__':
    unittest.main()
