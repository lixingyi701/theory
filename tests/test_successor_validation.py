"""Four-attempt orchestration and unequal-scene weighting; no Agent measurements."""
import copy
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from agent_reliability.io import digest
from prepare_successor_protocol import make_trajectories, PRIMARY, SECONDARY, SEEDS
from run_successor_validation import validate_matrix
from evaluate_successor_validation import score_rows
from run_multiround_diagnostic import execute_trajectory
from run_shared_calibration import initial_memory
from prepare_shared_calibration import ENTRIES
from test_multiround_diagnostic import fake_runner


def fixture():
    cases, counts, secondary = [], {}, []
    for scene in range(25):
        n = 2 if scene == 0 else (3 if scene < 3 else (5 if scene < 7 else 4))
        counts[str(scene)] = n
        for j in range(n):
            task = {'task_id': f'{scene}-{j}', 'scene_id': str(scene), 'task_type': 'pick_heat_then_place_in_recep'}
            cases.append({'task_id': task['task_id'], 'task': task, 'initial_observation_sha256': digest('public initial observation')})
            if j == 0:
                secondary.append(task['task_id'])
    manifest = {'tasks': [c['task'] for c in cases], 'counts_by_scene': counts,
        'secondary_recovery_task_ids': secondary, 'sha256': 'fixture'}
    protocol = {'cases': cases, 'entries': ENTRIES, 'manifest_sha256': 'fixture',
        'primary_conditions': PRIMARY, 'secondary_conditions': SECONDARY, 'crossed_seeds': SEEDS,
        'secondary_recovery_task_ids': secondary, 'trajectories': make_trajectories(cases, set(secondary))}
    return protocol, manifest


class SuccessorValidationTests(unittest.TestCase):
    def test_seven_feature_kernel_and_frozen_budget_interface(self):
        import numpy as np
        from prepare_successor_protocol import PARAMETERS, PLAN, checked, forecasts
        from calibrate_successor_success import component_kernel
        parameters=checked(PARAMETERS);plan=checked(PLAN)
        models={'full_seven':parameters['models']['seven']['model'],'full_six':parameters['models']['six']['model']}
        self.assertEqual(len(models['full_seven']['p_coefficients']),7)
        for state in [[{'type':'E','native':True}], [{'type':'N','native':False},{'type':'C','native':False}]]:
            kernel=component_kernel(models['full_seven'],state)
            self.assertEqual(kernel.shape,(3,6))
            self.assertTrue(np.allclose(kernel.sum(axis=1),1))
            self.assertTrue(np.all(kernel>=0))
        for name,conditions in forecasts(models).items():
            for condition,curve in conditions.items():
                self.assertTrue(np.allclose(curve,plan['forecasts'][name][condition],atol=1e-12,rtol=0))

    def test_crossed_complete_panels_and_four_attempt_fifo(self):
        protocol, manifest = fixture()
        self.assertEqual(validate_matrix(protocol, manifest)['trajectories'], 550)
        for corruption in ['missing', 'seed', 'order']:
            changed = copy.deepcopy(protocol)
            if corruption == 'missing':
                changed['trajectories'].pop()
            elif corruption == 'seed':
                changed['trajectories'][0]['seed'] = 999
            else:
                changed['trajectories'].reverse()
            with self.assertRaises(ValueError):
                validate_matrix(changed, manifest)
        trajectory = protocol['trajectories'][0]
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            summary = execute_trajectory(Path(directory), protocol, protocol['cases'][0], trajectory,
                None, lambda: None, attempt_runner=fake_runner(observed, [False, False, False, True]), memory_resolver=initial_memory)
        self.assertEqual(summary['attempts_completed'], 4)
        self.assertEqual(len(observed[-1]), 6)
        self.assertEqual(observed[-1][-3:], [f'Native reflection {j}.' for j in [1, 2, 3]])
        self.assertEqual(summary['attempts'][3]['visible_memory_sha256'], digest(observed[-1][-3:]))
        with tempfile.TemporaryDirectory() as directory:
            summary = execute_trajectory(Path(directory), protocol, protocol['cases'][0], trajectory,
                None, lambda: None, attempt_runner=fake_runner([], [True, False, False, False]), memory_resolver=initial_memory)
        self.assertEqual(summary['attempts_completed'], 1)

    def test_scene_equal_score_and_incomplete_matrix_rejection(self):
        protocol, _ = fixture()
        lookup = {c['task_id']: c['task']['scene_id'] for c in protocol['cases']}
        observed = [{**{k: t[k] for k in ['task_id', 'condition', 'seed', 'panel']},
            'trajectory_id': t['id'], 'scene_id': lookup[t['task_id']],
            'success_curve': [False, False, False, lookup[t['task_id']] == '0']} for t in protocol['trajectories']]
        protocol['forecasts'] = {name: {c: [p]*4 for c in PRIMARY+SECONDARY}
            for name, p in [('full_seven', .2), ('unchanged_seven', .8), ('full_six', .5)]}
        report = score_rows(protocol, observed)
        self.assertAlmostEqual(report['primary']['mean'], .6*(1-2/25))
        with self.assertRaises(ValueError):
            score_rows(protocol, observed[:-1])
        changed = copy.deepcopy(observed)
        changed[0]['seed'] = 999
        with self.assertRaises(ValueError):
            score_rows(protocol, changed)


if __name__ == '__main__':
    unittest.main()
