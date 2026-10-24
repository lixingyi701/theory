"""Check intervention coverage and native memory chaining without model calls."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_reliability.io import digest, write_json
from run_multiround_diagnostic import execute_trajectory, validate_matrix


def fixture():
    entries = {'E': 'Wrong heating premise.', 'C': 'Correct heating premise.', 'N': 'Generic plan.'}
    protocol = {'entries': entries, 'cases': [], 'trajectories': [],
                'ports': [8001, 8002, 8003], 'gpu_indices': [0, 1, 3]}
    for index in range(12):
        task = {'task_id': f'task-{index:02d}', 'scene_id': f'scene-{index:02d}', 'task_type': 'pick_heat_then_place_in_recep'}
        protocol['cases'].append({'task_id': task['task_id'], 'task': task, 'initial_observation_sha256': digest('public initial observation')})
        for history_key, history in [('single_error', ['E']), ('repeated_error', ['E', 'N', 'E'])]:
            # Explicit fixture expectations, independent of the production mapping helper.
            outputs = ([['E'], ['C'], []] if history_key == 'single_error'
                       else [['E', 'N', 'E'], ['E', 'N', 'C'], ['E', 'N']])
            for operation, labels in zip(['retain', 'correct_latest', 'delete_latest'], outputs):
                trajectory = {'id': f'{task["task_id"]}-{history_key}-{operation}', 'task_id': task['task_id'],
                    'seed': 2103, 'history_key': history_key, 'initial_history': history,
                    'operation': operation, 'initial_labels': labels,
                    'initial_memory_sha256': digest([entries[label] for label in labels]),
                    'worker': index % 3, 'max_attempts': 3}
                protocol['trajectories'].append(trajectory)
    return protocol


def resolve(protocol, trajectory):
    return [protocol['entries'][label] for label in trajectory['initial_labels']]


def fake_runner(observed, outcomes, corrupt_prefix=False):
    def run(task, memory, trial, directory, transport, data_root, seed, allowance, **kwargs):
        observed.append(list(memory))
        directory.mkdir(parents=True, exist_ok=True)
        identity = {'task': task, 'memory': memory, 'trial': trial, 'seed': seed, **kwargs}
        identity['sha256'] = digest(identity)
        write_json(directory/'identity.json', identity)
        success = outcomes[trial-1]
        after = list(memory) if success else list(memory)+[f'Native reflection {trial}.']
        if corrupt_prefix:
            after[0] = 'Unexpected replacement of an older reflection.'
        result = {**{key: task[key] for key in ['task_id', 'scene_id', 'task_type']},
            'trial': trial, 'success': success, 'decisions': 10,
            'memory_before': memory, 'memory_after': after,
            'initial_observation_sha256': digest('public initial observation'),
            'identity_sha256': identity['sha256'], 'trajectory_sha256': digest(['actions', trial])}
        write_json(directory/'result.json', result)
        return result
    return run


class MultiroundDiagnosticTests(unittest.TestCase):
    def test_matrix_rejects_missing_duplicate_and_wrong_intervention(self):
        protocol = fixture()
        self.assertEqual(validate_matrix(protocol, resolve)['trajectories'], 72)
        for change in ['missing', 'duplicate', 'deletion_exposes_history', 'corrects_all', 'wrong_seed']:
            with self.subTest(change=change):
                altered = copy.deepcopy(protocol)
                if change == 'missing':
                    altered['trajectories'].pop()
                elif change == 'duplicate':
                    altered['trajectories'][1] = copy.deepcopy(altered['trajectories'][0])
                elif change == 'deletion_exposes_history':
                    altered['trajectories'][2]['initial_labels'] = ['N']
                elif change == 'corrects_all':
                    altered['trajectories'][4]['initial_labels'] = ['C', 'N', 'C']
                else:
                    altered['trajectories'][0]['seed'] = 2101
                with self.assertRaises(ValueError):
                    validate_matrix(altered, resolve)

    def test_native_history_appends_and_success_stops(self):
        protocol = fixture()
        trajectory = protocol['trajectories'][3]  # [E,N,E], retain.
        case, observed, callbacks = protocol['cases'][0], [], []
        with tempfile.TemporaryDirectory() as directory:
            summary = execute_trajectory(Path(directory), protocol, case, trajectory, None, lambda: None,
                lambda row, state: callbacks.append((row, state)),
                fake_runner(observed, [False, True, False]), resolve)
        initial = resolve(protocol, trajectory)
        self.assertEqual(observed, [initial, initial+['Native reflection 1.']])
        self.assertEqual(summary['attempts_completed'], 2)
        self.assertTrue(summary['complete'])
        self.assertTrue(summary['success'])
        self.assertFalse(callbacks[0][1]['complete'])
        self.assertEqual(summary['attempts'][1]['visible_memory_sha256'], digest([initial[1], initial[2], 'Native reflection 1.']))

    def test_three_failures_shift_visible_window_without_recovery_again(self):
        protocol = fixture()
        trajectory = protocol['trajectories'][5]  # [E,N,E] -> [E,N].
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            summary = execute_trajectory(Path(directory), protocol, protocol['cases'][0], trajectory,
                None, lambda: None, attempt_runner=fake_runner(observed, [False]*3), memory_resolver=resolve)
        initial = resolve(protocol, trajectory)
        self.assertEqual(observed, [initial, initial+['Native reflection 1.'],
                                    initial+['Native reflection 1.', 'Native reflection 2.']])
        self.assertEqual(summary['attempts_completed'], 3)
        self.assertTrue(summary['complete'])
        self.assertFalse(summary['success'])
        self.assertEqual(summary['attempts'][2]['visible_memory_sha256'],
                         digest([initial[1], 'Native reflection 1.', 'Native reflection 2.']))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'preceding history'):
                execute_trajectory(Path(directory), protocol, protocol['cases'][0], trajectory, None,
                    lambda: None, attempt_runner=fake_runner([], [False]*3, True), memory_resolver=resolve)


if __name__ == '__main__':
    unittest.main()
