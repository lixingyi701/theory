"""Critical residual-text identity and factorial-coverage validation."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_reliability.io import digest
from run_residual_context import CalibrationStop, validate_matrix


def fixture():
    entries = {'N': 'Generic plan.', 'E': 'Wrong target rule.', 'C': 'Correct target rule.'}
    protocol = {'cases': [{'task_id': f'task-{i:02d}'} for i in range(12)], 'entries': entries,
                'ports': [8001, 8002], 'phases': ['factorial'], 'phase_cells': {'factorial': 72},
                'expected_cells': 72, 'blocks': [], 'reused_controls': []}
    for index, case in enumerate(protocol['cases']):
        task = case['task_id']
        residual = 'Other advice for '+task
        for seed in [2101, 2102]:
            cells = []
            for condition in ['N_E', 'N_C', 'R_E', 'R_C']:
                prefix, target = condition.split('_')
                memory = [entries['N'] if prefix == 'N' else residual, entries[target], entries['N']]
                cell = {'id': f'{task}-{seed}-{condition}', 'task_id': task, 'seed': seed,
                        'condition': condition, 'labels': [prefix, target, 'N'],
                        'memory_sha256': digest(memory), '_memory': memory}
                if prefix == 'R':
                    cell['residual_sha256'] = digest(residual)
                if seed == 2101 and prefix == 'N':
                    protocol['reused_controls'].append(cell)
                else:
                    cells.append(cell)
            protocol['blocks'].append({'task_id': task, 'seed': seed, 'attempts': 1,
                                       'panel': 'factorial', 'worker': index % 2, 'cells': cells})
    return protocol


def resolve(protocol, cell):
    return cell['_memory']


class ResidualContextTests(unittest.TestCase):
    def test_residual_position_and_hash_are_enforced(self):
        self.assertTrue(issubclass(CalibrationStop, BaseException))
        self.assertFalse(issubclass(CalibrationStop, Exception))
        for mutation in ['wrong_label_slot', 'wrong_text_slot', 'wrong_residual_hash', 'wrong_memory_hash']:
            with self.subTest(mutation=mutation):
                protocol = fixture()
                cell = protocol['blocks'][0]['cells'][0]
                if mutation == 'wrong_label_slot':
                    cell['labels'] = ['N', 'E', 'R']
                elif mutation == 'wrong_text_slot':
                    cell['_memory'] = list(reversed(cell['_memory']))
                    cell['memory_sha256'] = digest(cell['_memory'])
                elif mutation == 'wrong_residual_hash':
                    cell['residual_sha256'] = digest('replacement advice')
                else:
                    cell['memory_sha256'] = digest(['changed memory'])
                with self.assertRaises(ValueError):
                    validate_matrix(protocol, resolve)

    def test_complete_matrix_requires_72_new_and_24_reused_cells(self):
        protocol = fixture()
        self.assertEqual(validate_matrix(protocol, resolve),
                         {'new_cells': 72, 'reused_cells': 24, 'factorial_cells': 96})
        for mutation in ['missing_new', 'missing_reused', 'duplicate_cell', 'wrong_reuse_seed', 'different_residual']:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(protocol)
                if mutation == 'missing_new':
                    changed['blocks'][0]['cells'].pop()
                elif mutation == 'missing_reused':
                    changed['reused_controls'].pop()
                elif mutation == 'duplicate_cell':
                    changed['blocks'][0]['cells'].append(copy.deepcopy(changed['blocks'][0]['cells'][0]))
                elif mutation == 'wrong_reuse_seed':
                    changed['reused_controls'][0]['seed'] = 2102
                else:
                    cell = changed['blocks'][0]['cells'][1]
                    cell['_memory'][0] = 'different source advice'
                    cell['residual_sha256'] = digest(cell['_memory'][0])
                    cell['memory_sha256'] = digest(cell['_memory'])
                with self.assertRaises(ValueError):
                    validate_matrix(changed, resolve)


if __name__ == '__main__':
    unittest.main()
