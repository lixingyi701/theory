"""Public goal extraction and one-token intervention regression."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_reliability.io import digest, read_json, write_json
import prepare_reflexion_goal_targets as selector
from run_reflexion_forks import memories


class TargetTests(unittest.TestCase):
    def test_public_prefix_only_and_exact_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'tasks/example/trial-00'
            initial = ('You see a fridge 1, and a countertop 1.\n'
                       'Your task is to: put a clean lettuce in countertop.')
            wrapper = read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
            prompt = wrapper['prefix']+'Here is the task:\n'+initial+'\n>'+wrapper['suffix']
            write_json(directory/'calls/actor-0000/attempt-000.request.json', {'payload': {'prompt': prompt}})
            parent = {'success': False, 'memory_before': [],
                'memory_after': ['Plan: I should clean the lettuce, then place it on the countertop.'],
                'initial_observation_sha256': digest(initial)}
            write_json(directory/'result.json', parent)
            with patch.object(selector, 'PILOT', root):
                target, status = selector.candidate({'task_id': 'example'})
                write_json(root/'tasks/example/trial-01/result.json', {'future': 'must not affect selection'})
                again, _ = selector.candidate({'task_id': 'example'})
                self.assertEqual(target, again)
                self.assertEqual(status, 'eligible')
                self.assertEqual(target['original_text'], 'countertop')
                self.assertEqual(target['replacement_text'], 'fridge')
                variants = memories(parent, target)
                self.assertEqual(variants['correct_target'], variants['original'])
                self.assertEqual(variants['sham'], variants['original'])
                self.assertEqual(variants['retain_error'][0], parent['memory_after'][0].replace('countertop', 'fridge'))
                initial = 'You see a cabinet 12, and a diningtable 1.\nYour task is to: put some bowl in cabinet.'
                prompt = wrapper['prefix']+'Here is the task:\n'+initial+'\n>'+wrapper['suffix']
                write_json(directory/'calls/actor-0000/attempt-000.request.json', {'payload': {'prompt': prompt}})
                parent['initial_observation_sha256'] = digest(initial)
                parent['memory_after'] = ['Plan: I should put bowl 1 in cabinet 12.']
                write_json(directory/'result.json', parent)
                numbered, status = selector.candidate({'task_id': 'example'})
                self.assertEqual((numbered['original_text'], numbered['replacement_text']), ('cabinet 12', 'diningtable 1'))
                parent['initial_observation_sha256'] = 'tampered'
                write_json(directory/'result.json', parent)
                with self.assertRaisesRegex(ValueError, 'observation hash'):
                    selector.candidate({'task_id': 'example'})


if __name__ == '__main__':
    unittest.main()
