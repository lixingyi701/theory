"""Concrete regressions: syntax mismatch, thought budgets, history and resume."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from agent_reliability.io import read_json
from agent_reliability.env import ReplayEnvironment
from agent_reliability.llm import InfrastructureError, UncertainRequest
from agent_reliability.react import environment_action, parse_action, react_messages, run_react_episode
from test_execution import FakeBackend, MODEL, TASK


class Provider:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail
    def complete(self, payload):
        self.calls.append(copy.deepcopy(payload))
        if len(self.calls) == self.fail:
            raise InfrastructureError('injected interruption')
        # One thought, then environment actions; seed makes resumed request stable.
        text = 'think: I should advance.' if len(self.calls) == 1 else 'advance'
        return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}


class ReactTests(unittest.TestCase):
    def settings(self):
        c = read_json(Path('configs/react_base.json'))
        return {**c, 'max_decisions': 6}
    def task(self):
        return {**TASK, 'task_type': 'pick_and_place_simple'}
    def test_fixed_alias_and_no_action_selection(self):
        self.assertEqual(parse_action('> go to cabinet 1  '), 'go to cabinet 1')
        self.assertEqual(parse_action('> think: I should advance.'), 'think: I should advance.')
        self.assertEqual(environment_action('put apple 1 in/on table 1'), 'move apple 1 to table 1')
        for action in ['take apple 1 from table 1', 'put imaginary 9 on nowhere 2', 'look']:
            self.assertEqual(environment_action(action), action)
    def test_full_history_and_no_oracle_fields(self):
        demos = read_json(Path('prompts/react_alfworld.json'))
        history = [{'action': f'go to desk {i}', 'observation': f'public observation {i}'} for i in range(20)]
        message = react_messages('pick_and_place_simple', 'public task', history, demos)
        self.assertIn('public observation 0', message[0]['content'])
        self.assertIn('public observation 19', message[0]['content'])
        self.assertNotIn('admissible_commands', str(message))
        self.assertIn(demos['react_put_1'] + demos['react_put_0'], message[0]['content'])
    def test_thought_does_not_step_environment_and_resume_uses_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); p = Provider(fail=2)
            with self.assertRaises(InfrastructureError):
                run_react_episode(ReplayEnvironment(FakeBackend()), p, self.task(), self.settings(), MODEL, 42, 'base', d)
            checkpoint = read_json(d / 'checkpoint.json')
            self.assertEqual((checkpoint['decisions'], checkpoint['env_steps']), (1, 0))
            with self.assertRaises(UncertainRequest):
                run_react_episode(ReplayEnvironment(FakeBackend()), p, self.task(), self.settings(), MODEL, 42, 'base', d, True)
            result = run_react_episode(ReplayEnvironment(FakeBackend()), p, self.task(), self.settings(), MODEL, 42, 'base', d, True, True)
            self.assertTrue(result['success'])
            self.assertEqual((result['decisions'], result['env_steps']), (5, 4))
            self.assertEqual(result['dispatched_calls'], 6)
            count = len(p.calls)
            run_react_episode(ReplayEnvironment(FakeBackend()), p, self.task(), self.settings(), MODEL, 42, 'base', d, True)
            self.assertEqual(len(p.calls), count)
    def test_thoughts_consume_decision_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            c = {**self.settings(), 'max_decisions': 1}
            result = run_react_episode(ReplayEnvironment(FakeBackend()), Provider(), self.task(), c, MODEL, 42, 'base', Path(temp))
            self.assertEqual(result['status'], 'timeout_decision_budget')
            self.assertEqual(result['env_steps'], 0)

    def test_memory_and_sham_have_identical_payloads(self):
        class MemoryProvider(Provider):
            def complete(self, payload):
                self.calls.append(copy.deepcopy(payload))
                is_memory = payload['messages'][0]['role'] == 'system'
                text = json.dumps({'updates': [{'entity': 'apple 1', 'attribute': 'location', 'value': 'table 1'}]}) if is_memory else 'advance'
                return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as temp:
            providers = []
            for condition in ['memory', 'sham']:
                p = MemoryProvider();providers.append(p)
                r = run_react_episode(ReplayEnvironment(FakeBackend()), p, self.task(), self.settings(), MODEL, 42, condition, Path(temp) / condition)
                self.assertTrue(r['success'])
                self.assertEqual(r['sham_applied'], condition == 'sham')
            self.assertEqual(providers[0].calls, providers[1].calls)
            for call in providers[0].calls:
                self.assertNotIn('admissible_commands', str(call['messages']))
                self.assertNotIn('expert_commands', str(call['messages']))
