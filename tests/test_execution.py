"""Offline contract tests. Fake worlds/providers below are not benchmark evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from agent_reliability.audit import audit_manifest, verify_audit
from agent_reliability.env import Observation, ReplayEnvironment, ReplayMismatch
from agent_reliability.io import digest, read_json, write_json
from agent_reliability.llm import CallBudgetExceeded, CallJournal, InfrastructureError, UncertainRequest
from agent_reliability.memory import FormatError, Memory
from agent_reliability.runner import run_episode


def truth():
    return [{"name": "in", "arguments": [{"name": "apple 1", "type": "o"},
                                         {"name": "table 1", "type": "r"}]}]


class FakeBackend:
    identity = {"backend": "offline_fake", "version": 1}

    def __init__(self):
        self.position = 0
        self.drift = False

    def current(self):
        return Observation(text=f"stage{self.position}", won=self.position == 4,
                           terminated=self.position == 4, reward=float(self.position == 4),
                           facts=[] if self.drift else truth(), moves=self.position,
                           admissible_commands=["advance"], expert_commands=["advance"])

    def reset(self, task, seed):
        self.position = 0
        return self.current()

    def step(self, action):
        if action == "advance":
            self.position += 1
        return self.current()

    def close(self):
        pass


class FakeProvider:
    def __init__(self, fail_call=None, malformed_policy=False):
        self.calls = []
        self.fail_call = fail_call
        self.malformed_policy = malformed_policy

    def complete(self, payload):
        self.calls.append(copy.deepcopy(payload))
        if len(self.calls) == self.fail_call:
            raise InfrastructureError("offline injected interruption")
        if "updates" in payload["messages"][0]["content"]:
            value = {"updates": [{"entity": "apple 1", "attribute": "location", "value": "table 1"}]}
        elif self.malformed_policy:
            value = {"plan": "bad", "action": ["advance", "advance"]}
        else:
            value = {"plan": "continue", "action": "advance"}
        return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


TASK = {"task_id": "offline_task", "scene_id": "offline_scene", "task_type": "offline_type"}
MODEL = {"served_name": "offline_fake", "revision": "not_a_real_model"}


def config():
    return {"context_window": 2, "memory_capacity": 2, "max_env_steps": 8,
            "max_llm_calls": 20, "format_retries": 1, "policy_max_tokens": 64,
            "memory_max_tokens": 64, "timeout_seconds": 1, "sham_start_step": 1,
            "sham_last_step": 3, "conditions": ["clean", "sham"], "sampling": {"temperature": 0.7}}


class ExecutionTests(unittest.TestCase):
    def test_replay_detects_fact_drift_and_checkpoint_tampering(self):
        backend = FakeBackend()
        env = ReplayEnvironment(backend)
        env.reset(TASK, 42)
        env.step("advance")
        checkpoint = env.checkpoint()
        env.step("advance")
        self.assertEqual(env.restore(checkpoint)["replayed_actions"], 1)
        self.assertEqual(env.current.text, "stage1")
        backend.drift = True
        with self.assertRaises(ReplayMismatch):
            env.restore(checkpoint)
        checkpoint["seed"] = 99
        with self.assertRaises(ReplayMismatch):
            env.restore(checkpoint)

    def test_10_task_40_restore_gate_and_failed_report(self):
        tasks = [{**TASK, "task_id": f"offline_{i}", "task_type": f"kind{i % 6}"} for i in range(10)]
        manifest = {"tasks": tasks, "sha256": "offline_manifest"}
        factory = lambda: ReplayEnvironment(FakeBackend())
        report = audit_manifest(factory, manifest, reference_factory=factory)
        self.assertTrue(report["passed"])
        self.assertEqual(report["restore_checks_passed"], 40)
        verify_audit(report, manifest)
        report["restore_checks_passed"] = 0
        with self.assertRaises(ValueError):
            verify_audit(report, manifest)

    def test_memory_delta_is_atomic_and_provenance_is_runner_owned(self):
        memory = Memory(1)
        memory.update({"updates": [{"entity": "apple 1", "attribute": "location", "value": "table 1",
                                   "source_step": 999}]}, 2)
        self.assertEqual(memory.export()[0]["source_step"], 2)
        before = memory.export()
        with self.assertRaises(FormatError):
            memory.update({"updates": [{"entity": 1, "attribute": "location", "value": "table 1"}]}, 3)
        self.assertEqual(memory.export(), before)
        self.assertIsNotNone(memory.sham_verified_location(truth()))
        self.assertEqual(memory.export(), before)
        memory.update({"updates": [{"entity": "book 1", "attribute": "location", "value": "desk 1"}]}, 4)
        self.assertEqual(memory.export()[0]["entity"], "book 1")

    def test_request_cache_uncertainty_and_hard_budget(self):
        with tempfile.TemporaryDirectory() as root:
            provider = FakeProvider(fail_call=1)
            journal = CallJournal(Path(root), provider, max_calls=2)
            payload = {"messages": [{"content": "policy"}]}
            with self.assertRaises(InfrastructureError):
                journal.complete("first", payload)
            with self.assertRaises(UncertainRequest):
                journal.complete("first", payload)
            self.assertEqual(len(provider.calls), 1)
            journal.retry_uncertain = True
            saved = journal.complete("first", payload)
            self.assertEqual(journal.dispatched, 2)
            self.assertEqual(saved, journal.complete("first", payload))
            with self.assertRaises(CallBudgetExceeded):
                journal.complete("second", payload)

    def test_runner_clean_sham_and_no_oracle_prompt_leak(self):
        with tempfile.TemporaryDirectory() as root:
            for condition in ["clean", "sham"]:
                provider = FakeProvider()
                result = run_episode(ReplayEnvironment(FakeBackend()), provider, TASK, config(), MODEL,
                                     42, condition, Path(root) / condition)
                self.assertTrue(result["success"])
                self.assertEqual(result["env_steps"], 4)
                self.assertEqual(result["dispatched_calls"], 8)
                self.assertEqual(result["sham_applied"], condition == "sham")
                for call in provider.calls:
                    user = json.loads(call["messages"][1]["content"])
                    self.assertNotIn("facts", user)
                    self.assertNotIn("expert_commands", user)
                    self.assertNotIn("admissible_commands", user)

    def test_interrupted_step_resume_reuses_saved_memory_call(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            provider = FakeProvider(fail_call=2)
            with self.assertRaises(InfrastructureError):
                run_episode(ReplayEnvironment(FakeBackend()), provider, TASK, config(), MODEL,
                            42, "clean", directory)
            self.assertFalse((directory / "result.json").exists())
            self.assertEqual(read_json(directory / "checkpoint.json")["env_steps"], 0)
            with self.assertRaises(UncertainRequest):
                run_episode(ReplayEnvironment(FakeBackend()), provider, TASK, config(), MODEL,
                            42, "clean", directory, resume=True)
            result = run_episode(ReplayEnvironment(FakeBackend()), provider, TASK, config(), MODEL,
                                 42, "clean", directory, resume=True, retry_uncertain=True)
            self.assertTrue(result["success"])
            self.assertEqual(result["dispatched_calls"], 9)  # 8 successful + 1 uncertain
            self.assertEqual(len(provider.calls), 9)
            before = len(provider.calls)
            run_episode(ReplayEnvironment(FakeBackend()), provider, TASK, config(), MODEL,
                        42, "clean", directory, resume=True)
            self.assertEqual(len(provider.calls), before)

    def test_invalid_policy_is_distinct_from_environment_failure(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_episode(ReplayEnvironment(FakeBackend()), FakeProvider(malformed_policy=True),
                                 TASK, config(), MODEL, 42, "clean", Path(root))
            self.assertEqual(result["status"], "agent_format_failure")
            self.assertEqual(result["env_steps"], 0)
            self.assertEqual(result["dispatched_calls"], 3)

    def test_environment_and_model_budgets_are_separate(self):
        with tempfile.TemporaryDirectory() as root:
            settings = config() | {"max_env_steps": 2, "sham_start_step": 0, "sham_last_step": 1}
            result = run_episode(ReplayEnvironment(FakeBackend()), FakeProvider(), TASK, settings, MODEL,
                                 42, "clean", Path(root) / "env")
            self.assertEqual(result["status"], "timeout_env_budget")
            settings = config() | {"max_llm_calls": 1}
            result = run_episode(ReplayEnvironment(FakeBackend()), FakeProvider(), TASK, settings, MODEL,
                                 42, "clean", Path(root) / "llm")
            self.assertEqual(result["status"], "timeout_llm_budget")
            self.assertEqual(result["dispatched_calls"], 1)


if __name__ == "__main__":
    unittest.main()
