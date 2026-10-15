"""Replay environment contract. The budget belongs to the runner, not the world."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .io import digest


@dataclass
class Observation:
    text: str
    reward: float = 0.0
    won: bool = False
    terminated: bool = False
    facts: list[dict] | None = None
    admissible_commands: list[str] = field(default_factory=list)
    expert_commands: list[str] = field(default_factory=list)
    moves: int | None = None

    def signature(self) -> dict:
        # Planner output is not semantic state and is deliberately excluded.
        return {
            "text": self.text, "reward": self.reward, "won": self.won,
            "terminated": self.terminated, "facts": self.facts,
            "admissible_commands": sorted(self.admissible_commands), "moves": self.moves,
        }

    def fingerprint(self) -> str:
        return digest(self.signature())


class Backend(Protocol):
    identity: dict

    def reset(self, task: dict, seed: int) -> Observation: ...
    def step(self, action: str) -> Observation: ...
    def close(self) -> None: ...


class ReplayMismatch(RuntimeError):
    pass


class ReplayEnvironment:
    """Verify every reset/replayed transition, not just the final visible text."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.task: dict | None = None
        self.seed = 0
        self.actions: list[str] = []
        self.fingerprints: list[str] = []
        self.current: Observation | None = None

    def reset(self, task: dict, seed: int) -> Observation:
        self.task, self.seed = task, seed
        self.actions = []
        self.current = self.backend.reset(task, seed)
        self.fingerprints = [self.current.fingerprint()]
        return self.current

    def step(self, action: str) -> Observation:
        if self.current is None or self.current.terminated:
            raise RuntimeError("Cannot step an uninitialized or terminated environment")
        self.current = self.backend.step(action)
        self.actions.append(action)
        self.fingerprints.append(self.current.fingerprint())
        return self.current

    def checkpoint(self) -> dict:
        if self.task is None or self.current is None:
            raise RuntimeError("Reset before checkpoint")
        payload = {
            "schema_version": 1, "backend": self.backend.identity,
            "task": self.task, "seed": self.seed, "actions": list(self.actions),
            "fingerprints": list(self.fingerprints), "endpoint": asdict(self.current),
        }
        return {**payload, "sha256": digest(payload)}

    def restore(self, checkpoint: dict) -> dict:
        payload = {k: v for k, v in checkpoint.items() if k != "sha256"}
        if digest(payload) != checkpoint.get("sha256"):
            raise ReplayMismatch("Checkpoint integrity mismatch")
        if checkpoint["backend"] != self.backend.identity:
            raise ReplayMismatch("Backend version/configuration changed")
        if len(checkpoint["fingerprints"]) != len(checkpoint["actions"]) + 1:
            raise ReplayMismatch("Invalid checkpoint prefix length")
        self.reset(checkpoint["task"], checkpoint["seed"])
        if self.current.fingerprint() != checkpoint["fingerprints"][0]:
            raise ReplayMismatch("Reset signature mismatch")
        for index, action in enumerate(checkpoint["actions"], 1):
            observation = self.step(action)
            if observation.fingerprint() != checkpoint["fingerprints"][index]:
                raise ReplayMismatch(f"Replay signature mismatch at action {index}: {action}")
        level = "facts_and_observations" if self.current.facts is not None else "observations_only"
        return {"passed": True, "level": level, "replayed_actions": len(self.actions),
                "checkpoint_sha256": checkpoint["sha256"]}

    def close(self) -> None:
        self.backend.close()
