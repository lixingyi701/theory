"""Direct, deterministic-game TextWorld backend; oracle fields never enter prompts."""
from __future__ import annotations

import importlib.metadata
import json
import random
from pathlib import Path

from .env import Observation
from .io import canonical, file_hash

ALFWORLD_COMMIT = "aaba6870f86c5be6a08a491f32a50b906227bc3e"


class ALFWorldBackend:
    def __init__(self, data_root: Path, expert: bool = False):
        self.data_root = data_root.resolve()
        self.expert = expert
        self.env = None
        self.identity = {"backend": "alfworld_textworld", "expert": expert,
                         "alfworld": importlib.metadata.version("alfworld"),
                         "textworld": importlib.metadata.version("textworld"),
                         "domain_randomization": False}

    def reset(self, task: dict, seed: int) -> Observation:
        import numpy as np
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        self.close()
        game = (self.data_root / task["gamefile"]).resolve()
        if not game.is_relative_to(self.data_root):
            raise ValueError("Game path escapes data root")
        if file_hash(game) != task["game_sha256"]:
            raise ValueError(f"Game bytes changed for {task['task_id']}")
        random.seed(seed)
        np.random.seed(seed % (2**32))
        infos = textworld.EnvInfos(won=True, lost=True, facts=True, moves=True,
                                   admissible_commands=True, policy_commands=self.expert)
        # textworld.start loads the file after wrapping, as in the official backend.
        self.env = textworld.start(str(game), request_infos=infos,
                                   wrappers=[AlfredDemangler(shuffle=False)])
        return self._convert(self.env.reset(), 0.0, False)

    @staticmethod
    def _convert(state, reward: float, done: bool) -> Observation:
        facts = state.get("facts")
        normalized = None if facts is None else sorted([
            {"name": fact.name,
             "arguments": [{"name": v.name, "type": v.type} for v in fact.arguments]}
            for fact in facts
        ], key=canonical)
        won = bool(state.get("won", False))
        return Observation(text=str(state.get("feedback", "")), reward=float(reward),
                           won=won, terminated=bool(done or won or state.get("lost", False)),
                           facts=normalized, moves=state.get("moves"),
                           admissible_commands=list(state.get("admissible_commands") or []),
                           expert_commands=list(state.get("policy_commands") or []))

    def step(self, action: str) -> Observation:
        state, reward, done = self.env.step(action)
        return self._convert(state, reward, done)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
