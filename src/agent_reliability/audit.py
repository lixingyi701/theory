"""No-LLM checkpoint/replay audit using evaluator-only planner actions."""
from __future__ import annotations

from collections import defaultdict

from .env import ReplayEnvironment
from .io import digest


def select_audit_tasks(tasks: list[dict], count: int = 10) -> list[dict]:
    groups = defaultdict(list)
    for task in tasks:
        groups[task["task_type"]].append(task)
    selected = []
    while len(selected) < count:
        before = len(selected)
        for kind in sorted(groups):
            if groups[kind] and len(selected) < count:
                selected.append(groups[kind].pop(0))
        if len(selected) == before:
            raise ValueError("Not enough tasks for audit")
    return selected


def audit_task(env: ReplayEnvironment, task: dict, seed: int = 42,
               max_reference_steps: int = 100, repeats: int = 2,
               reference_env: ReplayEnvironment | None = None) -> dict:
    reference = reference_env or env
    observation = reference.reset(task, seed)
    checkpoints = [reference.checkpoint()]
    observations = [observation.fingerprint()]
    actions = []
    while not observation.terminated and len(actions) < max_reference_steps:
        if not observation.expert_commands:
            raise RuntimeError("Evaluator planner returned no reference action")
        action = observation.expert_commands[0]
        observation = reference.step(action)
        actions.append(action)
        observations.append(observation.fingerprint())
        checkpoints.append(reference.checkpoint())
    if not observation.won:
        raise RuntimeError("Reference plan did not reach a successful state")
    if len(actions) < 2:
        raise RuntimeError("Reference too short for two distinct checkpoints")
    if reference_env is not None:
        # Requesting a planner must not change the audited execution semantics.
        initial = env.reset(task, seed)
        if initial.fingerprint() != observations[0]:
            raise RuntimeError("Planner-enabled reset differs from execution backend")
        checkpoints = [env.checkpoint()]
        for index, action in enumerate(actions, 1):
            if env.step(action).fingerprint() != observations[index]:
                raise RuntimeError("Planner-enabled reference differs from execution backend")
            checkpoints.append(env.checkpoint())
    first = min(len(actions) - 2, len(actions) // 3)
    second = max(first + 1, min(len(actions) - 1, 2 * len(actions) // 3))
    checks = []
    for prefix in [first, second]:
        for repetition in range(repeats):
            restore = env.restore(checkpoints[prefix])
            probes = actions[prefix:prefix + 2]
            for offset, action in enumerate(probes, prefix + 1):
                got = env.step(action)
                if got.fingerprint() != observations[offset]:
                    raise RuntimeError(f"Fixed-suffix replay differs at {offset}")
            if restore["level"] != "facts_and_observations":
                raise RuntimeError("Facts missing: observation-only audit is insufficient")
            checks.append({**restore, "prefix_length": prefix, "repetition": repetition,
                           "fixed_suffix": probes})
    return {"task_id": task["task_id"], "passed": True, "reference_length": len(actions),
            "reference_actions": actions, "checks": checks}


def audit_manifest(factory, manifest: dict, count: int = 10, reference_factory=None) -> dict:
    tasks = select_audit_tasks(manifest["tasks"], count)
    rows = []
    backend_identity = None
    for task in tasks:
        env = None
        reference = None
        try:
            env = factory()
            reference = reference_factory() if reference_factory else None
            backend_identity = env.backend.identity
            row = audit_task(env, task, reference_env=reference)
        except Exception as exc:
            row = {"task_id": task["task_id"], "passed": False,
                   "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            if env is not None:
                env.close()
            if reference is not None:
                reference.close()
        rows.append(row)
    payload = {"schema_version": 1, "manifest_sha256": manifest["sha256"],
               "backend": backend_identity, "requested_tasks": count,
               "passed": len(rows) == count and all(r["passed"] for r in rows),
               "restore_checks_passed": sum(len(r.get("checks", [])) for r in rows),
               "level": "facts_and_observations", "tasks": rows,
               "scope": "Recorded symbolic facts, outputs and fixed suffix; not proof of all hidden-state equality"}
    return {**payload, "sha256": digest(payload)}


def verify_audit(report: dict, manifest: dict) -> None:
    payload = {k: v for k, v in report.items() if k != "sha256"}
    if digest(payload) != report.get("sha256") or report["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Audit integrity/manifest mismatch")
    if not report["passed"] or report["requested_tasks"] < 10 or report["restore_checks_passed"] < 40:
        raise ValueError("Run requires at least 10 tasks and 40 passed replay checks")
