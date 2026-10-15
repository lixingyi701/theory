"""Clean/sham execution with bounded history, explicit memory and resumable steps."""
from __future__ import annotations

import json
from pathlib import Path

from .env import ReplayEnvironment
from .io import canonical, digest, read_json, seed_for, write_json
from .llm import CallBudgetExceeded, CallJournal, response_content
from .memory import FormatError, Memory, short_text
from .prompts import memory_messages, policy_messages


class OutputFormatExhausted(RuntimeError):
    pass


def validate_config(config: dict) -> None:
    for name in ["context_window", "memory_capacity", "max_env_steps", "max_llm_calls",
                 "policy_max_tokens", "memory_max_tokens", "timeout_seconds"]:
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.get("format_retries") not in [0, 1]:
        raise ValueError("format_retries must be 0 or 1")
    if set(config["conditions"]) - {"clean", "sham"}:
        raise ValueError("This unit implements clean/sham only")
    if not 0 <= config["sham_start_step"] <= config["sham_last_step"] < config["max_env_steps"]:
        raise ValueError("Invalid sham window")


def _call_json(journal, model: dict, config: dict, task_id: str, seed: int,
               step: int, role: str, messages: list[dict], validate) -> tuple[dict, list[dict]]:
    records = []
    for attempt in range(config["format_retries"] + 1):
        prompt = messages if attempt == 0 else messages + [{"role": "user", "content":
            "The previous response was not usable. Return only the requested JSON schema; no markdown."}]
        payload = {"model": model["served_name"], "messages": prompt,
                   "max_tokens": config[role + "_max_tokens"], **config["sampling"],
                   "seed": seed_for(task_id, seed, step, role, attempt),
                   "chat_template_kwargs": {"enable_thinking": False}}
        logical_id = f"step-{step:04d}-{role}-{attempt}"
        saved = journal.complete(logical_id, payload)
        text, finish = response_content(saved)
        record = {"logical_id": logical_id, "prompt_sha256": digest(prompt),
                  "finish_reason": finish, "truncated_output": finish == "length"}
        try:
            if finish == "length":
                raise FormatError("Truncated model output")
            value = json.loads(text)
            validate(value)
            record["valid"] = True
            records.append(record)
            write_json(journal.directory / logical_id / "validation.json", record)
            return value, records
        except (json.JSONDecodeError, FormatError, TypeError) as exc:
            record.update({"valid": False, "format_error": str(exc)})
            records.append(record)
            # Raw responses/requests remain available even when no step commits.
            write_json(journal.directory / logical_id / "validation.json", record)
    raise OutputFormatExhausted(f"Unusable {role} output after format retry")


def run_episode(env: ReplayEnvironment, transport, task: dict, config: dict, model: dict,
                seed: int, condition: str, directory: Path, resume: bool = False,
                retry_uncertain: bool = False) -> dict:
    validate_config(config)
    if condition not in {"clean", "sham"}:
        raise ValueError("Unknown condition")
    identity = {"task": task, "config": config, "model": model,
                "seed": seed, "condition": condition}
    identity_hash = digest(identity)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path, result_path = directory / "checkpoint.json", directory / "result.json"
    identity_path = directory / "identity.json"
    if identity_path.exists():
        if read_json(identity_path)["sha256"] != identity_hash:
            raise ValueError("Episode identity/config mismatch; use a new run directory")
        if not resume:
            raise FileExistsError("Episode already exists; use --resume or a new output directory")
    else:
        write_json(identity_path, {**identity, "sha256": identity_hash})
    if result_path.exists():
        return read_json(result_path)
    journal = CallJournal(directory / "calls", transport, config["max_llm_calls"], retry_uncertain)
    if checkpoint_path.exists():
        state = read_json(checkpoint_path)
        env.restore(state["environment"])
    else:
        initial = env.reset(task, seed)
        state = {"schema_version": 1, "env_steps": 0, "initial_observation": initial.text,
                 "history": [], "memory": [], "sham": None,
                 "environment": env.checkpoint(), "status": "active"}
        write_json(checkpoint_path, state)
    status = state["status"]
    while status == "active":
        observation = env.current
        if observation.won:
            status = "success"
            break
        if observation.terminated:
            status = "environment_failure"
            break
        step = state["env_steps"]
        if step >= config["max_env_steps"]:
            status = "timeout_env_budget"
            break
        memory = Memory(config["memory_capacity"], state["memory"])
        before = memory.export()
        records = []
        sham_event = state["sham"]
        try:
            # Validation uses a separate Memory so failed deltas cannot mutate state.
            def validate_memory(value):
                Memory(memory.capacity, before).update(value, step)

            delta, calls = _call_json(journal, model, config, task["task_id"], seed, step,
                                      "memory", memory_messages(state["initial_observation"], observation.text, before),
                                      validate_memory)
            records.extend(calls)
            memory_event = memory.update(delta, step)
            if (condition == "sham" and sham_event is None
                    and config["sham_start_step"] <= step <= config["sham_last_step"]):
                sham_event = memory.sham_verified_location(observation.facts)

            def validate_action(value):
                if not isinstance(value, dict):
                    raise FormatError("Policy response must be an object")
                action = short_text(value.get("action"), "action", 200)
                if "\n" in action or action.startswith("think:"):
                    raise FormatError("Exactly one environment action required")
                short_text(value.get("plan"), "plan", 500)

            messages = policy_messages(state["initial_observation"], observation.text,
                                       state["history"][-config["context_window"]:], memory.export())
            policy, calls = _call_json(journal, model, config, task["task_id"], seed, step,
                                       "policy", messages, validate_action)
            records.extend(calls)
        except CallBudgetExceeded:
            status = "timeout_llm_budget"
            break
        except OutputFormatExhausted:
            status = "agent_format_failure"
            break
        action = policy["action"].strip()
        following = env.step(action)
        interaction = {"plan": policy["plan"], "action": action, "observation": following.text}
        # Public and oracle records are separate. Prompts never receive facts or plans.
        write_json(directory / "steps" / f"{step:04d}.public.json", {
            "env_step": step, "observation": observation.text, "memory_before": before,
            "memory_after": memory.export(), "memory_event": memory_event,
            "policy": policy, "following_observation": following.text, "calls": records,
            "reward": following.reward, "won": following.won,
            "terminated": following.terminated,
        })
        write_json(directory / "steps" / f"{step:04d}.evaluator.json", {
            "env_step": step, "facts_before": observation.facts, "facts_after": following.facts,
            "admissible_before": observation.admissible_commands,
            "action_admissible": action in observation.admissible_commands,
            "sham": sham_event, "target_contamination": None,
        })
        state = {**state, "env_steps": step + 1,
                 "history": (state["history"] + [interaction])[-config["context_window"]:],
                 "memory": memory.export(), "sham": sham_event, "environment": env.checkpoint()}
        # This is the commit point. An interrupted uncommitted step is reconstructed
        # from cached calls; files for that step are overwritten deterministically.
        write_json(checkpoint_path, state)
    state["status"] = status
    write_json(checkpoint_path, state)
    result = {"task_id": task["task_id"], "scene_id": task["scene_id"],
              "task_type": task["task_type"], "seed": seed, "condition": condition,
              "status": status, "success": status == "success", "env_steps": state["env_steps"],
              "remaining_env_budget": config["max_env_steps"] - state["env_steps"],
              "sham_applied": bool(state["sham"]), "identity_sha256": identity_hash,
              **journal.totals()}
    write_json(result_path, result)
    return result
