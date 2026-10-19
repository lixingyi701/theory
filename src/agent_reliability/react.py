"""Public ReAct ALFWorld protocol adapted to Qwen chat and the pinned grammar.

This is a baseline capability/instrumentation check, not a paper reproduction.
Only public feedback and agent-authored state enter prompts.
"""
from __future__ import annotations

import re
from pathlib import Path

from .io import canonical, digest, file_hash, read_json, seed_for, write_json
from .llm import CallBudgetExceeded, CallJournal, response_content
from .memory import FormatError, Memory
from .prompts import memory_messages
from .runner import OutputFormatExhausted, _call_json

TYPES = {"pick_and_place_simple": "put", "look_at_obj_in_light": "examine",
         "pick_clean_then_place_in_recep": "clean", "pick_heat_then_place_in_recep": "heat",
         "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "puttwo"}


def validate_react_config(config: dict) -> None:
    if config.get("agent") != "react_public_v2":
        raise ValueError("Unknown baseline agent version")
    for name in ["max_decisions", "max_env_steps", "max_llm_calls", "policy_max_tokens",
                 "memory_max_tokens", "memory_capacity", "timeout_seconds"]:
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not config.get("conditions") or set(config["conditions"]) - {"base", "memory", "sham"}:
        raise ValueError("ReAct conditions are base/memory/sham")
    if config.get("format_retries") not in [0, 1]:
        raise ValueError("Memory format retries must be 0 or 1")
    if config.get("action_adapter") != "put_in_on_to_move_v1":
        raise ValueError("Unknown action adapter")
    if not 0 <= config["sham_start_step"] <= config["sham_last_step"] < config["max_env_steps"]:
        raise ValueError("Invalid sham window")
    if file_hash(Path(config["demonstrations"])) != config["demonstrations_sha256"]:
        raise ValueError("Public demonstration bytes changed")


def environment_action(action: str) -> str:
    """A fixed syntax alias; never consults admissible commands or oracle facts."""
    match = re.fullmatch(r"put (.+) in/on (.+)", action)
    return f"move {match[1]} to {match[2]}" if match else action


def parse_action(text: str) -> str:
    # Chat models may repeat the demonstration's prompt marker; it is not part
    # of an ALFWorld command. Remove one marker, without repairing action content.
    action = text.strip()
    return action[1:].strip() if action.startswith(">") else action


def initial_observation(text: str) -> str:
    # The public notebook removes the TextWorld welcome block at reset.
    return text.split("\n\n", 1)[1].strip() if text.startswith("-= Welcome") else text.strip()


def process_observation(text: str) -> str:
    # Preserve the released notebook's processing, without stripping named locations.
    if text.startswith("You arrive at loc ") and ". " in text:
        text = text.split(". ", 1)[1]
    return text.strip()


def react_messages(task_type: str, initial: str, history: list[dict], demos: dict,
                   memory: list[dict] | None = None) -> list[dict]:
    kind = TYPES[task_type]
    prompt = ("Interact with a household to solve a task. Here are two examples.\n"
              + demos[f"react_{kind}_1"] + demos[f"react_{kind}_0"]
              + "\nHere is the task.\n" + initial + "\n>")
    for item in history:
        prompt += f" {item['action']}\n{item['observation']}\n>"
    if memory is not None:
        prompt += ("\nAgent-maintained factual memory (fallible; verify when needed):\n"
                   + canonical(memory) + "\n>")
    return [{"role": "user", "content": prompt}]


def run_react_episode(env, transport, task: dict, config: dict, model: dict,
                      seed: int, condition: str, directory: Path, resume: bool = False,
                      retry_uncertain: bool = False) -> dict:
    validate_react_config(config)
    if condition not in {"base", "memory", "sham"}:
        raise ValueError("Unknown ReAct condition")
    identity = {"task": task, "config": config, "model": model, "seed": seed, "condition": condition}
    identity_hash = digest(identity)
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "identity.json"
    checkpoint_path, result_path = directory / "checkpoint.json", directory / "result.json"
    if identity_path.exists():
        if read_json(identity_path)["sha256"] != identity_hash:
            raise ValueError("Episode identity/config mismatch; use a new run directory")
        if not resume:
            raise FileExistsError("Episode exists; use --resume")
    else:
        write_json(identity_path, {**identity, "sha256": identity_hash})
    if result_path.exists():
        return read_json(result_path)
    journal = CallJournal(directory / "calls", transport, config["max_llm_calls"], retry_uncertain)
    demos = read_json(Path(config["demonstrations"]))
    if checkpoint_path.exists():
        state = read_json(checkpoint_path)
        env.restore(state["environment"])
    else:
        initial = env.reset(task, seed)
        state = {"schema_version": 1, "decisions": 0, "env_steps": 0,
                 "initial_observation": initial_observation(initial.text), "history": [],
                 "memory": [], "memory_observation_step": -1, "sham": None,
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
        decision, step = state["decisions"], state["env_steps"]
        if decision >= config["max_decisions"]:
            status = "timeout_decision_budget"
            break
        if step >= config["max_env_steps"]:
            status = "timeout_env_budget"
            break
        memory = Memory(config["memory_capacity"], state["memory"])
        before, event, sham = memory.export(), None, state["sham"]
        memory_step = state["memory_observation_step"]
        try:
            if condition != "base" and memory_step != step:
                def validate_delta(value):
                    Memory(memory.capacity, before).update(value, step)
                delta, _ = _call_json(journal, model, config, task["task_id"], seed,
                                     decision, "memory",
                                     memory_messages(state["initial_observation"], observation.text, before),
                                     validate_delta)
                event = memory.update(delta, step)
                memory_step = step
            if condition == "sham" and sham is None and config["sham_start_step"] <= step <= config["sham_last_step"]:
                sham = memory.sham_verified_location(observation.facts)
            messages = react_messages(task["task_type"], state["initial_observation"],
                                      state["history"], demos, memory.export() if condition != "base" else None)
            payload = {"model": model["served_name"], "messages": messages,
                       "max_tokens": config["policy_max_tokens"], **config["sampling"],
                       "stop": ["\n"], "seed": seed_for(task["task_id"], seed, decision, "react_policy", 0),
                       "chat_template_kwargs": {"enable_thinking": False}}
            logical_id = f"decision-{decision:04d}-policy"
            saved = journal.complete(logical_id, payload)
            text, finish = response_content(saved)
            action = parse_action(text)
            valid = bool(action) and "\n" not in action and len(action) <= 500 and finish != "length"
            write_json(journal.directory / logical_id / "validation.json", {
                "logical_id": logical_id, "valid": valid, "finish_reason": finish,
                "truncated_output": finish == "length", "prompt_sha256": digest(messages)})
            if not valid:
                raise OutputFormatExhausted("Empty/multiline/truncated ReAct decision")
        except CallBudgetExceeded:
            status = "timeout_llm_budget"
            break
        except OutputFormatExhausted:
            status = "agent_format_failure"
            break
        thought = action.startswith("think:")
        submitted = None if thought else environment_action(action)
        # A thought consumes a decision and model call but does not mutate the world.
        following = observation if thought else env.step(submitted)
        feedback = "OK." if thought else process_observation(following.text)
        write_json(directory / "steps" / f"{decision:04d}.public.json", {
            "decision": decision, "env_step": step, "action": action, "submitted_action": submitted,
            "is_thought": thought, "observation": observation.text, "following_observation": feedback,
            "memory_before": before, "memory_after": memory.export(), "memory_event": event,
            "policy_prompt_sha256": digest(messages), "won": following.won})
        write_json(directory / "steps" / f"{decision:04d}.evaluator.json", {
            "facts_before": observation.facts, "facts_after": following.facts,
            "action_admissible": None if thought else submitted in observation.admissible_commands,
            "sham": sham})
        state = {**state, "decisions": decision + 1, "env_steps": step + int(not thought),
                 "history": state["history"] + [{"action": action, "observation": feedback}],
                 "memory": memory.export(), "memory_observation_step": memory_step,
                 "sham": sham, "environment": env.checkpoint()}
        write_json(checkpoint_path, state)
    state["status"] = status
    write_json(checkpoint_path, state)
    result = {"task_id": task["task_id"], "scene_id": task["scene_id"], "task_type": task["task_type"],
              "seed": seed, "condition": condition, "status": status, "success": status == "success",
              "decisions": state["decisions"], "env_steps": state["env_steps"],
              "remaining_env_budget": config["max_env_steps"] - state["env_steps"],
              "sham_applied": bool(state["sham"]), "identity_sha256": identity_hash,
              "trajectory_sha256": digest(state["history"]), **journal.totals()}
    write_json(result_path, result)
    return result
