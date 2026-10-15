"""Only public observations and the agent's own state may be formatted here."""
from .io import canonical

POLICY = """You are a household-task agent operating a text environment.
Use the observation, recent interactions and your fallible memory to complete the goal.
Return only JSON: {"plan":"a short plan", "action":"one environment command"}.
Do not output a list of commands. Use concrete numbered object names from observations.
Common commands: look; inventory; go to <receptacle>; open <receptacle>;
close <receptacle>; take <object> from <receptacle>;
put <object> in/on <receptacle>; clean <object> with <sinkbasin>;
heat <object> with <microwave>; cool <object> with <fridge>; use <lamp>.
Memory may be incorrect or outdated: inspect the environment when needed.
"""

UPDATER = """Maintain explicit factual memory for a text household task.
Return only JSON: {"updates":[{"entity":"apple 1","attribute":"location","value":"table 1"}]}.
Use only the supplied new observation and existing memory. Do not invent facts.
Use attribute location for an object's current receptacle; retain exact numbered names.
Other factual state attributes are allowed. An empty updates list is valid.
New entries replace the same entity/attribute. Do not emit unchanged facts just to refresh them.
"""


def memory_messages(task: str, observation: str, memory: list[dict]) -> list[dict]:
    return [{"role": "system", "content": UPDATER},
            {"role": "user", "content": canonical({"task": task, "new_observation": observation, "memory": memory})}]


def policy_messages(task: str, observation: str, history: list[dict], memory: list[dict]) -> list[dict]:
    return [{"role": "system", "content": POLICY},
            {"role": "user", "content": canonical({"task": task, "observation": observation,
                                                      "recent_interactions": history, "memory": memory})}]
