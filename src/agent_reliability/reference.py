"""Execute the author's ReAct ALFWorld functions; adapters supply I/O only.

No policy loop, new memory updater, recovery policy or prompt edits live here.
The original notebook cells are loaded after checking their fixed source bytes.
"""
from __future__ import annotations
import ast
import json
import os
import re
import sys
from pathlib import Path

from .io import digest, file_hash, read_json, seed_for, write_json
from .llm import CallJournal, HTTPTransport, InfrastructureError

UPSTREAM_COMMIT = '6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9'
NOTEBOOK_SHA = 'bf13b2d204e89d2a078eec19d44b03ebbebb55114322d2573fa070b78316645b'


def author_code(root: Path) -> tuple[dict, dict]:
    receipt = read_json(root / 'receipt.json')
    if receipt['commit'] != UPSTREAM_COMMIT or file_hash(root / 'alfworld.ipynb') != NOTEBOOK_SHA:
        raise ValueError('Author notebook revision mismatch')
    for name, expected in receipt['files'].items():
        if file_hash(root / name) != expected:
            raise ValueError(f'Upstream file changed: {name}')
    functions, selected = [], {}
    for cell in read_json(root / 'alfworld.ipynb')['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in {'process_ob', 'alfworld_run'}:
                functions.append(node)
                selected[node.name] = digest(ast.get_source_segment(source, node))
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                if target == 'prefixes' and isinstance(node.value, ast.Dict):
                    selected['prefixes'] = ast.literal_eval(node.value)
                elif target == 'prompt' and 'Here are two examples.' in (ast.get_source_segment(source, node) or ''):
                    selected['prompt_assignment'] = node
                elif target == 'ob' and "split('\\n\\n')" in (ast.get_source_segment(source, node) or ''):
                    selected['observation_assignment'] = node
    if {f.name for f in functions} != {'process_ob', 'alfworld_run'} or not all(k in selected for k in ['prefixes', 'prompt_assignment', 'observation_assignment']):
        raise ValueError('Expected author ALFWorld cells missing')
    namespace = {'sys': sys}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / 'alfworld.ipynb'), 'exec'), namespace)
    metadata = {'repository': receipt['repository'], 'commit': UPSTREAM_COMMIT,
                'notebook_sha256': NOTEBOOK_SHA,
                'function_source_hashes': {k: selected[k] for k in ['process_ob', 'alfworld_run']}}
    namespace['_selected'] = selected
    return namespace, metadata


def author_inputs(namespace: dict, demos: dict, task_type: str, observations: list[str]) -> tuple[str, str]:
    """Run the author's original prompt/initial-observation assignments verbatim."""
    selected = namespace['_selected']
    v = next(value for key, value in selected['prefixes'].items() if task_type.startswith(key))
    variables = {'d': demos, 'v': v, 'ob': observations}
    nodes = [selected['prompt_assignment'], selected['observation_assignment']]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<author input assignments>', 'exec'), variables)
    return variables['prompt'], variables['ob']


ACTION_ALIAS_PROFILE = 'alfworld_put_in_on_variants_v2'


def submitted_command(action: str) -> str:
    # Public spellings of the same placement operation. Preserve chosen entities;
    # only translate the tool grammar, without consulting state or admissibility.
    match = re.fullmatch(r'put (.+) (?:in/on|in|on) (.+)', action)
    return f'move {match[1]} to {match[2]}' if match else action


def official_environment(data_root: Path, task: dict, seed: int, upstream: Path):
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    root = data_root.resolve();game = (root / task['gamefile']).resolve()
    if not game.is_relative_to(root) or file_hash(game) != task['game_sha256']:
        raise ValueError('Game path/hash mismatch')
    if file_hash(game.with_name('traj_data.json')) != task['trajectory_sha256']:
        raise ValueError('Trajectory hash mismatch')
    config = yaml.safe_load((upstream / 'base_config.yaml').read_text())
    # Evaluation mode prevents expert/planner attachment. Task provenance stays
    # the manifest's actual split (train for development), not the mode's name.
    config['dataset']['eval_ood_data_path'] = str(game.parent)
    instance = AlfredTWEnv(config, train_eval='eval_out_of_distribution')
    if instance.game_files != [str(game)]:
        raise ValueError('Official environment did not select exactly the frozen game')
    env = instance.init_env(batch_size=1)
    env.seed(seed)
    return env


class CompletionTransport(HTTPTransport):
    def complete(self, payload: dict) -> dict:
        return self._request('/completions', payload)


class RecordedBatchEnvironment:
    """Observe calls made by the unchanged author loop; do not choose actions."""
    def __init__(self, env, directory: Path, checkpoint: dict | None):
        self.env, self.directory = env, directory
        self.committed = checkpoint['transitions'] if checkpoint else []
        self.transitions, self.last = [], None

    def step(self, actions):
        if len(actions) != 1:
            raise ValueError('Single-environment batch required')
        action = actions[0];command = submitted_command(action)
        observation, reward, done, info = self.env.step([command])
        public = {'action': action, 'submitted_action': command, 'observation': observation[0],
                  'reward': float(reward[0]), 'done': bool(done[0]), 'won': bool(info['won'][0])}
        index = len(self.transitions)
        if index < len(self.committed) and public != self.committed[index]:
            raise InfrastructureError('Official environment replay differs from committed checkpoint')
        self.transitions.append(public);self.last = public
        if index >= len(self.committed):
            write_json(self.directory / 'steps' / f'{index:04d}.public.json', public)
            # Presence of uncommitted step files is never treated as progress.
            write_json(self.directory / 'checkpoint.json', {'schema_version': 1,
                       'transitions': self.transitions, 'status': 'active'})
        return observation, reward, done, info


def run_reference_episode(env, transport, task, config, model, directory, seed=42, resume=False,
                          retry_uncertain=False):
    namespace, upstream = author_code(Path(config['upstream_dir']))
    identity = {'task': task, 'config': config, 'model': model, 'seed': seed, 'upstream': upstream}
    identity_hash = digest(identity)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'identity.json').exists():
        if not resume or read_json(directory / 'identity.json')['sha256'] != identity_hash:
            raise ValueError('Existing episode; require matching identity and --resume')
    else:
        write_json(directory / 'identity.json', {**identity, 'sha256': identity_hash})
    if (directory / 'result.json').exists():
        return read_json(directory / 'result.json')
    checkpoint = read_json(directory / 'checkpoint.json') if (directory / 'checkpoint.json').exists() else None
    observations, _ = env.reset()
    prompt, ob = author_inputs(namespace, read_json(Path(config['upstream_dir']) / 'alfworld_3prompts.json'),
                              task['task_type'], observations)
    if checkpoint is None:
        write_json(directory / 'checkpoint.json', {'schema_version': 1, 'transitions': [], 'status': 'active'})
    journal = CallJournal(directory / 'calls', transport, 49, retry_uncertain)
    batch = RecordedBatchEnvironment(env, directory, checkpoint)
    decision = 0

    def llm(prompt, stop):
        nonlocal decision
        payload = {'model': model['served_name'], 'prompt': prompt, 'temperature': 0,
                   'top_p': 1, 'max_tokens': 100, 'frequency_penalty': 0.0,
                   'presence_penalty': 0.0, 'stop': stop,
                   'seed': seed_for(task['task_id'], seed, decision, 'author_react')}
        logical = f'decision-{decision:04d}';saved = journal.complete(logical, payload)
        try:
            choice = saved['response']['choices'][0];text, finish = choice['text'], choice['finish_reason']
            if not isinstance(text, str):
                raise ValueError('Not text')
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InfrastructureError('Malformed completion response') from exc
        write_json(journal.directory / logical / 'validation.json', {
            'valid': bool(text.strip()), 'truncated_output': finish == 'length',
            'finish_reason': finish, 'prompt_sha256': digest(prompt)})
        decision += 1
        # Exactly the upstream contract: return text. Do not remove prompt
        # markers, repair decisions or turn truncated thoughts into new failures.
        return text

    namespace['env'], namespace['llm'] = batch, llm
    won = bool(namespace['alfworld_run'](prompt, to_print=False, ob=ob))
    if len(batch.transitions) < len(batch.committed):
        raise InfrastructureError('Resumed execution ended before committed checkpoint')
    status = 'success' if won else ('environment_failure' if batch.last and batch.last['done'] else 'timeout_decision_budget')
    write_json(directory / 'checkpoint.json', {'schema_version': 1, 'transitions': batch.transitions, 'status': status})
    result = {'task_id': task['task_id'], 'scene_id': task['scene_id'], 'task_type': task['task_type'],
              'condition': 'author_react_base', 'seed': seed, 'success': won, 'status': status,
              'decisions': decision, 'env_calls_including_thoughts': len(batch.transitions),
              'actual_commands_excluding_thoughts': sum(not x['action'].startswith('think:') for x in batch.transitions),
              'identity_sha256': identity_hash, 'trajectory_sha256': digest(batch.transitions), **journal.totals()}
    write_json(directory / 'result.json', result)
    return result
