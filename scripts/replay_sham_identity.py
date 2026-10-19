#!/usr/bin/env python3
"""Replay recorded memory trajectories with the real backend and sham hook.

No model calls: checks hook-to-input invariance under fixed recorded actions.
It does not replace an independently generated sham rollout.
"""
import argparse
from pathlib import Path
from agent_reliability.cli import backend
from agent_reliability.io import digest, read_json, write_json
from agent_reliability.memory import Memory
from agent_reliability.react import environment_action, process_observation, react_messages


def replay(root, data_root, output):
    meta = read_json(root / 'run.json');config = meta['config']
    demos = read_json(Path(config['demonstrations']));rows = []
    for ep in sorted(root.glob('episodes/*-memory')):
        if not (ep / 'result.json').exists():
            raise ValueError('All memory episodes must complete before replay audit')
        identity = read_json(ep / 'identity.json');checkpoint = read_json(ep / 'checkpoint.json')
        env = backend(data_root);env.reset(identity['task'], identity['seed'])
        history, hook = [], None
        count = 0
        try:
            for p in sorted((ep / 'steps').glob('*.public.json')):
                frame = read_json(p);ev = read_json(p.with_name(p.name.replace('.public.', '.evaluator.')))
                if env.current.facts != ev['facts_before']:
                    raise ValueError('Real facts differ from recorded prefix')
                memory = Memory(config['memory_capacity'], frame['memory_after']);before = memory.export()
                if hook is None and config['sham_start_step'] <= frame['env_step'] <= config['sham_last_step']:
                    hook = memory.sham_verified_location(env.current.facts)
                if memory.export() != before:
                    raise ValueError('Sham changed memory')
                messages = react_messages(identity['task']['task_type'], checkpoint['initial_observation'], history, demos, memory.export())
                if digest(messages) != frame['policy_prompt_sha256']:
                    raise ValueError('Sham replay changed policy input')
                if frame['is_thought']:
                    feedback = 'OK.'
                else:
                    command = environment_action(frame['action'])
                    if command != frame['submitted_action']:
                        raise ValueError('Action alias changed')
                    feedback = process_observation(env.step(command).text)
                if feedback != frame['following_observation']:
                    raise ValueError('Real environment output changed')
                history.append({'action': frame['action'], 'observation': feedback});count += 1
            rows.append({'task_id': identity['task']['task_id'], 'passed': True, 'decisions_checked': count,
                         'sham_applied': hook is not None, 'trajectory_sha256': digest(history)})
        finally:
            env.close()
    result = {'scope': 'real backend + recorded responses/actions; no new model inference or success estimate',
              'tasks': rows, 'all_passed': bool(rows) and all(x['passed'] for x in rows),
              'sham_applied_tasks': sum(x['sham_applied'] for x in rows), 'llm_dispatches': 0}
    write_json(output, result);print({'passed': result['all_passed'], 'tasks': len(rows), 'hooks': result['sham_applied_tasks']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True);p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True);a = p.parse_args()
    replay(a.source, a.data_root, a.output)
