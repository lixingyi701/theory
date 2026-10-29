#!/usr/bin/env python3
"""Observable procedural chains from accepted public object-handling feedback."""
import argparse
import re
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reference import submitted_command
from summarize_reflexion import costs
from analyze_reflexion_pilot_forks import object_type
from run_reflexion_precondition_forks import repair_text


def procedure_events(steps, goal_object, verb, device):
    held = None
    last_handling, last_placement_attempt = {}, {}
    wrong_chains, successful_macros, other_failures, competing, unexplained = [], [], [], [], []
    placement_attempts = []
    target_picked_up = False
    for index, step in enumerate(steps):
        action, feedback = step['action'], step['observation']
        command = step.get('submitted_action', submitted_command(action))
        placement = re.fullmatch(r'move ([a-z]+ \d+) to '+re.escape(device), command)
        if placement and placement[1].split()[0] == goal_object:
            event = {'index': index, 'action': action, 'observation': feedback, 'held_before': held,
                     'accepted': feedback.startswith(f'You move the {placement[1]} to the {device}')}
            placement_attempts.append(event); last_placement_attempt[placement[1]] = event
        macro = re.fullmatch(r'(heat|cool) ([a-z]+ \d+) with ([a-z]+ \d+)', action)
        if macro:
            operation, obj, appliance = macro.groups()
            event = {'index': index, 'action': action, 'observation': feedback, 'held_before': held}
            if operation != verb or obj.split()[0] != goal_object or appliance != device:
                competing.append(event)
            elif feedback.startswith(f'You {verb} the {obj} using the {device}') and held == obj:
                previous = last_placement_attempt.get(obj)
                event['preceding_rejected_placement_index'] = previous['index'] if previous and not previous['accepted'] else None
                successful_macros.append(event)
            elif feedback.startswith(f'You {verb} the {obj} using the {device}'):
                unexplained.append(event)
            elif feedback == 'Nothing happens.' and held != obj and last_handling.get(obj, {}).get('destination') == device:
                wrong_chains.append({**event, 'accepted_placement_index': last_handling[obj]['index']})
            elif feedback == 'Nothing happens.':
                other_failures.append(event)
        take = re.match(r'You pick up the ([a-z]+ \d+) from the ([a-z]+ \d+)\.', feedback)
        move = re.match(r'You (?:move|put) the ([a-z]+ \d+) (?:to|in/on|in|on) the ([a-z]+ \d+)\.', feedback)
        if take:
            held = take[1]
            last_handling[held] = {'source': take[2], 'index': index}
            target_picked_up |= held.split()[0] == goal_object
        elif move:
            last_handling[move[1]] = {'destination': move[2], 'index': index}
            if held == move[1]:
                held = None
    return {'target_picked_up': target_picked_up, 'wrong_precondition_chains': wrong_chains,
            'target_device_placement_attempts': placement_attempts,
            'held_successful_target_macros': successful_macros,
            'other_failed_target_macros': other_failures, 'competing_macros': competing,
            'unexplained_successful_macros': unexplained}


def main():
    parser = argparse.ArgumentParser()
    for name in ['run', 'targets', 'parent-run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    registry = read_json(args.targets); identity = read_json(args.run/'run.json')['identity']
    if registry['sha256'] != identity['targets_sha256'] or registry['sha256'] != digest({k: v for k, v in registry.items() if k != 'sha256'}):
        raise ValueError('Registry identity mismatch')
    reports = []
    for target in registry['targets']:
        parent_file = args.parent_run/'tasks'/target['task_id']/'trial-00/result.json'
        if file_hash(parent_file) != target['parent_result_sha256']:
            raise ValueError('Parent prefix changed')
        source = read_json(parent_file)['memory_after'][-1]
        corrected = repair_text(source, target)
        fragments = [edit['original_text'] for edit in target.get('repairs', [target])]
        candidate = read_json(Path(target['probe_file']))['candidate']
        prompt = read_json(parent_file.parent/'calls/actor-0000/attempt-000.request.json')['payload']['prompt']
        wrapper = read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
        raw = prompt[len(wrapper['prefix']):-len(wrapper['suffix'])]
        initial = raw.rsplit('Here is the task:\n', 1)[1][:-2]
        if digest(initial) != read_json(parent_file)['initial_observation_sha256']:
            raise ValueError('Public task extraction changed')
        goal_object = object_type(initial.rsplit('Your task is to: ', 1)[1])
        verb, device = candidate['verb'], candidate['device']
        if goal_object != candidate['object'].split()[0]:
            raise ValueError('Precondition probe used a non-goal object')
        branches = {}
        for branch in ['original', 'sham', 'retain_error', 'correct_target', 'discard_new']:
            root = args.run/'targets'/target['task_id']/'branches'/branch
            attempts = []
            for path in sorted(root.glob('trial-*/result.json')):
                result = read_json(path)
                steps = read_json(path.parent/'checkpoint.json')['transitions']
                new = result['memory_after'][len(result['memory_before']):]
                attempts.append({'trial': result['trial'], 'success': result['success'], 'status': result['status'],
                    'decisions': result['decisions'], 'trajectory_sha256': result['trajectory_sha256'],
                    'native_error_entry_visible': source in result['memory_before'][-3:],
                    'target_wrong_fragment_visible': any(target['original_text'] in m for m in result['memory_before'][-3:]),
                    'target_wrong_fragments_visible': [any(fragment in m for m in result['memory_before'][-3:]) for fragment in fragments],
                    'repaired_entry_visible': corrected in result['memory_before'][-3:],
                    'new_reflection_sha256': [digest(m) for m in new],
                    **procedure_events(steps, goal_object, verb, device)})
            complete = any(r['success'] for r in attempts) or len(attempts) == 2
            branches[branch] = {'completed': complete, 'success': any(r['success'] for r in attempts) if complete else None,
                                'attempts': attempts, 'costs': costs(root)}
        reports.append({'task_id': target['task_id'], 'target_object': goal_object, 'operation': verb,
                        'device': device, 'branches': branches})
    report = {'scope': 'Observed accepted placement followed by target macro without retrieving the object; no latent-belief label',
        'targets_sha256': registry['sha256'], 'driver_sha256': file_hash(Path(__file__)),
        'independent_tasks': len(reports), 'completed_targets': sum(all(b['completed'] for b in t['branches'].values()) for t in reports),
        'targets': reports, 'costs': costs(args.run)}
    report['sha256'] = digest(report); write_json(args.output, report)
    print({k: v for k, v in report.items() if k != 'targets'})


if __name__ == '__main__':
    main()
