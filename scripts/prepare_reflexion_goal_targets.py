#!/usr/bin/env python3
"""Select native plan destination bindings from public first-attempt prefixes.

This stratum is distinct from an observed object-location fact. The script never
reads future native attempts or intervention outcomes. Review its public evidence
before freezing a registry with --freeze.
"""
import argparse
import re
import time
from pathlib import Path

from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, write_json
from continue_reflexion_pilot import PILOT, MANIFEST


DESTINATIONS = ['diningtable', 'countertop', 'desk', 'shelf', 'sidetable',
                'fridge', 'cabinet', 'drawer', 'sinkbasin', 'garbagecan']
RULE = {'stratum': 'native_plan_goal_destination', 'parent_trial': 0, 'max_targets': 4,
        'order': 'frozen pilot manifest order, first eligible destination clause in native reflection',
        'source': 'public initial task plus newly generated native reflection only',
        'replacement_order': DESTINATIONS,
        'intervention': 'Replace one destination entity in one native put/place clause; keep any replacement instance publicly valid and preserve all other bytes',
        'selection_excludes': ['future attempts', 'branch outcomes', 'evaluator facts', 'expert plans']}


def candidate(task):
    directory = PILOT/'tasks'/task['task_id']/'trial-00'
    if not (directory/'result.json').exists():
        return None, 'first_attempt_pending'
    result = read_json(directory/'result.json')
    if result['success']:
        return None, 'first_attempt_success_no_reflection'
    native = result['memory_after'][-1]
    request = directory/'calls/actor-0000/attempt-000.request.json'
    prompt = read_json(request)['payload']['prompt']
    wrapper = read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
    if not prompt.startswith(wrapper['prefix']) or not prompt.endswith(wrapper['suffix']):
        raise ValueError('Unexpected first prompt wrapper')
    raw = prompt[len(wrapper['prefix']):-len(wrapper['suffix'])]
    initial = raw.rsplit('Here is the task:\n', 1)[1]
    if not initial.endswith('\n>'):
        raise ValueError('Unexpected public initial observation boundary')
    initial = initial[:-2]
    if digest(initial) != result['initial_observation_sha256']:
        raise ValueError('Public observation hash mismatch')
    goal = initial.rsplit('Your task is to: ', 1)[1]
    match = re.search(r'\b(?:in|on) ([a-z]+)\.$', goal)
    if not match:
        return None, 'no_placement_destination'
    destination = match.group(1)
    match = re.search(r'\b(?:put|place)\b[^.!?\n]{0,85}?\b(?:in|on|into|onto)\s+(?:the\s+)?('+re.escape(destination)+r'(?:\s+\d+)?)\b', native, re.I)
    if not match:
        return None, 'no_correct_destination_clause_in_native_plan'
    alternatives = [x for x in DESTINATIONS if x != destination and re.search(r'\ba '+x+r' \d+\b', initial)]
    if not alternatives:
        return None, 'no_public_alternative_receptacle'
    start, end = match.span(1)
    replacement = alternatives[0]
    if len(native[start:end].split()) > 1:
        if not re.search(r'\ba '+re.escape(native[start:end])+r'\b', initial):
            return None, 'native_destination_instance_not_publicly_valid'
        ids = re.findall(r'\ba '+replacement+r' (\d+)\b', initial)
        replacement += ' '+min(ids, key=int)
    target = {'task_id': task['task_id'], 'parent_trial': 0, 'stratum': RULE['stratum'],
        'parent_result_sha256': file_hash(directory/'result.json'),
        'span': [start, end], 'original_text': native[start:end], 'replacement_text': replacement,
        'public_evidence_reviewed': False,
        'public_evidence': [{'source': 'initial_public_task', 'request_sha256': file_hash(request),
            'initial_observation_sha256': result['initial_observation_sha256'], 'goal': goal,
            'goal_destination': destination, 'correct_destination': native[start:end], 'wrong_destination': replacement}],
        'native_clause': match.group(0), 'native_destination_mentions': len(re.findall(r'\b'+destination+r'\b', native, re.I)),
        'selection': RULE['order'], 'false_claim': f'Goal destination is {replacement}',
        'intervention': RULE['intervention']}
    return target, 'eligible'


def main():
    global PILOT,MANIFEST
    p = argparse.ArgumentParser()
    p.add_argument('--parent-run',type=Path,default=PILOT)
    p.add_argument('--manifest',type=Path,default=MANIFEST)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--freeze', action='store_true')
    a = p.parse_args()
    PILOT,MANIFEST=a.parent_run,a.manifest
    if a.output.exists() and (a.freeze or read_json(a.output)['frozen_unix'] is not None):
        raise FileExistsError('Target registries are immutable')
    manifest = load_manifest(MANIFEST)
    targets, coverage = [], []
    for task in manifest['tasks']:
        target, reason = candidate(task)
        coverage.append({'task_id': task['task_id'], 'status': reason})
        if target and len(targets) < RULE['max_targets']:
            target['public_evidence_reviewed'] = a.freeze
            target['sha256'] = digest(target)
            targets.append(target)
    registry = {'manifest_sha256': manifest['sha256'], 'selection_rule': RULE,
        'selector_sha256': file_hash(Path(__file__)),
        'frozen_unix': time.time() if a.freeze else None, 'coverage': coverage, 'targets': targets}
    if a.freeze and any(r['status'] == 'first_attempt_pending' for r in coverage):
        raise ValueError('Finish first-attempt cohort coverage before freezing')
    registry['sha256'] = digest(registry)
    write_json(a.output, registry)
    print({'targets': len(targets), 'sha256': registry['sha256'], 'frozen': a.freeze})
    for target in targets:
        print({k: target[k] for k in ['task_id', 'original_text', 'replacement_text', 'native_clause', 'public_evidence']})


if __name__ == '__main__':
    main()
