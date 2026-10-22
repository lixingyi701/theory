#!/usr/bin/env python3
"""Review candidates using native first attempts, never their future branches.

A preview proposes native advice and its public replay. Freezing additionally
requires a reviewed exact edit for each accepted target. The review file does
not contain future outcomes; all parent tasks retain their coverage status.
"""
import argparse
import re
import time
from pathlib import Path

from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reflexion import ROOT
from analyze_reflexion_pilot_forks import object_type
from probe_reflexion_preconditions import find_candidate, replay_probe
from run_reflexion_precondition_forks import repair_text


RULE = {'stratum': 'native_false_holding_precondition', 'parent_trial': 0, 'max_targets': 4,
        'order': 'Frozen thermal manifest order; first four independently reviewed native errors supported by public replay',
        'evidence': 'Accepted goal-object placement followed by failed thermal macro, native advice recommending placement before that macro, and held/placed/retrieved public replay',
        'excludes': ['future branch outcomes', 'evaluator facts', 'expert plans', 'non-goal-object failures', 'negated or retrospective descriptions mislabeled as advice']}


def native_advice_spans(text, device_type):
    proposals = []
    patterns = [r'\b(?:put|place|placed|putting|placing)\b[^.!?\n]{0,85}?\b',
                r'\b(?:ensure|confirm|verify|check|ensured|confirmed|verified|checked|make sure|made sure)\b'
                r'[^.!?\n]{0,85}?\b(?:is|was|be|been|being)\b[^.!?\n]{0,40}?\b(?:in|inside|into|within)\s+(?:the\s+)?']
    # Mask abbreviation periods without moving native character offsets.
    masked = re.sub(r'\b(?:e\.g\.|i\.e\.)', lambda m: m[0].replace('.', '\u00b7'), text, flags=re.I)
    for boundary in re.finditer(r'[^.!?\n]+[.!?]?', masked):
        sentence = text[boundary.start():boundary.end()]
        for pattern in patterns:
            for placement in re.finditer(pattern+re.escape(device_type)+r'(?:\s+\d+)?\b', sentence, re.I):
                before = sentence[:placement.start()]
                modals = list(re.finditer(r'\b(?:should|will|must|need to)\b', before, re.I))
                if not modals:
                    continue
                operative = before[modals[-1].start():]+placement[0]
                if re.search(r'\b(?:not|never|avoid|without|instead|rather than)\b', operative, re.I):
                    continue
                start = boundary.start()+placement.start()
                end = boundary.start()+placement.end()
                proposals.append({'span': [start, end], 'text': text[start:end],
                                  'sentence_excerpt': sentence.strip()})
    return proposals


def public_goal_object(parent, result):
    request = read_json(parent/'calls/actor-0000/attempt-000.request.json')['payload']['prompt']
    wrapper = read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
    raw = request[len(wrapper['prefix']):-len(wrapper['suffix'])]
    initial = raw.rsplit('Here is the task:\n', 1)[1][:-2]
    if digest(initial) != result['initial_observation_sha256']:
        raise ValueError('Public task extraction mismatch')
    return object_type(initial.rsplit('Your task is to: ', 1)[1])


def main():
    parser = argparse.ArgumentParser()
    for name in ['parent-run', 'manifest', 'evidence-dir', 'output']:
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--reviewed-edits', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep existing target evidence')
    manifest = load_manifest(args.manifest)
    summary = read_json(args.parent_run/'summary.json')
    if summary['completed_tasks'] != len(manifest['tasks']):
        raise ValueError('Complete all prospective prefixes first')
    reviewed = read_json(args.reviewed_edits) if args.reviewed_edits else {}
    edits, rejected = reviewed.get('edits', {}), reviewed.get('rejected', {})
    coverage, targets, previews = [], [], []
    for task in manifest['tasks']:
        parent = args.parent_run/'tasks'/task['task_id']/'trial-00'
        result = read_json(parent/'result.json')
        row = {'task_id': task['task_id']}
        if result['success']:
            row['status'] = 'first_attempt_success_no_reflection'; coverage.append(row); continue
        steps = read_json(parent/'checkpoint.json')['transitions']
        candidate = find_candidate(steps)
        if candidate is None:
            row['status'] = 'no_observed_placement_before_macro_failure'; coverage.append(row); continue
        if candidate['object'].split()[0] != public_goal_object(parent, result):
            row['status'] = 'competing_non_goal_object'; coverage.append(row); continue
        native = result['memory_after'][-1]
        proposals = native_advice_spans(native, candidate['device'].split()[0])
        if not proposals:
            row['status'] = 'no_candidate_native_placement_advice'; coverage.append(row); continue
        probe_file = args.evidence_dir/(task['task_id']+'-replay.json')
        if probe_file.exists():
            probe = read_json(probe_file)
            if probe['parent_result_sha256'] != file_hash(parent/'result.json') or probe['sha256'] != digest({k: v for k, v in probe.items() if k != 'sha256'}):
                raise ValueError('Probe identity mismatch')
        else:
            probe = replay_probe(task, parent, candidate); write_json(probe_file, probe)
        if not probe['holding_precondition_supported']:
            row['status'] = 'holding_premise_not_supported_by_replay'; coverage.append(row); continue
        previews.append({'task_id': task['task_id'], 'proposals': proposals,
                         'probe_file': str(probe_file), 'probe_sha256': probe['sha256']})
        if task['task_id'] in rejected:
            if not rejected[task['task_id']]:
                raise ValueError('A rejected candidate needs a contextual reason')
            row['status'] = 'text_review_rejected'; row['reason'] = rejected[task['task_id']]
            coverage.append(row); continue
        edit = edits.get(task['task_id'])
        if edit is None:
            row['status'] = 'publicly_supported_pending_text_review'; coverage.append(row); continue
        if not edit['reviewed_native_advice']:
            raise ValueError('Native advice needs contextual review')
        repairs = edit.get('repairs', [edit])
        normative = []
        for change in repairs:
            within = any(change['span'][0] >= p['span'][0] and change['span'][1] <= p['span'][1] for p in proposals)
            if within:
                normative.append(change)
            elif not change.get('coreferent_same_precondition_reviewed') or not change.get('review_note'):
                raise ValueError('An additional predicate reference needs its own review')
        if not normative:
            raise ValueError('At least one edit must correct the proposed native advice')
        repair_text(native, {'repairs': repairs})
        primary = normative[0]
        key = 'react_heat_1' if candidate['verb'] == 'heat' else 'react_cool_1'
        demos = read_json(ROOT/'prompts/alfworld_3prompts.json')
        target = {'task_id': task['task_id'], 'parent_trial': 0, 'stratum': RULE['stratum'],
            'original_truth': 'error', 'parent_result_sha256': file_hash(parent/'result.json'),
            'span': primary['span'], 'original_text': primary['original_text'], 'corrected_text': primary['corrected_text'],
            'repairs': repairs, 'precondition_references_repaired': len(repairs),
            'repair_scope': 'All reviewed explicit references to one inside-device precondition; other advice unchanged',
            'public_evidence_reviewed': True, 'probe_file': str(probe_file), 'probe_sha256': probe['sha256'],
            'shared_demo_key': key, 'shared_demo_sha256': digest(demos[key]),
            'unchanged_competing_advice': edit['unchanged_competing_advice'], 'review': edit['review_note']}
        target['sha256'] = digest(target)
        if len(targets) < 4:
            targets.append(target); row['status'] = 'eligible_selected'
        else:
            row['status'] = 'eligible_beyond_declared_limit'
        coverage.append(row)
    if args.reviewed_edits and any(r['status'] == 'publicly_supported_pending_text_review' for r in coverage):
        raise ValueError('Review all supported candidate texts before freezing')
    report = {'scope': 'prospective thermal development native-error forks', 'manifest_sha256': manifest['sha256'],
              'selection_rule': RULE, 'selector_sha256': file_hash(Path(__file__)), 'coverage': coverage,
              'previews': previews, 'targets': targets,
              'reviewed_edits_sha256': file_hash(args.reviewed_edits) if args.reviewed_edits else None,
              'frozen_unix': time.time() if args.reviewed_edits else None}
    report['sha256'] = digest(report); write_json(args.output, report)
    print({'targets': len(targets), 'review_candidates': len(previews), 'frozen': bool(args.reviewed_edits), 'sha256': report['sha256']})


if __name__ == '__main__':
    main()
