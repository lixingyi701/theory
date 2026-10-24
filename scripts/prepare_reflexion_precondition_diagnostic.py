#!/usr/bin/env python3
"""Freeze one development target after public replay, before future branches."""
import argparse
import time
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from agent_reliability.reflexion import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Target registry immutable')
    manifest = Path('repro/reflexion-20261023/pilot-prefix24-v2.json')
    parent = Path('runs/reflexion-normalized-prefix24-20261023/tasks/762a7487ace813d7/trial-00')
    probe_file = Path('repro/reflexion-20261023/precondition-diagnostic-762.json')
    result = read_json(parent/'result.json'); probe = read_json(probe_file)
    if result['success'] or not probe['holding_precondition_supported']:
        raise ValueError('Failed native attempt and successful public premise test required')
    text = result['memory_after'][-1]
    original = 'placed it in the microwave'
    corrected = 'kept hold of it near the microwave'
    if text.count(original) != 1:
        raise ValueError('Reviewed native fragment changed')
    start = text.index(original)
    demos = read_json(ROOT/'prompts/alfworld_3prompts.json')
    target = {'task_id': result['task_id'], 'parent_trial': 0,
        'stratum': 'native_false_holding_precondition', 'original_truth': 'error',
        'parent_result_sha256': file_hash(parent/'result.json'),
        'span': [start, start+len(original)], 'original_text': original, 'corrected_text': corrected,
        'public_evidence_reviewed': True, 'probe_file': str(probe_file), 'probe_sha256': probe['sha256'],
        'shared_demo_key': 'react_heat_1', 'shared_demo_sha256': digest(demos['react_heat_1']),
        'unchanged_competing_advice': 'The later stoveburner fallback is retained exactly as in the native reflection',
        'review': 'The native reflection recommends placement before the macro. Public paired replay succeeds while held, fails after placement, and succeeds after retrieval, with both lid states tested.'}
    target['sha256'] = digest(target)
    registry = {'scope': 'selected development procedural-premise diagnostic; prospective thermal cohort remains separate',
        'selection': 'Observed native wrong premise in an unused failed prefix; chosen from development evidence, not future branch effects',
        'manifest_sha256': read_json(manifest)['sha256'], 'frozen_unix': time.time(),
        'selector_sha256': file_hash(Path(__file__)), 'targets': [target]}
    registry['sha256'] = digest(registry); write_json(args.output, registry)
    print({'targets': 1, 'sha256': registry['sha256']})


if __name__ == '__main__':
    main()
