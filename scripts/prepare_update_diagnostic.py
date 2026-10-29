#!/usr/bin/env python3
"""Select existing first-failure texts for offline diagnosis, without model calls."""
from pathlib import Path
from collections import Counter
from agent_reliability.io import read_json, file_hash, digest, write_json
from prepare_shared_calibration import save
from prepare_shared_validation import PROTOCOL
from run_shared_validation import ROOT, verify_protocol

PUBLIC = PROTOCOL.parent
PRIVATE = Path('runs/reflexion-update-diagnostic-20261025')


def main():
    protocol = verify_protocol()
    summary = read_json(ROOT / 'summary.json')
    frozen = read_json(PUBLIC / 'shared-validation-results.json')
    assert frozen['raw_summary_file_sha256'] == file_hash(ROOT / 'summary.json')
    assert summary['status'] == 'complete'
    outcomes = {r['trajectory_id']: r for r in frozen['observed']}
    selected, reviews = [], []
    rows = sorted(summary['attempts_rows'], key=lambda r: (r['trajectory_id'], r['trial']))
    for row in rows:
        if row['trial'] != 1 or row['success']:
            continue
        outcome = outcomes[row['trajectory_id']]
        if row['condition'] not in ['correct_latest', 'delete_latest'] and outcome['scene_id'] != 'FloorPlan26':
            continue
        path = Path(row['result_path'])
        assert file_hash(path) == row['result_file_sha256']
        result = read_json(path)
        reflection = result['memory_after'][-1]
        assert [digest(reflection)] == row['new_reflection_sha256']
        index = len(reviews)
        selected.append({
            'index': index, 'trajectory_id': row['trajectory_id'], 'task_id': row['task_id'],
            'scene_id': outcome['scene_id'], 'condition': row['condition'],
            'reflection_sha256': digest(reflection), 'result_file_sha256': row['result_file_sha256'],
        })
        # The reading view deliberately omits second-attempt outcomes.
        reviews.append({'index': index, 'reflection_sha256': digest(reflection), 'reflection': reflection})
    assert len({r['reflection_sha256'] for r in selected}) == len(selected)
    for condition in ['correct_latest', 'delete_latest']:
        expected = sum(r['trial'] == 1 and not r['success'] and r['condition'] == condition for r in rows)
        assert sum(r['condition'] == condition for r in selected) == expected == 53
    PRIVATE.mkdir(parents=True, exist_ok=True)
    write_json(PRIVATE / 'reading-view.json', reviews)
    manifest = save(PUBLIC / 'update-diagnostic-manifest.json', {
        'scope': 'Post-collection existing-record diagnosis; no new Agent attempts and no coefficient fitting',
        'selection': 'All first failures in correct_latest/delete_latest, plus all first failures in FloorPlan26; deduplicated union',
        'selection_rationale': 'Deletion second-attempt underprediction and negative primary scene score identified after validation',
        'reading_view': 'Full text and content hash; second-attempt outcomes omitted during annotation',
        'protocol_sha256': protocol['sha256'], 'frozen_results_sha256': frozen['sha256'],
        'raw_summary_file_sha256': file_hash(ROOT / 'summary.json'),
        'private_reading_file_sha256': file_hash(PRIVATE / 'reading-view.json'),
        'preparation_source_sha256': file_hash(Path(__file__)),
        'selected': selected, 'counts_by_condition': dict(Counter(r['condition'] for r in selected)),
        'total_validation_reflections': 610, 'selected_reflections': len(selected),
        'new_model_calls': 0, 'parameter_refits': 0,
    })
    # Inspect successful deletion traces as cases, not as an unbiased population sample.
    cases = []
    for row in rows:
        if row['condition'] == 'delete_latest' and row['trial'] == 2 and row['success']:
            path = Path(row['result_path'])
            checkpoint = path.with_name('checkpoint.json')
            value = read_json(checkpoint)
            cases.append({'trajectory_id': row['trajectory_id'], 'task_id': row['task_id'],
                          'scene_id': outcomes[row['trajectory_id']]['scene_id'],
                          'result_file_sha256': row['result_file_sha256'],
                          'checkpoint_sha256': file_hash(checkpoint), 'transitions': value['transitions']})
    assert len(cases) == 7
    write_json(PRIVATE / 'successful-deletion-cases.json', cases)
    print({'selected': len(selected), 'conditions': manifest['counts_by_condition'], 'sha256': manifest['sha256']})


if __name__ == '__main__':
    main()
