#!/usr/bin/env python3
"""Prepare existing reflections needed to inspect unsupported successor states."""
from collections import Counter
from pathlib import Path
from agent_reliability.io import digest, file_hash, read_json, write_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save

PUBLIC = Path('repro/reflexion-20261025')
ROOT = Path('runs/reflexion-four-attempt-validation-20261025')
PRIVATE = Path('runs/reflexion-successor-support-review-20261026')


def main():
    protocol = checked(PUBLIC/'four-attempt-validation-protocol.json')
    evaluation = checked(PUBLIC/'four-attempt-validation-evaluation.json')
    summary = checked(ROOT/'summary.json')
    assert summary['status']=='complete'
    assert evaluation['protocol_sha256']==protocol['sha256']
    assert evaluation['raw_summary_file_sha256']==file_hash(ROOT/'summary.json')
    tasks = set(protocol['secondary_recovery_task_ids'])
    trajectories = [t for t in protocol['trajectories'] if t['panel']=='primary' and t['task_id'] in tasks]
    ids = {t['id'] for t in trajectories}
    assert len(tasks)==27 and len(trajectories)==108
    selected = []; texts = {}
    for row in summary['attempts_rows']:
        if row['trajectory_id'] not in ids or row['trial']>3 or row['success']: continue
        path = Path(row['result_path']); result = read_json(path)
        assert file_hash(path)==row['result_file_sha256']
        assert not result['success'] and result['trial']==row['trial']
        text = result['memory_after'][-1]; h = digest(text)
        assert [h]==row['new_reflection_sha256']
        texts[h] = text
        selected.append({'trajectory_id':row['trajectory_id'], 'task_id':row['task_id'],
                         'condition':row['condition'], 'seed':row['seed'], 'trial':row['trial'],
                         'native_count_before':row['trial']-1, 'reflection_sha256':h,
                         'result_file_sha256':row['result_file_sha256']})
    # Stable text-only order hides task, condition, trial and subsequent outcome.
    hashes = sorted(texts, key=lambda h:digest(['successor-support-review-20261026',h]))
    reading = [{'index':i, 'reflection_sha256':h, 'reflection':texts[h]} for i,h in enumerate(hashes)]
    index = {r['reflection_sha256']:r['index'] for r in reading}
    selected = [{**r, 'index':index[r['reflection_sha256']]} for r in selected]
    PRIVATE.mkdir(parents=True, exist_ok=True)
    write_json(PRIVATE/'reading-view.json',reading)
    output = save(PUBLIC/'successor-support-review-manifest-20261026.json', {
        'scope':'Existing-record development diagnosis; semantic annotation pending',
        'selection':'Use all primary C/N trajectories on the27 tasks fixed for the secondary panel before collection, both original seeds; collect every failed write after trials1..3, never select by later success',
        'selection_rationale':'Inspect native-count0/1/2 write mechanisms and memory contents before attempts3/4 without new inference',
        'protocol_sha256':protocol['sha256'], 'evaluation_sha256':evaluation['sha256'],
        'raw_summary_file_sha256':file_hash(ROOT/'summary.json'),
        'source_sha256':file_hash(Path(__file__)), 'private_reading_file_sha256':file_hash(PRIVATE/'reading-view.json'),
        'selected_trajectories':len(ids), 'selected_tasks':len(tasks), 'selected_scenes':27,
        'selected_write_events':len(selected), 'unique_texts':len(texts),
        'events_by_native_count_before':dict(Counter(r['native_count_before'] for r in selected)),
        'reading_view':'Full reflection text and digest only. Annotator has seen aggregate outcomes; not independent blinded review.',
        'label_schema':'Reuse E/C/B/U/A definitions in shared-calibration-semantic-labels.json; N is an initial constructed plan, not a generated category',
        'selected':selected, 'new_model_calls':0, 'parameter_refits':0,
        'status':'prepared_not_annotated'})
    print({k:output[k] for k in ['sha256','selected_write_events','unique_texts','events_by_native_count_before','status']})


if __name__=='__main__':
    main()
