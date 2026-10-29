#!/usr/bin/env python3
"""Prepare genuine native-error parents; does not edit memories or run inference."""
from pathlib import Path
from collections import defaultdict, Counter
from agent_reliability.io import digest, file_hash, read_json, write_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from diagnose_second_attempt import raw_index, read_attempt

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-recovery-mechanism-20261029')


def main():
    output=PUBLIC/'recovery-mechanism-parents-20261029.json'
    assert not output.exists(), 'Preserve prepared parents'
    prior=checked(PUBLIC/'stage-calibration-protocol-20261029.json')
    rows=read_json(Path('runs/reflexion-seven-parameter-development-20261027/input-events.json'))
    assert digest(rows)==prior['input_rows_sha256']
    pool=[r for r in rows if r['trial']==1 and r['condition']=='neutral_full' and not r['success'] and r['outcome']=='E']
    assert len(pool)==70
    grouped=defaultdict(list)
    for row in pool: grouped[row['task_id']].append(row)
    selected=[min(v,key=lambda r:digest([r['trajectory_id'],r['trial']])) for task,v in sorted(grouped.items())]
    assert len(selected)==51 and len({r['scene_id'] for r in selected})==23
    raw,_=raw_index(); reading=[]; public=[]
    for index,row in enumerate(selected):
        assert len(row['state'])==3 and all(s=={'type':'N','native':False} for s in row['state'])
        rr,path,result,checkpoint,identity=read_attempt(row,raw)
        assert not result['success'] and len(result['memory_after'])==len(result['memory_before'])+1
        window=result['memory_after'][-3:]
        assert len(window)==3 and window[-1].strip()
        record=dict(index=index,task_id=row['task_id'],scene_id=row['scene_id'],source=row['source'],
            parent_failure_id=[row['trajectory_id'],1],parent_result_file_sha256=file_hash(path),
            memory_sha256=digest(window),native_text_sha256=digest(window[-1]),
            fork_trial=2,state=[{'type':'N','native':False},{'type':'N','native':False},{'type':'E','native':True}],
            initial_observation_sha256=result['initial_observation_sha256'])
        public.append(record);reading.append(dict(record,memory=window,native_text=window[-1]))
    PRIVATE.mkdir(parents=True,exist_ok=True);write_json(PRIVATE/'parent-reading.json',reading)
    result=save(output,dict(scope='Prepared first-failure native-E parent pool, not a completed intervention',
        selection='First-attempt neutral_full failure with new reflection previously labelled E; one identity-only minimum per task, no later-outcome filter',
        eligible_records=70,tasks=51,scenes=23,by_source=dict(Counter(r['source'] for r in public)),
        cases=public,private_reading_sha256=file_hash(PRIVATE/'parent-reading.json'),
        source_sha256=file_hash(Path(__file__)),development_input_sha256=digest(rows),
        status='pending_full_text_edits_interface_precision_and_inference_freeze',
        semantic_review='Prior single-Codex E labels; new full-text edit coverage not yet reviewed',
        new_model_calls=0,parameter_fits=0,holdout134_calls=0))
    write_json(PRIVATE/'preparation.json',dict(status=result['status'],parents_sha256=result['sha256'],
        scope='Parent preparation only; no inference allowance or started GPU job',new_model_calls=0,parameter_fits=0))
    print(dict(manifest_sha256=result['sha256'],tasks=51,scenes=23,new_model_calls=0),flush=True)


if __name__=='__main__': main()
