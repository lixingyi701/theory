#!/usr/bin/env python3
"""Prepare same-parent reflection edits without examining new branch outcomes."""
from pathlib import Path
from collections import defaultdict
import argparse
from agent_reliability.io import digest, file_hash, read_json, write_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from diagnose_second_attempt import ROOTS, raw_index
from calibrate_successor_success import features

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-content-intervention-20261028')
PARENT=Path('runs/reflexion-second-attempt-diagnostic-20261028')
MANIFEST=PUBLIC/'content-intervention-manifest-20261028.json'


def select():
    assert not MANIFEST.exists() and not (PRIVATE/'preparation.json').exists(),'Preserve original parent selection'
    result=checked(PUBLIC/'second-attempt-results-20261028.json')
    closed=checked(PUBLIC/'second-attempt-closeout-20261028.json');assert closed['status']=='complete'
    assert result['sha256']==closed['result_sha256']
    pool=result['future_intervention_parent_pool'];all_rows=read_json(PARENT/'input-rows.json')
    by_id={(r['trajectory_id'],r['trial']):r for r in all_rows};raw,_=raw_index()
    grouped=defaultdict(list)
    for identity in pool['attempt_ids']:
        r=by_id[tuple(identity)];assert r['trial']==2
        grouped[r['task_id']].append(r)
    # The sorting key contains identities only, not any result/outcome hash.
    chosen=[min(group,key=lambda r:digest([r['trajectory_id'],r['trial']])) for task,group in sorted(grouped.items())]
    assert len(chosen)==46 and len(pool['attempt_ids'])==59
    reading=[];selected=[]
    for i,r in enumerate(chosen):
        prior=raw[r['source'],r['trajectory_id'],1];path=Path(prior['result_path'])
        assert file_hash(path)==prior['result_file_sha256']
        rr=read_json(path);identity=checked(path.parent/'identity.json');memory=rr['memory_after'][-3:]
        assert not rr['success'] and identity['task']['task_id']==r['task_id']
        assert len(memory)==len(r['state']) and sum(s['native'] for s in r['state'])==1
        slot=next(j for j,s in enumerate(r['state']) if s['native'])
        assert r['state'][slot]['type'] in ['C','U'] and all(s['type'] not in ['E','B','A'] for s in r['state'])
        history=[by_id[r['trajectory_id'],j]['state'] for j in range(1,3)]
        row={'index':i,'task_id':r['task_id'],'scene_id':r['scene_id'],'parent_attempt_id':[r['trajectory_id'],2],
             'source':r['source'],'parent_trial1_result_file_sha256':prior['result_file_sha256'],
             'original_memory_sha256':digest(memory),'native_text_sha256':digest(memory[slot]),'native_slot':slot,
             'native_type':r['state'][slot]['type'],'state':r['state'],'seven_history':[features(s,'seven') for s in history],
             'task':identity['task'],'initial_observation_sha256':rr['initial_observation_sha256']}
        selected.append(row)
        reading.append({**{k:row[k] for k in ['index','task_id','scene_id','native_type','native_slot']},
                        'text':memory[slot],'memory':memory})
    PRIVATE.mkdir(parents=True,exist_ok=True)
    write_json(PRIVATE/'reading-view.json',reading)
    manifest=save(MANIFEST,{'scope':'One identity-hash-selected parent per eligible task, no trial2 success filtering',
        'parent_result_sha256':result['sha256'],'parent_closeout_sha256':closed['sha256'],
        'selection_rule':'Within task choose minimum SHA256 canonical[trajectory_id,trial]; no result/outcome hash',
        'candidate_records':59,'selected_tasks':46,'selected_scenes':len({r['scene_id'] for r in selected}),
        'reading_file_sha256':file_hash(PRIVATE/'reading-view.json'),'selected':selected,
        'preparation_source_sha256':file_hash(Path(__file__)),'new_model_calls':0})
    write_json(PRIVATE/'preparation.json',{'status':'selected_pending_text_edits','manifest_sha256':manifest['sha256'],
        'cpu_budget_seconds':900,'new_model_calls':0,'parameter_fits':0,
        'scope':'Text preparation only; a separate finite collection contract and public freeze precede inference'})
    print({'manifest':manifest['sha256'],'parents':46,'scenes':manifest['selected_scenes'],'new_model_calls':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['select']);args=parser.parse_args();select()
