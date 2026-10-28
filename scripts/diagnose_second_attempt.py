#!/usr/bin/env python3
"""Read completed traces; separate compressed-memory labels from physical failure stages."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,re,time
import numpy as np
from agent_reliability.io import digest,file_hash,read_json,write_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from analyze_seven_reached_risk import independent_score
from calibrate_successor_success import features
from audit_reflexion_action_shapes import action_shape

PUBLIC=Path('repro/reflexion-20261025');PRIVATE=Path('runs/reflexion-second-attempt-diagnostic-20261028')
PROTOCOL=PUBLIC/'second-attempt-protocol-20261028.json'
MANIFEST=PUBLIC/'second-attempt-manifest-20261028.json'
SOURCE=Path('scripts/diagnose_second_attempt.py')
DESIGN=Path('docs/experiments/second-attempt-design-20261028.md')
ROOTS={'original_calibration':Path('runs/reflexion-shared-calibration-20261025'),
    'successor_review':Path('runs/reflexion-four-attempt-validation-20261025'),
    'completed_successor_primary':Path('runs/reflexion-successor-validation-20261026')}
LIMITS={'new_model_calls':0,'parameter_fits':0,'cpu_seconds':600,'primary_attempts':441,'priority_error_absent_attempts':126,'matched_blocks':37}
FLAGS=['location_search','operational_state','command_format','alternative_heating','environment_speculation']


def raw_index():
    out={};summaries={}
    for source,root in ROOTS.items():
        summary=checked(root/'summary.json');assert summary['status']=='complete';summaries[source]=summary
        for row in summary['attempts_rows']:out[source,row['trajectory_id'],row['trial']]=row
    return out,summaries


def read_attempt(row,index):
    raw=index[row['source'],row['trajectory_id'],row['trial']];path=Path(raw['result_path'])
    assert raw['result_file_sha256']==row['result_file_sha256']==file_hash(path)
    result=read_json(path);checkpoint=read_json(path.parent/'checkpoint.json');identity=checked(path.parent/'identity.json')
    assert result['success']==row['success']==bool(checkpoint['transitions'][-1]['won'])
    assert identity['task']['task_id']==row['task_id'] and result['scene_id']==row['scene_id']
    assert digest(result['memory_before'][-3:])==raw['visible_memory_sha256']
    assert len(result['memory_before'][-3:])==len(row['state'])
    return raw,path,result,checkpoint,identity


def target_names(identity):
    directory=Path(identity['task']['gamefile']).parts[-3].split('-')
    assert directory[0]=='pick_heat_then_place_in_recep'
    return directory[1].lower(),directory[3].lower()


def trace_features(checkpoint,target):
    transitions=checkpoint['transitions'];held=set();put_in_microwave=set();picked=[];heated=[];bad_order=[]
    physical=thought=no_effect=unmatched=0;repeat=Counter();heat_attempts=0
    for step,t in enumerate(transitions):
        action=t['action'];submitted=t.get('submitted_action',action);observation=t['observation']
        shape=action_shape(action,submitted)
        if shape=='thought':thought+=1;continue
        physical+=1
        if shape!='public_command_shape':unmatched+=1
        ineffective=observation.strip()=='Nothing happens.'
        if ineffective:no_effect+=1;repeat[submitted]+=1
        pickup=re.search(r'(?:^|:\s*)You pick up the ('+re.escape(target)+r' \d+)(?: from |\.|$)',observation,re.I)
        if pickup:
            name=pickup[1].lower();held.add(name);put_in_microwave.discard(name);picked.append(step)
        placed=re.search(r'(?:^|:\s*)You (?:put|move) the ('+re.escape(target)+r' \d+) (?:in|on|into|to) (?:the )?(.+?)\.?$',observation,re.I)
        if placed:
            name=placed[1].lower();held.discard(name)
            if placed[2].lower().startswith('microwave '):put_in_microwave.add(name)
        heating=re.fullmatch(r'heat ('+re.escape(target)+r' \d+) with (.+)',submitted,re.I)
        if heating:
            heat_attempts+=1;name=heating[1].lower()
            if name in put_in_microwave and name not in held:
                bad_order.append({'step':step,'command_sha256':digest(submitted),'no_effect':ineffective})
        if re.match(r'^You heat the '+re.escape(target)+r' \d+ using ',observation,re.I):heated.append(step)
    won=bool(transitions[-1]['won']);stage='won' if won else 'no_target_pickup' if not picked else 'pickup_without_heat' if not heated else 'heated_without_win'
    return {'native_won':won,'failure_stage':stage,'thought_steps':thought,'physical_steps':physical,
        'physical_no_effect':no_effect,'unmatched_physical_shapes':unmatched,'target_pickup_steps':picked,
        'target_heat_success_steps':heated,'target_heat_attempts':heat_attempts,'placed_then_heat_without_pickup':bad_order,
        'max_repeated_no_effect_action':max(repeat.values(),default=0),'checkpoint_sha256':digest(checkpoint)}


def prepare():
    assert not (PRIVATE/'contract.json').exists(),'Preserve the new diagnostic input selection'
    PRIVATE.mkdir(parents=True,exist_ok=True)
    test=[('think: test','Nothing happens.'),('take mug 1 from countertop 1','You pick up the mug 1 from the countertop 1.'),
        ('move mug 1 to microwave 1','You move the mug 1 to the microwave 1.'),('heat mug 1 with microwave 1','Nothing happens.')]
    parsed=trace_features({'transitions':[{'action':a,'submitted_action':a,'observation':o,'won':False} for a,o in test]},'mug')
    assert parsed['thought_steps']==1 and parsed['physical_no_effect']==1 and len(parsed['placed_then_heat_without_pickup'])==1
    parent=checked(PUBLIC/'seven-compatibility-protocol-20261027.json')
    audit=checked(PUBLIC/'seven-compatibility-audit-20261027.json')
    rows=read_json(Path('runs/reflexion-seven-compatibility-20261027/input-events.json'));assert digest(rows)==parent['input_rows_sha256']
    second=[r for r in rows if r['trial']==2]
    clean=[r for r in second if not any(s['type'] in ['E','B'] for s in r['state'])]
    groups=defaultdict(lambda:defaultdict(list));sequences=defaultdict(list)
    for r in rows:sequences[r['trajectory_id']].append(r)
    for tid,sequence in sequences.items():
        history=[]
        for r in sorted(sequence,key=lambda r:r['trial']):
            history.append(tuple(features(r['state'],'seven')));b=any(s['type']=='B' for s in r['state']);e=any(s['type']=='E' for s in r['state'])
            if b or e:groups[r['task_id'],r['trial'],tuple(history)]['B' if b else 'E'].append(r)
    matched=[];matched_rows={}
    for key,value in sorted(groups.items()):
        if not ('B' in value and 'E' in value):continue
        matched.append({'task_id':key[0],'trial':key[1],'feature_history_sha256':digest(key[2]),
            'B_attempt_ids':[[r['trajectory_id'],r['trial']] for r in value['B']],
            'E_attempt_ids':[[r['trajectory_id'],r['trial']] for r in value['E']]})
        for r in value['B']+value['E']:matched_rows[r['trajectory_id'],r['trial']]=r
    assert len(second)==441 and len(clean)==126 and len(matched)==37
    assert len(matched_rows)==99 and len({r['task_id'] for r in matched_rows.values()})==22
    focus={(r['trajectory_id'],r['trial']):r for r in clean};focus.update(matched_rows)
    index,summaries=raw_index();records=[];text_uses=defaultdict(list);texts={};private=[]
    for row in sorted({**{(r['trajectory_id'],r['trial']):r for r in second},**matched_rows}.values(),key=lambda r:(r['trajectory_id'],r['trial'])):
        raw,path,result,checkpoint,identity=read_attempt(row,index);target,destination=target_names(identity)
        previous=next(r for r in rows if r['trajectory_id']==row['trajectory_id'] and r['trial']==row['trial']-1)
        _,previous_path,previous_result,previous_checkpoint,_=read_attempt(previous,index)
        memory_hashes=[digest(t) for t in result['memory_before'][-3:]]
        in_focus=(row['trajectory_id'],row['trial']) in focus
        native_hashes=[]
        for state,text in zip(row['state'],result['memory_before'][-3:]):
            if state['native']:
                h=digest(text);native_hashes.append(h)
                if in_focus:texts[h]=text;text_uses[h].append({'trajectory_id':row['trajectory_id'],'trial':row['trial'],'target':target,'destination':destination,'legacy_type':state['type']})
        features_now=trace_features(checkpoint,target);features_prev=trace_features(previous_checkpoint,target)
        entry={**row,'priority_error_absent':(row['trajectory_id'],row['trial']) in {(r['trajectory_id'],r['trial']) for r in clean},
            'matched_focus':(row['trajectory_id'],row['trial']) in matched_rows,'memory_hashes':memory_hashes,
            'native_memory_hashes':native_hashes,'target_type':target,'destination_type':destination,
            'task_type':result['task_type'],'result_status':result['status'],'current_trace':features_now,'previous_trace':features_prev,
            'raw_checkpoint_file_sha256':file_hash(path.parent/'checkpoint.json'),'identity_file_sha256':file_hash(path.parent/'identity.json'),
            'previous_result_file_sha256':file_hash(previous_path),'truncated_responses':result['truncated_responses'],
            'format_invalid_responses':result['format_invalid_responses']}
        records.append(entry)
        if in_focus:
            private.append({'attempt_id':[row['trajectory_id'],row['trial']],'target':target,'destination':destination,
                'state':row['state'],'visible_memory':result['memory_before'][-3:],
                'previous_transitions':previous_checkpoint['transitions'],'current_transitions':checkpoint['transitions']})
    assert len(records)==len({(r['trajectory_id'],r['trial']) for r in second}|set(matched_rows))
    reading=[{'index':i,'text_sha256':h,'reflection':texts[h],
        'goals':sorted({(u['target'],u['destination']) for u in text_uses[h]}),'legacy_types':sorted({u['legacy_type'] for u in text_uses[h]})} for i,h in enumerate(sorted(texts))]
    write_json(PRIVATE/'input-rows.json',rows);write_json(PRIVATE/'records.json',records);write_json(PRIVATE/'reading-view.json',reading)
    write_json(PRIVATE/'full-traces.json',private);write_json(PRIVATE/'text-uses.json',dict(text_uses))
    manifest=save(MANIFEST,{'scope':'Outcome-independent identity selection for second-attempt and same-feature-prefix diagnosis',
        'parent_audit_sha256':audit['sha256'],'input_rows_sha256':digest(rows),
        'second_attempt_ids':[[r['trajectory_id'],r['trial']] for r in second],
        'priority_error_absent_ids':[[r['trajectory_id'],r['trial']] for r in clean],
        'matched_blocks':matched,'full_text_focus_unique_attempts':len(focus),'all_trace_records':len(records),
        'unique_native_texts_to_review':len(reading),'private_reading_file_sha256':file_hash(PRIVATE/'reading-view.json'),
        'private_full_trace_file_sha256':file_hash(PRIVATE/'full-traces.json'),'private_records_file_sha256':file_hash(PRIVATE/'records.json'),
        'selection':'All441 trial2; all126 trial2 states withoutE/B; all37 same task/full7feature-prefix blocks with bothB-present andE-withoutB, no success filtering',
        'outcome_visibility':'Aggregate outcomes and earlier diagnostics already known. Text content view omits current outcome/actor trace, not independent blinded review',
        'new_model_calls':0})
    protected=dict(parent['protected_files'])
    for path in [SOURCE,DESIGN,MANIFEST,PUBLIC/'seven-compatibility-protocol-20261027.json',PUBLIC/'seven-compatibility-results-20261027.json',PUBLIC/'seven-compatibility-audit-20261027.json']:
        protected[path.as_posix()]=file_hash(path)
    protocol=save(PROTOCOL,{'scope':'Zero-inference trace and text diagnosis, no parameter fitting',
        'manifest_sha256':manifest['sha256'],'limits':LIMITS,'input_rows_sha256':digest(rows),
        'mle_prediction_file':parent['mle_reference']['prediction_file'],'mle_prediction_sha256':parent['mle_reference']['prediction_sha256'],
        'content_flags':FLAGS,'flag_meanings':{
            'location_search':'Named search/location/object retrieval advice; does not assert correctness or utility',
            'operational_state':'Specific object/tool state or action ordering advice',
            'command_format':'Discussion of command syntax, accepted action form, or system recognition',
            'alternative_heating':'Alternative appliance, heat mode or unrelated object heating proposed',
            'environment_speculation':'Appliance broken/unusable, object unheatable or unsupported world capability explanation'},
        'annotation':'Read every selected native text before joining current actor/outcome; preserve original E/C/B/U/A labels; content properties only, no causal utility labels',
        'trace_rules':'Exclude native think: from physical no-effect counts. Pickup/heat/placement require matching public feedback. Put-then-heat-without-take is an observed sequence, not attribution to memory',
        'statistics':'Fixed-start predictable weighting, per-source/task/scene/type decomposition; prior-trace features are diagnostics not automatic new predictors',
        'parser_smoke':'Public pickup/move/heat feedback chain and thought exclusion checked on constructed fixture; not Agent evidence',
        'protected_files':protected,'new_model_calls':0,'parameter_fits':0,'holdout134_calls':0})
    print({'protocol_sha256':protocol['sha256'],'manifest_sha256':manifest['sha256'],
        'records':len(records),'focus_attempts':len(focus),'reading_texts':len(reading),'protected':len(protected)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare']);args=parser.parse_args();prepare()
