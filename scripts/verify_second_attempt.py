#!/usr/bin/env python3
"""Independently verify raw-feedback counts and fixed-start subgroup scores."""
from pathlib import Path
from collections import Counter, defaultdict
import re, time, math
from agent_reliability.io import digest, file_hash, read_json, write_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from continue_reflexion_pilot import ledger

P=Path('repro/reflexion-20261025')
D=Path('runs/reflexion-second-attempt-diagnostic-20261028')
ROOTS={'original_calibration':'reflexion-shared-calibration-20261025',
       'successor_review':'reflexion-four-attempt-validation-20261025',
       'completed_successor_primary':'reflexion-successor-validation-20261026'}


def extract(checkpoint, target):
    """Use latest observed take/deposit times instead of the production sets."""
    take_time={}; deposit_time={}; pickup=[]; heated=[]; bad=[]
    thoughts=physical=no_effect=0
    for i,t in enumerate(checkpoint['transitions']):
        action=t['action']; submitted=t.get('submitted_action',action); ob=t['observation']
        if action.startswith('think:'):
            thoughts+=1
            continue
        physical+=1; no_effect+=ob.strip()=='Nothing happens.'
        # Official feedback names the moved object before its source/destination.
        m=re.search(r'(?:^|:\s*)You pick up the ([a-z]+ \d+)(?: from |\.|$)',ob,re.I)
        if m and m[1].lower().startswith(target+' '):
            take_time[m[1].lower()]=i; deposit_time.pop(m[1].lower(),None); pickup.append(i)
        m=re.search(r'(?:^|:\s*)You (?:put|move) the ([a-z]+ \d+) (?:in|on|into|to) (?:the )?(.+?)\.?$',ob,re.I)
        if m and m[1].lower().startswith(target+' '):
            name=m[1].lower(); take_time[name]=-1
            if m[2].lower().startswith('microwave '): deposit_time[name]=i
        m=re.fullmatch(r'heat ([a-z]+ \d+) with (.+)',submitted,re.I)
        if m and m[1].lower().startswith(target+' '):
            name=m[1].lower()
            if name in deposit_time and take_time.get(name,-1)<deposit_time[name]:
                bad.append({'step':i,'command_sha256':digest(submitted),'no_effect':ob.strip()=='Nothing happens.'})
        if re.match(r'^You heat the '+re.escape(target)+r' \d+ using ',ob,re.I): heated.append(i)
    won=bool(checkpoint['transitions'][-1]['won'])
    return {'native_won':won,'thought_steps':thoughts,'physical_steps':physical,
            'physical_no_effect':no_effect,'target_pickup_steps':pickup,'target_heat_success_steps':heated,
            'placed_then_heat_without_pickup':bad,
            'failure_stage':'won' if won else 'no_target_pickup' if not pickup else 'pickup_without_heat' if not heated else 'heated_without_win'}


def main():
    started=time.process_time(); protocol=checked(P/'second-attempt-protocol-20261028.json')
    manifest=checked(P/'second-attempt-manifest-20261028.json'); labels=checked(P/'second-attempt-content-labels-20261028.json')
    result=checked(P/'second-attempt-results-20261028.json'); index=checked(P/'second-attempt-records-index-20261028.json')
    contract=read_json(D/'contract.json'); assert contract['freeze_commit']=='9c9a75af717346860986ac3c1984c1a7b0ac3d98'
    for path,h in protocol['protected_files'].items(): assert file_hash(Path(path))==h,path
    for key,name in [('private_records_file_sha256','records.json'),('private_reading_file_sha256','reading-view.json'),('private_full_trace_file_sha256','full-traces.json')]:
        assert file_hash(D/name)==manifest[key]
    assert file_hash(Path('scripts/analyze_second_attempt.py'))==result['analysis_source_sha256']
    view=read_json(D/'reading-view.json'); decisions=read_json(D/'manual-decisions.json')
    assert len(view)==len(labels['entries'])==len(decisions)==232
    assert file_hash(D/'manual-decisions.json')==labels['decisions_file_sha256']
    for text,lab,decision in zip(view,labels['entries'],decisions):
        assert digest(text['reflection'])==lab['reflection_sha256']==text['text_sha256']
        assert decision[0]==lab['index']==text['index']
        assert lab['original_target_rule_categories']==text['legacy_types']
        assert set(lab['flags'])==set(lab['evidence_spans'])
        for a,b in lab['evidence_spans'].values(): assert 0<=a<b<=len(text['reflection'])
    assert dict(Counter(f for x in labels['entries'] for f in x['flags']))==labels['flag_counts']
    records=[]
    for part in index['parts']:
        data=checked(P/part['file']); assert data['sha256']==part['sha256'] and len(data['records'])==part['records']
        records.extend(data['records'])
    assert len(records)==491==index['records']; by_id={(r['trajectory_id'],r['trial']):r for r in records}
    assert len(by_id)==491 and result['record_index_sha256']==index['sha256']
    rows=read_json(D/'input-rows.json'); assert len(rows)==1553 and digest(rows)==protocol['input_rows_sha256']
    original={(r['trajectory_id'],r['trial']):r for r in rows}
    raw={}
    for source,root in ROOTS.items():
        for r in checked(Path('runs')/root/'summary.json')['attempts_rows']:
            raw[source,r['trajectory_id'],r['trial']]=r
    lm={e['reflection_sha256']:e for e in labels['entries']}; trace_checks=0
    for r in records:
        for k,v in original[r['trajectory_id'],r['trial']].items(): assert r[k]==v
        path=Path(raw[r['source'],r['trajectory_id'],r['trial']]['result_path'])
        assert file_hash(path)==r['result_file_sha256']
        rr=read_json(path); identity=checked(path.parent/'identity.json')
        assert identity['task']['task_id']==r['task_id'] and rr['scene_id']==r['scene_id']
        names=Path(identity['task']['gamefile']).parts[-3].split('-')
        assert names[0]=='pick_heat_then_place_in_recep' and names[1].lower()==r['target_type'] and names[3].lower()==r['destination_type']
        assert [digest(t) for t in rr['memory_before'][-3:]]==r['memory_hashes']
        assert [digest(t) for s,t in zip(r['state'],rr['memory_before'][-3:]) if s['native']]==r['native_memory_hashes']
        prev_path=Path(raw[r['source'],r['trajectory_id'],r['trial']-1]['result_path'])
        prev=read_json(prev_path); assert rr['memory_before']==prev['memory_after']
        assert file_hash(prev_path)==r['previous_result_file_sha256']
        for prefix,cp_path in [('current',path.parent/'checkpoint.json'),('previous',prev_path.parent/'checkpoint.json')]:
            cp=read_json(cp_path); obs=extract(cp,r['target_type']); expected=r[prefix+'_trace']
            assert digest(cp)==expected['checkpoint_sha256']
            for k,v in obs.items(): assert expected[k]==v,(r['trajectory_id'],prefix,k)
            trace_checks+=1
        assert rr['success']==r['success']==r['current_trace']['native_won']
        reviewed=all(h in lm for h in r['native_memory_hashes'])
        assert reviewed==r['native_texts_content_reviewed']
        if reviewed: assert sorted({f for h in r['native_memory_hashes'] for f in lm[h]['flags']})==r['content_flags']
    second=[r for r in records if r['trial']==2]; clean=[r for r in second if r['priority_error_absent']]
    matched=[r for r in records if r['matched_focus']]
    assert len(second)==441 and len(clean)==126 and len(matched)==99
    assert {tuple(x) for x in manifest['second_attempt_ids']}=={(r['trajectory_id'],2) for r in second}
    assert {tuple(x) for x in manifest['priority_error_absent_ids']}=={(r['trajectory_id'],2) for r in clean}
    assert all(not any(s['type'] in ['E','B'] for s in r['state']) for r in clean)
    assert len(manifest['matched_blocks'])==37 and len({r['task_id'] for r in matched})==22
    # Every weight is reconstructed from all preplanned branches, before selecting reached records.
    tasks=defaultdict(set); branches=defaultdict(set)
    for r in rows: tasks[r['scene_id']].add(r['task_id']); branches[r['task_id']].add(r['trajectory_id'])
    pred=checked(P/protocol['mle_prediction_file']); assert pred['sha256']==protocol['mle_prediction_sha256']
    pp={(r['trajectory_id'],r['trial']):r['probability'] for r in pred['predictions']}
    def compare(selected, expected):
        n=len(selected); obs=sum(r['success'] for r in selected); exp=sum(pp[r['trajectory_id'],r['trial']] for r in selected)
        assert n==expected['attempts'] and obs==expected['observed_successes'] and abs(exp-expected['expected_successes'])<1e-10
        weights=[1/(4*len(tasks)*len(tasks[r['scene_id']])*len(branches[r['task_id']])) for r in selected]
        if n:
            denominator=sum(weights)
            for k,val in [('weighted_observed_success_fraction',sum(w*r['success'] for w,r in zip(weights,selected))/denominator),
                          ('weighted_expected_success_fraction',sum(w*pp[r['trajectory_id'],r['trial']] for w,r in zip(weights,selected))/denominator),
                          ('weighted_conditional_brier',sum(w*(pp[r['trajectory_id'],r['trial']]-r['success'])**2 for w,r in zip(weights,selected))/denominator)]:
                assert abs(val-expected[k])<1e-12,(k,val,expected[k])
        for k,prefix in [('failure_stage_counts','current'),('previous_failure_stage_counts','previous')]:
            assert dict(Counter(r[prefix+'_trace']['failure_stage'] for r in selected))==expected[k]
        assert sum(bool(r['current_trace']['placed_then_heat_without_pickup']) for r in selected)==expected['put_then_heat_without_pickup_attempts']
    compare(second,result['overall']); compare(clean,result['priority_no_E_B'])
    compare([r for r in clean if all(s['type']!='A' for s in r['state'])],result['priority_excluding_uncertain_A'])
    for field,groups in [('source',result['by_source']),('target_type',result['by_object'])]:
        for k,v in groups.items(): compare([r for r in second if r[field]==k],v)
    for k,v in result['by_window_label'].items():
        compare([r for r in second if ('B' if any(s['type']=='B' for s in r['state']) else 'E' if any(s['type']=='E' for s in r['state']) else 'no_E_B')==k],v)
    for selection,name in [(second,'by_previous_failure_stage'),(clean,'priority_by_previous_stage')]:
        for k,v in result[name].items(): compare([r for r in selection if r['previous_trace']['failure_stage']==k],v)
    for k,v in result['priority_by_native_type'].items(): compare([r for r in clean if r['state'][-1]['type']==k],v)
    for k,v in result['priority_content_flag_groups'].items(): compare([r for r in clean if k in r['content_flags']],v)
    review=checked(P/'second-attempt-trace-review-20261028.json')
    assert review['sha256']==result['trace_review_sha256'] and len(review['entries'])==8
    for case,selected in zip(review['entries'],result['trace_example_selection']):
        key=case['trajectory_id'],case['trial']; assert key==(selected['trajectory_id'],selected['trial'])
        r=by_id[key]
        for prefix in ['current','previous']:
            j=case['trial']-(prefix=='previous'); path=Path(raw[r['source'],case['trajectory_id'],j]['result_path']).parent/'checkpoint.json'
            trans=read_json(path)['transitions']
            assert all(0<=s<len(trans) and not trans[s]['action'].startswith('think:') for s in case['evidence_steps'][prefix])
    eligible=[r for r in clean if all(s['type']!='A' for s in r['state']) and r['previous_trace']['failure_stage']=='pickup_without_heat'
              and any(f in r['content_flags'] for f in ['alternative_heating','environment_speculation'])]
    assert [[r['trajectory_id'],r['trial']] for r in eligible]==result['future_intervention_parent_pool']['attempt_ids']
    assert len(eligible)==59 and len({r['task_id'] for r in eligible})==46 and len({r['scene_id'] for r in eligible})==24
    counts=ledger(); dispatched=sum(v['dispatched'] for v in counts.values()); responses=sum(v['responses'] for v in counts.values())
    assert dispatched==responses==200396
    cpu=time.process_time()-started
    total=70+contract.get('annotation_cpu_seconds',0)+contract.get('analysis_cpu_seconds',0)+cpu
    assert total<protocol['limits']['cpu_seconds']
    audit=save(P/'second-attempt-audit-20261028.json',{'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],
        'source_sha256':file_hash(Path(__file__)),'checks_passed':True,'protected_files':len(protocol['protected_files']),
        'raw_attempt_pairs':491,'independently_parsed_traces':trace_checks,'native_texts_verified':232,'manual_trace_cases':8,
        'source':'Independent action-time parser and subgroup weighting implementation; annotation meaning remains single-coder reviewed',
        'public_projection_parts':len(index['parts']),'global_calls_responses':dispatched,
        'new_model_calls':0,'parameter_fits':0,'holdout134_calls':0,'audit_cpu_seconds':cpu})
    close=save(P/'second-attempt-closeout-20261028.json',{'status':'complete','freeze_commit':contract['freeze_commit'],
        'protocol_sha256':protocol['sha256'],'result_sha256':result['sha256'],'audit_sha256':audit['sha256'],
        'cpu_budget_seconds':600,'cpu_accounting_upper_estimate_seconds':total,
        'cpu_accounting':'60-second preparation and10-second prior-pass reserves plus measured annotation/latest aggregation/audit process CPU',
        'new_model_calls':0,'parameter_fits':0,'holdout134_calls':0,'global_calls_responses':200396})
    contract.update(status='complete',audit_sha256=audit['sha256'],closeout_sha256=close['sha256'],cpu_accounting_upper_estimate_seconds=total)
    write_json(D/'contract.json',contract)
    print({'audit':audit['sha256'],'closeout':close['sha256'],'cpu_upper':total,'traces':trace_checks,'checks_passed':True})


if __name__=='__main__': main()
