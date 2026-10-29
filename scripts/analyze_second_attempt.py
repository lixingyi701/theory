#!/usr/bin/env python3
"""Join full-text content decisions to completed action traces; no inference or fitting."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,re,time
import numpy as np
from agent_reliability.io import digest,file_hash,read_json,write_json
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from analyze_seven_reached_risk import independent_score
from diagnose_second_attempt import PRIVATE,PUBLIC,PROTOCOL,MANIFEST,raw_index,read_attempt,trace_features
from continue_reflexion_pilot import ledger

LABELS=PUBLIC/'second-attempt-content-labels-20261028.json'
CODES={'L':'location_search','S':'operational_state','C':'command_format','A':'alternative_heating','E':'environment_speculation'}
PATTERNS={
    'L':[r'\b(?:search\w*|locat\w*|retriev\w*)\b',r'\b(?:fridges?|countertops?|cabinets?|drawers?|shel(?:f|ves)|sinkbasins?|garbagecans?|diningtables?|coffeemachine)\b',r'\bfind\b'],
    'S':[r'\binventory\b',r'\b(?:inside|closed|open|placed|placing|holding|held|state|sequence|hot|heat|heated)\b',r'\b(?:check|verify|confirm)\w*\b'],
    'C':[r'\b(?:syntax|format\w*|recogniz\w*|structure)\b',r'commands?\s*\(?(?:like|such as)',r'"(?:heat|turn on|put|discard|drop|use|place)'],
    'A':[r'\b(?:stoveburner\w*|toaster|stove|oven|pan|pot)\b',r'\b(?:alternative|different|another|pre-heated|pre-hot)\b',r'\bturn(?:ed)? on\b'],
    'E':[r'\b(?:functional\w*|functioning|operational|malfunction\w*|broken|unresponsive|heatable|heatability|limitations?)\b',
        r'\b(?:not present|not be present|absence|required item is missing|cannot be completed|not working|not responding|did not work|did not function|appropriate tool|system)\b',
        r'\btask\b',r'\balready\b']}


def annotate():
    started=time.process_time();p=checked(PROTOCOL);manifest=checked(MANIFEST)
    view=read_json(PRIVATE/'reading-view.json');decisions=read_json(PRIVATE/'manual-decisions.json')
    assert file_hash(PRIVATE/'reading-view.json')==manifest['private_reading_file_sha256']
    assert [i for i,f in decisions]==list(range(len(view)))==list(range(232))
    entries=[];missing=[]
    for index,codes in decisions:
        row=view[index];text=row['reflection'];assert digest(text)==row['text_sha256']
        spans={}
        for code in codes:
            matches=[m for pattern in PATTERNS[code] for m in [re.search(pattern,text,re.I)] if m is not None]
            if not matches:missing.append({'index':index,'code':code,'text':text});continue
            m=matches[0];start=max(0,m.start()-12);end=min(len(text),m.end()+24)
            spans[CODES[code]]=[start,end]
        entries.append({'index':index,'reflection_sha256':row['text_sha256'],'flags':[CODES[c] for c in codes],
            'evidence_spans':spans,'original_target_rule_categories':row['legacy_types']})
    if missing:
        print({'missing_evidence':missing},flush=True);raise ValueError('Review content-code evidence before publication')
    result=save(LABELS,{'scope':'Content-property reading of232 distinct native reflection texts, not utility or causal labels',
        'protocol_sha256':p['sha256'],'manifest_sha256':manifest['sha256'],'reading_file_sha256':manifest['private_reading_file_sha256'],
        'decisions_file_sha256':file_hash(PRIVATE/'manual-decisions.json'),'annotation':'Codex full-text read and single-coder self-review before current-trace aggregation; earlier aggregate outcomes known',
        'independent_blinded_annotation':False,'human_second_review':False,'flags':p['flag_meanings'],
        'evidence':'Offsets into server-private full texts, indexed by exact text digest; raw model responses are not published',
        'entries':entries,'flag_counts':dict(Counter(f for e in entries for f in e['flags'])),'new_model_calls':0})
    c=read_json(PRIVATE/'contract.json');c['status']='annotated';c['labels_sha256']=result['sha256'];c['annotation_cpu_seconds']=time.process_time()-started
    write_json(PRIVATE/'contract.json',c)
    print({'labels_sha256':result['sha256'],'texts':len(entries),'counts':result['flag_counts']},flush=True)


def extra_actions(checkpoint,target,destination):
    nonmicro=[];goal_placements=[];same_target=[]
    for i,step in enumerate(checkpoint['transitions']):
        a=step.get('submitted_action',step['action']);ob=step['observation']
        if step['action'].startswith('think:'):continue
        m=re.fullmatch(r'heat ('+re.escape(target)+r' \d+) with ([a-z]+) \d+',a)
        if m:
            same_target.append(i)
            if m[2]!='microwave':nonmicro.append(i)
        if re.search(r'(?:^|:\s*)You (?:move|put) the '+re.escape(target)+r' \d+ (?:to|in|on|into) (?:the )?'+re.escape(destination)+r' \d+\.',ob,re.I):goal_placements.append(i)
    return {'non_microwave_target_heat_steps':nonmicro,'goal_placement_feedback_steps':goal_placements,'target_heat_command_steps':same_target}


def analyse():
    started=time.process_time();p=checked(PROTOCOL);manifest=checked(MANIFEST);labels=checked(LABELS);contract=read_json(PRIVATE/'contract.json')
    assert contract['labels_sha256']==labels['sha256'] and contract['freeze_commit']=='9c9a75af717346860986ac3c1984c1a7b0ac3d98'
    for path,h in p['protected_files'].items():assert file_hash(Path(path))==h,path
    rows=read_json(PRIVATE/'input-rows.json');records=read_json(PRIVATE/'records.json')
    assert digest(rows)==p['input_rows_sha256'] and file_hash(PRIVATE/'records.json')==manifest['private_records_file_sha256']
    index,_=raw_index();labelmap={e['reflection_sha256']:e for e in labels['entries']};enriched=[]
    for record in records:
        raw,path,result,checkpoint,identity=read_attempt(record,index)
        assert trace_features(checkpoint,record['target_type'])==record['current_trace']
        native=record['native_memory_hashes'];reviewed=all(h in labelmap for h in native)
        flags=sorted({f for h in native for f in labelmap[h]['flags']}) if reviewed else None
        enriched.append({**record,'native_texts_content_reviewed':reviewed,'content_flags':flags,
            'current_extra_actions':extra_actions(checkpoint,record['target_type'],record['destination_type'])})
    by_id={(r['trajectory_id'],r['trial']):r for r in enriched};second=[r for r in enriched if r['trial']==2]
    clean=[r for r in second if r['priority_error_absent']];matched=[r for r in enriched if r['matched_focus']]
    assert len(second)==441 and len(clean)==126 and len(matched)==99
    assert all(r['native_texts_content_reviewed'] for r in clean+matched)
    frozen=checked(PUBLIC/p['mle_prediction_file']);assert frozen['sha256']==p['mle_prediction_sha256']
    probabilities={(r['trajectory_id'],r['trial']):r['probability'] for r in frozen['predictions']}
    def summary(predicate):
        scored=independent_score(rows,probabilities,lambda r:r['trial']==2 and predicate(by_id[r['trajectory_id'],2]))
        selected=[r for r in second if predicate(r)]
        scored.update({'tasks':len({r['task_id'] for r in selected}),'scenes':len({r['scene_id'] for r in selected}),
            'failure_stage_counts':dict(Counter(r['current_trace']['failure_stage'] for r in selected)),
            'previous_failure_stage_counts':dict(Counter(r['previous_trace']['failure_stage'] for r in selected)),
            'put_then_heat_without_pickup_attempts':sum(bool(r['current_trace']['placed_then_heat_without_pickup']) for r in selected),
            'non_microwave_target_heat_attempts':sum(bool(r['current_extra_actions']['non_microwave_target_heat_steps']) for r in selected),
            'unmatched_physical_action_attempts':sum(r['current_trace']['unmatched_physical_shapes']>0 for r in selected),
            'physical_steps':sum(r['current_trace']['physical_steps'] for r in selected),
            'physical_no_effect_steps':sum(r['current_trace']['physical_no_effect'] for r in selected),
            'thought_steps_excluded':sum(r['current_trace']['thought_steps'] for r in selected)})
        return scored
    overall=summary(lambda r:True);clean_summary=summary(lambda r:r['priority_error_absent'])
    reference=checked(PUBLIC/'seven-compatibility-audit-20261027.json')['reference_cv_scores']['second_attempt']
    assert abs(overall['weighted_conditional_brier']-reference['weighted_conditional_brier'])<1e-12
    source_groups={s:summary(lambda r,s=s:r['source']==s) for s in sorted({r['source'] for r in second})}
    object_groups={s:summary(lambda r,s=s:r['target_type']==s) for s in sorted({r['target_type'] for r in second})}
    memory_groups={k:summary(lambda r,k=k:('B' if any(s['type']=='B' for s in r['state']) else 'E' if any(s['type']=='E' for s in r['state']) else 'no_E_B')==k) for k in ['B','E','no_E_B']}
    previous_groups={k:summary(lambda r,k=k:r['previous_trace']['failure_stage']==k) for k in sorted({r['previous_trace']['failure_stage'] for r in second})}
    clean_by_type={k:summary(lambda r,k=k:r['priority_error_absent'] and r['state'][-1]['type']==k) for k in sorted({r['state'][-1]['type'] for r in clean})}
    clean_no_A=summary(lambda r:r['priority_error_absent'] and all(s['type']!='A' for s in r['state']))
    clean_by_previous={k:summary(lambda r,k=k:r['priority_error_absent'] and r['previous_trace']['failure_stage']==k) for k in sorted({r['previous_trace']['failure_stage'] for r in clean})}
    content_groups={f:summary(lambda r,f=f:r['priority_error_absent'] and f in r['content_flags']) for f in p['content_flags']}
    behavior_groups={k:summary(lambda r,k=k:(r['previous_trace']['failure_stage']==k) and not any(s['type'] in ['E','B'] for s in r['state'])) for k in sorted({r['previous_trace']['failure_stage'] for r in clean})}
    # Deterministic examples illustrate trace rules; the statistical cohort stays complete.
    cases=[];seen=set()
    def add(records,reason,number):
        for r in sorted(records,key=lambda x:x['result_file_sha256']):
            key=r['trajectory_id'],r['trial']
            if key in seen:continue
            cases.append({'trajectory_id':key[0],'trial':key[1],'selection_reason':reason});seen.add(key)
            number-=1
            if not number:break
    for stage in ['no_target_pickup','pickup_without_heat','heated_without_win','won']:
        add([r for r in clean if r['current_trace']['failure_stage']==stage],stage,1)
    add([r for r in clean if r['current_trace']['placed_then_heat_without_pickup']],'observable_put_then_heat_sequence',2)
    block=manifest['matched_blocks'][0]
    for kind in ['B','E']:
        ids={tuple(x) for x in block[kind+'_attempt_ids']};add([by_id[x] for x in ids],'same_prefix_'+kind,1)
    review=None
    if (PRIVATE/'trace-review.json').exists():
        reviewed=read_json(PRIVATE/'trace-review.json')
        assert [(r['trajectory_id'],r['trial']) for r in reviewed['entries']]==[(r['trajectory_id'],r['trial']) for r in cases]
        review=save(PUBLIC/'second-attempt-trace-review-20261028.json',{'protocol_sha256':p['sha256'],
            'manifest_sha256':manifest['sha256'],'case_selection':cases,**reviewed})
    eligible=[r for r in clean if all(s['type']!='A' for s in r['state']) and r['previous_trace']['failure_stage']=='pickup_without_heat'
        and any(f in r['content_flags'] for f in ['alternative_heating','environment_speculation'])]
    pool={'scope':'Possible future same-state text intervention parents, chosen without current-attempt outcome filtering; no experiment launched',
        'attempt_ids':[[r['trajectory_id'],r['trial']] for r in eligible],
        'attempts':len(eligible),'tasks':len({r['task_id'] for r in eligible}),'scenes':len({r['scene_id'] for r in eligible}),
        'rule':'Trial2 window withoutE/B/A, previous attempt picked target but did not heat, reviewed text proposes alternative heating or environment speculation'}
    # Full public rows are compact feature projections, not raw responses or dialogue.
    parts=[]
    for i,start in enumerate(range(0,len(enriched),75)):
        name=f'second-attempt-records-{i+1:02d}-20261028.json'
        value=save(PUBLIC/name,{'protocol_sha256':p['sha256'],'manifest_sha256':manifest['sha256'],'start':start,'records':enriched[start:start+75]})
        parts.append({'file':name,'sha256':value['sha256'],'records':len(value['records'])})
    record_index=save(PUBLIC/'second-attempt-records-index-20261028.json',{'protocol_sha256':p['sha256'],'manifest_sha256':manifest['sha256'],
        'records':len(enriched),'parts':parts,'projection':'Original row identity and labels, public-feedback trace features, text hashes and reviewed content flags; no raw model responses'})
    result=save(PUBLIC/'second-attempt-results-20261028.json',{'scope':'Existing second-attempt trace/content diagnosis, not a new predictor or causal comparison',
        'protocol_sha256':p['sha256'],'manifest_sha256':manifest['sha256'],'labels_sha256':labels['sha256'],'record_index_sha256':record_index['sha256'],
        'analysis_source_sha256':file_hash(Path(__file__)),
        'overall':overall,'priority_no_E_B':clean_summary,'by_source':source_groups,'by_object':object_groups,'by_window_label':memory_groups,
        'by_previous_failure_stage':previous_groups,'priority_by_native_type':clean_by_type,'priority_excluding_uncertain_A':clean_no_A,'priority_by_previous_stage':clean_by_previous,
        'priority_content_flag_groups':content_groups,'trace_example_selection':cases,
        'annotation_coverage':{'unique_native_texts':232,'planned_focus_attempts':225,'all_trace_records':len(enriched),
            'manual_trace_cases_reviewed':0 if review is None else len(reviewed['entries']),'manual_trace_cases_pending':len(cases) if review is None else 0},
        'trace_review_sha256':None if review is None else review['sha256'],'future_intervention_parent_pool':pool,
        'weights':'Every subset uses original complete1553-row preplanned scene/task/branch weights; never recompute branches from reached subset',
        'new_model_calls':0,'parameter_fits':0,'holdout134_calls':0,'analysis_cpu_seconds':time.process_time()-started})
    write_json(PRIVATE/'enriched-records.json',enriched);write_json(PRIVATE/'example-selection.json',cases)
    c=read_json(PRIVATE/'contract.json');c['status']='trace_cases_pending';c['results_sha256']=result['sha256'];c['analysis_cpu_seconds']=time.process_time()-started
    assert c.get('annotation_cpu_seconds',0)+c['analysis_cpu_seconds']+60<600
    write_json(PRIVATE/'contract.json',c)
    small=lambda s:{k:s[k] for k in ['attempts','observed_successes','expected_successes','weighted_observed_success_fraction','weighted_expected_success_fraction','failure_stage_counts','previous_failure_stage_counts','put_then_heat_without_pickup_attempts','non_microwave_target_heat_attempts']}
    print({'sha256':result['sha256'],'overall':small(overall),'no_E_B':small(clean_summary),
        'source':{k:small(v) for k,v in source_groups.items()},'clean_by_previous':{k:small(v) for k,v in clean_by_previous.items()},
        'content':{k:small(v) for k,v in content_groups.items()},'case_count':len(cases)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['annotate','analyse']);a=parser.parse_args()
    annotate() if a.action=='annotate' else analyse()
