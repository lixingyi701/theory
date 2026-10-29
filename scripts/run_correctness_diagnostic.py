#!/usr/bin/env python3
"""Freeze, execute and analyse five paired starting memories; no parameter fit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, run_lock, write_json
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import author_namespace, run_attempt
from analyze_reflexion_precondition_forks import procedure_events
from audit_reflexion_action_shapes import action_shape
from continue_reflexion_pilot import ledger
from run_middle_context_transport import CalibrationStop, reserve_call, verify_contract, replica_audit
from run_multiround_diagnostic import verify_attempt
from run_residual_context import verify_dependencies
from state_representation_model import scores, predict
from summarize_reflexion import costs

PREFIX = 'correctness-diagnostic'
PROTOCOL = Path('repro/reflexion-20261024')/(PREFIX+'-protocol.json')
PARENT = Path('repro/reflexion-20261024/multiround-diagnostic-protocol.json')
CONDITIONS = {'C':['C'], 'N':['N'], 'empty':[], 'EC':['E','N','C'], 'EN':['E','N','N']}
SEEDS = [2104,2105,2106]
CONTRASTS = {'EC_minus_EN':('EC','EN'), 'C_minus_N':('C','N'),
             'N_minus_empty':('N','empty'), 'C_minus_empty':('C','empty')}
BOOTSTRAP_SEED = 2026102403
FILES = [str(PARENT), 'scripts/run_correctness_diagnostic.py', 'tests/test_correctness_diagnostic.py',
         'docs/experiments/correctness-diagnostic-design-20261024.md',
         'docs/research/recovery_state_sufficiency_analysis.md',
         'scripts/run_multiround_diagnostic.py', 'scripts/run_middle_context_transport.py',
         'scripts/run_residual_context.py', 'scripts/analyze_reflexion_precondition_forks.py',
         'scripts/audit_reflexion_action_shapes.py', 'scripts/state_representation_model.py',
         'scripts/continue_reflexion_pilot.py', 'scripts/summarize_reflexion.py']
LABEL_FILES = [f'repro/reflexion-20261023/{p}-semantic-labels.json'
               for p in ['state-representation','middle-context']]+[
    f'repro/reflexion-20261024/{p}-semantic-labels.json' for p in ['residual-context','multiround-diagnostic']]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check(value, name):
    require(value.get('sha256') == digest({k:v for k,v in value.items() if k!='sha256'}), name+' integrity')


def save(path, value):
    value['sha256'] = digest(value)
    write_json(path, value)
    return value


def cells(protocol):
    return [c for block in protocol['blocks'] for c in block['cells']]


def memory_for(protocol, cell):
    memory = [protocol['entries'][label] for label in cell['labels']]
    require(digest(memory) == cell['memory_sha256'], 'Starting memory hash changed')
    return memory


def validate_matrix(protocol):
    tasks = [c['task_id'] for c in protocol['cases']]
    require(len(tasks)==len(set(tasks))==12, 'Twelve distinct tasks required')
    require(protocol['seeds']==SEEDS and protocol['ports']==[8001,8002,8003,8004]
            and protocol['gpu_indices']==[0,1,2,3], 'Frozen seeds or services changed')
    rows = cells(protocol)
    require(len(rows)==len({c['id'] for c in rows})==180, 'Exactly 180 unique cells required')
    seen = set()
    for block in protocol['blocks']:
        task,seed = block['task_id'],block['seed']
        require(task in tasks and seed in SEEDS and block['worker']==tasks.index(task)%4, 'Task/seed/worker mismatch')
        require(len(block['cells'])==5, 'Each task/seed requires all five conditions')
        for cell in block['cells']:
            require(cell['task_id']==task and cell['seed']==seed and cell['condition'] in CONDITIONS, 'Cell identity mismatch')
            require(cell['labels']==CONDITIONS[cell['condition']], 'Unexpected memory or alignment')
            key=(task,seed,cell['condition'])
            require(key not in seen, 'Duplicate task/seed/condition')
            seen.add(key)
            memory_for(protocol,cell)
    require(seen=={(t,s,c) for t in tasks for s in SEEDS for c in CONDITIONS}, 'Incomplete matrix')
    return {'cells':180,'base_tasks':12,'seeds_per_task':3}


def verify_protocol(path):
    protocol=read_json(path); check(protocol,'Protocol'); verify_dependencies(protocol)
    require(provenance()['source_hashes']==protocol['source_hashes'], 'Agent source changed')
    require(protocol['limits']=={'additional_calls_including_replica_checks':10500,'seconds':10800}, 'Finite allowance changed')
    for name,text in protocol['entries'].items():
        require(digest(text)==protocol['entry_hashes'][name], 'Initial text changed')
    validate_matrix(protocol)
    return protocol


def prepare():
    require(not PROTOCOL.exists(), 'Preserve the existing frozen protocol')
    from run_multiround_diagnostic import verify_protocol as verify_parent
    parent=verify_parent(PARENT)
    cases=sorted(parent['cases'],key=lambda c:c['task_id']); ids={c['task_id'] for c in cases}
    for path in Path('runs').glob('reflexion-*/**/identity.json'):
        prior=read_json(path)
        require(not(prior.get('task',{}).get('task_id') in ids and prior.get('seed') in SEEDS), 'Selected new seed already used')
    holdout=read_json(Path('repro/baseline-20261019/holdout134.json'))
    require(not ids & {t['task_id'] for t in holdout['tasks']}, 'Holdout task selected')
    presence=read_json(Path('repro/reflexion-20261023/state-representation-presence-baseline.json'))
    primary=read_json(Path('repro/reflexion-20261023/state-representation-model-freeze.json'))
    check(presence,'Frozen presence model'); check(primary,'Frozen primary model')
    joint={bool(label):next(r['joint'] for r in presence['one_attempt_forecasts'] if r['labels']==label)
           for label in [[],['E']]}
    blocks=[]; forecasts=[]
    for index,case in enumerate(cases):
        # Each task stays on one replica; rotate seed and condition order independently of outcomes.
        ordered_seeds=SEEDS[index%3:]+SEEDS[:index%3]
        for seed in ordered_seeds:
            order=list(CONDITIONS); offset=(index+seed-SEEDS[0])%5; order=order[offset:]+order[:offset]
            block={'task_id':case['task_id'],'seed':seed,'worker':index%4,'cells':[]}
            for condition in order:
                labels=CONDITIONS[condition]
                cell={'id':f'correctness-{case["task_id"]}-{seed}-{condition}',
                      'task_id':case['task_id'],'seed':seed,'condition':condition,'labels':labels,
                      'memory_sha256':digest([parent['entries'][x] for x in labels])}
                block['cells'].append(cell)
                forecasts.append({'cell_id':cell['id'],'presence':joint['E' in labels],
                    'constant':predict(primary['all_fitted_models']['constant'],[])})
            blocks.append(block)
    protected=dict(parent['protected_files'])
    for name in FILES+LABEL_FILES:
        protected[name]=file_hash(Path(name))
    protocol={'scope':'Known-task diagnosis of explicit correct advice versus generic or empty memory',
        'parent_commit':'9c68d4dede1f4193e482b2fde1d89a131102b1da',
        'parent_protocol_sha256':parent['sha256'],'cases':cases,'entries':parent['entries'],
        'entry_hashes':parent['entry_hashes'],'blocks':blocks,'predictions':forecasts,
        'seeds':SEEDS,'ports':[8001,8002,8003,8004],'gpu_indices':[0,1,2,3],
        'limits':{'additional_calls_including_replica_checks':10500,'seconds':10800},
        'expected_cells':180,'maximum_attempts':180,'prior_global_calls':34591,
        'analysis':{'primary':'EC_minus_EN task-success probability difference',
            'contrasts':CONTRASTS,'bootstrap_replicates':20000,'bootstrap_seed':BOOTSTRAP_SEED,
            'aggregation':'Mean three paired seeds within task, then equal mean of twelve tasks',
            'primary_interval':'Task-cluster bootstrap; paired seed bootstrap is a sensitivity analysis',
            'interpretation':'Diagnostic, not equivalence or formal independent validation; degenerate empirical intervals do not establish exact zero probabilities',
            'semantic':'Full-text E/C/U/A; existing hashes immutable; human review only if actually performed',
            'conditional_failure':'Report conditional E proportions and unconditional joint failure-and-E separately; success selects different failure populations'},
        'parameter_refits':0,'holdout_model_calls':0,'source_presence_freeze_sha256':presence['sha256'],
        'protected_files':protected,'source_hashes':provenance()['source_hashes'],
        'seed_audit':{'selected_tasks':12,'new_seeds':SEEDS,'previous_actual_matches':0}}
    validate_matrix(protocol); save(PROTOCOL,protocol)
    print({'protocol_sha256':protocol['sha256'],'cells':180,'model_calls':0})


def execute_cell(root,protocol,case,cell,transport,allowance,attempt_runner=run_attempt):
    memory=memory_for(protocol,cell); path=root/'cells'/cell['id']/'trial-01'
    value=attempt_runner(case['task'],memory,1,path,transport,Path('data/alfworld'),cell['seed'],allowance,
        input_format='qwen3_non_thinking_actionline',decoding_profile='qwen3_non_thinking_recommended_v1')
    verify_attempt(value,path,case,cell,memory,1)
    checkpoint=read_json(path/'checkpoint.json')
    require(digest(checkpoint['transitions'])==value['trajectory_sha256'], 'Committed trajectory hash mismatch')
    return {'cell_id':cell['id'],'task_id':cell['task_id'],'seed':cell['seed'],
            'condition':cell['condition'],'success':value['success'],'decisions':value['decisions'],
            'result_sha256':file_hash(path/'result.json'),'memory_sha256':digest(memory)}


def worker(root,protocol,index):
    contract=verify_contract(root,protocol); rows=[]; status='complete'
    transport=CompletionTransport(f'http://127.0.0.1:{protocol["ports"][index]}/v1','local-development',180)
    require(transport._request('/models')==contract['serving'],'Worker service identity changed')
    cases={c['task_id']:c for c in protocol['cases']}
    try:
        for block in protocol['blocks']:
            if block['worker']!=index: continue
            for cell in block['cells']:
                row=execute_cell(root,protocol,cases[cell['task_id']],cell,transport,lambda:reserve_call(root,contract))
                rows.append(row); write_json(root/f'worker-{index}.json',{'status':'running','rows':rows})
                print(json.dumps(row),flush=True)
    except CalibrationStop as exc:
        status='allowance_stop'; (root/'STOP').write_text(str(exc),encoding='utf-8')
    except BaseException:
        status='integrity_or_infrastructure_stop'; (root/'STOP').write_text('Inspect worker; no automatic retry',encoding='utf-8')
        raise
    finally:
        write_json(root/f'worker-{index}.json',{'status':status,'rows':rows})


def run(root,protocol,path):
    with run_lock(root):
        require(not(root/'contract.json').exists(),'Existing campaign cannot be reset')
        before=ledger(); total=sum(v['dispatched'] for v in before.values())
        require(total==34591 and all(v['dispatched']==v['responses'] for v in before.values()),'Prior ledger mismatch or uncertainty')
        transport=CompletionTransport(f'http://127.0.0.1:{protocol["ports"][0]}/v1','local-development',180)
        health=transport._request('/models'); now=time.time(); current=provenance(); _,upstream=author_namespace()
        contract=save(root/'contract.json',{'protocol_sha256':protocol['sha256'],'created_unix':now,
            'deadline_unix':now+10800,'max_calls':10500,'prior_ledger':before,'source_hashes':current['source_hashes'],
            'native_hashes':current['native_hashes'],'serving':health,
            'authorization':'User continued the revised theory and minimal diagnosis roadmap on 2026-10-24; idle GPUs authorized'})
        write_json(root/'run.json',{'identity':{'protocol_sha256':protocol['sha256'],'campaign_sha256':contract['sha256'],
            'source_hashes':current['source_hashes'],'serving':health,'upstream':upstream,
            'input_format':'qwen3_non_thinking_actionline','decoding_profile':'qwen3_non_thinking_recommended_v1',
            'environment_action_alias':ACTION_ALIAS_PROFILE,
            'wrapper_sha256':file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256':file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))},'created_unix':now})
        replica_audit(root,protocol,contract)
        children=[]
        for i in range(4):
            log=(root/f'worker-{i}.log').open('a',encoding='utf-8')
            children.append((subprocess.Popen([sys.executable,__file__,'run','--protocol',str(path),'--run',str(root),'--worker',str(i)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[]
        for child,log in children: codes.append(child.wait());log.close()
        saved=[read_json(root/f'worker-{i}.json') for i in range(4)]
        rows=[r for s in saved for r in s['rows']]; actual=costs(root); reserved=read_json(root/'budget.json')['reserved']
        complete=len(rows)==len({r['cell_id'] for r in rows})==180 and not any(codes) and all(s['status']=='complete' for s in saved)
        summary=save(root/'summary.json',{'protocol_sha256':protocol['sha256'],'status':'complete' if complete else 'partial',
            'expected_cells':180,'completed_cells':len(rows),'rows':sorted(rows,key=lambda r:r['cell_id']),
            'worker_exit_codes':codes,'worker_statuses':[s['status'] for s in saved],
            'costs_incremental':actual,'reserved_calls':reserved,'elapsed_seconds':time.time()-now})
        print({k:v for k,v in summary.items() if k!='rows'},flush=True)
        require(complete and not actual['uncertain_or_unrecorded_attempts'] and reserved==actual['dispatched_calls'], 'Incomplete or unresolved campaign')


def collect(protocol,root):
    known={digest(text):'U' if name=='N' else name for name,text in protocol['entries'].items()}
    for name in LABEL_FILES:
        value=read_json(Path(name));check(value,'Previous labels')
        for e in value['entries']:
            require(e['sha256'] not in known or known[e['sha256']]==e['label'],'Previous label conflict')
            known[e['sha256']]=e['label']
    rows=[]; review={}; cases={c['task_id']:c for c in protocol['cases']}
    expected={c['id'] for c in cells(protocol)}
    require(all(p.parent.parent.name in expected and p.parent.name=='trial-01' for p in root.glob('cells/*/trial-*/result.json')), 'Unexpected raw result')
    for cell in cells(protocol):
        directory=root/'cells'/cell['id']/'trial-01'; result_path=directory/'result.json'
        if not result_path.exists():continue
        value=read_json(result_path); memory=memory_for(protocol,cell); case=cases[cell['task_id']]
        verify_attempt(value,directory,case,cell,memory,1)
        steps=read_json(directory/'checkpoint.json')['transitions']
        require(digest(steps)==value['trajectory_sha256'],'Trajectory changed')
        fresh=value['memory_after'][len(memory):]
        for text in fresh:
            sha=digest(text)
            if sha not in known:review.setdefault(sha,{'sha256':sha,'text':text,'sources':[]})['sources'].append(cell['id'])
        goal=Path(case['task']['gamefile']).parts[-3].split('-')[1].lower()
        events=procedure_events(steps,goal,'heat','microwave 1')
        rows.append({'cell_id':cell['id'],'task_id':cell['task_id'],'seed':cell['seed'],'condition':cell['condition'],
            'initial_labels':cell['labels'],'success':value['success'],'decisions':value['decisions'],
            'result_sha256':file_hash(result_path),'identity_sha256':value['identity_sha256'],
            'memory_sha256':digest(memory),'trajectory_sha256':value['trajectory_sha256'],
            'new_reflection_hashes':[digest(t) for t in fresh],
            'observable_events':{k:v if isinstance(v,bool) else len(v) for k,v in events.items()},
            'action_shapes':dict(Counter(action_shape(s['action'],s['submitted_action']) for s in steps)),
            'no_effect_shapes':dict(Counter(action_shape(s['action'],s['submitted_action']) for s in steps if s['observation']=='Nothing happens.'))})
    return rows,known,review


def effect_summary(rows):
    tasks=sorted({r['task_id'] for r in rows}); lookup={(r['task_id'],r['seed'],r['condition']):int(r['success']) for r in rows}
    require(len(rows)==len(lookup)==180,'Complete paired results required')
    rng=np.random.default_rng(BOOTSTRAP_SEED); ti=rng.integers(0,12,size=(20000,12))
    si=rng.integers(0,3,size=(20000,12,3)); out={}
    for name,(a,b) in CONTRASTS.items():
        matrix=np.asarray([[lookup[t,s,a]-lookup[t,s,b] for s in SEEDS] for t in tasks],dtype=float)
        means=matrix.mean(axis=1); distribution=means[ti].mean(axis=1)
        nested=matrix[ti[:,:,None],si].mean(axis=(1,2))
        out[name]={'mean':float(means.mean()),'task_bootstrap_95_interval':np.quantile(distribution,[.025,.975]).tolist(),
            'paired_seed_sensitivity_95_interval':np.quantile(nested,[.025,.975]).tolist(),
            'by_seed':{str(s):float(matrix[:,i].mean()) for i,s in enumerate(SEEDS)},
            'task_effects':[{'task_id':t,'mean':float(means[i]),'paired_differences':matrix[i].tolist()} for i,t in enumerate(tasks)],
            'paired_outcomes':dict(Counter('both_success' if lookup[t,s,a] and lookup[t,s,b] else 'first_only_success' if lookup[t,s,a] else 'second_only_success' if lookup[t,s,b] else 'both_failure' for t in tasks for s in SEEDS))}
    return {'primary':'EC_minus_EN','estimates':out,'bootstrap_seed':BOOTSTRAP_SEED,'replicates':20000,
            'independent_base_tasks':12,'seed_replicates_per_task':3,
            'interpretation':'Empirical task intervals and paired-seed sensitivity; not equivalence, iid scene inference or calibrated uniform transition bounds'}


def analyse(protocol,root,labels_path=None):
    rows,known,review=collect(protocol,root)
    write_json(root/'semantic-review.json',{'protocol_sha256':protocol['sha256'],'items':list(review.values())})
    if labels_path is None:
        print({'completed_cells':len(rows),'unique_new_texts':len(review)});return
    labels=read_json(labels_path);check(labels,'Semantic labels')
    require(labels['protocol_sha256']==protocol['sha256'],'Label/protocol binding')
    entries=labels['entries']
    require(len(entries)==len({e['sha256'] for e in entries})==len(review) and {e['sha256'] for e in entries}==set(review),'Semantic coverage')
    for e in entries:
        require(e['label'] in {'E','C','U','A'} and bool(e.get('reason')),'Invalid semantic decision')
        known[e['sha256']]=e['label']
    require(len(rows)==180,'All cells required')
    forecasts={p['cell_id']:p for p in protocol['predictions']}; groups=[]; metric_rows=[]
    for row in rows:
        row['outcome']='S' if row['success'] else known[row['new_reflection_hashes'][0]]
        row['E_event']=('restatement' if 'E' in row['initial_labels'] else 'generation') if row['outcome']=='E' else None
        for model in ['presence','constant']:
            metric_rows.append({'task_id':row['task_id'],'condition':row['condition'],'model':model,
                               **scores(forecasts[row['cell_id']][model],row['outcome'])})
    for condition in CONDITIONS:
        selected=[r for r in rows if r['condition']==condition]; counts=Counter(r['outcome'] for r in selected)
        groups.append({'condition':condition,'cells':len(selected),'successes':counts['S'],'joint_counts':dict(counts),
            'failure_conditional_E':counts['E']/(len(selected)-counts['S']) if counts['S']<len(selected) else None,
            'failure_and_E_joint_probability':counts['E']/len(selected),
            'observable_events':{k:sum(r['observable_events'][k] for r in selected) for k in selected[0]['observable_events']},
            'action_shapes':dict(sum((Counter(r['action_shapes']) for r in selected),Counter())),
            'no_effect_shapes':dict(sum((Counter(r['no_effect_shapes']) for r in selected),Counter()))})
    score_summary={m:{metric:float(np.mean([r[metric] for r in metric_rows if r['model']==m]))
                      for metric in ['brier','success_brier','log_loss']} for m in ['presence','constant']}
    summary=read_json(root/'summary.json');check(summary,'Run summary')
    analysis=save(root/'analysis.json',{'protocol_sha256':protocol['sha256'],'summary_sha256':summary['sha256'],
        'semantic_labels_sha256':labels['sha256'],'completed_cells':180,'independent_base_tasks':12,'seed_replicates_per_task':3,
        'successes':sum(r['success'] for r in rows),'rows':rows,'by_condition':groups,'effects':effect_summary(rows),
        'frozen_scores':score_summary,'semantic_review':{'reviewer':labels.get('reviewer'),'human_second_review':labels.get('human_second_review',False)},
        'parameter_refits':0,'holdout_model_calls':0,'analysis_driver_sha256':file_hash(Path(__file__))})
    print({'analysis_sha256':analysis['sha256'],'successes':analysis['successes'],'primary':analysis['effects']['estimates']['EC_minus_EN']})


def close(protocol,root,freeze):
    import re
    require(re.fullmatch('[0-9a-f]{40}',freeze) is not None,'Full freeze commit required')
    summary=read_json(root/'summary.json');analysis=read_json(root/'analysis.json');contract=read_json(root/'contract.json')
    audit=read_json(root/'protocol-audit.json');replica=read_json(root/'replica-audit.json')
    labels=read_json(root/'semantic-labels.json')
    for value,name in [(summary,'Summary'),(analysis,'Analysis'),(contract,'Contract'),(audit,'Audit'),(replica,'Replica'),(labels,'Labels')]:check(value,name)
    require(summary['completed_cells']==analysis['completed_cells']==180 and summary['worker_exit_codes']==[0]*4,'Incomplete campaign')
    require(all(v['protocol_sha256']==protocol['sha256'] for v in [summary,analysis,contract,labels]),'Artifact/protocol mismatch')
    require(analysis['summary_sha256']==summary['sha256'] and analysis['semantic_labels_sha256']==labels['sha256'],'Artifact binding changed')
    require(audit['passed'] and not audit['errors'] and replica['passed'] and len(replica['rows'])==8,'Audit failure')
    raw,_,_=collect(protocol,root); require(len(raw)==180,'Raw coverage')
    analysed={r['cell_id']:r for r in analysis['rows']}
    for row in raw:require(all(analysed[row['cell_id']].get(k)==v for k,v in row.items()),'Raw evidence changed')
    worker_rows={r['cell_id']:r for r in summary['rows']}
    require(len(worker_rows)==180 and set(worker_rows)==set(analysed),'Worker summary coverage')
    for key,row in worker_rows.items():
        require(all(analysed[key].get(k)==v for k,v in row.items()),'Worker summary differs from raw result')
    actual=costs(root);reserved=read_json(root/'budget.json')['reserved']
    require(actual==summary['costs_incremental'] and reserved==actual['dispatched_calls']<=10500,'Cost/reservation mismatch')
    require(actual['saved_responses']==reserved and actual['uncertain_or_unrecorded_attempts']==0,'Unconfirmed calls')
    stamps=[read_json(p)['dispatched_unix'] for p in root.glob('**/calls/*/attempt-*.request.json')]
    require(len(stamps)==reserved and min(stamps)>=contract['created_unix'] and max(stamps)<contract['deadline_unix'],'Dispatch envelope')
    current=ledger();before=contract['prior_ledger'];total=sum(v['dispatched'] for v in current.values())
    require(sum(v['dispatched'] for v in before.values())==34591 and total==34591+reserved==sum(v['responses'] for v in current.values()),'Global ledger mismatch')
    for name in set(current)|set(before):
        if name!=root.resolve().name:require(current.get(name,{'dispatched':0,'responses':0})==before.get(name,{'dispatched':0,'responses':0}),'Other journal changed')
    output=PROTOCOL.parent;artifacts={}
    for source,target in [('contract','contract'),('run','run'),('summary','raw-summary'),('analysis','results'),
                          ('replica-audit','replica-audit'),('protocol-audit','protocol-audit'),('action-shapes','action-shapes'),('semantic-labels','semantic-labels')]:
        src=root/(source+'.json');dst=output/(PREFIX+'-'+target+'.json')
        if dst.exists():require(dst.read_bytes()==src.read_bytes(),'Published evidence changed')
        else:shutil.copyfile(src,dst)
        artifacts[dst.name]=file_hash(dst)
    result={'protocol_sha256':protocol['sha256'],'freeze_commit':freeze,'stop_reason':'frozen_180_cells_complete',
        'completed_cells':180,'successes':analysis['successes'],'independent_base_tasks':12,'seed_replicates_per_task':3,
        'costs_incremental':actual,'replica_check_calls':8,'elapsed_seconds':summary['elapsed_seconds'],
        'prior_calls':34591,'all_calls':total,'all_responses':total,'parameter_refits':0,'holdout_model_calls':0,
        'artifacts':artifacts,'driver_sha256':file_hash(Path(__file__))}
    dst=output/(PREFIX+'-closeout.json')
    if dst.exists():require(read_json(dst)=={**result,'sha256':digest(result)},'Closeout changed')
    else:save(dst,result)
    print({k:v for k,v in result.items() if k!='artifacts'})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','validate','run','analyse','close'])
    p.add_argument('--protocol',type=Path,default=PROTOCOL);p.add_argument('--run',type=Path,default=Path('runs/reflexion-correctness-diagnostic-20261024'))
    p.add_argument('--worker',type=int);p.add_argument('--labels',type=Path);p.add_argument('--freeze-commit');a=p.parse_args()
    if a.mode=='prepare':prepare();return
    protocol=verify_protocol(a.protocol)
    if a.mode=='validate':print(validate_matrix(protocol));return
    if a.mode=='run':
        if a.worker is None:run(a.run,protocol,a.protocol)
        else:worker(a.run,protocol,a.worker)
    elif a.mode=='analyse':analyse(protocol,a.run,a.labels)
    else:close(protocol,a.run,a.freeze_commit)


if __name__=='__main__':main()
