#!/usr/bin/env python3
"""Exact native-text edits, paired actor attempts and fixed-input critic probes."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, run_lock, seed_for, write_json
from agent_reliability.llm import CallJournal
from agent_reliability.reference import ACTION_ALIAS_PROFILE, CompletionTransport
from agent_reliability.reflexion import CompletionIO, Events, author_namespace, run_attempt
from analyze_reflexion_precondition_forks import procedure_events
from audit_reflexion_action_shapes import action_shape
from continue_reflexion_pilot import ledger
from prepare_middle_context_transport import SOURCE_RUN, native_text
from run_correctness_diagnostic import check, require, save
from run_middle_context_transport import CalibrationStop, replica_audit, reserve_call, verify_contract
from run_multiround_diagnostic import verify_attempt
from run_residual_context import verify_dependencies
from summarize_reflexion import costs

PREFIX = 'native-clause'
PROTOCOL = Path('repro/reflexion-20261025/native-clause-protocol.json')
REGISTRY = PROTOCOL.with_name('native-clause-registry.json')
PARENT = Path('repro/reflexion-20261024/correctness-diagnostic-protocol.json')
SEEDS = [2107, 2108, 2109]
CONDITIONS = ['E', 'C', 'remove']
PANELS = ['actor', 'critic']
LABEL_FILES = [f'repro/reflexion-20261023/{p}-semantic-labels.json' for p in ['state-representation', 'middle-context']] + [
    f'repro/reflexion-20261024/{p}-semantic-labels.json' for p in ['residual-context', 'multiround-diagnostic', 'correctness-diagnostic']]
FILES = ['scripts/run_native_clause.py', 'tests/test_native_clause.py',
         'docs/experiments/native-clause-design-20261025.md', str(REGISTRY), str(PARENT),
         'scripts/run_correctness_diagnostic.py', 'scripts/run_multiround_diagnostic.py',
         'scripts/run_middle_context_transport.py', 'scripts/run_residual_context.py',
         'scripts/prepare_middle_context_transport.py', 'scripts/audit_reflexion_protocol.py',
         'scripts/audit_reflexion_action_shapes.py', 'scripts/analyze_reflexion_precondition_forks.py',
         'scripts/continue_reflexion_pilot.py', 'scripts/summarize_reflexion.py']


def edited_text(original, record, condition):
    require(digest(original) == record['source']['reflection_sha256'], 'Original reflection hash')
    require(condition in CONDITIONS, 'Unknown condition')
    previous = 0
    parts = []
    for edit in record['edits']:
        start, end = edit['start'], edit['end']
        require(previous <= start < end <= len(original), 'Overlapping or invalid source spans')
        require(digest(original[start:end]) == edit['source_span_sha256'], 'Target span hash')
        parts.append(original[previous:start])
        parts.append(original[start:end] if condition == 'E' else edit[
            'correct_replacement' if condition == 'C' else 'remove_replacement'])
        previous = end
    parts.append(original[previous:])
    text = ''.join(parts)
    require(text.strip() and digest(text) == record['variant_hashes'][condition], 'Edited reflection hash')
    return text


def cells(protocol):
    return [cell for block in protocol['blocks'] for cell in block['cells']]


def record_for(protocol, task):
    return next(r for r in protocol['registry']['entries'] if r['task_id'] == task)


def memory_for(protocol, cell):
    record = record_for(protocol, cell['task_id'])
    text = edited_text(native_text(record['source']), record, cell['condition'])
    memory = [protocol['entries']['N'], text, protocol['entries']['N']]
    require(digest(memory) == cell['memory_sha256'], 'Starting memory hash')
    return memory


def validate_matrix(protocol, resolver=memory_for):
    tasks = [c['task_id'] for c in protocol['cases']]
    require(len(tasks) == len(set(tasks)) == 12, 'Twelve fixed tasks required')
    require(protocol['seeds'] == SEEDS and protocol['ports'] == [8001,8002,8003,8004]
            and protocol['gpu_indices'] == [0,1,2,3], 'Frozen seed/replica matrix')
    rows = cells(protocol)
    require(len(rows) == len({c['id'] for c in rows}) == 216, '216 unique cells required')
    seen = set()
    for block in protocol['blocks']:
        require(block['worker'] == tasks.index(block['task_id']) % 4, 'Task/worker assignment')
        for cell in block['cells']:
            require(cell['task_id'] == block['task_id'] and cell['seed'] == block['seed'], 'Block binding')
            key = (cell['task_id'], cell['seed'], cell['panel'], cell['condition'])
            require(key not in seen, 'Duplicate cell'); seen.add(key)
            require(len(resolver(protocol, cell)) == 3, 'Three actual memory entries required')
    require(seen == {(t,s,p,c) for t in tasks for s in SEEDS for p in PANELS for c in CONDITIONS}, 'Incomplete matrix')
    return {'actor_attempts':108, 'fixed_critic_calls':108, 'base_tasks':12, 'seeds_per_task':3}


def fixed_source(record):
    path = Path(SOURCE_RUN)/'cells'/record['source']['cell_id']/'trial-01'
    result = read_json(path/'result.json')
    native_text(record['source'])
    require(result['task_id']==record['task_id'] and not result['success'] and file_hash(path/'trial.log') == record['fixed_failure']['trial_log_sha256'], 'Fixed failure log')
    require(result['decisions'] == record['fixed_failure']['decisions'], 'Fixed actor decision count')
    return path, result


def verify_protocol(path):
    value = read_json(path); check(value,'Protocol'); verify_dependencies(value)
    require(provenance()['source_hashes'] == value['source_hashes'], 'Agent source changed')
    check(value['registry'],'Registry')
    require(value['registry']['sha256'] == value['registry_sha256'], 'Registry binding')
    require(value['limits'] == {'additional_calls_including_replica_checks':6500,'seconds':10800}, 'Finite allowance changed')
    for record in value['registry']['entries']: fixed_source(record)
    validate_matrix(value)
    return value


def prepare():
    require(not PROTOCOL.exists(), 'Preserve frozen protocol')
    from run_correctness_diagnostic import verify_protocol as verify_parent
    parent = verify_parent(PARENT); registry = read_json(REGISTRY); check(registry,'Registry')
    records = registry['entries']; cases = sorted(parent['cases'],key=lambda c:c['task_id'])
    require({r['task_id'] for r in records} == {c['task_id'] for c in cases}, 'Source cohort changed')
    source_registry = read_json(Path('repro/reflexion-20261024/residual-context-registry.json'))
    require(registry['source_registry_sha256'] == source_registry['sha256'], 'Prior source registry binding')
    original_sources = {r['task_id']:r['source'] for r in source_registry['entries']}
    protected = {**parent['protected_files']}
    for record in records:
        require(record['source'] == original_sources[record['task_id']], 'Native source selection changed')
        source_dir = Path(SOURCE_RUN)/'cells'/record['source']['cell_id']/'trial-01'
        result = read_json(source_dir/'result.json')
        record['fixed_failure'] = {'trial_log_sha256':file_hash(source_dir/'trial.log'),'decisions':result['decisions']}
        for name in ['trial.log','result.json']: protected[str(source_dir/name)] = file_hash(source_dir/name)
        fixed_source(record)
        for condition in CONDITIONS: edited_text(native_text(record['source']),record,condition)
    # The embedded registry adds source-log identity, retaining the public edit registry's separate hash.
    registry.pop('sha256'); registry['sha256'] = digest(registry)
    for path in Path('runs').glob('reflexion-*/**/identity.json'):
        prior = read_json(path)
        require(not(prior.get('task',{}).get('task_id') in original_sources and prior.get('seed') in SEEDS), 'New seed already used')
    holdout = read_json(Path('repro/baseline-20261019/holdout134.json'))
    require(not set(original_sources)&{t['task_id'] for t in holdout['tasks']}, 'Holdout selected')
    blocks = []; working = {'entries':parent['entries'],'registry':registry}
    for index, case in enumerate(cases):
        for seed in SEEDS[index%3:]+SEEDS[:index%3]:
            block = {'task_id':case['task_id'],'seed':seed,'worker':index%4,'cells':[]}
            order = CONDITIONS[(index+seed-SEEDS[0])%3:]+CONDITIONS[:(index+seed-SEEDS[0])%3]
            for panel in PANELS if index%2==0 else PANELS[::-1]:
                for condition in order:
                    cell = {'id':f'native-{panel}-{case["task_id"]}-{seed}-{condition}',
                            'task_id':case['task_id'],'seed':seed,'panel':panel,'condition':condition}
                    record = record_for(working,case['task_id'])
                    middle = edited_text(native_text(record['source']),record,condition)
                    cell['memory_sha256'] = digest([parent['entries']['N'],middle,parent['entries']['N']])
                    block['cells'].append(cell)
            blocks.append(block)
    for name in FILES+LABEL_FILES: protected[name] = file_hash(Path(name))
    protocol = {'scope':'Native target-clause edits with actor attempts and fixed failure-input critic probes',
        'parent_commit':'d68642201524155152ba6d40105f88c3ef9fabfc','parent_protocol_sha256':parent['sha256'],
        'cases':cases,'entries':parent['entries'],'registry':registry,'registry_sha256':registry['sha256'],
        'blocks':blocks,'seeds':SEEDS,'ports':[8001,8002,8003,8004],'gpu_indices':[0,1,2,3],
        'limits':{'additional_calls_including_replica_checks':6500,'seconds':10800},
        'expected_cells':216,'maximum_actor_attempts':108,'fixed_critic_calls':108,'prior_global_calls':41181,
        'analysis':{'co_primary':['actor C-minus-E success','fixed critic C-minus-E probability of writing E'],
            'intervals':'Task-cluster bootstrap 20000, seed2026102501; simultaneous Bonferroni97.5% per primary;95% descriptive and paired-seed sensitivity',
            'aggregation':'Average three paired seeds within task,then equal mean over12 tasks',
            'secondary':['remove-minus-E','C-minus-remove','actor failure joint E,C,U,A','fixed critic output E,C,U,A'],
            'unknown_A':'For critic E probability, report A-as-nonE point estimates plus A-as-E alternative and [E,E+A] outcome bounds; A is never silently discarded',
            'selection':'Known development tasks and fixed twelve E sources; new seeds do not create independent tasks',
            'decision':'No refitting or equivalence claim; actor and fixed-input critic effects describe distinct mechanisms'},
        'parameter_refits':0,'holdout_model_calls':0,'protected_files':protected,'source_hashes':provenance()['source_hashes']}
    validate_matrix(protocol); save(PROTOCOL,protocol)
    print({'protocol_sha256':protocol['sha256'],**validate_matrix(protocol),'new_model_calls':0})


def critic_probe(root, protocol, case, cell, transport, allowance):
    record = record_for(protocol,cell['task_id']); source_dir, source = fixed_source(record)
    memory = memory_for(protocol,cell); path = root/'critic-cells'/cell['id']/'trial-01'
    ns, receipt = author_namespace()
    query = ns['_generate_reflection_query']((source_dir/'trial.log').read_text(),memory)
    identity = {'task':case['task'],'seed':cell['seed'],'trial':1,'memory':memory,
        'upstream':receipt,'source_log_sha256':record['fixed_failure']['trial_log_sha256'],
        'query_sha256':digest(query),'source_actor_decisions':source['decisions'],
        'input_format':'qwen3_non_thinking_actionline','decoding_profile':'qwen3_non_thinking_recommended_v1'}
    identity['sha256'] = digest(identity)
    require(not(path/'identity.json').exists(), 'Do not repeat a critic probe automatically')
    write_json(path/'identity.json',identity)
    journal = CallJournal(path/'calls',transport,2)
    io = CompletionIO(journal,case['task'],1,cell['seed'],Events(path/'events.jsonl'),allowance,
                      'qwen3_non_thinking_actionline','qwen3_non_thinking_recommended_v1')
    io.role = 'reflector'; io.index = source['decisions']; ns['get_completion'] = io
    config = [{'name':'env_0','memory':list(memory),'is_success':False,'skip':False}]
    ns['update_memory'](str(source_dir/'trial.log'),config)
    after = config[0]['memory']; require(after[:-1]==memory and len(after)==4 and after[-1].strip(), 'Critic append contract')
    result = {'cell_id':cell['id'],'task_id':cell['task_id'],'seed':cell['seed'],'condition':cell['condition'],
        'identity_sha256':identity['sha256'],'memory_sha256':digest(memory),'query_sha256':digest(query),
        'source_log_sha256':identity['source_log_sha256'],'reflection':after[-1],
        'reflection_sha256':digest(after[-1]),'source_actor_decisions':source['decisions']}
    save(path/'result.json',result)
    return {k:v for k,v in result.items() if k!='reflection'} | {'panel':'critic','result_sha256':file_hash(path/'result.json')}


def actor_attempt(root, protocol, case, cell, transport, allowance):
    memory = memory_for(protocol,cell); path = root/'cells'/cell['id']/'trial-01'
    result = run_attempt(case['task'],memory,1,path,transport,Path('data/alfworld'),cell['seed'],allowance,
        input_format='qwen3_non_thinking_actionline',decoding_profile='qwen3_non_thinking_recommended_v1')
    verify_attempt(result,path,case,cell,memory,1)
    return {'cell_id':cell['id'],'task_id':cell['task_id'],'seed':cell['seed'],'panel':'actor',
            'condition':cell['condition'],'success':result['success'],'decisions':result['decisions'],
            'memory_sha256':digest(memory),'result_sha256':file_hash(path/'result.json')}


def worker(root, protocol, index):
    contract=verify_contract(root,protocol); cases={c['task_id']:c for c in protocol['cases']}; rows=[];status='complete'
    transport=CompletionTransport(f'http://127.0.0.1:{protocol["ports"][index]}/v1','local-development',180)
    require(transport._request('/models')==contract['serving'],'Worker identity changed')
    try:
        for block in protocol['blocks']:
            if block['worker']!=index:continue
            for cell in block['cells']:
                runner=actor_attempt if cell['panel']=='actor' else critic_probe
                row=runner(root,protocol,cases[cell['task_id']],cell,transport,lambda:reserve_call(root,contract))
                rows.append(row);write_json(root/f'worker-{index}.json',{'status':'running','rows':rows})
                print({k:row[k] for k in ['cell_id','panel','condition']},flush=True)
    except CalibrationStop as exc:
        status='allowance_stop';(root/'STOP').write_text(str(exc))
    except BaseException:
        status='integrity_or_infrastructure_stop';(root/'STOP').write_text('Inspect evidence before any retry')
        raise
    finally:write_json(root/f'worker-{index}.json',{'status':status,'rows':rows})


def run(root,protocol,path):
    with run_lock(root):
        require(not(root/'contract.json').exists(),'Existing campaign cannot be reset')
        before=ledger();require(sum(v['dispatched'] for v in before.values())==41181 and all(v['dispatched']==v['responses'] for v in before.values()),'Prior call ledger')
        transport=CompletionTransport('http://127.0.0.1:8001/v1','local-development',180)
        health=transport._request('/models');now=time.time();current=provenance();_,upstream=author_namespace()
        contract=save(root/'contract.json',{'protocol_sha256':protocol['sha256'],'created_unix':now,
            'deadline_unix':now+10800,'max_calls':6500,'prior_ledger':before,'source_hashes':current['source_hashes'],
            'native_hashes':current['native_hashes'],'serving':health,'authorization':'User authorized the next native-reflection clause diagnostic on2026-10-25 and use of idle GPUs'})
        write_json(root/'run.json',{'identity':{'protocol_sha256':protocol['sha256'],'campaign_sha256':contract['sha256'],
            'source_hashes':current['source_hashes'],'serving':health,'upstream':upstream,
            'input_format':'qwen3_non_thinking_actionline','decoding_profile':'qwen3_non_thinking_recommended_v1',
            'environment_action_alias':ACTION_ALIAS_PROFILE,'wrapper_sha256':file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256':file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))},'created_unix':now})
        replica_audit(root,protocol,contract);children=[]
        for index in range(4):
            log=(root/f'worker-{index}.log').open('a')
            children.append((subprocess.Popen([sys.executable,__file__,'run','--protocol',str(path),'--run',str(root),'--worker',str(index)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[]
        for child,log in children:codes.append(child.wait());log.close()
        workers=[read_json(root/f'worker-{i}.json') for i in range(4)];rows=[r for w in workers for r in w['rows']]
        complete=len(rows)==len({r['cell_id'] for r in rows})==216 and codes==[0]*4 and all(w['status']=='complete' for w in workers)
        actual=costs(root);reserved=read_json(root/'budget.json')['reserved']
        completed_tasks=sum(all(any(r['cell_id']==c['id'] for r in rows) for b in protocol['blocks'] if b['task_id']==task for c in b['cells']) for task in {c['task_id'] for c in protocol['cases']})
        save(root/'summary.json',{'protocol_sha256':protocol['sha256'],'status':'complete' if complete else 'partial',
            'expected_cells':216,'completed_cells':len(rows),'expected_tasks':12,'completed_tasks':completed_tasks,
            'worker_exit_codes':codes,'worker_statuses':[w['status'] for w in workers],'rows':sorted(rows,key=lambda r:r['cell_id']),
            'costs_incremental':actual,'reserved_calls':reserved,'elapsed_seconds':time.time()-now})
        require(complete and reserved==actual['dispatched_calls']==actual['saved_responses'] and not actual['uncertain_or_unrecorded_attempts'],'Incomplete or unconfirmed campaign')
        print({'complete':True,'cells':216,'costs':actual,'elapsed_seconds':time.time()-now})


def collect(protocol,root):
    known={}
    for name in LABEL_FILES:
        value=read_json(Path(name));check(value,'Prior labels')
        for e in value['entries']:
            require(e['sha256'] not in known or known[e['sha256']]==e['label'],'Prior label conflict');known[e['sha256']]=e['label']
    for record in protocol['registry']['entries']:
        for condition,label in record['variant_labels'].items():
            sha=record['variant_hashes'][condition]
            require(sha not in known or known[sha]==label,'Curated variant conflicts with prior label');known[sha]=label
    rows=[];review={};cases={c['task_id']:c for c in protocol['cases']}
    expected_ids={c['id'] for c in cells(protocol)}
    for folder in ['cells','critic-cells']:
        require(all(p.parent.parent.name in expected_ids for p in (root/folder).glob('*/trial-*/result.json')),'Unexpected result cell')
    for cell in cells(protocol):
        folder='cells' if cell['panel']=='actor' else 'critic-cells';path=root/folder/cell['id']/'trial-01'
        if not(path/'result.json').exists():continue
        result=read_json(path/'result.json');memory=memory_for(protocol,cell)
        row={'cell_id':cell['id'],'task_id':cell['task_id'],'seed':cell['seed'],'panel':cell['panel'],'condition':cell['condition'],
             'memory_sha256':digest(memory),'result_sha256':file_hash(path/'result.json')}
        if cell['panel']=='actor':
            verify_attempt(result,path,cases[cell['task_id']],cell,memory,1)
            steps=read_json(path/'checkpoint.json')['transitions'];require(digest(steps)==result['trajectory_sha256'],'Actor trajectory integrity')
            fresh=result['memory_after'][len(memory):]
            goal=Path(cases[cell['task_id']]['task']['gamefile']).parts[-3].split('-')[1].lower()
            events=procedure_events(steps,goal,'heat','microwave 1')
            row.update(success=result['success'],decisions=result['decisions'],trajectory_sha256=result['trajectory_sha256'],
                observable_events={k:v if isinstance(v,bool) else len(v) for k,v in events.items()},
                action_shapes=dict(Counter(action_shape(s['action'],s['submitted_action']) for s in steps)))
        else:
            check(result,'Critic result');identity=read_json(path/'identity.json');check(identity,'Critic identity')
            record=record_for(protocol,cell['task_id']);source_dir,source=fixed_source(record)
            ns,_=author_namespace();query=ns['_generate_reflection_query']((source_dir/'trial.log').read_text(),memory)
            require(identity['task']==cases[cell['task_id']]['task'] and identity['seed']==cell['seed'] and identity['memory']==memory,'Critic identity binding')
            require(identity['sha256']==result['identity_sha256'] and digest(query)==identity['query_sha256']==result['query_sha256'],'Critic query binding')
            require(identity['source_log_sha256']==result['source_log_sha256']==record['fixed_failure']['trial_log_sha256'],'Critic failure-log binding')
            response=read_json(path/'calls'/f'reflector-{source["decisions"]:04d}'/'response.json')
            request=read_json(path/'calls'/f'reflector-{source["decisions"]:04d}'/'attempt-000.request.json')
            wrapper=read_json(Path('configs/qwen3_8b_non_thinking_wrapper.json'))
            require(request['payload']['prompt']==wrapper['prefix']+query+wrapper['suffix'],'Actual fixed-critic prompt')
            require(request['payload']['seed']==seed_for(cell['task_id'],cell['seed'],1,source['decisions'],'reflector'),'Actual fixed-critic seed')
            require(response['payload_sha256']==request['payload_sha256']==digest(request['payload']),'Critic request/response binding')
            require(response['response']['choices'][0]['text']==result['reflection'] and digest(result['reflection'])==result['reflection_sha256'],'Critic response binding')
            fresh=[result['reflection']];row.update({k:v for k,v in result.items() if k not in ['reflection','sha256']})
        row['new_reflection_hashes']=[digest(t) for t in fresh]
        for text in fresh:
            sha=digest(text)
            if sha not in known:review.setdefault(sha,{'sha256':sha,'text':text,'sources':[]})['sources'].append(cell['id'])
        rows.append(row)
    return rows,known,review


def contrasts(rows,panel,indicator):
    selected=[r for r in rows if r['panel']==panel];tasks=sorted({r['task_id'] for r in selected})
    lookup={(r['task_id'],r['seed'],r['condition']):indicator(r) for r in selected}
    require(len(lookup)==108 and len(tasks)==12,'Full paired panel required')
    rng=np.random.default_rng(2026102501);ti=rng.integers(0,12,size=(20000,12));si=rng.integers(0,3,size=(20000,12,3));out={}
    for a,b in [('C','E'),('remove','E'),('C','remove')]:
        matrix=np.asarray([[lookup[t,s,a]-lookup[t,s,b] for s in SEEDS] for t in tasks],float)
        means=matrix.mean(axis=1);dist=means[ti].mean(axis=1);nested=matrix[ti[:,:,None],si].mean(axis=(1,2))
        out[a+'_minus_'+b]={'mean':float(means.mean()),'task_95':np.quantile(dist,[.025,.975]).tolist(),
            'task_97_5':np.quantile(dist,[.0125,.9875]).tolist(),'paired_seed_95':np.quantile(nested,[.025,.975]).tolist(),
            'by_seed':{str(s):float(matrix[:,i].mean()) for i,s in enumerate(SEEDS)},
            'task_effects':[{'task_id':t,'mean':float(means[i]),'paired_differences':matrix[i].tolist()} for i,t in enumerate(tasks)],
            'paired_outcomes':dict(Counter(('both_positive' if lookup[t,s,a] and lookup[t,s,b] else 'first_only' if lookup[t,s,a] else 'second_only' if lookup[t,s,b] else 'both_negative') for t in tasks for s in SEEDS))}
    return out


def analyse(protocol,root,labels_path=None):
    rows,known,review=collect(protocol,root);write_json(root/'semantic-review.json',{'protocol_sha256':protocol['sha256'],'items':list(review.values())})
    if labels_path is None:print({'cells':len(rows),'new_texts':len(review)});return
    labels=read_json(labels_path);check(labels,'Labels');require(labels['protocol_sha256']==protocol['sha256'],'Label/protocol binding')
    require(len(labels['entries'])==len({e['sha256'] for e in labels['entries']})==len(review) and {e['sha256'] for e in labels['entries']}==set(review),'Full semantic coverage')
    for e in labels['entries']:require(e['label'] in {'E','C','U','A'} and e.get('reason'),'Semantic category');known[e['sha256']]=e['label']
    require(len(rows)==216,'Full matrix required')
    for row in rows:
        row['outcome']='S' if row.get('success') else known[row['new_reflection_hashes'][0]]
        row['E_event']=('restatement' if row['condition']=='E' else 'generation') if row['outcome']=='E' else None
    groups=[]
    for panel in PANELS:
        for condition in CONDITIONS:
            chosen=[r for r in rows if r['panel']==panel and r['condition']==condition];counts=Counter(r['outcome'] for r in chosen)
            groups.append({'panel':panel,'condition':condition,'cells':len(chosen),'outcome_counts':dict(counts),
                'E_proportion_bounds':[counts['E']/36,(counts['E']+counts['A'])/36],
                'failure_conditional_E':counts['E']/(36-counts['S']) if counts['S']<36 else None})
    actor=contrasts(rows,'actor',lambda r:int(r['success']))
    critic=contrasts(rows,'critic',lambda r:int(r['outcome']=='E'))
    critic_a=contrasts(rows,'critic',lambda r:int(r['outcome'] in {'E','A'}))
    ambiguity_bounds={}
    for a,b in [('C','E'),('remove','E'),('C','remove')]:
        first=next(g for g in groups if g['panel']=='critic' and g['condition']==a)['E_proportion_bounds']
        second=next(g for g in groups if g['panel']=='critic' and g['condition']==b)['E_proportion_bounds']
        ambiguity_bounds[a+'_minus_'+b]=[first[0]-second[1],first[1]-second[0]]
    summary=read_json(root/'summary.json');check(summary,'Summary')
    result=save(root/'analysis.json',{'protocol_sha256':protocol['sha256'],'summary_sha256':summary['sha256'],
        'semantic_labels_sha256':labels['sha256'],'completed_cells':216,'actor_attempts':108,'fixed_critic_calls':108,
        'actor_successes':sum(r.get('success',False) for r in rows),'base_tasks':12,'seeds_per_task':3,
        'rows':rows,'by_condition':groups,'actor_success_contrasts':actor,'fixed_critic_E_contrasts':critic,
        'fixed_critic_E_or_A_sensitivity':critic_a,'fixed_critic_ambiguity_bounds':ambiguity_bounds,
        'primary_intervals':'Bonferroni97.5% per co-primary;95% intervals descriptive',
        'semantic_review':{'reviewer':labels.get('reviewer'),'human_second_review':labels.get('human_second_review',False)},
        'parameter_refits':0,'holdout_model_calls':0,'driver_sha256':file_hash(Path(__file__))})
    print({'analysis_sha256':result['sha256'],'actor_primary':actor['C_minus_E'],'critic_primary':critic['C_minus_E']})


def audit(root,protocol):
    summary=read_json(root/'summary.json');check(summary,'Summary')
    require(summary['status']=='complete' and summary['worker_exit_codes']==[0]*4 and summary['completed_cells']==216 and summary['completed_tasks']==12,'Incomplete cohort')
    rows,_,_=collect(protocol,root)
    require({r['cell_id'] for r in rows}=={c['id'] for c in cells(protocol)},'Audit raw coverage')
    subprocess.run([sys.executable,'scripts/audit_reflexion_protocol.py','--run',str(root),'--output',str(root/'protocol-audit.json')],check=True)
    import alfworld
    import audit_reflexion_action_shapes as shapes
    write_json(root/'action-shapes.json',shapes.audit(root,Path(alfworld.__file__).parent/'data/alfred.twl2'))


def close(root,protocol,freeze):
    import re
    require(re.fullmatch('[0-9a-f]{40}',freeze or '') is not None,'Full precollection freeze required')
    summary=read_json(root/'summary.json');analysis=read_json(root/'analysis.json');contract=read_json(root/'contract.json')
    labels=read_json(root/'semantic-labels.json');audit_report=read_json(root/'protocol-audit.json');replica=read_json(root/'replica-audit.json')
    for v in [summary,analysis,contract,labels,audit_report,replica]:check(v,'Final artifact')
    require(all(v['protocol_sha256']==protocol['sha256'] for v in [summary,analysis,contract,labels]),'Protocol binding')
    require(analysis['summary_sha256']==summary['sha256'] and analysis['semantic_labels_sha256']==labels['sha256'],'Analysis binding')
    require(audit_report['passed'] and not audit_report['errors'] and replica['passed'] and len(replica['rows'])==8,'Audit failure')
    raw,_,_=collect(protocol,root);lookup={r['cell_id']:r for r in analysis['rows']}
    require(len(raw)==len(lookup)==216 and summary['worker_exit_codes']==[0]*4,'Complete coverage')
    for row in raw:require(all(lookup[row['cell_id']].get(k)==v for k,v in row.items()),'Changed raw result')
    for row in summary['rows']:require(all(lookup[row['cell_id']].get(k)==v for k,v in row.items()),'Changed worker row')
    actual=costs(root);reserved=read_json(root/'budget.json')['reserved']
    require(actual==summary['costs_incremental'] and reserved==actual['dispatched_calls']==actual['saved_responses']<=6500 and not actual['uncertain_or_unrecorded_attempts'],'Final cost ledger')
    stamps=[read_json(p)['dispatched_unix'] for p in root.glob('**/calls/*/attempt-*.request.json')]
    require(len(stamps)==reserved and min(stamps)>=contract['created_unix'] and max(stamps)<contract['deadline_unix'],'Dispatch envelope')
    current=ledger();before=contract['prior_ledger'];total=sum(v['dispatched'] for v in current.values())
    require(total==41181+reserved==sum(v['responses'] for v in current.values()),'Global cost ledger')
    for name in set(current)|set(before):
        if name!=root.resolve().name:require(current.get(name,{'dispatched':0,'responses':0})==before.get(name,{'dispatched':0,'responses':0}),'Other campaign changed')
    output=PROTOCOL.parent;artifacts={}
    for source,target in [('contract','contract'),('run','run'),('summary','raw-summary'),('analysis','results'),('semantic-labels','semantic-labels'),('replica-audit','replica-audit'),('protocol-audit','protocol-audit'),('action-shapes','action-shapes')]:
        src=root/(source+'.json');dst=output/(PREFIX+'-'+target+'.json')
        if dst.exists():require(dst.read_bytes()==src.read_bytes(),'Existing published evidence changed')
        else:shutil.copyfile(src,dst)
        artifacts[dst.name]=file_hash(dst)
    result={'protocol_sha256':protocol['sha256'],'freeze_commit':freeze,'stop_reason':'frozen108actor_and108critic_complete',
        'actor_attempts':108,'fixed_critic_calls':108,'actor_successes':analysis['actor_successes'],'base_tasks':12,'seeds_per_task':3,
        'costs_incremental':actual,'replica_check_calls':8,'elapsed_seconds':summary['elapsed_seconds'],
        'prior_calls':41181,'all_calls':total,'all_responses':total,'parameter_refits':0,'holdout_model_calls':0,
        'artifacts':artifacts,'driver_sha256':file_hash(Path(__file__))}
    target=output/(PREFIX+'-closeout.json')
    if target.exists():require(read_json(target)=={**result,'sha256':digest(result)},'Closeout changed')
    else:save(target,result)
    print({k:v for k,v in result.items() if k!='artifacts'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','validate','run','analyse','audit','close'])
    parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--run',type=Path,default=Path('runs/reflexion-native-clause-20261025'))
    parser.add_argument('--worker',type=int);parser.add_argument('--labels',type=Path);parser.add_argument('--freeze-commit');args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    protocol=verify_protocol(args.protocol)
    if args.mode=='validate':print(validate_matrix(protocol))
    elif args.mode=='run':run(args.run,protocol,args.protocol) if args.worker is None else worker(args.run,protocol,args.worker)
    elif args.mode=='analyse':analyse(protocol,args.run,args.labels)
    elif args.mode=='audit':audit(args.run,protocol)
    else:close(args.run,protocol,args.freeze_commit)


if __name__=='__main__':main()
