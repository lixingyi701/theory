#!/usr/bin/env python3
"""Bounded common-kernel calibration with original actor and natural reflection updates."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time
from agent_reliability.cli import provenance
from agent_reliability.io import digest,file_hash,read_json,run_lock,write_json
from agent_reliability.reference import ACTION_ALIAS_PROFILE,CompletionTransport
from agent_reliability.reflexion import author_namespace
from continue_reflexion_pilot import ledger
from prepare_shared_calibration import PROTOCOL,PARTITIONS,STATES,save
from run_middle_context_transport import CalibrationStop,replica_audit,reserve_call,verify_contract
from run_multiround_diagnostic import execute_trajectory,trajectory_summary
from run_residual_context import verify_dependencies
from summarize_reflexion import costs

ROOT = Path('runs/reflexion-shared-calibration-20261025')
PUBLIC = PROTOCOL.parent


def initial_memory(protocol,trajectory):
    memory = [protocol['entries'][k] for k in trajectory['initial_labels']]
    if digest(memory)!=trajectory['initial_memory_sha256']: raise ValueError('Initial memory changed')
    return memory


def validate_matrix(protocol):
    tasks = [c['task_id'] for c in protocol['cases']]
    if len(tasks)!=len(set(tasks)) or len(tasks)!=36: raise ValueError('36 unique calibration tasks required')
    counts = Counter(c['task']['scene_id'] for c in protocol['cases'])
    if len(counts)!=18 or set(counts.values())!={2}: raise ValueError('18 scenes, two tasks each required')
    expected = {(t,s) for t in tasks for s in STATES}; seen = set()
    for row in protocol['trajectories']:
        key = (row['task_id'],row['condition'])
        if key in seen or key not in expected: raise ValueError('Duplicate/unknown trajectory')
        seen.add(key)
        if row['initial_labels']!=STATES[row['condition']] or row['seed']!=2110+list(STATES).index(row['condition']) or row['max_attempts']!=2:
            raise ValueError('Frozen memory, seed or horizon changed')
        if row['worker']!=tasks.index(row['task_id'])%3: raise ValueError('Worker assignment changed')
        initial_memory(protocol,row)
    if seen!=expected or len(protocol['trajectories'])!=252: raise ValueError('Incomplete matrix')
    if protocol['ports']!=[8001,8003,8004] or protocol['gpu_indices']!=[0,2,3]: raise ValueError('Replica mapping changed')
    partition = read_json(PARTITIONS)
    if partition['sha256']!=digest({k:v for k,v in partition.items() if k!='sha256'}) or protocol['partitions_sha256']!=partition['sha256']:
        raise ValueError('Partition integrity')
    if {c['task_id'] for c in protocol['cases']}!={t['task_id'] for t in partition['calibration_tasks']}:
        raise ValueError('Calibration partition changed')
    forbidden = {t['task_id'] for t in partition['reserved_validation_tasks']}|{t['task_id'] for t in read_json(Path('repro/baseline-20261019/holdout134.json'))['tasks']}
    if set(tasks)&forbidden: raise ValueError('Validation task selected')
    return {'tasks':36,'scenes':18,'trajectories':252,'maximum_attempts':504}


def verify_protocol():
    protocol = read_json(PROTOCOL)
    if protocol['sha256']!=digest({k:v for k,v in protocol.items() if k!='sha256'}): raise ValueError('Protocol integrity')
    verify_dependencies(protocol)
    if provenance()['source_hashes']!=protocol['source_hashes']: raise ValueError('Agent source changed')
    if protocol['limits']!={'additional_calls_including_replica_checks':28000,'seconds':14400}: raise ValueError('Allowance changed')
    for k,v in protocol['entries'].items():
        if digest(v)!=protocol['entry_hashes'][k]: raise ValueError('Entry changed')
    validate_matrix(protocol)
    return protocol


def worker(root,protocol,index):
    contract = verify_contract(root,protocol); port = protocol['ports'][index]
    transport = CompletionTransport(f'http://127.0.0.1:{port}/v1','local-development',180)
    if transport._request('/models')!=contract['serving']: raise ValueError('Worker serving identity changed')
    cases = {c['task_id']:c for c in protocol['cases']}; rows=[]; trajectories=[]; status='complete'
    def store(state): write_json(root/f'worker-{index}.json',{'status':state,'rows':rows,'trajectories':trajectories})
    try:
        for trajectory in protocol['trajectories']:
            if trajectory['worker']!=index: continue
            trajectories.append(trajectory_summary(trajectory,[])); store('running')
            def observed(row,summary):
                row['condition'] = trajectory['condition']; row['port'] = port
                rows.append(row); trajectories[-1]=summary; store('running')
                print({k:row[k] for k in ['trajectory_id','trial','success']},flush=True)
            trajectories[-1] = execute_trajectory(root,protocol,cases[trajectory['task_id']],trajectory,
                transport,lambda:reserve_call(root,contract),observed,memory_resolver=initial_memory)
            store('running')
    except CalibrationStop as exc:
        status='allowance_stop';(root/'STOP').write_text(str(exc))
    except BaseException:
        status='infrastructure_or_integrity_stop';(root/'STOP').write_text('Inspect before any retry');raise
    finally: store(status)


def run(root,protocol):
    with run_lock(root):
        if (root/'contract.json').exists(): raise FileExistsError('Do not reset campaign')
        before=ledger()
        if sum(v['dispatched'] for v in before.values())!=44939 or any(v['dispatched']!=v['responses'] for v in before.values()):
            raise ValueError('Preceding ledger changed')
        health=CompletionTransport('http://127.0.0.1:8001/v1','local-development',180)._request('/models')
        current=provenance();_,upstream=author_namespace(); now=time.time()
        freeze = Path('runs/setup/shared-calibration-freeze-commit.txt').read_text().strip()
        if len(freeze)!=40: raise ValueError('Public precollection freeze missing')
        contract=save(root/'contract.json',{'protocol_sha256':protocol['sha256'],'freeze_commit':freeze,'created_unix':now,
            'deadline_unix':now+14400,'max_calls':28000,'prior_ledger':before,'serving':health,
            'source_hashes':current['source_hashes'],'native_hashes':current['native_hashes'],
            'authorization':'User authorized next shared-model calibration and idle GPUs on 2026-10-25'})
        write_json(root/'run.json',{'created_unix':now,'identity':{'protocol_sha256':protocol['sha256'],
            'campaign_sha256':contract['sha256'],'serving':health,'upstream':upstream,
            'source_hashes':current['source_hashes'],'input_format':'qwen3_non_thinking_actionline',
            'decoding_profile':'qwen3_non_thinking_recommended_v1','environment_action_alias':ACTION_ALIAS_PROFILE,
            'wrapper_sha256':file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256':file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))}})
        replica_audit(root,protocol,contract); children=[]
        for index in range(3):
            log=(root/f'worker-{index}.log').open('a')
            children.append((subprocess.Popen([sys.executable,__file__,'run','--worker',str(index)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[]
        for child,log in children: codes.append(child.wait());log.close()
        workers=[read_json(root/f'worker-{i}.json') for i in range(3)]
        rows=[r for w in workers for r in w['rows']]; trajectories=[t for w in workers for t in w['trajectories']]
        complete = codes==[0]*3 and len(trajectories)==252 and all(t['complete'] for t in trajectories) and all(w['status']=='complete' for w in workers)
        actual=costs(root); reserved=read_json(root/'budget.json')['reserved']
        summary=save(root/'summary.json',{'protocol_sha256':protocol['sha256'],'status':'complete' if complete else 'partial',
            'expected_tasks':36,'completed_tasks':36 if complete else 0,'expected_trajectories':252,
            'completed_trajectories':sum(t['complete'] for t in trajectories),'attempts_completed':len(rows),
            'attempts_rows':sorted(rows,key=lambda r:(r['trajectory_id'],r['trial'])),
            'trajectories':sorted(trajectories,key=lambda t:t['trajectory_id']),'worker_exit_codes':codes,
            'worker_statuses':[w['status'] for w in workers],'costs_incremental':actual,'reserved_calls':reserved,
            'elapsed_seconds':time.time()-now,'holdout_model_calls':0,'reserved_validation_model_calls':0})
        print({k:v for k,v in summary.items() if k not in ['attempts_rows','trajectories']},flush=True)
        if not complete or reserved!=actual['dispatched_calls'] or reserved!=actual['saved_responses'] or actual['uncertain_or_unrecorded_attempts']:
            raise RuntimeError('Campaign incomplete/unconfirmed')


def collect(root,protocol):
    summary=read_json(root/'summary.json'); texts={}; rows=[]
    if summary['status']!='complete': raise ValueError('Complete collection required')
    for row in summary['attempts_rows']:
        path=Path(row['result_path']); result=read_json(path)
        if file_hash(path)!=row['result_file_sha256']: raise ValueError('Result changed')
        for text in result['memory_after']:
            if text not in protocol['entries'].values(): texts[digest(text)]=text
        rows.append({**row,'scene_id':result['scene_id'],'memory_before_hashes':[digest(x) for x in result['memory_before'][-3:]],
            'outcome':'S' if result['success'] else None})
    write_json(root/'semantic-review-private.json',{'protocol_sha256':protocol['sha256'],'texts':[{'sha256':k,'text':v} for k,v in sorted(texts.items())]})
    save(root/'raw-events.json',{'protocol_sha256':protocol['sha256'],'summary_file_sha256':file_hash(root/'summary.json'),
        'rows':rows,'new_unique_texts':len(texts),'annotation_required':len(texts),'holdout_model_calls':0})
    print({'attempts':len(rows),'new_unique_texts':len(texts),'first_successes':sum(r['success'] and r['trial']==1 for r in rows)})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','run','collect']);p.add_argument('--worker',type=int);a=p.parse_args()
    protocol=verify_protocol()
    if a.action=='validate': print({**validate_matrix(protocol),'new_model_calls':0})
    elif a.action=='collect': collect(ROOT,protocol)
    elif a.worker is None: run(ROOT,protocol)
    else: worker(ROOT,protocol,a.worker)


if __name__=='__main__': main()
