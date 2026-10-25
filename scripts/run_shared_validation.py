#!/usr/bin/env python3
"""Execute frozen validation through the unchanged shared-calibration worker."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time
from agent_reliability.cli import provenance
from agent_reliability.io import digest,file_hash,read_json,run_lock,write_json
from agent_reliability.reference import CompletionTransport,ACTION_ALIAS_PROFILE
from agent_reliability.reflexion import author_namespace
from continue_reflexion_pilot import ledger
from prepare_shared_calibration import STATES,save
from prepare_shared_validation import PROTOCOL,MANIFEST
from run_shared_calibration import worker
from run_residual_context import verify_dependencies
from run_middle_context_transport import replica_audit
from shared_recovery_model import budget_probability
from summarize_reflexion import costs

ROOT=Path('runs/reflexion-shared-validation-20261025')


def validate_matrix(protocol):
    cases=protocol['cases'];ids=[c['task_id'] for c in cases]
    if len(ids)!=len(set(ids)) or len(ids)!=54:raise ValueError('54 unique tasks required')
    counts=Counter(c['task']['scene_id'] for c in cases)
    if len(counts)!=9 or set(counts.values())!={6}:raise ValueError('Six tasks in nine scenes required')
    manifest=read_json(MANIFEST)
    if protocol['manifest_sha256']!=manifest['sha256'] or {t['task_id'] for t in manifest['tasks']}!=set(ids):raise ValueError('Manifest binding')
    if set(manifest['calibration_scenes'])&set(counts):raise ValueError('Calibration scene selected')
    expected={(t,c) for t in ids for c in STATES};seen=set()
    for row in protocol['trajectories']:
        key=(row['task_id'],row['condition'])
        if key not in expected or key in seen:raise ValueError('Invalid or duplicate trajectory')
        seen.add(key);labels=STATES[row['condition']]
        if row['initial_labels']!=labels or row['seed']!=2120+list(STATES).index(row['condition']) or row['max_attempts']!=2:
            raise ValueError('State, seed or budget changed')
        if row['worker']!=ids.index(row['task_id'])%4 or row['initial_memory_sha256']!=digest([protocol['entries'][k] for k in labels]):
            raise ValueError('Memory/worker binding')
    if seen!=expected or len(protocol['trajectories'])!=378:raise ValueError('Incomplete validation matrix')
    return {'tasks':54,'scenes':9,'trajectories':378,'maximum_attempts':756}


def verify_protocol():
    protocol=read_json(PROTOCOL)
    if protocol['sha256']!=digest({k:v for k,v in protocol.items() if k!='sha256'}):raise ValueError('Protocol digest')
    verify_dependencies(protocol)
    if provenance()['source_hashes']!=protocol['source_hashes']:raise ValueError('Agent source changed')
    if protocol['limits']!={'additional_calls_including_replica_checks':42000,'seconds':21600}:raise ValueError('Allowance changed')
    if protocol['ports']!=[8001,8002,8003,8004] or protocol['gpu_indices']!=[0,1,2,3]:raise ValueError('Replica matrix changed')
    for name,model in protocol['models'].items():
        for condition,labels in STATES.items():
            state=[{'type':k,'native':False} for k in labels]
            for b,p in enumerate(protocol['forecasts'][name][condition],1):
                if abs(p-budget_probability(model,state,b))>1e-12:raise ValueError('Frozen forecast mismatch')
    validate_matrix(protocol);return protocol


def run(protocol):
    root=ROOT
    with run_lock(root):
        if (root/'contract.json').exists():raise FileExistsError('Do not restart validation')
        before=ledger()
        if sum(v['dispatched'] for v in before.values())!=64027 or any(v['dispatched']!=v['responses'] for v in before.values()):raise ValueError('Prior ledger changed')
        freeze=Path('runs/setup/shared-validation-freeze-commit.txt').read_text().strip()
        if len(freeze)!=40:raise ValueError('Public freeze missing')
        health=CompletionTransport('http://127.0.0.1:8001/v1','local-development',180)._request('/models')
        current=provenance();_,upstream=author_namespace();now=time.time()
        contract=save(root/'contract.json',{'protocol_sha256':protocol['sha256'],'freeze_commit':freeze,
            'created_unix':now,'deadline_unix':now+21600,'max_calls':42000,'prior_ledger':before,
            'serving':health,'source_hashes':current['source_hashes'],'native_hashes':current['native_hashes'],
            'authorization':'User authorized completed-calibration analysis and next frozen validation on2026-10-25; remaining GPU memory authorized'})
        write_json(root/'run.json',{'created_unix':now,'identity':{'protocol_sha256':protocol['sha256'],
            'campaign_sha256':contract['sha256'],'serving':health,'upstream':upstream,'source_hashes':current['source_hashes'],
            'input_format':'qwen3_non_thinking_actionline','decoding_profile':'qwen3_non_thinking_recommended_v1',
            'environment_action_alias':ACTION_ALIAS_PROFILE,'wrapper_sha256':file_hash(Path('configs/qwen3_8b_non_thinking_wrapper.json')),
            'model_protocol_sha256':file_hash(Path('configs/reflexion_qwen3_8b_protocol.json'))}})
        replica_audit(root,protocol,contract);children=[]
        for index in range(4):
            log=(root/f'worker-{index}.log').open('a')
            children.append((subprocess.Popen([sys.executable,__file__,'run','--worker',str(index)],stdout=log,stderr=subprocess.STDOUT),log))
        codes=[]
        for process,log in children:codes.append(process.wait());log.close()
        workers=[read_json(root/f'worker-{i}.json') for i in range(4)]
        rows=[r for w in workers for r in w['rows']];trajectories=[t for w in workers for t in w['trajectories']]
        expected={r['id'] for r in protocol['trajectories']};observed=[r['trajectory_id'] for r in trajectories]
        complete=codes==[0]*4 and len(observed)==len(set(observed))==378 and set(observed)==expected and all(t['complete'] for t in trajectories) and all(w['status']=='complete' for w in workers)
        actual=costs(root);reserved=read_json(root/'budget.json')['reserved']
        summary=save(root/'summary.json',{'protocol_sha256':protocol['sha256'],'status':'complete' if complete else 'partial',
            'expected_tasks':54,'completed_tasks':54 if complete else 0,'expected_trajectories':378,
            'completed_trajectories':sum(t['complete'] for t in trajectories),'attempts_completed':len(rows),
            'attempts_rows':sorted(rows,key=lambda r:(r['trajectory_id'],r['trial'])),
            'trajectories':sorted(trajectories,key=lambda t:t['trajectory_id']),'worker_exit_codes':codes,
            'worker_statuses':[w['status'] for w in workers],'costs_incremental':actual,'reserved_calls':reserved,
            'elapsed_seconds':time.time()-now,'parameter_refits':0,'holdout134_model_calls':0})
        print({k:v for k,v in summary.items() if k not in ['attempts_rows','trajectories']},flush=True)
        if not complete or reserved!=actual['dispatched_calls'] or reserved!=actual['saved_responses'] or actual['uncertain_or_unrecorded_attempts']:
            raise RuntimeError('Incomplete/unconfirmed validation')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','run']);p.add_argument('--worker',type=int);a=p.parse_args()
    protocol=verify_protocol()
    if a.action=='validate':print({**validate_matrix(protocol),'new_model_calls':0})
    elif a.worker is None:run(protocol)
    else:worker(ROOT,protocol,a.worker)


if __name__=='__main__':main()
