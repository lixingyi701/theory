#!/usr/bin/env python3
"""Close the sampled calibration without changing its frozen sources or raw records."""
from collections import Counter
from pathlib import Path
from agent_reliability.io import digest,file_hash,read_json
from prepare_shared_calibration import PROTOCOL,PARTITIONS,save
from run_shared_calibration import ROOT,verify_protocol
from continue_reflexion_pilot import ledger
from summarize_reflexion import costs


def main():
    protocol=verify_protocol();pub=PROTOCOL.parent
    summary=read_json(ROOT/'summary.json');integrity=read_json(ROOT/'collection-integrity.json')
    for name,expected in integrity['artifacts'].items():
        assert file_hash(ROOT/name)==expected,name
    labels=read_json(pub/'shared-calibration-semantic-labels.json')
    events=read_json(pub/'shared-calibration-events.json')
    parameters=read_json(pub/'shared-calibration-parameters.json')
    results=read_json(pub/'shared-calibration-results.json')
    precision=read_json(pub/'shared-calibration-precision.json')
    for value in [summary,integrity,labels,events,parameters,results,precision]:
        assert value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'})
    assert summary['status']=='complete' and summary['completed_trajectories']==252
    assert summary['worker_exit_codes']==[0,0,0] and summary['attempts_completed']==len(events['rows'])==463
    assert len(labels['entries'])==402 and len({e['reflection_sha256'] for e in labels['entries']})==402
    assert parameters['events_sha256']==events['sha256'] and results['parameters_sha256']==parameters['sha256']
    assert precision['parameters_sha256']==parameters['sha256']
    assert parameters['model_source_sha256']==protocol['protected_files']['scripts/shared_recovery_model.py']
    assert parameters['analysis_source_sha256']==file_hash(Path('scripts/analyze_shared_calibration.py'))
    assert len(parameters['comparison']['all_fitted_models'])==6
    actual=costs(ROOT);state=read_json(ROOT/'budget.json')
    assert actual['dispatched_calls']==actual['saved_responses']==state['reserved']==19088
    assert not actual['uncertain_or_unrecorded_attempts']
    assert summary['elapsed_seconds']<=protocol['limits']['seconds'] and actual['dispatched_calls']<=28000
    assert read_json(ROOT/'protocol-audit.json')['passed']
    used=set()
    for p in ROOT.glob('**/identity.json'):
        task=read_json(p).get('task',{}).get('task_id')
        if task:used.add(task)
    partition=read_json(PARTITIONS)
    forbidden={t['task_id'] for t in partition['reserved_validation_tasks']}|{t['task_id'] for t in read_json(Path('repro/baseline-20261019/holdout134.json'))['tasks']}
    assert not used&forbidden and used=={c['task_id'] for c in protocol['cases']}
    all_runs=ledger();calls=sum(v['dispatched'] for v in all_runs.values());responses=sum(v['responses'] for v in all_runs.values())
    assert calls==responses==64027
    truncations=[]
    for path in ROOT.glob('**/calls/*/validation.json'):
        if read_json(path).get('truncated_output'):
            relative=path.relative_to(ROOT);truncations.append({'path':relative.as_posix(),'trajectory_id':relative.parts[1],
                'role':relative.parts[-2].split('-')[0]})
    names=['shared-calibration-'+n+'.json' for n in ['semantic-labels','events','parameters','results','precision']]
    value={'scope':'Completed new-task calibration; prospective validation not yet sampled',
        'protocol_sha256':protocol['sha256'],'sampling_freeze':'513a93173d14f3ae4f04c836577e7bf0bacdc148',
        'tasks':36,'scenes':18,'trajectories':252,'attempts':463,'successes_by_two':61,
        'annotation_counts':dict(Counter(e['category'] for e in labels['entries'])),'human_second_review':False,
        'costs_incremental':actual,'collection_elapsed_seconds':summary['elapsed_seconds'],
        'all_calls':calls,'all_responses':responses,'allowance_status':'closed',
        'truncations':truncations,'holdout_model_calls':0,'reserved_validation_model_calls':0,
        'artifacts':{n:file_hash(pub/n) for n in names},'collection_integrity_sha256':integrity['sha256'],
        'closeout_source_sha256':file_hash(Path(__file__))}
    report=save(pub/'shared-calibration-closeout.json',value)
    print({k:report[k] for k in ['sha256','tasks','trajectories','attempts','annotation_counts','all_calls','allowance_status']})


if __name__=='__main__':main()
