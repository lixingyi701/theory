#!/usr/bin/env python3
"""Close the frozen run, explicitly verifying critic worker result digests."""
import argparse
from pathlib import Path
import re
import shutil

from agent_reliability.io import digest, file_hash, read_json
from continue_reflexion_pilot import ledger
from run_correctness_diagnostic import check, require, save
from run_native_clause import PROTOCOL, PREFIX, collect, verify_protocol
from summarize_reflexion import costs


def verify_worker_row(worker, analysed, raw):
    if worker['panel']=='critic':
        check(raw,'Raw fixed-critic result')
        require(worker.get('sha256')==raw['sha256'],'Worker critic digest does not match raw result')
    else:
        require('sha256' not in worker,'Unexpected actor digest metadata')
    require(all(analysed.get(k)==v for k,v in worker.items() if k!='sha256'),'Worker result differs from analysis')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=Path('runs/reflexion-native-clause-20261025'))
    parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--freeze-commit',required=True)
    args=parser.parse_args();root=args.run;protocol=verify_protocol(args.protocol)
    require(re.fullmatch('[0-9a-f]{40}',args.freeze_commit) is not None,'Full precollection freeze required')
    summary=read_json(root/'summary.json');analysis=read_json(root/'analysis.json');contract=read_json(root/'contract.json')
    labels=read_json(root/'semantic-labels.json');audit=read_json(root/'protocol-audit.json');replica=read_json(root/'replica-audit.json')
    for value in [summary,analysis,contract,labels,audit,replica]:check(value,'Final artifact')
    require(all(v['protocol_sha256']==protocol['sha256'] for v in [summary,analysis,contract,labels]),'Protocol binding')
    require(analysis['summary_sha256']==summary['sha256'] and analysis['semantic_labels_sha256']==labels['sha256'],'Analysis binding')
    require(audit['passed'] and not audit['errors'] and replica['passed'] and len(replica['rows'])==8,'Audit failure')
    raw_rows,_,_=collect(protocol,root);lookup={r['cell_id']:r for r in analysis['rows']}
    require(len(raw_rows)==len(lookup)==summary['completed_cells']==216 and summary['worker_exit_codes']==[0]*4
            and summary['worker_statuses']==['complete']*4 and summary['status']=='complete','Complete coverage')
    for row in raw_rows:require(all(lookup[row['cell_id']].get(k)==v for k,v in row.items()),'Changed raw result')
    require(len(summary['rows'])==len({r['cell_id'] for r in summary['rows']})==216
            and {r['cell_id'] for r in summary['rows']}==set(lookup),'Worker coverage')
    for row in summary['rows']:
        folder='cells' if row['panel']=='actor' else 'critic-cells'
        path=root/folder/row['cell_id']/'trial-01/result.json'
        require(row['result_sha256']==file_hash(path),'Worker raw-file digest')
        verify_worker_row(row,lookup[row['cell_id']],read_json(path))
    actual=costs(root);reserved=read_json(root/'budget.json')['reserved']
    require(actual==summary['costs_incremental'] and reserved==actual['dispatched_calls']==actual['saved_responses']<=6500
            and not actual['uncertain_or_unrecorded_attempts'],'Final cost ledger')
    stamps=[read_json(p)['dispatched_unix'] for p in root.glob('**/calls/*/attempt-*.request.json')]
    require(len(stamps)==reserved and min(stamps)>=contract['created_unix'] and max(stamps)<contract['deadline_unix'],'Dispatch envelope')
    current=ledger();before=contract['prior_ledger'];total=sum(v['dispatched'] for v in current.values())
    require(sum(v['dispatched'] for v in before.values())==41181
            and total==41181+reserved==sum(v['responses'] for v in current.values()),'Global cost ledger')
    for name in set(current)|set(before):
        if name!=root.resolve().name:require(current.get(name,{'dispatched':0,'responses':0})==before.get(name,{'dispatched':0,'responses':0}),'Other campaign changed')
    output=args.protocol.parent;artifacts={}
    for source,target in [('contract','contract'),('run','run'),('summary','raw-summary'),('analysis','results'),
                          ('semantic-labels','semantic-labels'),('replica-audit','replica-audit'),
                          ('protocol-audit','protocol-audit'),('action-shapes','action-shapes')]:
        src=root/(source+'.json');dst=output/(PREFIX+'-'+target+'.json')
        if dst.exists():require(dst.read_bytes()==src.read_bytes(),'Existing evidence changed')
        else:shutil.copyfile(src,dst)
        artifacts[dst.name]=file_hash(dst)
    result={'protocol_sha256':protocol['sha256'],'freeze_commit':args.freeze_commit,
        'stop_reason':'frozen108actor_and108critic_complete','actor_attempts':108,'fixed_critic_calls':108,
        'actor_successes':analysis['actor_successes'],'base_tasks':12,'seeds_per_task':3,
        'costs_incremental':actual,'replica_check_calls':8,'elapsed_seconds':summary['elapsed_seconds'],
        'prior_calls':41181,'all_calls':total,'all_responses':total,'parameter_refits':0,'holdout_model_calls':0,
        'artifacts':artifacts,'driver_sha256':file_hash(Path('scripts/run_native_clause.py')),
        'closeout_adapter':{'path':'scripts/close_native_clause.py','driver_sha256':file_hash(Path(__file__)),
            'source_summary_sha256':file_hash(root/'summary.json'),'source_analysis_sha256':file_hash(root/'analysis.json'),
            'critic_result_digests_verified':sum(r['panel']=='critic' for r in summary['rows']),
            'reason':'Frozen worker rows contain the internal critic-result sha256, while analysis rows store result_sha256. Verify the extra digest directly against the unchanged raw result, then compare all shared fields.'}}
    target=output/(PREFIX+'-closeout.json')
    if target.exists():require(read_json(target)=={**result,'sha256':digest(result)},'Closeout changed')
    else:save(target,result)
    print({k:v for k,v in result.items() if k not in ['artifacts','closeout_adapter']})


if __name__=='__main__':main()
