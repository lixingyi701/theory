#!/usr/bin/env python3
"""Close both prospective phases under their single immutable allowance."""
import argparse
import shutil
from pathlib import Path
from agent_reliability.io import digest,file_hash,read_json,write_json
from continue_reflexion_pilot import ledger
from run_middle_context_transport import verify_protocol
from summarize_reflexion import costs


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args(); root=args.run
    protocol=verify_protocol(Path('repro/reflexion-20261023/middle-context-protocol.json'))
    summary=read_json(root/'summary.json'); analysis=read_json(root/'analysis.json')
    contract=read_json(root/'contract.json'); audit=read_json(root/'protocol-audit.json')
    for value in [protocol,summary,analysis,contract,audit]:
        assert value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'})
    assert analysis['protocol_sha256']==summary['protocol_sha256']==contract['protocol_sha256']==protocol['sha256']
    assert analysis['summary_sha256']==summary['sha256'] and audit['passed']
    assert summary['completed_cells']==analysis['completed_cells']==48
    assert summary['worker_exit_codes']==[0,0,0,0]
    assert [v['completed_cells'] for v in summary['phases']]==[24,24]
    actual=costs(root); assert actual==summary['costs_incremental']
    reserved=read_json(root/'budget.json')['reserved']
    assert reserved==actual['dispatched_calls']<=contract['max_calls']==3000
    assert actual['saved_responses']==actual['dispatched_calls'] and actual['uncertain_or_unrecorded_attempts']==0
    times=[read_json(p)['dispatched_unix'] for p in root.glob('**/calls/*/attempt-*.request.json')]
    assert min(times)>=contract['created_unix'] and max(times)<contract['deadline_unix']
    counts=ledger(); total=sum(v['dispatched'] for v in counts.values())
    responses=sum(v['responses'] for v in counts.values())
    before=sum(v['dispatched'] for v in contract['prior_ledger'].values())
    assert total==responses==before+actual['dispatched_calls'] and before==22132
    supplement=read_json(root/'mixture-analysis.json')
    assert supplement['analysis_sha256']==analysis['sha256']
    args.output.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for name in ['contract','run','summary','analysis','replica-audit','protocol-audit','action-shapes','semantic-labels','mixture-forecast','mixture-analysis']:
        destination=args.output/f'middle-context-{"raw-summary" if name=="summary" else "results" if name=="analysis" else name}.json'
        if destination.exists():
            assert destination.read_bytes().replace(b'\r\n',b'\n')==(root/(name+'.json')).read_bytes().replace(b'\r\n',b'\n')
        else:
            shutil.copyfile(root/(name+'.json'),destination)
        artifacts[destination.name]=file_hash(destination)
    value={'protocol_sha256':protocol['sha256'],'freeze_commit':'1a3e3c94d866c567f5775fc190f494301506a575',
        'stop_reason':'both_frozen_phases_complete','completed_cells':48,'phase_cells':{'middle':24,'native':24},
        'successes':analysis['successes'],'independent_base_tasks':12,'new_base_tasks':0,
        'costs_incremental':actual,'replica_check_calls':4,'reserved_calls':reserved,
        'prior_calls':before,'all_calls':total,'all_responses':responses,'elapsed_seconds':summary['elapsed_seconds'],
        'holdout_model_calls':0,'parameter_refits':0,'gpu_indices':[0,1],
        'request_configuration_errors':len(audit['errors']),'artifacts':artifacts,
        'source_presence_forecast_sha256':protocol['source_presence_freeze_sha256'],
        'supplement_freeze_commit':'fe2ce68f3bd992ef2ce27b67a22f9c3f21e3bb1a',
        'driver_sha256':file_hash(Path(__file__))}
    value['sha256']=digest(value); write_json(args.output/'middle-context-closeout.json',value)
    print({k:v for k,v in value.items() if k!='artifacts'})


if __name__=='__main__':
    main()
