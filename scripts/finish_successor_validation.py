#!/usr/bin/env python3
"""Close only this allowance and release only recorded task-owned tmux replicas."""
from pathlib import Path
import argparse,subprocess
from agent_reliability.io import file_hash,read_json
from prepare_shared_calibration import save
from prepare_successor_protocol import ROOT,PROTOCOL,checked
from summarize_reflexion import costs
from continue_reflexion_pilot import ledger


def release_owned():
    receipt=Path('runs/setup/successor-service-ownership.json')
    if not receipt.exists():return []
    released=[]
    for row in read_json(receipt)['sessions']:
        assert row['gpu'] in [2,3] and row['name']==f'theory-successor-replica-{row["gpu"]}-20261026'
        current=subprocess.run(['tmux','display-message','-p','-t',row['name'],'#{session_id}|#{session_created}'],capture_output=True,text=True)
        if current.returncode==0 and current.stdout.strip()==row['session_id']+'|'+row['session_created']:
            subprocess.run(['tmux','kill-session','-t',row['name']],check=True);released.append(row['name'])
    return released


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exit-code',type=int,required=True);args=parser.parse_args()
    try:
        if not (ROOT/'contract.json').exists():return
        contract=checked(ROOT/'contract.json');protocol=checked(PROTOCOL);actual=costs(ROOT)
        complete=args.exit_code==0
        if complete:
            result=checked(PROTOCOL.with_name('successor-validation-results-20261026.json'))
            summary=checked(ROOT/'summary.json')
            audit=read_json(ROOT/'protocol-audit.json')
            assert summary['status']=='complete' and summary['completed_trajectories']==550
            assert result['protocol_sha256']==protocol['sha256'] and result['trajectories']==550
            assert audit['passed'] and actual['dispatched_calls']==actual['saved_responses']==read_json(ROOT/'budget.json')['reserved']
            assert actual['uncertain_or_unrecorded_attempts']==0
        else:(ROOT/'STOP').write_text('Execution stopped; preserve evidence and inspect before any continuation')
        totals=ledger()
        report=save(ROOT/'allowance-closeout.json',{'protocol_sha256':protocol['sha256'],'contract_sha256':contract['sha256'],
            'allowance_status':'closed','execution_status':'complete' if complete else 'stopped','master_exit_code':args.exit_code,
            'costs_incremental':actual,'unspent_calls':contract['max_calls']-actual['dispatched_calls'],
            'global_calls':sum(r['dispatched'] for r in totals.values()),'global_responses':sum(r['responses'] for r in totals.values()),
            'summary_file_sha256':file_hash(ROOT/'summary.json') if (ROOT/'summary.json').exists() else None,
            'parameter_refits':0,'holdout134_model_calls':0})
        print({'closed':report['sha256'],'status':report['execution_status']},flush=True)
    finally:print({'released_owned_replicas':release_owned()},flush=True)


if __name__=='__main__':main()
