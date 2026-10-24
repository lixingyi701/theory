#!/usr/bin/env python3
"""Adapt verified cell-cohort completion metadata for unchanged shared auditors."""
import argparse
from pathlib import Path
import sys

import alfworld
from agent_reliability.io import digest,file_hash,read_json,write_json
import audit_reflexion_protocol as interface
import audit_reflexion_action_shapes as shapes
from run_correctness_diagnostic import cells,verify_protocol


def compatible_summary(summary,protocol):
    expected={c['id']:(c['task_id'],c['seed'],c['condition']) for c in cells(protocol)}
    rows=summary['rows']
    if (summary['protocol_sha256']!=protocol['sha256'] or summary['status']!='complete'
        or summary['completed_cells']!=summary['expected_cells'] or summary['expected_cells']!=180
        or summary['worker_exit_codes']!=[0]*4 or summary['worker_statuses']!=['complete']*4
        or len(rows)!=len({r['cell_id'] for r in rows}) or len(rows)!=180):
        raise ValueError('The complete frozen cell cohort has not been established')
    observed={r['cell_id']:(r['task_id'],r['seed'],r['condition']) for r in rows}
    if observed!=expected:
        raise ValueError('Worker summary differs from the frozen task/seed/condition matrix')
    task_count=len({r['task_id'] for r in rows})
    if task_count!=12:
        raise ValueError('Unexpected task coverage')
    return {**summary,'completed_tasks':task_count,'expected_tasks':task_count}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('repro/reflexion-20261024/correctness-diagnostic-protocol.json'))
    a=p.parse_args();protocol=verify_protocol(a.protocol);summary_path=a.run/'summary.json'
    raw=read_json(summary_path);expected_digest=digest({k:v for k,v in raw.items() if k!='sha256'})
    if raw['sha256']!=expected_digest:raise ValueError('Raw summary integrity')
    adapted=compatible_summary(raw,protocol)
    # Only a read-time metadata view changes. Raw summary and all request checks remain unchanged.
    def reader(path):
        return adapted if Path(path).resolve()==summary_path.resolve() else read_json(path)
    interface.read_json=reader;shapes.read_json=reader
    output=a.run/'protocol-audit.json'
    sys.argv=['audit_reflexion_protocol.py','--run',str(a.run),'--output',str(output)]
    interface.main()
    grammar=Path(alfworld.__file__).parent/'data'/'alfred.twl2'
    action_report=shapes.audit(a.run,grammar)
    receipt={'driver_sha256':file_hash(Path(__file__)),'source_summary_sha256':file_hash(summary_path),
        'source_protocol_sha256':protocol['sha256'],'verified_cells':180,'verified_tasks':12,
        'mapping':'Derive completed_tasks and expected_tasks from the complete frozen cell matrix; do not change raw artifacts'}
    for path,report in [(output,read_json(output)),(a.run/'action-shapes.json',action_report)]:
        report.pop('sha256',None);report['summary_adapter']=receipt;report['sha256']=digest(report);write_json(path,report)
    if read_json(summary_path)!=raw:raise ValueError('Raw summary was unexpectedly modified')
    print({'adapter':receipt,'action_shapes':action_report['counts']})


if __name__=='__main__':main()
