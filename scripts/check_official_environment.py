#!/usr/bin/env python3
"""Compare official ALFWorld Gym outputs and replay against an audited backend.

Facts remain in the independent evaluator backend; official policy sees only
its original public interface. This does not assert unseen hidden-state equality.
"""
import argparse
from pathlib import Path
from agent_reliability.audit import verify_audit
from agent_reliability.cli import backend, provenance
from agent_reliability.dataset import load_manifest
from agent_reliability.io import digest, read_json, write_json
from agent_reliability.reference import author_code, official_environment


def check(root, manifest, audit, upstream, output):
    if output.exists():
        raise FileExistsError('Keep previous audit; choose a new file')
    verify_audit(audit, manifest)
    _, source = author_code(upstream);rows=[]
    for row in audit['tasks']:
        task = next(t for t in manifest['tasks'] if t['task_id'] == row['task_id'])
        checks=[]
        for check in row['checks']:
            direct = backend(root);official = official_environment(root, task, 42, upstream)
            try:
                a=direct.reset(task,42);b,info=official.reset()
                if a.text != b[0] or a.won != bool(info['won'][0]):
                    raise ValueError('Official reset differs')
                actions=row['reference_actions'][:check['prefix_length']]+check['fixed_suffix']
                for action in actions:
                    a=direct.step(action);b,r,d,info=official.step([action])
                    if a.text != b[0] or a.won != bool(info['won'][0]) or a.terminated != bool(d[0]) or a.reward != float(r[0]):
                        raise ValueError('Official transition differs')
                checks.append({'prefix_length':check['prefix_length'],'repetition':check['repetition'],'passed':True})
            finally:
                direct.close();official.close()
        rows.append({'task_id':task['task_id'],'checks':checks,'passed':all(x['passed'] for x in checks)})
    current=provenance()
    report={'schema_version':1,'manifest_sha256':manifest['sha256'],'direct_audit_sha256':audit['sha256'],
            'upstream':source,'passed':all(r['passed'] for r in rows),'tasks':rows,
            'restore_checks_passed':sum(len(x['checks']) for x in rows),'source_hashes':current['source_hashes'],
            'native_hashes':current['native_hashes'],
            'scope':'official Gym public outputs/fixed suffix vs facts-audited direct backend; no model calls'}
    write_json(output,{**report,'sha256':digest(report)})
    print({'passed':report['passed'],'tasks':len(rows),'checks':report['restore_checks_passed']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--audit-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('third_party/react'));a=p.parse_args()
    check(a.data_root,load_manifest(a.manifest),read_json(a.audit_report),a.upstream,a.output)
