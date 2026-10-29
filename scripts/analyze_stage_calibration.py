#!/usr/bin/env python3
"""Independent feature reconstruction and reached-risk verification; no fits."""
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from agent_reliability.io import read_json, file_hash
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save
from analyze_seven_reached_risk import independent_score, PREDICATES

PUBLIC=Path('repro/reflexion-20261025')
PRIVATE=Path('runs/reflexion-stage-calibration-20261029')


def vector(state, indicator):
    n=sum(a['native'] for a in state)
    error=any(a['type'] in ('E','B') for a in state)
    age=max([3-len(state)+i+1 for i,a in enumerate(state) if a['type'] in ('E','B')] or [0])
    return np.array([1,len(state)/3,(int(n==1) if indicator else n)/3,
        sum(a['type']=='A' for a in state)/3,int(error),age/3,int(error and n>0)],float)


def reconstruct(rows, folds, base, indicator):
    out={}; sequences=defaultdict(list)
    for r in rows: sequences[r['trajectory_id']].append(r)
    for tid,sequence in sequences.items():
        theta=np.asarray(base['task_offsets'])*base['task_scale']; w=np.asarray(base['task_prior']).copy()
        beta=np.asarray(folds[sequence[0]['scene_id']])
        for r in sorted(sequence,key=lambda a:a['trial']):
            values=1/(1+np.exp(-(vector(r['state'],indicator)@beta+theta)))
            out[tid,r['trial']]=float(np.dot(w,values))
            if not r['success']:
                w=w*(1-values); w=w/np.sum(w)
    return out


def main():
    assert not (PUBLIC/'stage-calibration-audit-20261029.json').exists(),'Preserve completed audit'
    p=checked(PUBLIC/'stage-calibration-protocol-20261029.json')
    result=checked(PUBLIC/'stage-calibration-results-20261029.json')
    close=checked(PRIVATE/'closeout.json'); assert close['status']=='complete' and close['fits']==30
    assert close['result_sha256']==result['sha256'] and result['protocol_sha256']==p['sha256']
    rows=read_json(Path('runs/reflexion-seven-parameter-development-20261027/input-events.json'))
    counts=Counter((r['trial'],sum(a['native'] for a in r['state'])) for r in rows)
    assert counts=={(1,0):760,(2,1):441,(3,2):188,(4,3):164}
    original=checked(PUBLIC/'seven-parameter-development-results-20261027.json')
    ref=next(c for c in original['configs'] if c['config']['name']=='ridge0.01_scale1.5')
    pp=[]
    for part in result['prediction_parts']:
        data=checked(PUBLIC/part['file']);assert data['sha256']==part['sha256'];pp+=data['predictions']
    linear=checked(PUBLIC/ref['prediction_file'])['predictions']
    maximum=0.; scores={}
    for indicator,name,parts,folds in [(False,'linear_count',linear,ref['folds']),(True,'one_native_indicator',pp,result['folds'])]:
        folds={f['scene_id']:f['coefficients'] for f in folds}
        values=reconstruct(rows,folds,ref['full_model'],indicator)
        saved={(r['trajectory_id'],r['trial']):r['probability'] for r in parts}
        assert len(saved)==len(parts)==1553 and saved.keys()==values.keys()
        difference=max(abs(values[k]-saved[k]) for k in values);assert difference<1e-12;maximum=max(maximum,difference)
        scores[name]={}
        for group,predicate in PREDICATES.items():
            s=independent_score(rows,values,predicate)
            target=result['scores'][name]['primary'] if group=='all' else result['scores'][name]['groups'][group]
            assert abs(s['risk_per_planned_horizon']-target['risk_per_planned_horizon'])<1e-12
            assert s['attempts']==target['attempts']
            scores[name][group]=s
    base=ref['full_model']; beta=np.array(result['candidate_full_model']['p_coefficients']); by_task=defaultdict(list)
    for r in rows: by_task[r['task_id']].append(r)
    loss=0.
    for task,rr in by_task.items():
        logs=np.log(np.array(base['task_prior']))
        for r in rr:
            z=vector(r['state'],True)@beta+base['task_scale']*np.array(base['task_offsets'])
            logs+=float(r['success'])*(-np.logaddexp(0,-z))+(1-float(r['success']))*(-np.logaddexp(0,z))
        m=logs.max();loss-=m+np.log(np.exp(logs-m).sum())
    loss+=.005*float(beta[1:]@beta[1:])+.5*np.logaddexp(0,-beta[0])+.5*np.logaddexp(0,beta[0])
    assert abs(loss-min(a['objective'] for a in result['full_start_audits']))<1e-8
    # Each of46 parent-failure nodes has6 children, but appears once in its posterior.
    graph=p['structure']['parent_graph']; assert len({tuple(g['parent_failure_id']) for g in graph})==46
    assert len({c for g in graph for c in g['children']})==276
    for name,h in p['protected_files'].items(): assert file_hash(Path(name))==h,name
    audit=save(PUBLIC/'stage-calibration-audit-20261029.json',dict(protocol_sha256=p['sha256'],result_sha256=result['sha256'],
        rebuilt_probabilities=3106,max_probability_difference=maximum,independent_full_objective=loss,
        independently_aggregated_scores=scores,stage_native_violations=0,unique_parent_failures=46,children=276,
        protected_files_checked=len(p['protected_files']),parameter_fits=0,new_model_calls=0,holdout134_calls=0,checks_passed=True))
    print(dict(audit_sha256=audit['sha256'],checks_passed=True,protected_files=len(p['protected_files'])),flush=True)


if __name__=='__main__':main()
