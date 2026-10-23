#!/usr/bin/env python3
"""Supplement: use the planned input mixture, while hiding each cell's state."""
import argparse
import time
from pathlib import Path
import numpy as np
from agent_reliability.io import digest,file_hash,read_json,write_json
from state_representation_model import CLASSES,scores


def pooled(protocol,phase):
    rows=[v for v in protocol['predictions'] if v['panel']==phase]
    tasks=sorted({v['task_id'] for v in rows})
    return {label:float(np.mean([np.mean([v['presence'][label] for v in rows if v['task_id']==task]) for task in tasks]))
            for label in CLASSES}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,default=Path('repro/reflexion-20261023/middle-context-protocol.json'))
    p.add_argument('--run',type=Path,required=True); p.add_argument('--prepare-only',action='store_true'); args=p.parse_args()
    protocol=read_json(args.protocol); destination=args.run/'mixture-forecast.json'
    if args.prepare_only:
        if destination.exists(): raise FileExistsError('Keep the supplemental prediction version')
        value={'protocol_sha256':protocol['sha256'],'created_unix':time.time(),
            'scope':'Supplementary input-mixture control added during collection; no response outcomes used to derive predictions; original primary comparison unchanged',
            'definition':'Within each task average frozen presence predictions over planned cells, then average tasks; use this one constant for every cell in the phase',
            'phases':{phase:pooled(protocol,phase) for phase in protocol['phases']},
            'driver_sha256':file_hash(Path(__file__))}
        value['sha256']=digest(value);write_json(destination,value);print(value);return
    frozen=read_json(destination); analysis=read_json(args.run/'analysis.json')
    assert frozen['protocol_sha256']==analysis['protocol_sha256']==protocol['sha256']
    forecasts={v['cell_id']:v['presence'] for v in protocol['predictions']}; reports=[]
    for phase in protocol['phases']:
        assert frozen['phases'][phase]==pooled(protocol,phase)
        selected=[r for r in analysis['rows'] if r['panel']==phase]; tasks=sorted({r['task_id'] for r in selected})
        task_scores=[]
        for task in tasks:
            rows=[r for r in selected if r['task_id']==task]
            for name in ['pooled_mixture','presence']:
                values=[scores(frozen['phases'][phase] if name=='pooled_mixture' else forecasts[r['cell_id']],r['outcome']) for r in rows]
                task_scores.append({'task_id':task,'model':name,**{k:float(np.mean([v[k] for v in values])) for k in values[0]}})
        differences=np.asarray([next(v['brier'] for v in task_scores if v['task_id']==task and v['model']=='pooled_mixture')
            -next(v['brier'] for v in task_scores if v['task_id']==task and v['model']=='presence') for task in tasks])
        indices=np.random.default_rng(2026102305).integers(0,len(tasks),size=(20000,len(tasks)))
        reports.append({'phase':phase,'scores':{name:{k:float(np.mean([v[k] for v in task_scores if v['model']==name]))
            for k in ['brier','log_loss','success_brier']} for name in ['pooled_mixture','presence']},
            'pooled_minus_presence_brier':float(differences.mean()),
            'task_bootstrap_95_interval':np.quantile(differences[indices].mean(axis=1),[.025,.975]).tolist()})
    value={'analysis_sha256':analysis['sha256'],'supplement_forecast_sha256':frozen['sha256'],
        'scope':frozen['scope'],'phases':reports,'parameter_refits':0,'driver_sha256':file_hash(Path(__file__))}
    value['sha256']=digest(value);write_json(args.run/'mixture-analysis.json',value);print(value)


if __name__=='__main__':main()
