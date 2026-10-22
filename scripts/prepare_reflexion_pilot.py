#!/usr/bin/env python3
"""Freeze a new scene-separated pilot cohort, without model calls."""
import argparse
from collections import Counter
from pathlib import Path
from agent_reliability.dataset import TASK_TYPES,enumerate_tasks,load_manifest
from agent_reliability.io import digest,read_json,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('data/alfworld'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--per-type',type=int,default=4);a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Manifests are immutable')
    frozen=[load_manifest(Path('repro/baseline-20261019/dev12.json')),load_manifest(Path('repro/baseline-20261019/holdout134.json'))]
    old=read_json(Path('repro/legacy-pilot24.json'))
    excluded={str(t['scene_id']) for m in frozen+[old] for t in m['tasks']}
    excluded.update(['FloorPlan302','FloorPlan319','FloorPlan218'])
    tasks,removed=enumerate_tasks(a.data_root,'train');seed=20261022
    pools={k:sorted([t for t in tasks if t['task_type']==k and t['scene_id'] not in excluded],key=lambda t:digest([seed,t['task_id']])) for k in TASK_TYPES}
    selected=[];used=set()
    for _ in range(a.per_type):
        for k in TASK_TYPES:
            choices=[t for t in pools[k] if t['scene_id'] not in used]
            if not choices: raise ValueError(f'Insufficient distinct scenes for {k}')
            selected.append(choices[0]);used.add(choices[0]['scene_id'])
    result={'schema_version':1,'split':'train','selection_seed':seed,'scope':'new pilot; not calibration or final validation',
            'eligible_count':len(tasks),'eligible_by_type':dict(Counter(t['task_type'] for t in tasks)),
            'excluded':removed,'excluded_scene_ids':sorted(excluded),'scene_grouping':'unique_floor_plan','tasks':selected,
            'frozen_input_manifest_sha256':[m['sha256'] for m in frozen]}
    result['sha256']=digest(result);write_json(a.output,result)
    print({'tasks':len(selected),'scenes':len(used),'sha256':result['sha256']})

if __name__=='__main__': main()
