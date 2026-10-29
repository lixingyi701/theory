#!/usr/bin/env python3
"""Expand the reserved50-task design before collection, preserving every old task."""
from pathlib import Path
from collections import Counter
import numpy as np
from agent_reliability.dataset import enumerate_tasks
from agent_reliability.io import read_json,file_hash,digest
from prepare_state_representation import walk_tasks
from prepare_four_attempt_protocol import checked
from prepare_shared_calibration import save

ROOT=Path('repro/reflexion-20261025')
OUTPUT=ROOT/'successor-validation-expanded-manifest-20261026.json'
SEED=2026102601


def main():
    parent=checked(ROOT/'successor-validation-manifest-20261026.json')
    plan=checked(ROOT/'successor-validation-plan-20261026.json')
    assert plan['manifest_sha256']==parent['sha256']
    assert not Path('runs/reflexion-successor-validation-20261026').exists(), 'Revise sample size only before collection'
    if OUTPUT.exists():
        expanded=checked(OUTPUT)
        assert expanded['parent_manifest_sha256']==parent['sha256']
    else:
        excluded={k:set() for k in ['task_id','game_sha256','trajectory_sha256']}
        for path in list(Path('repro').rglob('*.json'))+list(Path('runs').glob('**/identity.json')):
            for task in walk_tasks(read_json(path)):
                for k in excluded:excluded[k].add(task[k])
        tasks,removed=enumerate_tasks(Path('data/alfworld'),'train')
        pool=[t for t in tasks if t['task_type']=='pick_heat_then_place_in_recep' and
              t['scene_id'] in parent['counts_by_scene'] and all(t[k] not in excluded[k] for k in excluded)]
        assert len(pool)==200,'Recheck unused-task inventory before changing the reserved set'
        scenes=sorted(parent['counts_by_scene'],key=lambda s:digest([SEED,'scene',s]))
        by_scene={s:sorted([t for t in pool if t['scene_id']==s],key=lambda t:digest([SEED,t['task_id']])) for s in scenes}
        selected=list(parent['tasks']);added=[];counts=Counter(t['scene_id'] for t in selected)
        while len(selected)<100:
            changed=False
            for scene in scenes:
                used=sum(t['scene_id']==scene for t in added)
                if used<len(by_scene[scene]) and len(selected)<100:
                    task=by_scene[scene][used];added.append(task);selected.append(task);counts[scene]+=1;changed=True
            assert changed
        assert all(len({t[k] for t in selected})==100 for k in excluded)
        assert {t['task_id'] for t in parent['tasks']}<={t['task_id'] for t in selected}
        expanded=save(OUTPUT,{
            **{k:v for k,v in parent.items() if k not in ['sha256','source_sha256','tasks','counts_by_scene','scope',
                 'primary_trajectories','expected_trajectories','maximum_attempts']},
            'scope':'Precollection100-task expansion; all original50 tasks retained, no outcomes collected',
            'parent_manifest_sha256':parent['sha256'],'parent_plan_sha256':plan['sha256'],
            'revision_reason':'Initial50-task precision planning gives wide intervals at success0.70; increase independent tasks before any outcomes, keep metrics/model/decision unchanged',
            'tasks':selected,'counts_by_scene':dict(counts),'added_tasks':50,
            'selection':'Retain parent50, add50 unused tasks by the same hash order and capacity-capped round-robin over25 scenes',
            'primary_trajectories':400,'secondary_trajectories':150,'expected_trajectories':550,'maximum_attempts':2200,
            'source_sha256':file_hash(Path(__file__))})
    a=plan['forecasts']['unchanged_seven']['correct_middle'][3]
    v=plan['forecasts']['full_seven']['correct_middle'][3]
    rng=np.random.default_rng(2026102602);scenarios=[]
    for truth in [v,.70,(a+v)/2,.85,a]:
        for scene_rho,task_rho in [(.02,.10),(.05,.25),(.10,.50)]:
            values=[]
            for scene,n in expanded['counts_by_scene'].items():
                concentration=1/scene_rho-1
                scene_p=np.clip(rng.beta(truth*concentration,(1-truth)*concentration,size=(10000,1)),1e-12,1-1e-12)
                concentration=1/task_rho-1
                task_p=rng.beta(np.repeat(scene_p*concentration,n,axis=1),np.repeat((1-scene_p)*concentration,n,axis=1))
                y=rng.binomial(4,task_p).sum(axis=1)/(4*n)
                values.append((a-v)*(a+v-2*y))
            values=np.asarray(values).T;mean=values.mean(axis=1)
            half=2.063898562*values.std(axis=1,ddof=1)/5
            scenarios.append({'assumed_success_rate':truth,'assumed_scene_correlation':scene_rho,
                'assumed_conditional_task_correlation':task_rho,'analytic_loss_difference':(a-v)*(a+v-2*truth),
                'fraction_full_favored':float(np.mean(mean-half>0)),
                'fraction_static_favored':float(np.mean(mean+half<0)),
                'median_interval_halfwidth':float(np.median(half))})
    expected=2*sum(expanded['counts_by_scene'].values())*sum(4-sum(plan['forecasts']['full_seven'][c][:3]) for c in expanded['primary_conditions'])
    expected+=50*sum(4-sum(plan['forecasts']['full_seven'][c][:3]) for c in expanded['secondary_conditions'])
    closeout=checked(ROOT/'four-attempt-validation-closeout.json')
    rate=closeout['costs_incremental']['dispatched_calls']/closeout['collection_elapsed_seconds']
    calls=expected*plan['previous_calls_per_attempt']+4
    revised=save(ROOT/'successor-validation-expanded-plan-20261026.json',{
        **{k:v for k,v in plan.items() if k not in ['sha256','source_sha256','scope','manifest_sha256','precision_planning',
             'estimated_attempts','estimated_calls_including4replica_checks','estimated_hours_at_previous_two_replica_rate',
             'proposed_not_active_allowance','primary']},
        'scope':'Revised100-task prospective design; no new Agent data, allowance inactive',
        'manifest_sha256':expanded['sha256'],'superseded_50_task_plan_sha256':plan['sha256'],
        'primary':{**plan['primary'],'population':'400 C/N trajectories on100 fresh tasks in25 scenes',
            'aggregation':'Mean4 condition/seed trajectories per task, then equally average tasks within each scene, then25 scenes equally'},
        'precision_planning':{'replicates_per_scenario':10000,'seed':2026102602,'scenarios':scenarios,
            'interpretation':'Assumed correlations and rates; increasing tasks improves precision but does not guarantee80% power'},
        'estimated_attempts':expected,'estimated_calls_including4replica_checks':calls,
        'estimated_hours_at_previous_two_replica_rate':calls/rate/3600,
        'proposed_not_active_allowance':{'calls_including_checks_and_failures':125000,'seconds':86400},
        'precollection_requirements':[r.replace('100 official reset comparisons on50 fresh tasks','200 official reset comparisons on100 fresh tasks') for r in plan['precollection_requirements']],
        'runtime_plan':'GPU2/3 only; assess bounded concurrent workers per existing replica before launch. Historical two-replica throughput remains the resource estimate until measured.',
        'source_sha256':file_hash(Path(__file__))})
    print({'manifest_sha256':expanded['sha256'],'plan_sha256':revised['sha256'],
        'tasks':100,'scenes':25,'trajectories':550,'attempts_maximum':2200,
        'counts_by_scene':expanded['counts_by_scene'],'expected_calls':calls,'estimated_hours':calls/rate/3600,
        'planning_at070':[s for s in scenarios if s['assumed_success_rate']==.70]})


if __name__=='__main__':main()
