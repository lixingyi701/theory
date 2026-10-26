#!/usr/bin/env python3
"""Select unused tasks and freeze prospective forecasts without Agent calls."""
from pathlib import Path
from collections import Counter
import numpy as np
from agent_reliability.dataset import enumerate_tasks
from agent_reliability.io import read_json, file_hash, digest
from prepare_state_representation import walk_tasks
from prepare_shared_calibration import save, STATES
from prepare_four_attempt_protocol import checked
from calibrate_successor_success import budget_curve, success_components

PUBLIC=Path('repro/reflexion-20261025')
MANIFEST=PUBLIC/'successor-validation-manifest-20261026.json'
SEED=2026102601


def main():
    parameters=checked(PUBLIC/'successor-development-parameters-20261026.json')
    assert parameters['selection']['selected']=='seven'
    assert parameters['source_sha256']==file_hash(Path('scripts/calibrate_successor_success.py'))
    seven=parameters['models']['seven']['model'];six=parameters['models']['six']['model']
    forecasts={'full_seven':{},'unchanged_seven':{},'full_six':{}}
    for condition,labels in STATES.items():
        forecasts['full_seven'][condition]=budget_curve(seven,labels)
        forecasts['full_six'][condition]=budget_curve(six,labels)
        p=success_components(seven,[{'type':k,'native':False} for k in labels])
        forecasts['unchanged_seven'][condition]=[float(np.asarray(seven['task_prior'])@(1-(1-p)**b)) for b in range(1,5)]
        assert abs(forecasts['full_seven'][condition][0]-forecasts['unchanged_seven'][condition][0])<1e-12
    if MANIFEST.exists():
        manifest=checked(MANIFEST)
        assert manifest['parameters_sha256']==parameters['sha256'],'Preserve selected tasks and frozen parameters'
    else:
        excluded={k:set() for k in ['task_id','game_sha256','trajectory_sha256']}
        for path in list(Path('repro').rglob('*.json'))+list(Path('runs').glob('**/identity.json')):
            for task in walk_tasks(read_json(path)):
                for key in excluded:excluded[key].add(task[key])
        partitions=checked(PUBLIC/'shared-calibration-partitions.json')
        tasks,removed=enumerate_tasks(Path('data/alfworld'),'train')
        pool=[t for t in tasks if t['task_type']=='pick_heat_then_place_in_recep' and
              t['scene_id'] not in partitions['sealed134_scenes'] and all(t[k] not in excluded[k] for k in excluded)]
        scenes=sorted({t['scene_id'] for t in pool},key=lambda s:digest([SEED,'scene',s]))
        assert len(scenes)==25
        grouped={s:sorted([t for t in pool if t['scene_id']==s],key=lambda t:digest([SEED,t['task_id']])) for s in scenes}
        assert min(map(len,grouped.values()))>=2
        selected=[t for s in scenes for t in grouped[s][:2]]
        assert all(len({t[key] for t in selected})==50 for key in excluded)
        manifest=save(MANIFEST,{
            'scope':'Fresh-task prospective validation reserved after development; not launched',
            'parameters_sha256':parameters['sha256'],'selection_seed':SEED,
            'eligible_tasks_before_selection':len(pool),'available_by_scene_before_selection':dict(Counter(t['scene_id'] for t in pool)),
            'tasks':selected,'counts_by_scene':{s:2 for s in scenes},
            'secondary_recovery_task_ids':[grouped[s][0]['task_id'] for s in scenes],
            'selection':'All25 nonsealed heat scenes with unused tasks; first2 tasks by fixed hash order in every scene',
            'freshness':'Exclude task_id, game_sha256 and trajectory_sha256 in all prior repro task records and run identities',
            'scene_scope':'Fresh task/game identities within25 scenes also represented in development; no unseen-scene claim. FloorPlan4/26 have no fresh eligible task remaining.',
            'sealed_scenes_excluded':partitions['sealed134_scenes'],'dataset_removed':removed,
            'primary_conditions':['correct_middle','neutral_full'],
            'secondary_conditions':['repeated_error','correct_latest','delete_latest'],
            'crossed_seeds':[2140,2141],'maximum_attempts_per_trajectory':4,
            'primary_trajectories':200,'secondary_trajectories':150,'expected_trajectories':350,'maximum_attempts':1400,
            'new_model_calls':0,'holdout134_model_calls':0,'parameter_refits':0,
            'source_sha256':file_hash(Path(__file__))})
    a=forecasts['unchanged_seven']['correct_middle'][3];v=forecasts['full_seven']['correct_middle'][3]
    rng=np.random.default_rng(2026102602);scenarios=[]
    for truth in [v,.70,(a+v)/2,.85,a]:
        for scene_rho,task_rho in [(.02,.10),(.05,.25),(.10,.50)]:
            values=[]
            for _ in range(25):
                concentration=1/scene_rho-1
                scene_p=np.clip(rng.beta(truth*concentration,(1-truth)*concentration,size=(10000,1)),1e-12,1-1e-12)
                concentration=1/task_rho-1
                task_p=rng.beta(np.repeat(scene_p*concentration,2,axis=1),np.repeat((1-scene_p)*concentration,2,axis=1))
                outcome=rng.binomial(4,task_p).sum(axis=1)/8
                values.append((a-v)*(a+v-2*outcome))
            values=np.asarray(values).T;means=values.mean(axis=1)
            half=2.063898562*values.std(axis=1,ddof=1)/5
            scenarios.append({'assumed_success_rate':truth,'assumed_scene_correlation':scene_rho,
                'assumed_conditional_task_correlation':task_rho,'analytic_loss_difference':(a-v)*(a+v-2*truth),
                'fraction_full_favored':float(np.mean(means-half>0)),
                'fraction_static_favored':float(np.mean(means+half<0)),
                'median_interval_halfwidth':float(np.median(half))})
    expected_attempts=sum(100*(4-sum(forecasts['full_seven'][c][:3])) for c in manifest['primary_conditions'])
    expected_attempts+=sum(50*(4-sum(forecasts['full_seven'][c][:3])) for c in manifest['secondary_conditions'])
    previous=checked(PUBLIC/'four-attempt-validation-closeout.json')
    calls_per_attempt=(previous['costs_incremental']['dispatched_calls']-4)/previous['attempts']
    expected_calls=expected_attempts*calls_per_attempt+4
    measured_rate=previous['costs_incremental']['dispatched_calls']/previous['collection_elapsed_seconds']
    plan=save(PUBLIC/'successor-validation-plan-20261026.json',{
        'scope':'Prospective design and synthetic precision planning; no new Agent observations',
        'manifest_sha256':manifest['sha256'],'parameters_sha256':parameters['sha256'],
        'forecasts':forecasts,'primary':{'population':'200 C/N trajectories on50 fresh tasks in25 scenes',
          'budget':4,'loss_difference':'unchanged_seven Brier minus full_seven Brier',
          'aggregation':'Mean4 condition/seed trajectories per task, then2 tasks per scene, then25 scenes equally',
          'primary_interval':'Paired-scene Student t95, df24, critical2.063898562',
          'decision':'Lower bound>0 favors full; upper bound<0 favors unchanged; otherwise inconclusive',
          'secondary_interval':'20000 whole-scene bootstrap, seed2026102603',
          'secondary_comparisons':['full_seven versus full_six increment','Recovery contrasts on25 fixed fresh tasks, both seeds'],
          'no_replacement':'No parameter refitting or substitution of a favorable secondary comparison after collection'},
        'equal_expected_Brier_success_rate':(a+v)/2,
        'precision_planning':{'replicates_per_scenario':10000,'seed':2026102602,'scenarios':scenarios,
          'interpretation':'Assumed beta-binomial correlations and success rates, not empirical power or a guarantee'},
        'estimated_attempts':expected_attempts,'previous_calls_per_attempt':calls_per_attempt,
        'estimated_calls_including4replica_checks':expected_calls,
        'estimated_hours_at_previous_two_replica_rate':expected_calls/measured_rate/3600,
        'proposed_not_active_allowance':{'calls_including_checks_and_failures':75000,'seconds':64800},
        'precollection_requirements':['Dedicated new run namespace and evaluator; preserve old runs',
          'Kernel/interface and four-attempt chain checks','100 official reset comparisons on50 fresh tasks',
          'Current GPU2/3 ownership and free-memory check; preserve other jobs and GPU0/1',
          'Public execution-source/protocol/forecast freeze before calls; new bounded ledger contract'],
        'source_sha256':file_hash(Path(__file__)),'kernel_source_sha256':file_hash(Path('scripts/calibrate_successor_success.py')),
        'new_model_calls':0,'holdout134_model_calls':0,'parameter_refits':0})
    print({'manifest_sha256':manifest['sha256'],'plan_sha256':plan['sha256'],
           'fresh_tasks':50,'scenes':25,'trajectories':350,'full4':v,'static4':a,
           'equal_loss_rate':(a+v)/2,'estimated_calls':expected_calls,
           'estimated_hours':plan['estimated_hours_at_previous_two_replica_rate'],
           'planning_at_070':[r for r in scenarios if r['assumed_success_rate']==.70]})


if __name__=='__main__':main()
