#!/usr/bin/env python3
"""Freeze new-scene-partition predictions without reading validation outcomes."""
from collections import Counter
from pathlib import Path
from agent_reliability.cli import provenance
from agent_reliability.dataset import enumerate_tasks
from agent_reliability.io import digest,file_hash,read_json
from agent_reliability.reference import official_environment
from prepare_shared_calibration import STATES,ENTRIES,PARTITIONS,save
from prepare_state_representation import walk_tasks
from shared_recovery_model import budget_probability

PROTOCOL=Path('repro/reflexion-20261025/shared-validation-protocol.json')
MANIFEST=PROTOCOL.with_name('shared-validation-manifest.json')
FILES=['scripts/prepare_shared_validation.py','scripts/run_shared_validation.py','scripts/evaluate_shared_validation.py',
       'scripts/run_shared_calibration.py',
       'tests/test_shared_validation.py','scripts/shared_recovery_model.py','scripts/run_multiround_diagnostic.py',
       'scripts/prepare_shared_calibration.py','scripts/run_middle_context_transport.py','scripts/run_residual_context.py',
       'scripts/continue_reflexion_pilot.py','scripts/summarize_reflexion.py','scripts/audit_reflexion_protocol.py',
       'scripts/audit_reflexion_action_shapes.py','scripts/deployment/serve_reflexion_profiles.py',
       'configs/reflexion_qwen3_8b_protocol.json','configs/qwen3_8b_non_thinking_wrapper.json',
       'docs/experiments/shared-validation-design-20261025.md',str(PARTITIONS),
       'repro/reflexion-20261025/shared-calibration-parameters.json',
       'repro/reflexion-20261025/shared-calibration-closeout.json',
       'repro/reflexion-20261025/shared-calibration-precision.json','repro/baseline-20261019/holdout134.json']


def main():
    if PROTOCOL.exists() or MANIFEST.exists():raise FileExistsError('Preserve frozen validation selection')
    partition=read_json(PARTITIONS);base=partition['reserved_validation_tasks'];scenes=partition['reserved_validation_scenes']
    parameters=read_json(Path('repro/reflexion-20261025/shared-calibration-parameters.json'))
    excluded={k:set() for k in ['task_id','game_sha256','trajectory_sha256']};inputs={};used=set()
    for path in sorted(Path('repro').rglob('*.json')):
        values=list(walk_tasks(read_json(path)))
        if values:
            inputs[str(path)]=file_hash(path)
            for task in values:
                for key in excluded:excluded[key].add(task[key])
    for path in sorted(Path('runs').glob('reflexion-*/**/identity.json')):
        for task in walk_tasks(read_json(path)):
            used.add(task['task_id'])
            for key in excluded:excluded[key].add(task[key])
    if used&{t['task_id'] for t in base}:raise ValueError('Reserved tasks already called')
    all_tasks,removed=enumerate_tasks(Path('data/alfworld'),'train')
    pool=[t for t in all_tasks if t['task_type']=='pick_heat_then_place_in_recep' and t['scene_id'] in scenes
          and all(t[k] not in excluded[k] for k in excluded)]
    selected=[];seen={k:{t[k] for t in base} for k in excluded}
    for scene in scenes:
        fixed=[t for t in base if t['scene_id']==scene]
        if len(fixed)!=2:raise ValueError('Keep original two reserved tasks')
        additional=[]
        for task in sorted([t for t in pool if t['scene_id']==scene],key=lambda t:digest([2026102502,t['task_id']])):
            if any(task[k] in seen[k] for k in seen):continue
            additional.append(task)
            for key in seen:seen[key].add(task[key])
            if len(additional)==4:break
        if len(additional)!=4:raise ValueError('Insufficient fresh tasks in reserved scene')
        selected.extend(fixed+additional)
    for key in excluded:
        if len({t[key] for t in selected})!=54:raise ValueError('Duplicate selected identity '+key)
    manifest=save(MANIFEST,{'scope':'Prospective validation, fixed nine scenes, six new tasks per scene',
        'parent_partitions_sha256':partition['sha256'],'selection_seed':2026102502,'tasks':selected,
        'preserved_original_reserved_ids':[t['task_id'] for t in base],
        'calibration_scenes':partition['calibration_scenes'],'validation_scenes':scenes,
        'public_exclusion_inputs':inputs,'used_task_ids_sha256':digest(sorted(used)),
        'eligible_additional_tasks':len(pool),'dataset_removed':removed,'holdout134_used':False})
    cases=[]
    for task in selected:
        observations=[]
        for seed in [2120,2121]:
            env=official_environment(Path('data/alfworld'),task,seed,Path('third_party/react'))
            try:
                ob,_=env.reset();observations.append(digest('\n'.join(ob[0].split('\n\n')[1:])))
            finally:env.close()
        if observations[0]!=observations[1]:raise ValueError('Seed-dependent reset')
        cases.append({'task_id':task['task_id'],'task':task,'initial_observation_sha256':observations[0]})
    models=parameters['comparison']['all_fitted_models']
    pick=lambda f,s:next(m for m in models if m['family']==f and m['scale']==s)
    fitted={'selected':parameters['comparison']['selected_model'],'constant_persistent':pick('constant',1.5),
            'constant_pooled':pick('constant',0.),'error_pooled':pick('error_lifetime',0.)}
    forecasts={}
    for name,model in fitted.items():
        forecasts[name]={condition:[budget_probability(model,[{'type':k,'native':False} for k in labels],b)
                                    for b in [1,2]] for condition,labels in STATES.items()}
    trajectories=[]
    for index,case in enumerate(cases):
        order=list(STATES);shift=index%7;order=order[shift:]+order[:shift]
        for condition in order:
            seed=2120+list(STATES).index(condition);labels=STATES[condition]
            trajectories.append({'id':f'validation-{case["task_id"]}-{seed}-{condition}','task_id':case['task_id'],
                'condition':condition,'history_key':condition,'operation':'initial_state','initial_labels':labels,
                'initial_memory_sha256':digest([ENTRIES[k] for k in labels]),'seed':seed,'max_attempts':2,'worker':index%4})
    value={'scope':'Frozen shared-kernel prediction on new tasks in reserved scene partition',
        'parent_calibration_parameters_sha256':parameters['sha256'],'manifest_sha256':manifest['sha256'],
        'cases':cases,'trajectories':trajectories,'states':STATES,'entries':ENTRIES,
        'ports':[8001,8002,8003,8004],'gpu_indices':[0,1,2,3],
        'expected_trajectories':378,'maximum_attempts':756,'tasks':54,'scenes':9,
        'limits':{'additional_calls_including_replica_checks':42000,'seconds':21600},'prior_global_calls':64027,
        'models':fitted,'forecasts':forecasts,'parameter_refits':0,'holdout134_model_calls':0,
        'analysis':{'primary':'Two-budget Brier improvement: constant_persistent minus selected',
            'aggregation':'Mean seven trajectories within task, six tasks within scene, then nine scenes equally',
            'primary_interval':'Two-sided Student t95 across nine paired scene means; df8 critical2.306004135',
            'decision':'Positive lower endpoint supports prediction improvement; interval crossing0 is inconclusive; negative upper endpoint favors baseline',
            'bootstrap_sensitivity':{'replicates':20000,'seed':2026102503,'cluster':'scene'},
            'secondary':['first-budget Brier','constant_pooled baseline','error_pooled mixture ablation','per-condition calibration','descriptive recovery contrasts'],
            'semantic_labels':'Not required for primary budget-success score; native reflection annotation reserved for mechanism diagnostics',
            'truncation':'Retain all sampled attempts; report affected trajectories and a descriptive affected-scene sensitivity, no resampling'},
        'precision':{'goal':'Approximate two-sided95 halfwidth<=.025 in paired two-budget Brier improvement',
            'source':'shared-calibration-precision.json','calibration_scene_sd_for_m6':.031284909968407,
            'estimated_t8_halfwidth':.024047667462382183,
            'interpretation':'Variance components from18calibration scenes; actual precision uncertain; this is not an exact80% power guarantee'},
        'resources':'ExistingGPU0, idleGPU1, remaining memory onGPU2/3 if>=60000MiB free; preserve all other processes',
        'source_hashes':provenance()['source_hashes'],'protected_files':{p:file_hash(Path(p)) for p in FILES+[str(MANIFEST)]}}
    save(PROTOCOL,value)
    print({'protocol_sha256':read_json(PROTOCOL)['sha256'],'tasks':54,'scenes':9,'trajectories':378,'max_attempts':756,'new_model_calls':0})


if __name__=='__main__':main()
