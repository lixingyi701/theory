#!/usr/bin/env python3
"""Bind the100 reserved tasks to real resets and the frozen successor forecasts."""
from pathlib import Path
import numpy as np
from agent_reliability.cli import provenance
from agent_reliability.io import digest,file_hash,read_json
from agent_reliability.reference import official_environment
from continue_reflexion_pilot import ledger
from prepare_shared_calibration import STATES,ENTRIES,save
from prepare_four_attempt_protocol import checked
from calibrate_successor_success import budget_curve,success_components

PUBLIC=Path('repro/reflexion-20261025')
MANIFEST=PUBLIC/'successor-validation-expanded-manifest-20261026.json'
PLAN=PUBLIC/'successor-validation-expanded-plan-20261026.json'
PARAMETERS=PUBLIC/'successor-development-parameters-20261026.json'
PROTOCOL=PUBLIC/'successor-validation-protocol-20261026.json'
ROOT=Path('runs/reflexion-successor-validation-20261026')
PRIMARY=['correct_middle','neutral_full']
SECONDARY=['repeated_error','correct_latest','delete_latest']
SEEDS=[2140,2141]
LIMITS={'additional_calls_including_replica_checks':125000,'seconds':86400}
FILES=['scripts/prepare_successor_protocol.py','scripts/run_successor_validation.py',
 'scripts/evaluate_successor_validation.py','scripts/finish_successor_validation.py','scripts/run_successor_campaign.sh',
 'tests/test_successor_validation.py','tests/test_multiround_diagnostic.py',
 'scripts/calibrate_successor_success.py','scripts/diagnose_success_regularization.py','scripts/shared_recovery_model.py',
 'scripts/run_shared_calibration.py','scripts/run_multiround_diagnostic.py','scripts/run_middle_context_transport.py',
 'scripts/run_residual_context.py','scripts/continue_reflexion_pilot.py','scripts/summarize_reflexion.py',
 'scripts/prepare_shared_calibration.py','scripts/audit_reflexion_protocol.py','scripts/audit_reflexion_action_shapes.py',
 'scripts/deployment/serve_reflexion_profiles.py','configs/reflexion_qwen3_8b_protocol.json',
 'configs/qwen3_8b_non_thinking_wrapper.json','docs/experiments/successor-validation-design-20261026.md',
 str(MANIFEST),str(PLAN),str(PARAMETERS),'repro/reflexion-20261025/shared-validation-protocol.json',
 'repro/baseline-20261019/holdout134.json']


def make_trajectories(cases,secondary_ids):
    rows=[]
    for index,case in enumerate(cases):
        conditions=PRIMARY+(SECONDARY if case['task_id'] in secondary_ids else [])
        for seed_index,seed in enumerate(SEEDS):
            shift=(index+seed_index)%len(conditions)
            for condition in conditions[shift:]+conditions[:shift]:
                labels=STATES[condition]
                rows.append({'id':f'successor-{case["task_id"]}-{seed}-{condition}',
                    'task_id':case['task_id'],'seed':seed,'condition':condition,
                    'panel':'primary' if condition in PRIMARY else 'recovery','history_key':condition,
                    'operation':'initial_state','initial_labels':labels,
                    'initial_memory_sha256':digest([ENTRIES[k] for k in labels]),'max_attempts':4,'worker':seed_index})
    return rows


def forecasts(models):
    output={'full_seven':{},'unchanged_seven':{},'full_six':{}}
    for condition,labels in STATES.items():
        for name in ['full_seven','full_six']:
            output[name][condition]=budget_curve(models[name],labels)
        p=success_components(models['full_seven'],[{'type':k,'native':False} for k in labels])
        output['unchanged_seven'][condition]=[float(np.asarray(models['full_seven']['task_prior'])@(1-(1-p)**b)) for b in range(1,5)]
    return output


def main():
    if PROTOCOL.exists() or (ROOT/'contract.json').exists():raise FileExistsError('Preserve protocol/run; do not reset')
    manifest=checked(MANIFEST);plan=checked(PLAN);parameters=checked(PARAMETERS)
    assert manifest['sha256']==plan['manifest_sha256'] and parameters['sha256']==plan['parameters_sha256']
    assert parameters['source_sha256']==file_hash(Path('scripts/calibrate_successor_success.py'))
    assert parameters['selection']['selected']=='seven'
    before=ledger()
    assert sum(v['dispatched'] for v in before.values())==145135
    assert all(v['dispatched']==v['responses'] for v in before.values())
    selected={t['task_id'] for t in manifest['tasks']}
    assert len(selected)==100
    for path in Path('runs').glob('**/identity.json'):
        assert read_json(path).get('task',{}).get('task_id') not in selected, str(path)
    cases=[];resets=[]
    for index,task in enumerate(manifest['tasks']):
        observations=[]
        for seed in SEEDS:
            env=official_environment(Path('data/alfworld'),task,seed,Path('third_party/react'))
            try:
                ob,_=env.reset();observations.append(digest('\n'.join(ob[0].split('\n\n')[1:])))
            finally:env.close()
            resets.append({'task_id':task['task_id'],'seed':seed,'observation_sha256':observations[-1]})
        assert len(set(observations))==1, 'Seed-dependent reset observation'
        cases.append({'task_id':task['task_id'],'task':task,'initial_observation_sha256':observations[0]})
        if (index+1)%10==0:print({'tasks_reset':index+1,'checks':len(resets),'model_calls':0},flush=True)
    reset=save(PUBLIC/'successor-reset-checks-20261026.json',{'checks':resets,'passed':True,
        'scope':'200 official-environment seeded reset identity comparisons; zero model calls','manifest_sha256':manifest['sha256']})
    models={'full_seven':parameters['models']['seven']['model'],'full_six':parameters['models']['six']['model']}
    predictions=forecasts(models)
    for name in predictions:
        for condition in STATES:
            assert np.allclose(predictions[name][condition],plan['forecasts'][name][condition],atol=1e-12,rtol=0)
    protocol=save(PROTOCOL,{'scope':'Frozen100-task successor prediction with unchanged native Reflexion actor/reflector',
        'parent_calibration_parameters_sha256':parameters['sha256'],'manifest_sha256':manifest['sha256'],
        'plan_sha256':plan['sha256'],'reset_checks_sha256':reset['sha256'],'cases':cases,
        'trajectories':make_trajectories(cases,set(manifest['secondary_recovery_task_ids'])),
        'primary_conditions':PRIMARY,'secondary_conditions':SECONDARY,
        'secondary_recovery_task_ids':manifest['secondary_recovery_task_ids'],'crossed_seeds':SEEDS,
        'states':{c:STATES[c] for c in PRIMARY+SECONDARY},'entries':ENTRIES,
        'entry_hashes':{k:digest(v) for k,v in ENTRIES.items()},'models':models,'forecasts':predictions,
        'ports':[8003,8004],'gpu_indices':[2,3],'tasks':100,'scenes':25,'expected_trajectories':550,'maximum_attempts':2200,
        'replica_check_calls':4,'limits':LIMITS,'prior_global_calls':145135,
        'parameter_refits':0,'holdout134_model_calls':0,'estimated_two_replica_hours':plan['estimated_hours_at_previous_two_replica_rate'],
        'schedule':'Crossed seeds2140/2141 map to workers0/1 on GPU2/3; rotate conditions within task/seed by task_index+seed_index',
        'analysis':{'primary':'Four-budget Brier unchanged_seven minus full_seven on400C/N trajectories',
            'aggregation':'Equal condition/seeds per task, equal tasks per scene, equal25scenes','student_t_df':24,
            'student_t95_critical':2.063898562,'bootstrap_replicates':20000,'bootstrap_seed':2026102603,
            'secondary':['budgets1..4','full_seven versus full_six','recovery contrasts on25 fixed tasks'],
            'native_updates':'All actual trajectories use full native failure updates; static is an offline predictor',
            'truncation':'Retain every sampled record','missing':'Require complete550 trajectories; no partial scoring/imputation'},
        'serving':'Pinned serial raw profilev2, BF16,Qwen3-8B;one replica/GPU, max_num_seqs1, memory utilization0.70',
        'source_hashes':provenance()['source_hashes'],
        'protected_files':{p:file_hash(Path(p)) for p in FILES+[str(PUBLIC/'successor-reset-checks-20261026.json')]}})
    print({'protocol_sha256':protocol['sha256'],'resets':len(resets),'trajectories':len(protocol['trajectories']),
           'gpu_indices':[2,3],'new_model_calls':0},flush=True)


if __name__=='__main__':main()
