#!/usr/bin/env python3
"""Freeze new tasks, scene partitions, state candidates and a finite collection."""
from collections import Counter
from pathlib import Path
from agent_reliability.cli import provenance
from agent_reliability.dataset import enumerate_tasks
from agent_reliability.io import digest,file_hash,read_json,write_json
from agent_reliability.reference import official_environment
from prepare_state_representation import ENTRIES,walk_tasks

PROTOCOL = Path('repro/reflexion-20261025/shared-calibration-protocol.json')
PARTITIONS = PROTOCOL.with_name('shared-calibration-partitions.json')
SEED = 2026102502
STATES = {'error_middle':['N','E','N'],'correct_middle':['N','C','N'],'neutral_full':['N','N','N'],
          'repeated_error':['E','N','E'],'correct_latest':['E','N','C'],
          'delete_latest':['E','N'],'correct_oldest':['C','N','E']}
FILES = ['scripts/prepare_shared_calibration.py','scripts/run_shared_calibration.py','scripts/shared_recovery_model.py',
         'tests/test_shared_recovery.py','docs/experiments/shared-calibration-design-20261025.md',
         'scripts/prepare_state_representation.py','scripts/run_multiround_diagnostic.py',
         'scripts/run_middle_context_transport.py','scripts/run_residual_context.py','scripts/continue_reflexion_pilot.py',
         'scripts/summarize_reflexion.py','scripts/audit_reflexion_protocol.py','scripts/audit_reflexion_action_shapes.py',
         'configs/reflexion_qwen3_8b_protocol.json','configs/qwen3_8b_non_thinking_wrapper.json',
         'scripts/deployment/serve_reflexion_profiles.py','repro/baseline-20261019/holdout134.json']


def save(path,value):
    value = {k:v for k,v in value.items() if k!='sha256'}
    value['sha256'] = digest(value); write_json(path,value); return value


def main():
    if PROTOCOL.exists() or PARTITIONS.exists():
        raise FileExistsError('Preserve frozen selection')
    excluded = {k:set() for k in ['task_id','game_sha256','trajectory_sha256']}; inputs = {}; identities = []
    for path in sorted(Path('repro').rglob('*.json')):
        previous = list(walk_tasks(read_json(path)))
        if previous:
            inputs[str(path)] = file_hash(path)
            for task in previous:
                for key in excluded: excluded[key].add(task[key])
    for path in sorted(Path('runs').glob('**/identity.json')):
        for task in walk_tasks(read_json(path)):
            for key in excluded: excluded[key].add(task[key])
            identities.append({'task_id':task['task_id'],'identity_sha256':file_hash(path)})
    sealed = {t['scene_id'] for t in read_json(Path('repro/baseline-20261019/holdout134.json'))['tasks']}
    tasks,removed = enumerate_tasks(Path('data/alfworld'),'train')
    pool = [t for t in tasks if t['task_type']=='pick_heat_then_place_in_recep' and t['scene_id'] not in sealed
            and all(t[k] not in excluded[k] for k in excluded)]
    scenes = sorted({t['scene_id'] for t in pool},key=lambda s:digest([SEED,'scene',s]))
    if len(scenes)!=27: raise ValueError('Expected inspected 27-scene pool; inspect any dataset change')
    partition = {'selection_seed':SEED,'eligible_tasks':len(pool),'scene_counts':dict(Counter(t['scene_id'] for t in pool)),
        'calibration_scenes':scenes[:18],'reserved_validation_scenes':scenes[18:],
        'calibration_tasks':[],'reserved_validation_tasks':[], 'sealed134_scenes':sorted(sealed),
        'excluded_public_inputs':inputs,'private_identity_inventory_sha256':digest(identities),
        'excluded_counts':{k:len(v) for k,v in excluded.items()},'dataset_removed':removed,
        'meaning':'New task identities. Scenes have been seen in earlier development; this partition separates the new fit and prospective validation, not all historical knowledge.'}
    for role in ['calibration','reserved_validation']:
        for scene in partition[role+'_scenes']:
            selected = sorted([t for t in pool if t['scene_id']==scene],key=lambda t:digest([SEED,t['task_id']]))[:2]
            if len(selected)!=2: raise ValueError('Two fresh tasks required per scene')
            partition[role+'_tasks'].extend(selected)
    cases = []
    for task in partition['calibration_tasks']:
        observations = []
        for seed in [2110,2111]:
            env = official_environment(Path('data/alfworld'),task,seed,Path('third_party/react'))
            try:
                ob,_ = env.reset(); observations.append(digest('\n'.join(ob[0].split('\n\n')[1:])))
            finally: env.close()
        if observations[0]!=observations[1]: raise ValueError('Reset observation depends on seed')
        cases.append({'task_id':task['task_id'],'task':task,'initial_observation_sha256':observations[0]})
    trajectories = []
    for i,case in enumerate(cases):
        order = list(STATES); shift = i%7; order = order[shift:]+order[:shift]
        for condition in order:
            seed = 2110+list(STATES).index(condition)
            labels = STATES[condition]; memory = [ENTRIES[k] for k in labels]
            trajectories.append({'id':f'shared-{case["task_id"]}-{seed}-{condition}','task_id':case['task_id'],
                'condition':condition,'history_key':condition,'operation':'initial_state',
                'initial_labels':labels,'initial_memory_sha256':digest(memory),'seed':seed,'max_attempts':2,
                'worker':i%3})
    partition = save(PARTITIONS,partition)
    protocol = {'scope':'New-task calibration of shared success and natural reflection-write kernels',
        'parent_commit':'a473f305b32df8c0579ad48a77bf7241311dd996','selection_seed':SEED,
        'partitions_sha256':partition['sha256'],'cases':cases,'states':STATES,'entries':ENTRIES,
        'entry_hashes':{k:digest(v) for k,v in ENTRIES.items()},'trajectories':trajectories,
        'ports':[8001,8003,8004],'gpu_indices':[0,2,3],'visible_memory_entries':3,
        'expected_trajectories':252,'maximum_actor_attempts':504,'independent_base_tasks':36,'scene_clusters':18,
        'limits':{'additional_calls_including_replica_checks':28000,'seconds':14400},'prior_global_calls':44939,
        'question':'Can a small common memory kernel, with a persistent task-success mixture, describe first and naturally reached second attempts without operation-specific fits?',
        'models':{'families':['constant','error_lifetime','information_order'],'success_offset_scales':[0.,1.5],
            'persistent_offsets':[-1.,0.,1.],'prior':[1/6,2/3,1/6],'ridge':1.,'pseudocount_per_outcome':.5,
            'write_scope':'Failure-conditional E/C/B/U/A; no task offset in write model',
            'training':'Only this new cohort; all actual attempts. Same task retains its latent offset across all trajectories and trials. Distinct state-specific seeds2110..2116 avoid deliberate common-random-number dependence between trajectories.',
            'cross_validation':'Leave-one-scene-out, both tasks and all trajectories in a scene kept together. Test-trajectory prior is reset, updated only from its own previous observed attempts.',
            'selection':'Select success family/scale by equal-scene success Brier and write family separately by conditional write Brier. Paired-scene one-SE rule from each minimum; prefer constant,error_lifetime,information_order in order and scale0 before1.5. Joint Brier describes assembled kernel, not a reason to worsen success prediction.',
            'secondary':['success Brier','log loss','conditional write Brier','first-attempt contrasts','observed two-budget curves'],
            'freeze':'Fit all six predeclared candidates on all calibration data, freeze parameters. No reserved-validation calls in this campaign.'},
        'precision':{'primary_unit':'Scene (18); two tasks nested per scene, one seed per task/state',
            'first_attempt_pilot_task_sd_three_seed_average':.4133708031,'independent_task_halfwidth_optimistic_reference':1.96*.4133708031/(36**.5),
            'design_effect_sensitivity':{'rho0_halfwidth':1.96*.4133708031/(36**.5),
                'rho0.5_halfwidth':1.96*.4133708031/(24**.5),'rho1_halfwidth':1.96*.4133708031/(18**.5)},
            'prediction_score_worstcase_normal_halfwidth':1.96/(18**.5),
            'interpretation':'Pilot SD describes three-seed task averages, not this one-seed-per-state design. The displayed normal widths are optimistic sensitivity references; calibration is not a powered confirmatory15pp recovery-effect study. Prediction-score variance and formal validation size will be estimated from held-scene scores before validation calls.',
            'bootstrap':'20000 scene-cluster replicates, seed2026102502; all seven states and attempts remain nested'},
        'semantics':{'E':'Explicit target wrong precondition, no correct alternative',
            'C':'Explicit correct inventory precondition, no target wrong precondition',
            'B':'Both target wrong and correct preconditions are explicitly present',
            'U':'Neither target precondition is stated; other advice may remain',
            'A':'Target precondition cannot be assigned reliably, including unresolved referents/implicit ordering',
            'review':'New full-text Codex annotation by content hash, no human second review; B is not a synonym for A; no reused old A relabelling',
            'state':'Oldest-to-newest visible entry type and constructed/native origin; uncertain A remains explicit'},
        'support':'Constructed initial states plus native states reached after one failure. No claim for empty memory, native initial paragraphs, longer budgets or unseen error types.',
        'holdout_model_calls':0,'reserved_validation_model_calls':0,'source_hashes':provenance()['source_hashes'],
        'protected_files':{p:file_hash(Path(p)) for p in FILES+[str(PARTITIONS)]}}
    save(PROTOCOL,protocol)
    print({'protocol_sha256':read_json(PROTOCOL)['sha256'],'tasks':36,'scenes':18,'trajectories':252,'max_attempts':504,
           'reserved_tasks':18,'reserved_scenes':9,'new_model_calls':0})


if __name__=='__main__': main()
