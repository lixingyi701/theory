#!/usr/bin/env python3
"""Freeze native multi-attempt trajectories against an existing conditional forecast."""
from pathlib import Path
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, write_json
from predict_calibrated_finite_window import restore
from run_residual_context import verify_protocol as verify_parent

OUTPUT = Path('repro/reflexion-20261024/multiround-diagnostic-protocol.json')
PARENT = Path('repro/reflexion-20261024/residual-context-protocol.json')
FORECAST = Path('repro/theory-20261023/agent-calibrated-finite-window-forecast.json')
HISTORIES = {'single_error':['E'], 'repeated_error':['E','N','E']}
OPERATIONS = ['retain','correct_latest','delete_latest']


def initial_memory(protocol, trajectory):
    labels = trajectory['initial_labels']
    expected = restore(trajectory['initial_history'],trajectory['operation'])
    if labels != expected:
        raise ValueError('Starting recovery mapping changed')
    memory = [protocol['entries'][symbol] for symbol in labels]
    if digest(memory) != trajectory['initial_memory_sha256']:
        raise ValueError('Initial memory hash changed')
    return memory


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the frozen multiround diagnostic')
    parent = verify_parent(PARENT)
    forecast = read_json(FORECAST)
    if forecast['sha256'] != digest({k:v for k,v in forecast.items() if k!='sha256'}):
        raise ValueError('Existing conditional forecast integrity')
    if forecast['sha256'] != '2a2c4397ad026d73b87a1f7b1c83bdc9a2759e25085d6f5992e6b88e0fe0ffdb':
        raise ValueError('Do not silently replace the candidate forecast')
    cases = sorted(parent['cases'],key=lambda c:c['task_id'])
    tasks = {c['task_id'] for c in cases}
    for path in Path('runs').glob('reflexion-*/**/identity.json'):
        prior = read_json(path)
        if prior.get('task',{}).get('task_id') in tasks and prior.get('seed') == 2103:
            raise ValueError('Seed2103 was already used on a selected task')
    sealed = read_json(Path('repro/baseline-20261019/holdout134.json'))
    if {c['task_id'] for c in cases} & {c['task_id'] for c in sealed['tasks']}:
        raise ValueError('Known-task diagnostic cannot access sealed tasks')
    trajectories,forecasts = [],[]
    combinations = [(h,op) for h in HISTORIES for op in OPERATIONS]
    for index,case in enumerate(cases):
        offset=index%len(combinations)
        order=combinations[offset:]+combinations[:offset]
        for history_key,operation in order:
            history = HISTORIES[history_key]
            after = restore(history,operation)
            identifier=f'multiround-{case["task_id"]}-{history_key}-{operation}-2103'
            trajectory={'id':identifier,'task_id':case['task_id'],'seed':2103,
                'history_key':history_key,'initial_history':history,'operation':operation,
                'initial_labels':after,'initial_memory_sha256':digest([parent['entries'][v] for v in after]),
                'worker':index%3,'max_attempts':3}
            trajectories.append(trajectory)
            budgets=[]
            for budget in [1,2,3]:
                row=next(r for r in forecast['rows'] if r['initial_memory']==history and r['attempt_budget']==budget)
                prediction=row['operations'][operation]
                if prediction['memory_after'] != after:
                    raise ValueError('Candidate forecast uses another recovery mapping')
                budgets.append({'attempt_budget':budget,
                    'success_probability_bounds':prediction['success_probability_bounds']})
            forecasts.append({'trajectory_id':identifier,'task_id':case['task_id'],
                'history_key':history_key,'operation':operation,'budgets':budgets})
    dependencies=dict(parent['protected_files'])
    for name in [str(PARENT),str(FORECAST),'scripts/predict_calibrated_finite_window.py',
        'scripts/prepare_multiround_diagnostic.py','scripts/run_multiround_diagnostic.py',
        'scripts/analyze_multiround_diagnostic.py','scripts/close_multiround_diagnostic.py',
        'tests/test_multiround_diagnostic.py','docs/experiments/multiround-diagnostic-design-20261024.md',
        'repro/reflexion-20261024/residual-context-semantic-labels.json']:
        dependencies[name]=file_hash(Path(name))
    value={'schema_version':1,'scope':'Known-task multiround process diagnosis with one initial recovery and native later updates',
        'parent_commit':'e15f593ec4a4383cdd74d436d33faf07293b7306','parent_protocol_sha256':parent['sha256'],
        'source_forecast_sha256':forecast['sha256'],'source_forecast_file_sha256':file_hash(FORECAST),
        'source_presence_freeze_sha256':parent['source_presence_freeze_sha256'],
        'cases':cases,'entries':parent['entries'],'entry_hashes':parent['entry_hashes'],
        'histories':HISTORIES,'operations':OPERATIONS,'trajectories':trajectories,'forecasts':forecasts,
        'candidate_parameters':forecast['parameters'],'ports':[8001,8002,8003],'gpu_indices':[0,1,3],
        'new_seed_audit':{'seed':2103,'selected_tasks':12,'previous_actual_run_matches':0},
        'expected_trajectories':72,'maximum_attempts':216,'attempt_budgets':[1,2,3],
        'limits':{'additional_calls_including_replica_checks':11000,'seconds':10800},
        'scientific_questions':[
            'Do initial correction and deletion share a success curve for a single E, as the presence-only candidate predicts?',
            'For E,N,E, does correcting versus deleting the latest E produce the predicted budget-two difference when the older E exits at different updates?',
            'Do forecast departures begin at the first attempt or after native reflection updates, and which task paths accompany them?'],
        'analysis':{'primary':'For repeated_error at budget two, report task-paired correct_latest minus delete_latest success difference with task-bootstrap95 interval against the existing conditional candidate gain bounds',
            'curves':'Report all six history-operation success curves at budgets1/2/3, fixed candidate bounds, signed distance from the candidate interval, and task-bootstrap95 intervals for empirical values',
            'bootstrap':{'replicates':20000,'seed':2026102402,'unit':'12 base tasks; operations and nested budgets remain within each task'},
            'decision':'Diagnose predictive departures; no automatic gate, parameter refit, or theorem rejection from candidate bounds. A bounds cover semantic assignment only, not parameter or state error.',
            'native_update':'Run complete author attempts and append exactly one reflection after each failure, including terminal failure. Later attempts consume the full growing history through the author last-three visibility rule.',
            'old_error_tracking':'Track initial entries by origin index after the one recovery; corrected E is no longer an old error, deleted E has no origin, new reflections are new entries even if their text repeats.',
            'failure_labels':'Codex full-text E/C/U/A review with prior hashes reused. Starting visible E => restatement; definite absence with no A => generation; A-only uncertainty remains unresolved.',
            'interpretation':'Check initial E+C conflict and C versus empty first; later departures also involve native wording, failure-selected task composition, state compression, and sampling uncertainty.',
            'coverage':'A new seed on known development tasks; no independent-task or sealed-holdout validation'},
        'protected_files':dependencies,'source_hashes':provenance()['source_hashes'],
        'parameter_refits':0,'holdout_model_calls':0}
    from run_multiround_diagnostic import validate_matrix
    validate_matrix(value)
    value['sha256']=digest(value);write_json(OUTPUT,value)
    print({'protocol_sha256':value['sha256'],'trajectories':72,'maximum_attempts':216,'model_calls':0})


if __name__=='__main__':
    main()
