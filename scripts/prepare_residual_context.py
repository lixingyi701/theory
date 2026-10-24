#!/usr/bin/env python3
"""Freeze a paired target-rule by residual-context diagnostic, without refitting."""
from pathlib import Path
from agent_reliability.cli import provenance
from agent_reliability.io import digest, file_hash, read_json, write_json
from prepare_middle_context_transport import native_text
from run_middle_context_transport import verify_protocol as verify_parent

OUTPUT = Path('repro/reflexion-20261024/residual-context-protocol.json')
REGISTRY = Path('repro/reflexion-20261024/residual-context-registry.json')
PARENT = Path('repro/reflexion-20261023/middle-context-protocol.json')
PARENT_RUN = Path('runs/reflexion-middle-context-20261023')


def residual_for(record):
    original = native_text(record['source'])
    spans = record['retained_spans']
    if not spans or any(not 0 <= s['start'] < s['end'] <= len(original) for s in spans):
        raise ValueError('Residual source span out of range')
    if any(a['end'] > b['start'] for a, b in zip(spans, spans[1:])):
        raise ValueError('Residual spans must retain original order')
    if record['format_prefix'] not in {'', 'Plan: '}:
        raise ValueError('Only a format prefix may be added')
    text = record['format_prefix'] + ' '.join(original[s['start']:s['end']] for s in spans)
    if text != record['residual_text'] or digest(text) != record['residual_sha256']:
        raise ValueError('Residual reconstruction or hash changed')
    if record['residual_semantic_label'] != 'U':
        raise ValueError('Residual must omit the tracked positive and negative premise')
    return text


def memory_for(protocol, cell):
    labels = cell['labels']
    memory = []
    for symbol in labels:
        if symbol == 'R':
            record = next(r for r in protocol['residual_registry']['entries'] if r['task_id'] == cell['task_id'])
            if cell['residual_sha256'] != record['residual_sha256']:
                raise ValueError('Cell residual identity mismatch')
            memory.append(residual_for(record))
        else:
            memory.append(protocol['entries'][symbol])
    return memory


def main():
    if OUTPUT.exists():
        raise FileExistsError('Do not overwrite a frozen diagnostic')
    parent = verify_parent(PARENT)
    registry = read_json(REGISTRY)
    if registry['sha256'] != digest({k:v for k,v in registry.items() if k != 'sha256'}):
        raise ValueError('Registry integrity')
    if registry['parent_protocol_sha256'] != parent['sha256']:
        raise ValueError('Registry parent mismatch')
    for record in registry['entries']:
        residual_for(record)
    cases = sorted(parent['cases'], key=lambda c:c['task_id'])
    if {r['task_id'] for r in registry['entries']} != {c['task_id'] for c in cases}:
        raise ValueError('All twelve fixed E sources must be represented')
    presence = read_json(Path('repro/reflexion-20261023/state-representation-presence-baseline.json'))
    primary = read_json(Path('repro/reflexion-20261023/state-representation-model-freeze.json'))
    previous = read_json(Path('repro/reflexion-20261023/middle-context-results.json'))
    forecasts = {tuple(r['labels']):r['joint'] for r in presence['one_attempt_forecasts']}
    q_e, q_c = forecasts[('N','E','N')], forecasts[('N','C','N')]
    mixture = {k:(q_e[k]+q_c[k])/2 for k in q_e}
    from state_representation_model import predict
    blocks, reused, predictions = [], [], []
    protocol = {'entries':parent['entries'], 'residual_registry':registry}
    conditions = ['N_E','R_E','R_C','N_C']
    for index, case in enumerate(cases):
        task = case['task_id']
        record = next(r for r in registry['entries'] if r['task_id']==task)
        # Rotate order within each task/seed; no outcome-dependent selection.
        for seed in [2101,2102]:
            offset = (index + seed - 2101) % 4
            order = conditions[offset:]+conditions[:offset]
            cells = []
            for condition in order:
                context, rule = condition.split('_')
                labels = [context,rule,'N']
                cell = {'id':f'residual-{task}-{seed}-{condition}', 'task_id':task,
                        'condition':condition, 'labels':labels}
                if context == 'R':
                    cell['residual_sha256'] = record['residual_sha256']
                cell['memory_sha256'] = digest(memory_for(protocol,cell))
                predictions.append({'cell_id':cell['id'], 'task_id':task, 'seed':seed,
                    'condition':condition, 'presence':q_e if rule=='E' else q_c,
                    'constant':predict(primary['all_fitted_models']['constant'],[]),
                    'mixture_joint':mixture})
                if seed == 2101 and context == 'N':
                    source_id = f'middle-{task}-{rule}'
                    path = PARENT_RUN/'cells'/source_id/'trial-01'
                    result, identity = read_json(path/'result.json'), read_json(path/'identity.json')
                    row = next(r for r in previous['rows'] if r['cell_id']==source_id)
                    if file_hash(path/'result.json') != row['result_sha256']:
                        raise ValueError('Reused result changed')
                    if (identity['sha256'] != digest({k:v for k,v in identity.items() if k!='sha256'})
                        or identity['sha256'] != row['identity_sha256']
                        or identity['sha256'] != result['identity_sha256']):
                        raise ValueError('Reused identity integrity or binding mismatch')
                    if result['initial_observation_sha256'] != case['initial_observation_sha256']:
                        raise ValueError('Reused reset observation mismatch')
                    if identity['task'] != case['task'] or identity['seed'] != seed or identity['trial'] != 1:
                        raise ValueError('Reused task/seed/trial mismatch')
                    if result['memory_before'] != memory_for(protocol,cell) or identity['memory'] != result['memory_before']:
                        raise ValueError('Reused control memory mismatch')
                    reused.append({**cell,'seed':seed,'source_run':str(PARENT_RUN),'source_cell_id':source_id,
                        'result_sha256':file_hash(path/'result.json'), 'identity_sha256':identity['sha256']})
                else:
                    cells.append(cell)
            blocks.append({'task_id':task, 'seed':seed, 'worker':index%2,
                           'panel':'factorial', 'attempts':1, 'cells':cells})
    dependencies = dict(parent['protected_files'])
    for name in [str(PARENT),str(REGISTRY),'scripts/prepare_residual_context.py',
        'scripts/run_residual_context.py','scripts/analyze_residual_context.py',
        'scripts/close_residual_context.py','tests/test_residual_context.py',
        'docs/experiments/residual-context-design-20261024.md',
        'repro/reflexion-20261023/middle-context-results.json',
        'repro/reflexion-20261023/middle-context-semantic-labels.json']:
        dependencies[name] = file_hash(Path(name))
    protocol.update({'schema_version':1, 'scope':'Paired residual-text diagnostic on twelve known development tasks',
        'parent_commit':'9d711f3d60bee0430e37dfaeae2a5c8646286dca', 'parent_protocol_sha256':parent['sha256'],
        'scientific_questions':['Does residual native text alter task success with canonical middle E/C fixed?',
            'Does residual text change the benefit of correcting the target rule?',
            'Do the success and failure-reflection mechanisms require different starting information?'],
        'cases':cases,'entry_hashes':parent['entry_hashes'],'blocks':blocks,'reused_controls':reused,
        'predictions':predictions,'ports':[8001,8002],'gpu_indices':[0,1],'phases':['factorial'],
        'phase_cells':{'factorial':72},'expected_cells':72,'maximum_attempts':72,'analysis_cells':96,
        'seeds':[2101,2102], 'limits':{'additional_calls_including_replica_checks':4500,'seconds':7200},
        'analysis':{'primary':'Task-success interaction (RC-RE)-(NC-NE); average two seeds within task, then twelve tasks; task bootstrap 20000 seed 2026102401',
            'decomposition':['R_vs_N_under_E','R_vs_N_under_C','correction_under_N','correction_under_R'],
            'secondary':'Fixed joint and success Brier versus old constant and fixed balanced-mixture forecast; failed reflection E/C/U/A and observable action events separately',
            'decision':'An interaction interval excluding zero supports effect modification in this diagnostic; intervals including zero remain unresolved, not equivalence. No refit or automatic promotion from an auxiliary score.',
            'unit':'12 known tasks; seeds and conditions are nested repeated measurements, not new independent tasks',
            'reuse':'24 seed2101 N controls are audited historical results; 72 new attempts complete a 96-cell matrix. Report fresh seed2102 separately.',
            'semantic':'Codex full-text E/C/U/A annotation; retain A; prior labels immutable; no human second review claimed',
            'limits':'R is an edited subsequence of one prespecified E-source per task in the oldest entry. Its length, advice, token offsets and organization change together. This is not a reconstruction of the full native middle entry, and U does not mean error-free.'},
        'source_presence_freeze_sha256':presence['sha256'],'parameter_refits':0,'holdout_model_calls':0,
        'protected_files':dependencies,'source_hashes':provenance()['source_hashes']})
    from run_residual_context import validate_matrix
    validate_matrix(protocol)
    protocol['sha256']=digest(protocol)
    write_json(OUTPUT,protocol)
    print({'protocol_sha256':protocol['sha256'],'new_attempts':72,'reused_controls':24,'model_calls':0})


if __name__=='__main__':
    main()
