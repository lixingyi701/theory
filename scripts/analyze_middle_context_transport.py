#!/usr/bin/env python3
"""Audit prospective cells and score immutable forecasts without parameter refitting."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_reflexion_precondition_forks import procedure_events
from audit_reflexion_action_shapes import action_shape
from run_middle_context_transport import verify_protocol
from prepare_middle_context_transport import memory_for
from state_representation_model import scores
import numpy as np


def collect(protocol, root):
    cells = {c['id']: (b, c) for b in protocol['blocks'] for c in b['cells']}
    cases = {c['task_id']: c for c in protocol['cases']}
    known = {digest(v): ('U' if k == 'N' else k) for k, v in protocol['entries'].items()}
    prior = read_json(Path('repro/reflexion-20261023/state-representation-semantic-labels.json'))
    known.update({v['sha256']:v['label'] for v in prior['entries']})
    rows, review = [], {}
    for path in sorted(root.glob('cells/*/trial-*/result.json')):
        block, cell = cells[path.parent.parent.name]
        value = read_json(path); identity = read_json(path.parent/'identity.json')
        expected_memory = memory_for(protocol, cell)
        if identity['sha256'] != digest({k: v for k, v in identity.items() if k != 'sha256'}):
            raise ValueError('Attempt identity integrity')
        if identity['memory'] != expected_memory or value['memory_before'] != expected_memory:
            raise ValueError('Wrong constructed memory')
        if identity['task'] != cases[block['task_id']]['task'] or identity['seed'] != block['seed']:
            raise ValueError('Task/seed mismatch')
        if identity['trial'] != 1 or value['trial'] != 1 or value['identity_sha256'] != identity['sha256']:
            raise ValueError('Attempt identity mismatch')
        if value['initial_observation_sha256'] != cases[block['task_id']]['initial_observation_sha256']:
            raise ValueError('Task reset mismatch')
        steps = read_json(path.parent/'checkpoint.json')['transitions']
        if digest(steps) != value['trajectory_sha256']:
            raise ValueError('Trajectory integrity')
        fresh = value['memory_after'][len(expected_memory):]
        if value['memory_after'][:len(expected_memory)] != expected_memory or len(fresh) != int(not value['success']):
            raise ValueError('Native critic append mismatch')
        for text in fresh:
            sha = digest(text)
            if sha not in known:
                review.setdefault(sha, {'sha256': sha, 'text': text, 'sources': []})['sources'].append(cell['id'])
        goal = Path(cases[block['task_id']]['task']['gamefile']).parts[-3].split('-')[1].lower()
        events = procedure_events(steps, goal, 'heat', 'microwave 1')
        rows.append({'cell_id': cell['id'], 'task_id': block['task_id'], 'condition': cell['condition'],
            'panel': block['panel'], 'seed': block['seed'], 'initial_labels': cell['labels'], 'success': value['success'],
            'decisions': value['decisions'], 'status': value['status'], 'result_sha256': file_hash(path),
            'identity_sha256': identity['sha256'], 'trajectory_sha256': value['trajectory_sha256'],
            'memory_sha256': digest(expected_memory), 'new_reflection_hashes': [digest(v) for v in fresh],
            'observable_events': {k: v if isinstance(v, bool) else len(v) for k, v in events.items()},
            'action_shapes': dict(Counter(action_shape(s['action'], s['submitted_action']) for s in steps)),
            'no_effect_shapes': dict(Counter(action_shape(s['action'], s['submitted_action']) for s in steps if s['observation'] == 'Nothing happens.'))})
    if len({r['cell_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate committed cells')
    return rows, known, review


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261023/middle-context-protocol.json'))
    p.add_argument('--export-review', type=Path)
    p.add_argument('--labels', type=Path)
    args = p.parse_args()
    protocol = verify_protocol(args.protocol)
    rows, known, review = collect(protocol, args.run)
    if args.export_review:
        write_json(args.export_review, {'protocol_sha256': protocol['sha256'], 'items': list(review.values())})
    if not args.labels:
        print({'completed_cells': len(rows), 'unique_reflections_requiring_review': len(review)}); return
    labels_record = read_json(args.labels)
    labels = labels_record
    if labels['protocol_sha256'] != protocol['sha256'] or labels['sha256'] != digest({k: v for k, v in labels.items() if k != 'sha256'}):
        raise ValueError('Semantic label integrity')
    if set(review) != {r['sha256'] for r in labels['entries']}:
        raise ValueError('Review coverage mismatch')
    for entry in labels['entries']:
        if entry['label'] not in {'E', 'C', 'U', 'A'}:
            raise ValueError('Unknown semantic class')
        known[entry['sha256']] = entry['label']
    if len(rows) != protocol['expected_cells']:
        raise ValueError('Incomplete matrix is not a complete calibration fit')
    for row in rows:
        row['outcome'] = 'S' if row['success'] else known[row['new_reflection_hashes'][0]]
    forecasts={v['cell_id']:v for v in protocol['predictions']}
    phase_reports=[]
    for phase in protocol['phases']:
        selected=[r for r in rows if r['panel']==phase]
        task_scores=[]
        for task in sorted({r['task_id'] for r in selected}):
            block=[r for r in selected if r['task_id']==task]
            for model in ['presence','constant','remaining_lifetime']:
                values=[scores(forecasts[r['cell_id']][model],r['outcome']) for r in block]
                task_scores.append({'task_id':task,'model':model,
                    **{k:float(np.mean([v[k] for v in values])) for k in values[0]}})
        tasks=sorted({v['task_id'] for v in task_scores})
        rng=np.random.default_rng(2026102305); resamples=rng.integers(0,len(tasks),size=(20000,len(tasks)))
        delta=np.asarray([next(v['brier'] for v in task_scores if v['task_id']==t and v['model']=='constant')
            -next(v['brier'] for v in task_scores if v['task_id']==t and v['model']=='presence') for t in tasks])
        interval=np.quantile(delta[resamples].mean(axis=1),[.025,.975]).tolist()
        phase_reports.append({'phase':phase,'cells':len(selected),'independent_base_tasks':len(tasks),
            'outcomes':dict(Counter(r['outcome'] for r in selected)),
            'scores':{name:{metric:float(np.mean([v[metric] for v in task_scores if v['model']==name]))
                for metric in ['brier','log_loss','success_brier']} for name in ['presence','constant','remaining_lifetime']},
            'constant_minus_presence_brier':float(delta.mean()),'task_bootstrap_95_interval':interval,
            'predeclared_advantage_criterion_met':interval[0]>0,'task_scores':task_scores,
            'by_condition':[{'condition':condition,'outcomes':dict(Counter(r['outcome'] for r in selected if r['condition']==condition)),
                'cells':sum(r['condition']==condition for r in selected)} for condition in sorted({r['condition'] for r in selected})]})
    source_analysis=read_json(Path('repro/reflexion-20261023/state-representation-results.json'))
    old={r['cell_id']:r for r in source_analysis['rows']}; current={r['cell_id']:r for r in rows}
    contrasts=[]; cells={c['id']:c for b in protocol['blocks'] for c in b['cells']}
    for row in rows:
        if row['panel']!='native': continue
        cell=cells[row['cell_id']]; control=cell['compressed_control']
        peer=(old if control['source_run']=='runs/reflexion-state-representation-20261023' else current)[control['cell_id']]
        expected=protocol['entries']; labels=cell['labels']
        path=Path(control['source_run'])/'cells'/control['cell_id']/'trial-01'
        raw=read_json(path/'result.json'); identity=read_json(path/'identity.json')
        if raw['memory_before'] != [expected[v] for v in labels] or identity['seed']!=row['seed'] or identity['task']['task_id']!=row['task_id']:
            raise ValueError('Reused compressed control differs')
        if file_hash(path/'result.json')!=peer['result_sha256']:
            raise ValueError('Reused control result changed')
        contrasts.append({'task_id':row['task_id'],'source_label':cell['native_source']['label'],
            'native_cell_id':row['cell_id'],'control_cell_id':peer['cell_id'],
            'native_success':row['success'],'control_success':peer['success'],
            'native_outcome':row['outcome'],'control_outcome':peer['outcome'],
            'source_reflection_sha256':cell['native_source']['reflection_sha256']})
    source_cases={c['task_id']:c for c in protocol['cases']}
    summary=read_json(args.run/'summary.json')
    value={'protocol_sha256':protocol['sha256'],'summary_sha256':summary['sha256'],'semantic_labels_sha256':labels_record['sha256'],
        'completed_cells':len(rows),'successes':sum(r['success'] for r in rows),'rows':rows,
        'phases':phase_reports,'native_compressed_pairs':contrasts,
        'middle_edge_pairs':[{'task_id':r['task_id'],'condition':r['condition'],'middle_success':r['success'],
            'oldest_success':old[f"state-{r['task_id']}-2101-oldest_{'error' if r['condition']=='middle_E' else 'correct'}"]['success'],
            'newest_success':old[f"state-{r['task_id']}-2101-newest_{'error' if r['condition']=='middle_E' else 'correct'}"]['success']}
            for r in rows if r['panel']=='middle'],
        'analysis_driver_sha256':file_hash(Path(__file__)),'parameter_refits':0}
    value['sha256']=digest(value); write_json(args.run/'analysis.json',value)
    print({'completed_cells':len(rows),'successes':value['successes'],'phases':[{k:v for k,v in p.items() if k not in ['task_scores','by_condition']} for p in phase_reports],
           'analysis_sha256':value['sha256'],'parameter_refits':0})



if __name__ == '__main__':
    main()
