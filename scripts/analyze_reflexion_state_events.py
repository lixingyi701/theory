#!/usr/bin/env python3
"""Describe public procedural events in completed controlled-memory cells."""
import argparse
from pathlib import Path
from agent_reliability.io import digest, file_hash, read_json, write_json
from analyze_reflexion_precondition_forks import procedure_events


def main():
    parser = argparse.ArgumentParser()
    for name in ['protocol', 'run', 'output']:
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(); protocol = read_json(args.protocol)
    data = read_json(args.run/'summary.json')
    if data['completed_cells'] != data['expected_cells']:
        raise ValueError('Controlled-state measurements incomplete')
    cases = {case['task_id']: case for case in protocol['cases']}
    rows = []
    for row in data['rows']:
        case = cases[row['task_id']]
        registry = read_json(Path(case['target_registry']))
        target = next(t for t in registry['targets'] if t['task_id'] == case['task_id'])
        probe = read_json(Path(target['probe_file']))['candidate']
        result_path = Path(row['result_path'])
        if file_hash(result_path) != row['result_file_sha256']:
            raise ValueError('Source result changed')
        steps = read_json(result_path.parent/'checkpoint.json')['transitions']
        result = read_json(result_path)
        events = procedure_events(steps, probe['object'].split()[0], probe['verb'], probe['device'])
        fresh = result['memory_after'][len(result['memory_before']):]
        rows.append({'task_id': row['task_id'], 'state': row['state'], 'success': row['success'],
            'reused': row['reused'], 'trajectory_sha256': result['trajectory_sha256'],
            'new_reflection_sha256': [digest(entry) for entry in fresh], **events})
    report = {'scope': 'Observable actions and feedback; no automatic semantic belief labels',
        'summary_sha256': data['sha256'], 'protocol_sha256': protocol['sha256'],
        'driver_sha256': file_hash(Path(__file__)), 'rows': rows}
    report['sha256'] = digest(report); write_json(args.output, report)
    print({'cells': len(rows), 'wrong_chain_cells': sum(bool(r['wrong_precondition_chains']) for r in rows),
           'successful_macro_cells': sum(bool(r['held_successful_target_macros']) for r in rows)})


if __name__ == '__main__':
    main()
