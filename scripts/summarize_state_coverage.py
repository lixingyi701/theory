#!/usr/bin/env python3
"""Report calibration coverage without changing frozen features, scores or forecasts."""
import argparse
from pathlib import Path
import numpy as np
from agent_reliability.io import digest, file_hash, read_json, write_json
from state_representation_model import features, MODELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path('repro/reflexion-20261023/state-representation-protocol.json'))
    args = p.parse_args()
    protocol = read_json(args.protocol); fitted = read_json(args.run/'model-freeze.json')
    if protocol['sha256'] != fitted['protocol_sha256']:
        raise ValueError('Coverage/model protocol mismatch')
    patterns = list(protocol['states'].values())
    designs = {name: np.asarray([features(state, name) for state in patterns]) for name in MODELS}
    coverage = []
    for forecast in fitted['one_attempt_forecasts']:
        state = forecast['labels']; by_model = {}
        for name, x in designs.items():
            vector = np.asarray(features(state, name))
            never_observed = np.all(x == 0, axis=0)
            by_model[name] = {'active_never_observed_feature_indices': np.flatnonzero(never_observed & (vector != 0)).tolist(),
                'outside_observed_linear_span': bool(np.linalg.matrix_rank(np.vstack([x, vector])) > np.linalg.matrix_rank(x))}
        coverage.append({'labels': state, 'exact_starting_pattern_observed': state in patterns,
                         'by_candidate': by_model})
    value = {'protocol_sha256': protocol['sha256'], 'model_freeze_sha256': fitted['sha256'],
        'purpose': 'Descriptive coverage only; the predeclared model comparison and forecasts are unchanged',
        'constructed_patterns_observed': len(patterns), 'forecast_patterns': len(coverage),
        'by_candidate': {name: {'feature_dimension': int(x.shape[1]), 'observed_design_rank': int(np.linalg.matrix_rank(x)),
            'never_observed_feature_indices': np.flatnonzero(np.all(x == 0, axis=0)).tolist()} for name, x in designs.items()},
        'forecasts': coverage,
        'interpretation': ['The middle E/C positions and remaining error lifetime two were not sampled.',
            'Regularization yields coefficients for these features but does not provide empirical identification.',
            'Novel mixed states also require an additivity/compression transport check even when their feature vector is in the observed span.',
            'Native reflection U can carry other advice and is not established equivalent to constructed N.'],
        'driver_sha256': file_hash(Path(__file__))}
    value['sha256'] = digest(value); write_json(args.run/'state-coverage.json', value)
    print({k: value[k] for k in ['by_candidate', 'constructed_patterns_observed', 'forecast_patterns', 'sha256']})


if __name__ == '__main__':
    main()
