#!/usr/bin/env python3
"""Complete natural forks, calibrate controlled states, stop after freeze."""
import subprocess
import sys
import time
from pathlib import Path

from agent_reliability.io import file_hash, read_json, run_lock, write_json
from reflexion_campaign import verify


def main():
    campaign = Path('runs/reflexion-precondition-20261023')
    forks = Path('runs/reflexion-precondition-forks-20261023')
    calibration = Path('runs/reflexion-state-calibration-20261023')
    protocol = Path('repro/reflexion-20261023/state-prediction-protocol.json')
    with run_lock(campaign/'state-controller'):
        contract = verify(campaign)
        write_json(campaign/'state-controller.json', {'driver_sha256': file_hash(Path(__file__)),
                   'campaign_sha256': contract['sha256'], 'started_unix': time.time()})
        while time.time() < contract['deadline_unix']:
            live = subprocess.run(['tmux', 'has-session', '-t', 'theory-precondition-forks'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not live:
                report = read_json(forks/'summary.json')
                if report['completed_targets'] != report['expected_targets']:
                    raise RuntimeError('Natural precondition forks stopped before completion')
                break
            time.sleep(5)
        else:
            raise RuntimeError('Campaign deadline reached')

        def run(arguments, log):
            with log.open('a') as stream:
                result = subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT,
                    timeout=max(1, contract['deadline_unix']-time.time()+200))
            if result.returncode:
                raise RuntimeError('State prediction stage failed: '+arguments[0])

        log = Path('runs/setup/state-prediction-controller.log')
        run(['scripts/audit_reflexion_protocol.py', '--run', str(forks), '--output',
             str(campaign/'prospective-forks-protocol-audit.json')], log)
        run(['scripts/analyze_reflexion_precondition_forks.py', '--run', str(forks), '--parent-run',
             'runs/reflexion-precondition-prefix24-20261023', '--targets',
             'repro/reflexion-20261023/precondition-targets.json', '--output', str(forks/'analysis.json')], log)
        import alfworld
        grammar = str(Path(alfworld.__file__).resolve().parent/'data/alfred.twl2')
        run(['scripts/audit_reflexion_action_shapes.py', '--run', str(forks), '--grammar', grammar,
             '--output', str(campaign/'prospective-forks-action-shapes.json')], log)
        command = ['scripts/run_reflexion_state_prediction.py', '--campaign', str(campaign), '--protocol',
                   str(protocol), '--output', str(calibration), '--phase', 'calibration']
        run(command+['--validate-only'], log)
        run(command, Path('runs/setup/state-calibration.log'))
        run(['scripts/audit_reflexion_protocol.py', '--run', str(calibration), '--output',
             str(calibration/'protocol-audit.json')], log)
        run(['scripts/audit_reflexion_action_shapes.py', '--run', str(calibration), '--grammar', grammar,
             '--output', str(calibration/'action-shapes.json')], log)
        run(['scripts/fit_reflexion_state_prediction.py', '--protocol', str(protocol), '--calibration',
             str(calibration), '--validation-run', 'runs/reflexion-state-validation-20261023',
             '--parameters', str(calibration/'parameters.json'), '--predictions',
             str(calibration/'predictions.json')], log)
        write_json(campaign/'status.json', {'stage': 'development_parameters_frozen_validation_pending',
                   'parameters': str(calibration/'parameters.json'), 'predictions': str(calibration/'predictions.json'),
                   'updated_unix': time.time()})


if __name__ == '__main__':
    main()
