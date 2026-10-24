#!/usr/bin/env python3
"""Finish the selected diagnostic, audit it, then collect prospective prefixes."""
import subprocess
import sys
import time
from pathlib import Path

from agent_reliability.io import file_hash, read_json, run_lock, write_json
from reflexion_campaign import verify


def main():
    campaign = Path('runs/reflexion-precondition-20261023')
    diagnostic = Path('runs/reflexion-precondition-diagnostic-forks-20261023')
    prefix = Path('runs/reflexion-precondition-prefix24-20261023')
    manifest = Path('repro/reflexion-20261023/precondition-prefix24-v1.json')
    with run_lock(campaign/'controller'):
        contract = verify(campaign)
        write_json(campaign/'controller.json', {'driver_sha256': file_hash(Path(__file__)),
                   'campaign_sha256': contract['sha256'], 'started_unix': time.time()})
        while time.time() < contract['deadline_unix']:
            active = subprocess.run(['tmux', 'has-session', '-t', 'theory-precondition-diagnostic'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not active:
                report = read_json(diagnostic/'summary.json')
                if report['completed_targets'] != report['expected_targets']:
                    raise RuntimeError('Diagnostic stopped before completion')
                break
            time.sleep(5)
        else:
            raise RuntimeError('Campaign deadline reached')

        def run(arguments, log):
            with log.open('a') as stream:
                result = subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT,
                    timeout=max(1, contract['deadline_unix']-time.time()+200))
            if result.returncode:
                raise RuntimeError('Stage failed: '+arguments[0])

        log = Path('runs/setup/precondition-controller.log')
        run(['scripts/audit_reflexion_protocol.py', '--run', str(diagnostic), '--output',
             str(campaign/'diagnostic-protocol-audit.json')], log)
        import alfworld
        grammar = str(Path(alfworld.__file__).resolve().parent/'data/alfred.twl2')
        run(['scripts/audit_reflexion_action_shapes.py', '--run', str(diagnostic), '--grammar', grammar,
             '--output', str(campaign/'diagnostic-action-shapes.json')], log)
        run(['scripts/run_reflexion_precondition_prefixes.py', '--campaign', str(campaign), '--manifest', str(manifest),
             '--output', str(prefix), '--audit', 'runs/audit/precondition-prefix24-direct.json',
             '--environment-audit', 'runs/audit/precondition-prefix24-official.json'],
            Path('runs/setup/precondition-prefixes.log'))
        run(['scripts/audit_reflexion_protocol.py', '--run', str(prefix), '--output',
             str(campaign/'prefix-protocol-audit.json')], log)
        run(['scripts/audit_reflexion_action_shapes.py', '--run', str(prefix), '--grammar', grammar,
             '--output', str(campaign/'prefix-action-shapes.json')], log)
        write_json(campaign/'status.json', {'stage': 'prospective_prefixes_ready_for_target_review',
                    'prefix_run': str(prefix), 'updated_unix': time.time()})


if __name__ == '__main__':
    main()
