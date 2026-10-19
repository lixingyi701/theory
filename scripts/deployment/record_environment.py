#!/usr/bin/env python3
"""Collect publishable versions and hardware, without host/user/auth metadata."""
import argparse
import importlib.metadata as metadata
import json
import platform
import subprocess
from pathlib import Path


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(role):
    names = (['alfworld', 'textworld', 'fast-downward-textworld', 'numpy', 'PyYAML'] if role == 'runner'
             else ['vllm', 'torch', 'transformers', 'huggingface_hub', 'tokenizers', 'numpy'])
    packages = {}
    for name in names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    os_release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    return {'schema_version': 1, 'role': role, 'python': platform.python_version(),
            'os': os_release.get('PRETTY_NAME', '').strip('"'), 'kernel': platform.release(),
            'architecture': platform.machine(), 'packages': packages,
            'cpu_model': next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), None),
            'logical_cpus': command(['getconf', '_NPROCESSORS_ONLN']),
            'memory_kib': next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')),
            'nvidia': command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
            'scope': 'version/hardware snapshot; no credentials, network addresses, hostname or home paths'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['runner', 'server'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(collect(a.role), indent=2) + '\n')
