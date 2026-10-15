#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts
python scripts/download_model.py
python scripts/download_alfworld.py
bash scripts/serve_model.sh qwen3_8b --plan
