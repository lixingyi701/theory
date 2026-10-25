#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONPATH=src:scripts
PY="${THEORY_RUNNER_PYTHON:-python}"
ROOT=runs/reflexion-four-attempt-validation-20261025
release_replicas() {
  for gpu in 2 3; do
    name="theory-four-attempt-replica-$gpu-20261025"
    if tmux has-session -t "$name" 2>/dev/null; then tmux kill-session -t "$name"; fi
  done
}
trap release_replicas EXIT
"$PY" scripts/run_four_attempt_validation.py validate
"$PY" -u scripts/run_four_attempt_validation.py run
"$PY" scripts/audit_reflexion_protocol.py --run "$ROOT" --output "$ROOT/protocol-audit.json"
GRAMMAR=$("$PY" -c 'from pathlib import Path; import alfworld; print(Path(alfworld.__file__).parent/"data/alfred.twl2")')
"$PY" scripts/audit_reflexion_action_shapes.py --run "$ROOT" --grammar "$GRAMMAR" --output "$ROOT/action-shapes.json"
"$PY" -u scripts/evaluate_four_attempt_validation.py
echo FOUR_ATTEMPT_VALIDATION_AND_EVALUATION_COMPLETE
