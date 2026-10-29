#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="${THEORY_RUNNER_PYTHON:-python}"
ROOT=runs/reflexion-recovery-mechanism-20261029
finish() {
  code=$?
  trap - EXIT
  set +e
  "$PY" scripts/run_recovery_mechanism.py close --exit-code "$code"
  final_code=$?
  if [ "$code" -eq 0 ] && [ "$final_code" -ne 0 ]; then code=$final_code; fi
  exit "$code"
}
trap finish EXIT
"$PY" scripts/run_recovery_mechanism.py validate
"$PY" -u scripts/run_recovery_mechanism.py run
"$PY" scripts/audit_reflexion_protocol.py --run "$ROOT" --output "$ROOT/protocol-audit.json"
GRAMMAR=$("$PY" -c 'from pathlib import Path;import alfworld;print(Path(alfworld.__file__).parent/"data/alfred.twl2")')
"$PY" scripts/audit_reflexion_action_shapes.py --run "$ROOT" --grammar "$GRAMMAR" --output "$ROOT/action-shapes.json"
"$PY" -u scripts/evaluate_recovery_mechanism.py
echo RECOVERY_MECHANISM_AND_EVALUATION_COMPLETE
