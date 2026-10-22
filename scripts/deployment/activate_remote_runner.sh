#!/usr/bin/env bash
# Source on the server: THEORY_ROOT=/data02/<user>/theory source this file.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Source this file inside the server shell.' >&2
  exit 2
fi
: "${THEORY_ROOT:?Set THEORY_ROOT to the server experiment root}"
_theory_conda_root="${THEORY_CONDA_ROOT:-$HOME/miniconda3}"
_theory_restore_nounset=false
if [[ $- == *u* ]]; then _theory_restore_nounset=true; set +u; fi
source "$_theory_conda_root/etc/profile.d/conda.sh" || return
conda activate "$THEORY_ROOT/envs/theory-runner" || return
if $_theory_restore_nounset; then set -u; fi
export ALFWORLD_DATA="$THEORY_ROOT/project/data/alfworld"
export RELIABILITY_API_BASE=http://127.0.0.1:8000/v1
export NO_PROXY=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
cd "$THEORY_ROOT/project" || return
unset _theory_conda_root _theory_restore_nounset
