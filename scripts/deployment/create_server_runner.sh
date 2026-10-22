#!/usr/bin/env bash
# Plan by default. Execute inside an already checked-out server project.
set -euo pipefail
if [[ "${1:-}" != --execute ]]; then
  echo 'Plan: create isolated $THEORY_ROOT/envs/theory-runner, install pinned packages, build native code against server glibc.'
  echo 'Set THEORY_ROOT and optional THEORY_CONDA_ROOT, then add --execute.'
  exit 0
fi
: "${THEORY_ROOT:?Set the server experiment root}"
_theory_conda_root="${THEORY_CONDA_ROOT:-$HOME/miniconda3}"
cd "$THEORY_ROOT/project"
"$_theory_conda_root/bin/conda" create -y -p "$THEORY_ROOT/envs/theory-runner" \
  --file requirements/locks/runner-linux-64.conda.txt
source "$_theory_conda_root/etc/profile.d/conda.sh"
set +u
conda activate "$THEORY_ROOT/envs/theory-runner"
set -u
# The historical Conda toolchain has a glibc2.39 sysroot. Ubuntu20.04 must
# compile the planner with its native compiler, rather than that sysroot.
export CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONNOUSERSITE=1
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CONDA_BUILD_SYSROOT CMAKE_ARGS
python -m pip install --no-cache-dir -r requirements/locks/runner-pip.txt
python -m pip check
