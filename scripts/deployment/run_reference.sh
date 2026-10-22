#!/usr/bin/env bash
# All paths, native libraries, data and model requests are server-local.
set -euo pipefail
: "${THEORY_ROOT:?Set THEORY_ROOT to the server experiment root}"
cd "$THEORY_ROOT/project"
export ALFWORLD_DATA="$THEORY_ROOT/project/data/alfworld"
export RELIABILITY_API_BASE=http://127.0.0.1:8000/v1
export PYTHONNOUSERSITE=1 NO_PROXY=localhost,127.0.0.1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
exec "$THEORY_ROOT/envs/theory-runner/bin/python" -m agent_reliability reference-run "$@"
