#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"

bash_cmd="${BASH_CMD:-bash}"

"$bash_cmd" scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh \
  --smoke \
  --strict-preflight \
  "$@"
