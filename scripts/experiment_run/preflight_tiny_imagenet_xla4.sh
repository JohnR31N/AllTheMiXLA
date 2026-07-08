#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

required_venv_name="${VENV_NAME_REQUIRED-.venvxla}"
preflight_args=(
  --preset tiny-imagenet-xla4
  --format preflight
  --check-env
  --require-tpu-env
)
if [[ -n "$required_venv_name" ]]; then
  preflight_args+=(--require-venv-name "$required_venv_name")
fi
preflight_args+=(
  --min-free-gb "${MIN_FREE_GB:-10}"
  --require-existing-storage-roots
  --require-complete
)

python -m allthemix.cli.summarize \
  "${preflight_args[@]}" \
  "$@"
