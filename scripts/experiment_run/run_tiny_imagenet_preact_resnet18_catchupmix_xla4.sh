#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

python -m allthemix.cli.train \
  --config configs/tiny_imagenet/preact_resnet18/catchupmix_xla4.yaml \
  --device xla \
  --num-cores "${NUM_CORES:-4}" \
  --num-workers "${NUM_WORKERS:-0}" \
  "$@"
