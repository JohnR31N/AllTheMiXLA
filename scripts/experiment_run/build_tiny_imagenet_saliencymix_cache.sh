#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"

python -m allthemix.cli.build_saliency_cache \
  --config configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml \
  "$@"
