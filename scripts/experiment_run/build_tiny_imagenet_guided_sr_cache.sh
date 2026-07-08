#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"

python -m allthemix.cli.build_saliency_cache \
  --config configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml \
  --method spectral_residual \
  --output "${GUIDED_SR_SALIENCY_OUTPUT:-./data/tiny_imagenet_train_guided_sr_saliency.npy}" \
  "$@"
