#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"


python -m allthemix.cli.train \
  --config configs/imagenet_a/torch_resnet101/mixup_eval.yaml \
  "$@"
