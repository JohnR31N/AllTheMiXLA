#!/usr/bin/env bash
set -euo pipefail

python -m allthemix.cli.train \
  --config configs/imagenet_a/preact_resnet18/eval.yaml \
  "$@"
