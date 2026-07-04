#!/usr/bin/env bash
set -euo pipefail

python -m allthemix.cli.train \
  --config configs/imagenet_a/torch_resnet101/eval.yaml \
  "$@"
