#!/usr/bin/env bash
set -euo pipefail

python -m allthemix.cli.train \
  --config configs/cifar10/preact_resnet18/mixup.yaml \
  "$@"
