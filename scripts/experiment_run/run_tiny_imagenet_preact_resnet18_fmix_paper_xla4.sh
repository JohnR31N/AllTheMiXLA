#!/usr/bin/env bash
set -euo pipefail

python -m allthemix.cli.train \
  --config configs/tiny_imagenet/preact_resnet18/fmix_paper_xla4.yaml \
  "$@"
