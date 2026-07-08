#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

smoke=false
strict_preflight=false
extra_args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --smoke)
      smoke=true
      shift
      ;;
    --strict-preflight)
      strict_preflight=true
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ "$smoke" == true ]]; then
  user_extra_args=("${extra_args[@]}")
  smoke_checkpoint_dir="${SMOKE_CHECKPOINT_DIR:-./checkpoints/tiny_imagenet_xla4_smoke}"
  extra_args=(
    --train-arg=--epochs --train-arg=1
    --train-arg=--max-train-steps --train-arg=20
    --train-arg=--max-val-steps --train-arg=5
    --train-arg=--checkpoint-dir --train-arg="$smoke_checkpoint_dir"
    --train-arg=--saliency-source --train-arg=gradient
  )
  extra_args+=("${user_extra_args[@]}")
fi

preflight_extra_args=("${extra_args[@]}")
if [[ "$strict_preflight" == true ]]; then
  required_venv_name="${VENV_NAME_REQUIRED-.venvxla}"
  preflight_extra_args=(
    --check-env
    --require-tpu-env
  )
  if [[ -n "$required_venv_name" ]]; then
    preflight_extra_args+=(--require-venv-name "$required_venv_name")
  fi
  preflight_extra_args+=(
    --min-free-gb "${MIN_FREE_GB:-10}"
    --require-existing-storage-roots
  )
  preflight_extra_args+=("${extra_args[@]}")
fi

summary_dir="${SUMMARY_DIR:-outputs/tiny_imagenet_xla4_summary}"
mkdir -p "$summary_dir"

python - "$summary_dir/manifest.json" "$smoke" "$strict_preflight" "${extra_args[@]}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from allthemix.cli.summarize import TINY_IMAGENET_XLA4_PROTOCOL, TINY_IMAGENET_XLA4_PROTOCOL_ID


def git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


manifest_path = Path(sys.argv[1])
smoke = sys.argv[2].lower() == "true"
strict_preflight = sys.argv[3].lower() == "true"
extra_args = sys.argv[4:]
artifacts = [
    "manifest.json",
    "protocol.txt",
    "preflight.md",
    "commands.sh",
    "status.md" if smoke else "default.csv",
]
if not smoke:
    artifacts.extend(
        [
            "default.json",
            "default_latex_table.tex",
            "last10_median.csv",
            "last10_median.json",
            "last10_median_latex_table.tex",
        ]
    )

manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "preset": "tiny-imagenet-xla4",
    "protocol_id": TINY_IMAGENET_XLA4_PROTOCOL_ID,
    "protocol": TINY_IMAGENET_XLA4_PROTOCOL,
    "protocol_note": (
        "AllTheMiXLA table protocol: 200 epochs, global batch 128, step LR 150/180, "
        "Tiny-OpenMixup spatial augmentation, and final test on the best validation "
        "checkpoint. OpenMixup's public Tiny-ImageNet benchmark is a separate "
        "400-epoch, global-batch-100, cosine-LR protocol."
    ),
    "summary_dir": str(manifest_path.parent),
    "smoke": smoke,
    "strict_preflight": strict_preflight,
    "extra_command_args": extra_args,
    "metric_views": ["default"] if smoke else ["default", "last10_median"],
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_status_short": (git_output("status", "--short") or "").splitlines(),
    "artifacts": artifacts,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format protocol \
  --require-complete \
  "${extra_args[@]}" | tee "$summary_dir/protocol.txt"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format preflight \
  --require-complete \
  "${preflight_extra_args[@]}" | tee "$summary_dir/preflight.md"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format commands \
  "${extra_args[@]}" | tee "$summary_dir/commands.sh" | bash -e

if [[ "$smoke" == true ]]; then
  python -m allthemix.cli.summarize \
    --preset tiny-imagenet-xla4 \
    --format status \
    "${extra_args[@]}" | tee "$summary_dir/status.md"
  exit 0
fi

printf '\n# Default Tiny-ImageNet summary (best-validation-checkpoint final_test)\n'
python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format csv \
  --require-complete \
  "${extra_args[@]}" | tee "$summary_dir/default.csv"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format json \
  --require-complete \
  "${extra_args[@]}" | tee "$summary_dir/default.json"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format latex-table \
  --require-complete \
  "${extra_args[@]}" | tee "$summary_dir/default_latex_table.tex"

printf '\n# OpenMixup-style Tiny-ImageNet summary (median validation top-1 over last 10 epochs)\n'
python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --metric last10_median \
  --format csv \
  --require-complete \
  "${extra_args[@]}" | tee "$summary_dir/last10_median.csv"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --metric last10_median \
  --format json \
  --require-complete \
  "${extra_args[@]}" | tee "$summary_dir/last10_median.json"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --metric last10_median \
  --format latex-table \
  --require-complete \
  "${extra_args[@]}" | tee "$summary_dir/last10_median_latex_table.tex"
