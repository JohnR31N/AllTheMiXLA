#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Collection only reads metrics/config files and does not touch TPU devices.
# Keep user-site packages disabled to avoid stale local wheels while allowing
# table collection from any Python environment with the repo dependencies.
export PYTHONNOUSERSITE=1

require_complete=false
collect_args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --require-complete)
      require_complete=true
      shift
      ;;
    *)
      collect_args+=("$1")
      shift
      ;;
  esac
done

summary_dir="${SUMMARY_DIR:-outputs/tiny_imagenet_xla4_summary}"
mkdir -p "$summary_dir"

final_artifacts=(
  default.csv
  default.json
  default_latex_table.tex
  last10_median.csv
  last10_median.json
  last10_median_latex_table.tex
)
for artifact in "${final_artifacts[@]}"; do
  rm -f "$summary_dir/$artifact"
done

python - "$summary_dir/collect_manifest.json" "$summary_dir" "$require_complete" "${collect_args[@]}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from allthemix.cli.summarize import TINY_IMAGENET_XLA4_PROTOCOL, TINY_IMAGENET_XLA4_PROTOCOL_ID


def git_output(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


manifest_path = Path(sys.argv[1])
summary_dir = sys.argv[2]
require_complete = sys.argv[3].lower() == "true"
extra_args = sys.argv[4:]
artifacts = [
    "collect_manifest.json",
    "protocol.txt",
    "status.md",
    "default.csv",
    "default.json",
    "default_latex_table.tex",
    "last10_median.csv",
    "last10_median.json",
    "last10_median_latex_table.tex",
]
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": "collect",
    "preset": "tiny-imagenet-xla4",
    "protocol_id": TINY_IMAGENET_XLA4_PROTOCOL_ID,
    "protocol": TINY_IMAGENET_XLA4_PROTOCOL,
    "summary_dir": summary_dir,
    "require_complete": require_complete,
    "extra_command_args": extra_args,
    "metric_views": ["default", "last10_median"],
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_status_short": (git_output("status", "--short") or "").splitlines(),
    "artifacts": artifacts,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format protocol \
  "${collect_args[@]}" | tee "$summary_dir/protocol.txt"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format status \
  "${collect_args[@]}" | tee "$summary_dir/status.md"

if [[ "$require_complete" == true ]]; then
  python -m allthemix.cli.summarize \
    --preset tiny-imagenet-xla4 \
    --format csv \
    --require-complete \
    "${collect_args[@]}" >/dev/null

  python -m allthemix.cli.summarize \
    --preset tiny-imagenet-xla4 \
    --metric last10_median \
    --format csv \
    --require-complete \
    "${collect_args[@]}" >/dev/null
fi

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format csv \
  "${collect_args[@]}" | tee "$summary_dir/default.csv"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format json \
  "${collect_args[@]}" | tee "$summary_dir/default.json"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --format latex-table \
  "${collect_args[@]}" | tee "$summary_dir/default_latex_table.tex"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --metric last10_median \
  --format csv \
  "${collect_args[@]}" | tee "$summary_dir/last10_median.csv"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --metric last10_median \
  --format json \
  "${collect_args[@]}" | tee "$summary_dir/last10_median.json"

python -m allthemix.cli.summarize \
  --preset tiny-imagenet-xla4 \
  --metric last10_median \
  --format latex-table \
  "${collect_args[@]}" | tee "$summary_dir/last10_median_latex_table.tex"
