#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/experiment_run/prepare_tiny_imagenet_xla4.sh [download options] [-- preflight args]

Download options:
  --data-dir DIR   Directory that will contain tiny-imagenet-200. Default: ./data
  --url URL        Dataset zip URL. Default: official CS231n Tiny-ImageNet archive.
  --md5 MD5        Expected archive MD5. Default: official Tiny-ImageNet MD5.
  --skip-md5      Skip archive MD5 validation.
  --force         Re-extract even when tiny-imagenet-200 already exists.
  -h, --help      Show this help.

Everything after -- is forwarded to preflight_tiny_imagenet_xla4.sh.
EOF
}

data_dir="${DATA_DIR:-./data}"
bash_cmd="${BASH_CMD:-bash}"
download_args=()
preflight_args=()
data_dir_forwarded=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --data-dir)
      if [[ "$#" -lt 2 ]]; then
        echo "--data-dir requires a value" >&2
        exit 1
      fi
      data_dir="$2"
      download_args+=("--data-dir" "$2")
      preflight_args+=("--train-arg=--data-dir" "--train-arg=$2")
      data_dir_forwarded=true
      shift 2
      ;;
    --url|--md5)
      if [[ "$#" -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 1
      fi
      download_args+=("$1" "$2")
      shift 2
      ;;
    --skip-md5|--force)
      download_args+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ "$data_dir_forwarded" != true ]]; then
        download_args+=("--data-dir" "$data_dir")
        preflight_args+=("--train-arg=--data-dir" "--train-arg=$data_dir")
        data_dir_forwarded=true
      fi
      preflight_args+=("$@")
      break
      ;;
    *)
      echo "Unknown prepare/download argument before --: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$data_dir_forwarded" != true ]]; then
  download_args+=("--data-dir" "$data_dir")
  preflight_args+=("--train-arg=--data-dir" "--train-arg=$data_dir")
fi

"$bash_cmd" scripts/download_tiny_imagenet.sh "${download_args[@]}"
"$bash_cmd" scripts/experiment_run/preflight_tiny_imagenet_xla4.sh "${preflight_args[@]}"
