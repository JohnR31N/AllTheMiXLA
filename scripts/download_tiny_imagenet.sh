#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


usage() {
  cat <<'EOF'
Usage: bash scripts/download_tiny_imagenet.sh [options]

Options:
  --data-dir DIR   Directory that will contain tiny-imagenet-200. Default: ./data
  --url URL        Dataset zip URL. Default: https://cs231n.stanford.edu/tiny-imagenet-200.zip
  --md5 MD5        Expected archive MD5. Default: official Tiny-ImageNet MD5.
  --skip-md5      Skip archive MD5 validation.
  --force         Re-extract even when tiny-imagenet-200 already exists.
  -h, --help      Show this help.
EOF
}

data_dir="${DATA_DIR:-./data}"
tiny_url="${TINY_URL:-https://cs231n.stanford.edu/tiny-imagenet-200.zip}"
archive_name="${ARCHIVE_NAME:-tiny-imagenet-200.zip}"
expected_md5="${TINY_MD5:-90528d7ca1a48142e341f4ef8d21d0de}"
expected_train_images="${TINY_EXPECTED_TRAIN_IMAGES:-100000}"
expected_val_images="${TINY_EXPECTED_VAL_IMAGES:-10000}"
force=false
skip_md5=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --data-dir)
      if [[ "$#" -lt 2 ]]; then
        echo "--data-dir requires a value" >&2
        exit 1
      fi
      data_dir="$2"
      shift 2
      ;;
    --url)
      if [[ "$#" -lt 2 ]]; then
        echo "--url requires a value" >&2
        exit 1
      fi
      tiny_url="$2"
      shift 2
      ;;
    --md5)
      if [[ "$#" -lt 2 ]]; then
        echo "--md5 requires a value" >&2
        exit 1
      fi
      expected_md5="$2"
      shift 2
      ;;
    --skip-md5)
      skip_md5=true
      shift
      ;;
    --force)
      force=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$data_dir"
archive_path="$data_dir/$archive_name"
temp_archive_path="$data_dir/.$archive_name.tmp"
tiny_root="$data_dir/tiny-imagenet-200"

count_images() {
  find "$1" -type f \( \
    -iname '*.jpg' -o \
    -iname '*.jpeg' -o \
    -iname '*.png' -o \
    -iname '*.bmp' -o \
    -iname '*.ppm' \
  \) | wc -l | tr -d '[:space:]'
}

tiny_imagenet_ready() {
  [[ -d "$tiny_root/train" && -d "$tiny_root/val" ]] || return 1
  train_count="$(count_images "$tiny_root/train")"
  val_count="$(count_images "$tiny_root/val")"
  [[ "$train_count" == "$expected_train_images" && "$val_count" == "$expected_val_images" ]]
}

if [[ "$force" != true && -d "$tiny_root/train" && -d "$tiny_root/val" ]] && tiny_imagenet_ready; then
  echo "Tiny-ImageNet already exists at: $tiny_root"
  exit 0
fi

if [[ ! -f "$archive_path" ]]; then
  echo "Downloading Tiny-ImageNet to: $archive_path"
  rm -f "$temp_archive_path"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$tiny_url" -o "$temp_archive_path"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$temp_archive_path" "$tiny_url"
  else
    echo "Neither curl nor wget is available; install one or download $tiny_url manually." >&2
    exit 1
  fi
  mv "$temp_archive_path" "$archive_path"
else
  echo "Using existing archive: $archive_path"
fi

if [[ "$skip_md5" != true && -n "$expected_md5" ]] && command -v md5sum >/dev/null 2>&1; then
  actual_md5="$(md5sum "$archive_path" | awk '{print $1}')"
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    echo "MD5 mismatch for $archive_path" >&2
    echo "Expected: $expected_md5" >&2
    echo "Actual:   $actual_md5" >&2
    echo "Use --skip-md5 only if you intentionally use a trusted mirror/archive." >&2
    exit 1
  fi
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required to extract Tiny-ImageNet." >&2
  exit 1
fi

echo "Extracting Tiny-ImageNet under: $data_dir"
unzip -q -o "$archive_path" -d "$data_dir"

if ! tiny_imagenet_ready; then
  echo "Extraction did not produce expected Tiny-ImageNet layout at $tiny_root" >&2
  echo "Expected image counts: train=$expected_train_images val=$expected_val_images" >&2
  exit 1
fi

echo "Tiny-ImageNet is ready at: $tiny_root"
