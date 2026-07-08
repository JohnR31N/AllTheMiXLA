#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


VENV_DIR="${VENV_DIR:-.venvxla}"
VENV_NAME="$(basename "$VENV_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
RECREATE_VENV="${RECREATE_VENV:-0}"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
PIP_TMPDIR="${PIP_TMPDIR:-/dev/shm/allthemixla-pip-tmp}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-$TORCH_VERSION}"
EXPECTED_TPU_DEVICES="${EXPECTED_TPU_DEVICES:-4}"

export PIP_NO_CACHE_DIR
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
if [[ -d /dev/shm ]]; then
  mkdir -p "$PIP_TMPDIR"
  export TMPDIR="$PIP_TMPDIR"
fi

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3.10 python3.10-venv ca-certificates curl unzip libopenblas-dev
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
  echo "Expected Python 3.10, got Python $PYTHON_VERSION from $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$RECREATE_VENV" == "1" && -d "$VENV_DIR" ]]; then
  VENV_REAL="$(cd "$VENV_DIR" && pwd -P)"
  REPO_REAL="$(pwd -P)"
  if [[ "$VENV_REAL" == "$REPO_REAL" || "$VENV_REAL" != "$REPO_REAL"/* ]]; then
    echo "Refusing to recreate VENV_DIR outside this repo: $VENV_REAL" >&2
    echo "Use a repo-local VENV_DIR such as .venvxla." >&2
    exit 1
  fi
  rm -rf "$VENV_REAL"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

VENV_PYTHON_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$VENV_PYTHON_VERSION" != "3.10" ]]; then
  echo "$VENV_DIR uses Python $VENV_PYTHON_VERSION; expected Python 3.10." >&2
  echo "Recreate it with: RECREATE_VENV=1 bash scripts/setup_tpu_venvxla.sh" >&2
  exit 1
fi

python -m pip cache purge >/dev/null 2>&1 || true
python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir numpy opencv-contrib-python-headless pyyaml pillow
python -m pip uninstall -y torch torchvision torch_xla libtpu >/dev/null 2>&1 || true
python -m pip install --no-cache-dir \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torch_xla[tpu]==$TORCH_XLA_VERSION" \
  -f https://storage.googleapis.com/libtpu-releases/index.html
python -m pip check

python -m allthemix.cli.verify_xla_env --skip-device-check --require-venv-name "$VENV_NAME"

cat <<EOF

Created XLA virtual environment: $VENV_DIR

Activate it with:
  source $VENV_DIR/bin/activate
  export PJRT_DEVICE=TPU
  export PYTHONNOUSERSITE=1

Verify TPU visibility with:
  PJRT_DEVICE=TPU python -m allthemix.cli.verify_xla_env --require-tpu --expected-tpu-devices "$EXPECTED_TPU_DEVICES" --require-venv-name "$VENV_NAME"

After activating the environment, run a smoke check with:
  PJRT_DEVICE=TPU bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh --device xla --num-cores 4 --num-workers 0 --epochs 1 --max-train-steps 20 --max-val-steps 5 --log-interval 0
EOF
