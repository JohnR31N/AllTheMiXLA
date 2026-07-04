#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venvxla}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
RECREATE_VENV="${RECREATE_VENV:-0}"

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3.10 python3.10-venv libopenblas-dev
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
  rm -rf "$VENV_DIR"
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

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pyyaml pillow tqdm
python -m pip install torch torchvision 'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html

python - <<'PY'
import torch
import torchvision
import torch_xla

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torch_xla={torch_xla.__version__}")
PY

cat <<EOF

Created XLA virtual environment: $VENV_DIR

Activate it with:
  source $VENV_DIR/bin/activate
  export PJRT_DEVICE=TPU

Verify TPU visibility with:
  PJRT_DEVICE=TPU python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices('TPU'))"

Example smoke run:
  PJRT_DEVICE=TPU python -m allthemix.cli.train --config configs/cifar10/preact_resnet18/mixup.yaml --download --device xla --num-cores 8 --num-workers 0 --epochs 1 --batch-size 32 --max-train-steps 20 --max-val-steps 5
EOF
