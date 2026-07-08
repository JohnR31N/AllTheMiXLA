# Shared TPU Python environment guard for AllTheMiXLA shell entry points.
#
# This makes a missing .venvxla activation fail clearly instead of importing
# stale torch/torch_xla wheels from ~/.local and crashing with ABI errors.
export PYTHONNOUSERSITE=1

required_venv_name="${VENV_NAME_REQUIRED-.venvxla}"
if [[ -n "$required_venv_name" ]]; then
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "AllTheMiXLA TPU scripts require an active $required_venv_name virtualenv." >&2
    echo "Run: source $required_venv_name/bin/activate" >&2
    exit 1
  fi

  active_venv_name="$(basename "$VIRTUAL_ENV")"
  if [[ "$active_venv_name" != "$required_venv_name" ]]; then
    echo "AllTheMiXLA TPU scripts require virtualenv $required_venv_name, got $active_venv_name." >&2
    echo "Run: source $required_venv_name/bin/activate" >&2
    exit 1
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "AllTheMiXLA TPU scripts require python from $required_venv_name, but python is not on PATH." >&2
    echo "Run: source $required_venv_name/bin/activate" >&2
    exit 1
  fi

  if ! venv_real="$(cd "$VIRTUAL_ENV" && pwd -P)"; then
    echo "AllTheMiXLA TPU scripts cannot resolve VIRTUAL_ENV=$VIRTUAL_ENV." >&2
    echo "Run: source $required_venv_name/bin/activate" >&2
    exit 1
  fi
  if ! python_prefix="$(python - <<'PY'
import sys
print(sys.prefix)
PY
)"; then
    echo "AllTheMiXLA TPU scripts could not inspect the active python prefix." >&2
    echo "Run: source $required_venv_name/bin/activate" >&2
    exit 1
  fi
  if ! python_prefix_real="$(cd "$python_prefix" && pwd -P)"; then
    echo "AllTheMiXLA TPU scripts cannot resolve active python prefix: $python_prefix." >&2
    echo "Run: source $required_venv_name/bin/activate" >&2
    exit 1
  fi
  if [[ "$python_prefix_real" != "$venv_real" ]]; then
    echo "AllTheMiXLA TPU scripts require python from $required_venv_name." >&2
    echo "VIRTUAL_ENV resolves to: $venv_real" >&2
    echo "python resolves to prefix: $python_prefix_real ($(command -v python))" >&2
    echo "Run: source $required_venv_name/bin/activate" >&2
    exit 1
  fi
fi
