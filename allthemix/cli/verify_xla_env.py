"""Verify the local PyTorch/XLA environment before TPU runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import site
import sys
from typing import Sequence


DEFAULT_TORCH_VERSION = "2.9.0"
DEFAULT_TORCHVISION_VERSION = "0.24.0"
DEFAULT_TORCH_XLA_VERSION = "2.9.0"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _base_version(version: object) -> str:
    return str(version).split("+", 1)[0]


def _version_matches(actual: object, expected: str | None) -> bool:
    if expected in (None, ""):
        return True
    return _base_version(actual) == str(expected)


def _path_is_under(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def check_python_version(required: str = "3.10", version_info: object | None = None) -> CheckResult:
    info = version_info or sys.version_info
    actual = f"{int(info.major)}.{int(info.minor)}"
    return CheckResult(
        "python",
        actual == str(required),
        f"expected={required}; actual={actual}; executable={sys.executable}",
    )


def check_virtualenv(
    required_name: str | None = None,
    prefix: str | None = None,
    base_prefix: str | None = None,
) -> CheckResult:
    prefix_value = prefix or sys.prefix
    base_prefix_value = base_prefix or getattr(sys, "base_prefix", sys.prefix)
    required = str(required_name or "").strip()
    in_venv = Path(prefix_value).resolve() != Path(base_prefix_value).resolve()
    actual_name = Path(prefix_value).name
    ok = in_venv and (not required or actual_name == required)
    expected = required if required else "any virtualenv"
    return CheckResult(
        "venv",
        ok,
        f"expected={expected}; actual={actual_name}; prefix={prefix_value}; base_prefix={base_prefix_value}",
    )


def _user_site_on_path(user_site: str | None, path_entries: Sequence[str]) -> bool:
    if not user_site:
        return False
    for entry in path_entries:
        if not entry:
            continue
        if _path_is_under(entry, user_site) or _path_is_under(user_site, entry):
            return True
    return False


def check_user_site_disabled(
    env: dict[str, str] | None = None,
    enable_user_site: object | None = None,
    user_site: str | None = None,
    path_entries: Sequence[str] | None = None,
) -> CheckResult:
    env_values = env if env is not None else os.environ
    env_value = env_values.get("PYTHONNOUSERSITE", "")
    enable_value = site.ENABLE_USER_SITE if enable_user_site is None else enable_user_site
    user_site_value = user_site if user_site is not None else site.getusersitepackages()
    path_values = list(sys.path if path_entries is None else path_entries)
    user_site_on_path = _user_site_on_path(user_site_value, path_values)
    ok = enable_value is not True and not user_site_on_path
    detail = (
        f"PYTHONNOUSERSITE={env_value or '<unset>'}; "
        f"ENABLE_USER_SITE={enable_value}; user_site={user_site_value}; "
        f"user_site_on_path={'yes' if user_site_on_path else 'no'}"
    )
    if not ok:
        detail += "; launch Python with PYTHONNOUSERSITE=1 and no user-site path entries"
    return CheckResult("user_site", ok, detail)


def check_module_version(
    module_name: str,
    expected: str | None = None,
    label: str | None = None,
    required_prefix: str | None = None,
) -> CheckResult:
    name = label or module_name
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(name, False, f"import failed: {exc.__class__.__name__}: {exc}")

    actual = getattr(module, "__version__", "unknown")
    module_file = getattr(module, "__file__", "<built-in>")
    ok = _version_matches(actual, expected)
    expected_detail = "any" if expected in (None, "") else str(expected)
    detail = f"expected={expected_detail}; actual={actual}; file={module_file}"
    if ok and required_prefix:
        ok = isinstance(module_file, str) and _path_is_under(module_file, required_prefix)
        detail += f"; required_prefix={required_prefix}"
        if not ok:
            detail += "; module is outside the active virtualenv"
    return CheckResult(name, ok, detail)


def check_torch_xla_version_alignment() -> CheckResult:
    try:
        torch = importlib.import_module("torch")
        torch_xla = importlib.import_module("torch_xla")
    except Exception as exc:
        return CheckResult("torch_xla_alignment", False, f"import failed: {exc.__class__.__name__}: {exc}")

    torch_version = getattr(torch, "__version__", "unknown")
    torch_xla_version = getattr(torch_xla, "__version__", "unknown")
    torch_base = _base_version(torch_version)
    torch_xla_base = _base_version(torch_xla_version)
    ok = torch_base == torch_xla_base
    detail = f"torch={torch_version}; torch_xla={torch_xla_version}; base_match={'yes' if ok else 'no'}"
    if not ok:
        detail += "; reinstall matching torch and torch_xla wheels in the active virtualenv"
    return CheckResult("torch_xla_alignment", ok, detail)


def check_opencv_saliency() -> CheckResult:
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return CheckResult(
            "opencv_saliency",
            False,
            f"import failed: {exc.__class__.__name__}: {exc}; install opencv-contrib-python-headless",
        )

    version = getattr(cv2, "__version__", "unknown")
    has_saliency = hasattr(cv2, "saliency") and hasattr(cv2.saliency, "StaticSaliencyFineGrained_create")
    detail = f"cv2={version}; StaticSaliencyFineGrained_create={'yes' if has_saliency else 'no'}"
    if not has_saliency:
        detail += "; install opencv-contrib-python-headless for SaliencyMix table caches"
    return CheckResult("opencv_saliency", bool(has_saliency), detail)


def check_tpu_devices(require_tpu: bool = False, expected_count: int | None = None) -> CheckResult:
    expected_devices = None if expected_count in (None, "") else int(expected_count)
    if expected_devices is not None and expected_devices < 1:
        return CheckResult(
            "tpu_devices",
            False,
            f"expected_count must be >= 1 when provided; got {expected_devices}",
        )
    device_required = bool(require_tpu) or expected_devices is not None
    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
        devices = xm.get_xla_supported_devices()
    except Exception as exc:
        return CheckResult(
            "tpu_devices",
            not device_required,
            f"device check failed: {exc.__class__.__name__}: {exc}",
        )

    device_type = "unknown"
    try:
        xr = importlib.import_module("torch_xla.runtime")
        if hasattr(xr, "device_type"):
            device_type = str(xr.device_type())
    except Exception as exc:
        if require_tpu:
            return CheckResult(
                "tpu_devices",
                False,
                f"runtime device type check failed: {exc.__class__.__name__}: {exc}; devices={devices}",
            )

    visible_count = len(devices)
    ok = (not device_required) or (bool(devices) and device_type.upper() == "TPU")
    if expected_devices is not None:
        ok = ok and visible_count == expected_devices
    detail = f"device_type={device_type}; visible_count={visible_count}; devices={devices}"
    if expected_devices is not None:
        detail += f"; expected_count={expected_devices}"
        if visible_count != expected_devices:
            detail += "; visible TPU device count does not match the requested launch size"
    return CheckResult("tpu_devices", ok, detail)


def build_checks(args: argparse.Namespace) -> list[CheckResult]:
    checks = [
        check_python_version(args.python_version),
    ]
    require_venv_name = str(getattr(args, "require_venv_name", "") or "")
    if require_venv_name:
        checks.append(check_virtualenv(require_venv_name))
        checks.append(check_user_site_disabled())
    required_module_prefix = sys.prefix if require_venv_name else None
    torch_check = check_module_version("torch", args.torch_version, required_prefix=required_module_prefix)
    torchvision_check = check_module_version(
        "torchvision",
        args.torchvision_version,
        required_prefix=required_module_prefix,
    )
    torch_xla_check = check_module_version("torch_xla", args.torch_xla_version, required_prefix=required_module_prefix)
    checks.extend([torch_check, torchvision_check, torch_xla_check])
    if torch_check.ok and torch_xla_check.ok:
        checks.append(check_torch_xla_version_alignment())
        checks.append(
            check_module_version(
                "_XLAC",
                None,
                label="_XLAC",
                required_prefix=required_module_prefix,
            )
        )
    if not bool(args.skip_opencv_check):
        checks.append(check_opencv_saliency())
    if not bool(args.skip_device_check):
        checks.append(
            check_tpu_devices(
                require_tpu=bool(args.require_tpu),
                expected_count=getattr(args, "expected_tpu_devices", None),
            )
        )
    return checks


def render_checks(checks: Sequence[CheckResult]) -> str:
    lines = []
    for check in checks:
        status = "ok" if check.ok else "fail"
        lines.append(f"{check.name}: {status}: {check.detail}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PyTorch/XLA versions and optional TPU visibility.")
    parser.add_argument("--python-version", default=os.environ.get("PYTHON_VERSION_REQUIRED", "3.10"))
    parser.add_argument("--torch-version", default=os.environ.get("TORCH_VERSION", DEFAULT_TORCH_VERSION))
    parser.add_argument("--torchvision-version", default=os.environ.get("TORCHVISION_VERSION", DEFAULT_TORCHVISION_VERSION))
    parser.add_argument("--torch-xla-version", default=os.environ.get("TORCH_XLA_VERSION", DEFAULT_TORCH_XLA_VERSION))
    parser.add_argument(
        "--require-venv-name",
        default=os.environ.get("VENV_NAME_REQUIRED", ""),
        help="Fail unless Python is running inside a virtualenv with this directory name, e.g. .venvxla.",
    )
    parser.add_argument("--skip-device-check", action="store_true", help="Skip TPU device visibility checks.")
    parser.add_argument(
        "--skip-opencv-check",
        action="store_true",
        help="Skip the OpenCV contrib saliency backend check. Only use for non-SaliencyMix debug environments.",
    )
    parser.add_argument("--require-tpu", action="store_true", help="Fail if no TPU devices are visible.")
    parser.add_argument(
        "--expected-tpu-devices",
        type=int,
        default=None,
        help="Fail unless this many TPU devices are visible; useful for xla4 table runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checks = build_checks(args)
    print(render_checks(checks))
    if not all(check.ok for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
