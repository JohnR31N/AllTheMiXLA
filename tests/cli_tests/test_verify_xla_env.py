import argparse
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from allthemix.cli.verify_xla_env import (
    CheckResult,
    build_checks,
    check_module_version,
    check_opencv_saliency,
    check_python_version,
    check_torch_xla_version_alignment,
    check_tpu_devices,
    check_user_site_disabled,
    check_virtualenv,
    render_checks,
)


class VerifyXlaEnvTests(unittest.TestCase):
    def test_python_version_check_requires_310(self):
        ok = check_python_version("3.10", SimpleNamespace(major=3, minor=10))
        bad = check_python_version("3.10", SimpleNamespace(major=3, minor=11))

        self.assertTrue(ok.ok)
        self.assertFalse(bad.ok)
        self.assertIn("actual=3.11", bad.detail)

    def test_module_version_accepts_local_build_suffix(self):
        torch = ModuleType("torch")
        torch.__version__ = "2.9.0+cu128"
        torch.__file__ = "/repo/.venvxla/lib/python3.10/site-packages/torch/__init__.py"

        with patch.dict(sys.modules, {"torch": torch}):
            result = check_module_version("torch", "2.9.0")

        self.assertTrue(result.ok)
        self.assertIn("actual=2.9.0+cu128", result.detail)
        self.assertIn("/repo/.venvxla", result.detail)

    def test_module_version_can_require_active_venv_prefix(self):
        torch = ModuleType("torch")
        torch.__version__ = "2.9.0+cu128"
        torch.__file__ = "/home/user/AllTheMiXLA/.venvxla/lib/python3.10/site-packages/torch/__init__.py"

        with patch.dict(sys.modules, {"torch": torch}):
            result = check_module_version(
                "torch",
                "2.9.0",
                required_prefix="/home/user/AllTheMiXLA/.venvxla",
            )

        self.assertTrue(result.ok)
        self.assertIn("required_prefix=/home/user/AllTheMiXLA/.venvxla", result.detail)

    def test_module_version_rejects_user_site_package_when_venv_prefix_required(self):
        torch_xla = ModuleType("torch_xla")
        torch_xla.__version__ = "2.9.0"
        torch_xla.__file__ = "/home/user/.local/lib/python3.10/site-packages/torch_xla/__init__.py"

        with patch.dict(sys.modules, {"torch_xla": torch_xla}):
            result = check_module_version(
                "torch_xla",
                "2.9.0",
                required_prefix="/home/user/AllTheMiXLA/.venvxla",
            )

        self.assertFalse(result.ok)
        self.assertIn("module is outside the active virtualenv", result.detail)

    def test_module_version_reports_import_failure(self):
        def broken_import(name):
            if name == "torch_xla":
                raise ImportError("undefined symbol: _XLAC")
            return __import__(name)

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", side_effect=broken_import):
            result = check_module_version("torch_xla", "2.9.0")

        self.assertFalse(result.ok)
        self.assertIn("undefined symbol", result.detail)

    def test_virtualenv_check_requires_named_venv(self):
        ok = check_virtualenv(
            ".venvxla",
            prefix="/home/user/AllTheMiXLA/.venvxla",
            base_prefix="/usr",
        )
        wrong_name = check_virtualenv(
            ".venvxla",
            prefix="/home/user/AllTheMiXLA/.venv",
            base_prefix="/usr",
        )
        system_python = check_virtualenv(
            ".venvxla",
            prefix="/usr",
            base_prefix="/usr",
        )

        self.assertTrue(ok.ok)
        self.assertFalse(wrong_name.ok)
        self.assertFalse(system_python.ok)
        self.assertIn("expected=.venvxla", wrong_name.detail)
        self.assertIn("actual=.venv", wrong_name.detail)

    def test_user_site_check_accepts_guarded_interpreter_user_site(self):
        result = check_user_site_disabled(
            env={"PYTHONNOUSERSITE": "1"},
            enable_user_site=False,
            user_site="/home/user/.local/lib/python3.10/site-packages",
            path_entries=["/home/user/AllTheMiXLA/.venvxla/lib/python3.10/site-packages"],
        )

        self.assertTrue(result.ok)
        self.assertIn("PYTHONNOUSERSITE=1", result.detail)

    def test_user_site_check_accepts_disabled_interpreter_user_site(self):
        result = check_user_site_disabled(
            env={},
            enable_user_site=False,
            user_site="/home/user/.local/lib/python3.10/site-packages",
            path_entries=[],
        )

        self.assertTrue(result.ok)
        self.assertIn("ENABLE_USER_SITE=False", result.detail)

    def test_user_site_check_rejects_active_user_site_without_guard_env(self):
        result = check_user_site_disabled(
            env={},
            enable_user_site=True,
            user_site="/home/user/.local/lib/python3.10/site-packages",
            path_entries=["/home/user/.local/lib/python3.10/site-packages"],
        )

        self.assertFalse(result.ok)
        self.assertIn("user_site_on_path=yes", result.detail)
        self.assertIn("launch Python with PYTHONNOUSERSITE=1", result.detail)

    def test_user_site_check_rejects_active_user_site_even_when_env_is_set(self):
        result = check_user_site_disabled(
            env={"PYTHONNOUSERSITE": "1"},
            enable_user_site=True,
            user_site="/home/user/.local/lib/python3.10/site-packages",
            path_entries=["/home/user/.local/lib/python3.10/site-packages"],
        )

        self.assertFalse(result.ok)
        self.assertIn("PYTHONNOUSERSITE=1", result.detail)
        self.assertIn("ENABLE_USER_SITE=True", result.detail)
        self.assertIn("user_site_on_path=yes", result.detail)

    def test_torch_xla_alignment_accepts_matching_base_versions(self):
        torch = ModuleType("torch")
        torch.__version__ = "2.9.0+cu128"
        torch_xla = ModuleType("torch_xla")
        torch_xla.__version__ = "2.9.0"

        with patch.dict(sys.modules, {"torch": torch, "torch_xla": torch_xla}):
            result = check_torch_xla_version_alignment()

        self.assertTrue(result.ok)
        self.assertIn("base_match=yes", result.detail)

    def test_torch_xla_alignment_rejects_mismatched_base_versions(self):
        torch = ModuleType("torch")
        torch.__version__ = "2.9.0+cu128"
        torch_xla = ModuleType("torch_xla")
        torch_xla.__version__ = "2.8.0"

        with patch.dict(sys.modules, {"torch": torch, "torch_xla": torch_xla}):
            result = check_torch_xla_version_alignment()

        self.assertFalse(result.ok)
        self.assertIn("base_match=no", result.detail)
        self.assertIn("reinstall matching torch and torch_xla", result.detail)

    def test_tpu_device_check_can_be_optional_or_required(self):
        def broken_import(name):
            if name == "torch_xla.core.xla_model":
                raise RuntimeError("TPU initialization failed")
            return __import__(name)

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", side_effect=broken_import):
            optional = check_tpu_devices(require_tpu=False)
            required = check_tpu_devices(require_tpu=True)

        self.assertTrue(optional.ok)
        self.assertFalse(required.ok)
        self.assertIn("TPU initialization failed", required.detail)

    def test_tpu_device_check_reports_visible_devices(self):
        xm = ModuleType("torch_xla.core.xla_model")
        xm.get_xla_supported_devices = lambda: ["xla:0", "xla:1", "xla:2", "xla:3"]
        xr = ModuleType("torch_xla.runtime")
        xr.device_type = lambda: "TPU"

        def fake_import(name):
            if name == "torch_xla.core.xla_model":
                return xm
            if name == "torch_xla.runtime":
                return xr
            return __import__(name)

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", side_effect=fake_import):
            result = check_tpu_devices(require_tpu=True)

        self.assertTrue(result.ok)
        self.assertIn("device_type=TPU", result.detail)
        self.assertIn("visible_count=4", result.detail)
        self.assertIn("xla:3", result.detail)

    def test_tpu_device_check_accepts_expected_visible_device_count(self):
        xm = ModuleType("torch_xla.core.xla_model")
        xm.get_xla_supported_devices = lambda: ["xla:0", "xla:1", "xla:2", "xla:3"]
        xr = ModuleType("torch_xla.runtime")
        xr.device_type = lambda: "TPU"

        def fake_import(name):
            if name == "torch_xla.core.xla_model":
                return xm
            if name == "torch_xla.runtime":
                return xr
            return __import__(name)

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", side_effect=fake_import):
            result = check_tpu_devices(require_tpu=True, expected_count=4)

        self.assertTrue(result.ok)
        self.assertIn("expected_count=4", result.detail)

    def test_tpu_device_check_rejects_expected_visible_device_count_mismatch(self):
        xm = ModuleType("torch_xla.core.xla_model")
        xm.get_xla_supported_devices = lambda: ["xla:0", "xla:1"]
        xr = ModuleType("torch_xla.runtime")
        xr.device_type = lambda: "TPU"

        def fake_import(name):
            if name == "torch_xla.core.xla_model":
                return xm
            if name == "torch_xla.runtime":
                return xr
            return __import__(name)

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", side_effect=fake_import):
            result = check_tpu_devices(require_tpu=True, expected_count=4)

        self.assertFalse(result.ok)
        self.assertIn("visible_count=2", result.detail)
        self.assertIn("expected_count=4", result.detail)
        self.assertIn("does not match", result.detail)

    def test_tpu_device_check_rejects_cpu_xla_runtime_when_tpu_required(self):
        xm = ModuleType("torch_xla.core.xla_model")
        xm.get_xla_supported_devices = lambda: ["xla:0"]
        xr = ModuleType("torch_xla.runtime")
        xr.device_type = lambda: "CPU"

        def fake_import(name):
            if name == "torch_xla.core.xla_model":
                return xm
            if name == "torch_xla.runtime":
                return xr
            return __import__(name)

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", side_effect=fake_import):
            result = check_tpu_devices(require_tpu=True)

        self.assertFalse(result.ok)
        self.assertIn("device_type=CPU", result.detail)

    def test_opencv_saliency_check_requires_contrib_backend(self):
        cv2 = ModuleType("cv2")
        cv2.__version__ = "4.10.0"
        cv2.saliency = SimpleNamespace(StaticSaliencyFineGrained_create=lambda: object())

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", return_value=cv2):
            result = check_opencv_saliency()

        self.assertTrue(result.ok)
        self.assertIn("StaticSaliencyFineGrained_create=yes", result.detail)

    def test_opencv_saliency_check_rejects_plain_opencv(self):
        cv2 = ModuleType("cv2")
        cv2.__version__ = "4.10.0"

        with patch("allthemix.cli.verify_xla_env.importlib.import_module", return_value=cv2):
            result = check_opencv_saliency()

        self.assertFalse(result.ok)
        self.assertIn("opencv-contrib-python-headless", result.detail)

    def test_build_checks_includes_opencv_saliency_by_default(self):
        args = argparse.Namespace(
            python_version="3.10",
            torch_version="",
            torchvision_version="",
            torch_xla_version="",
            require_venv_name="",
            skip_device_check=True,
            skip_opencv_check=False,
            require_tpu=False,
        )

        with (
            patch("allthemix.cli.verify_xla_env.check_python_version", return_value=CheckResult("python", True, "ok")),
            patch("allthemix.cli.verify_xla_env.check_module_version", return_value=CheckResult("module", True, "ok")),
            patch("allthemix.cli.verify_xla_env.check_torch_xla_version_alignment", return_value=CheckResult("torch_xla_alignment", True, "ok")),
            patch("allthemix.cli.verify_xla_env.check_opencv_saliency", return_value=CheckResult("opencv_saliency", True, "ok")),
        ):
            checks = build_checks(args)

        self.assertIn("opencv_saliency", [check.name for check in checks])

    def test_build_checks_can_skip_opencv_saliency_for_debug_envs(self):
        args = argparse.Namespace(
            python_version="3.10",
            torch_version="",
            torchvision_version="",
            torch_xla_version="",
            require_venv_name="",
            skip_device_check=True,
            skip_opencv_check=True,
            require_tpu=False,
        )

        with (
            patch("allthemix.cli.verify_xla_env.check_python_version", return_value=CheckResult("python", True, "ok")),
            patch("allthemix.cli.verify_xla_env.check_module_version", return_value=CheckResult("module", True, "ok")),
            patch("allthemix.cli.verify_xla_env.check_torch_xla_version_alignment", return_value=CheckResult("torch_xla_alignment", True, "ok")),
            patch("allthemix.cli.verify_xla_env.check_opencv_saliency") as opencv_check,
        ):
            checks = build_checks(args)

        opencv_check.assert_not_called()
        self.assertNotIn("opencv_saliency", [check.name for check in checks])

    def test_build_checks_includes_named_venv_when_required(self):
        args = argparse.Namespace(
            python_version="3.10",
            torch_version="",
            torchvision_version="",
            torch_xla_version="",
            require_venv_name=".venvxla",
            skip_device_check=True,
            skip_opencv_check=True,
            require_tpu=False,
        )

        with (
            patch("allthemix.cli.verify_xla_env.check_python_version", return_value=CheckResult("python", True, "ok")),
            patch("allthemix.cli.verify_xla_env.check_virtualenv", return_value=CheckResult("venv", True, "ok")) as venv_check,
            patch("allthemix.cli.verify_xla_env.check_user_site_disabled", return_value=CheckResult("user_site", True, "ok")) as user_site_check,
            patch("allthemix.cli.verify_xla_env.check_module_version", return_value=CheckResult("module", True, "ok")) as module_check,
            patch("allthemix.cli.verify_xla_env.check_torch_xla_version_alignment", return_value=CheckResult("torch_xla_alignment", True, "ok")) as alignment_check,
        ):
            checks = build_checks(args)

        venv_check.assert_called_once_with(".venvxla")
        user_site_check.assert_called_once_with()
        alignment_check.assert_called_once_with()
        self.assertEqual(module_check.call_count, 4)
        for call in module_check.call_args_list:
            self.assertEqual(call.kwargs["required_prefix"], sys.prefix)
        self.assertIn("venv", [check.name for check in checks])
        self.assertIn("user_site", [check.name for check in checks])
        self.assertIn("torch_xla_alignment", [check.name for check in checks])
        self.assertIn("module", [check.name for check in checks])

    def test_render_checks_marks_status(self):
        rendered = render_checks(
            [
                CheckResult("torch", True, "expected=2.9.0; actual=2.9.0"),
                CheckResult("torch_xla", False, "import failed"),
            ]
        )

        self.assertIn("torch: ok:", rendered)
        self.assertIn("torch_xla: fail:", rendered)


if __name__ == "__main__":
    unittest.main()
