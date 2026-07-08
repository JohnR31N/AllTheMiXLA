import argparse
import os
import sys
import types
import unittest
from unittest.mock import patch

import allthemix.cli.train as train_cli
from allthemix.cli.train import (
    _optional_xla_launcher,
    _xla_rank,
    _xla_world_size,
    configure_xla_launch_environment,
    is_master,
    resolve_step_limit,
)


class XlaLauncherTests(unittest.TestCase):
    def test_configure_xla_launch_environment_sets_tpu_num_devices(self):
        args = argparse.Namespace(device="xla", num_cores=4)

        with patch.dict(os.environ, {}, clear=True):
            should_spawn = configure_xla_launch_environment(args)

            self.assertTrue(should_spawn)
            self.assertEqual(os.environ["TPU_NUM_DEVICES"], "4")

    def test_configure_xla_launch_environment_overrides_existing_tpu_num_devices(self):
        args = argparse.Namespace(device="xla", num_cores=4)

        with patch.dict(os.environ, {"TPU_NUM_DEVICES": "2"}, clear=True):
            should_spawn = configure_xla_launch_environment(args)

            self.assertTrue(should_spawn)
            self.assertEqual(os.environ["TPU_NUM_DEVICES"], "4")

    def test_configure_xla_launch_environment_sets_single_process_device_count_without_spawn(self):
        args = argparse.Namespace(device="xla", num_cores=1)

        with patch.dict(os.environ, {}, clear=True):
            should_spawn = configure_xla_launch_environment(args)

            self.assertFalse(should_spawn)
            self.assertEqual(os.environ["TPU_NUM_DEVICES"], "1")

    def test_configure_xla_launch_environment_overrides_stale_single_process_value(self):
        args = argparse.Namespace(device="xla", num_cores=1)

        with patch.dict(os.environ, {"TPU_NUM_DEVICES": "4"}, clear=True):
            should_spawn = configure_xla_launch_environment(args)

            self.assertFalse(should_spawn)
            self.assertEqual(os.environ["TPU_NUM_DEVICES"], "1")

    def test_configure_xla_launch_environment_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "--num-cores"):
            configure_xla_launch_environment(argparse.Namespace(device="xla", num_cores=0, num_workers=0))

        with self.assertRaisesRegex(ValueError, "--num-workers"):
            configure_xla_launch_environment(argparse.Namespace(device="xla", num_cores=1, num_workers=-1))

    def test_resolve_step_limit_treats_cli_negative_as_unlimited(self):
        self.assertIsNone(resolve_step_limit(-1, {"max_train_steps": 20}, "max_train_steps"))
        self.assertEqual(resolve_step_limit(None, {"max_train_steps": 20}, "max_train_steps"), 20)
        self.assertIsNone(resolve_step_limit(None, {"max_train_steps": -1}, "max_train_steps"))

    def test_resolve_step_limit_rejects_zero(self):
        with self.assertRaisesRegex(ValueError, "max_train_steps"):
            resolve_step_limit(0, {"max_train_steps": 20}, "max_train_steps")

        with self.assertRaisesRegex(ValueError, "max_train_steps"):
            resolve_step_limit(None, {"max_train_steps": 0}, "max_train_steps")

    def test_xla_rank_and_world_size_prefer_runtime_api(self):
        xm = types.SimpleNamespace(get_ordinal=lambda: 99, xrt_world_size=lambda: 99)
        xr = types.SimpleNamespace(global_ordinal=lambda: 3, world_size=lambda: 4)

        self.assertEqual(_xla_rank(xm, xr), 3)
        self.assertEqual(_xla_world_size(xm, xr), 4)

    def test_xla_master_check_prefers_runtime_api(self):
        xm = types.SimpleNamespace(is_master_ordinal=lambda: True, get_ordinal=lambda: 0)
        rank_zero = types.SimpleNamespace(global_ordinal=lambda: 0)
        rank_three = types.SimpleNamespace(global_ordinal=lambda: 3)

        self.assertTrue(is_master(True, xm, rank_zero))
        self.assertFalse(is_master(True, xm, rank_three))

    def test_xla_rank_world_size_fallbacks_support_old_xla_model_api(self):
        xm = types.SimpleNamespace(get_ordinal=lambda: 2, xrt_world_size=lambda: 4)

        self.assertEqual(_xla_rank(xm), 2)
        self.assertEqual(_xla_world_size(xm), 4)

    def test_xla_multiprocessing_launcher_uses_pjrt_default_nprocs(self):
        calls = {}
        torch_xla = types.ModuleType("torch_xla")
        distributed = types.ModuleType("torch_xla.distributed")
        distributed.__path__ = []
        xmp = types.ModuleType("torch_xla.distributed.xla_multiprocessing")

        def fake_spawn(fn, args=(), nprocs=None, start_method="spawn"):
            calls["fn"] = fn
            calls["args"] = args
            calls["nprocs"] = nprocs
            calls["start_method"] = start_method
            return "spawned"

        xmp.spawn = fake_spawn
        torch_xla.distributed = distributed
        distributed.xla_multiprocessing = xmp

        with patch.dict(
            sys.modules,
            {
                "torch_xla": torch_xla,
                "torch_xla.distributed": distributed,
                "torch_xla.distributed.xla_multiprocessing": xmp,
            },
        ):
            launcher = _optional_xla_launcher()
            result = launcher(lambda index, payload: None, args=("payload",), start_method="fork")

        self.assertEqual(result, "spawned")
        self.assertEqual(calls["args"], ("payload",))
        self.assertIsNone(calls["nprocs"])
        self.assertEqual(calls["start_method"], "fork")

    def test_torch_xla_launch_wrapper_does_not_forward_start_method(self):
        calls = {}
        torch_xla = types.ModuleType("torch_xla")

        def fake_launch(fn, args=(), debug_single_process=False):
            calls["fn"] = fn
            calls["args"] = args
            calls["debug_single_process"] = debug_single_process
            return "launched"

        torch_xla.launch = fake_launch

        with patch.dict(sys.modules, {"torch_xla": torch_xla}):
            launcher = _optional_xla_launcher()
            result = launcher(lambda index, payload: None, args=("payload",), start_method="fork")

        self.assertEqual(result, "launched")
        self.assertEqual(calls["args"], ("payload",))
        self.assertFalse(calls["debug_single_process"])

    def test_main_sets_tpu_num_devices_and_uses_launcher_for_xla4(self):
        calls = {}

        def fake_launcher(fn, args=(), start_method="spawn"):
            calls["fn"] = fn
            calls["args"] = args
            calls["start_method"] = start_method
            return "spawned"

        argv = [
            "train",
            "--config",
            "configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml",
            "--device",
            "xla",
            "--num-cores",
            "4",
            "--num-workers",
            "0",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {"TPU_NUM_DEVICES": "1"}, clear=True),
            patch("allthemix.cli.train._optional_xla_launcher", return_value=fake_launcher),
            patch("allthemix.cli.train.run_worker") as run_worker,
        ):
            train_cli.main()
            tpu_num_devices = os.environ["TPU_NUM_DEVICES"]

        run_worker.assert_not_called()
        self.assertEqual(tpu_num_devices, "4")
        self.assertIs(calls["fn"], run_worker)
        self.assertEqual(calls["start_method"], "spawn")
        self.assertEqual(calls["args"][0].device, "xla")
        self.assertEqual(calls["args"][0].num_cores, 4)


if __name__ == "__main__":
    unittest.main()
