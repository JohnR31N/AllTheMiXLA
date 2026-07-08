import csv
import io
import tempfile
import unittest
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from allthemix.cli.summarize import (
    checkpoint_resume_compatible,
    cache_args_from_train_args,
    ExperimentSpec,
    ExperimentSummary,
    command_rows_for_generation,
    collect_preflight_disk_paths,
    collect_preflight_storage_roots,
    disk_space_preflight_status,
    disk_space_preflight_status_for_paths,
    expected_resume_config,
    filter_specs_by_method,
    format_error_percent,
    generated_launch_validation_status,
    render_commands,
    render_csv,
    render_json,
    render_latex,
    render_latex_table,
    render_markdown,
    render_preflight,
    render_protocol,
    render_status,
    preflight_has_blockers,
    require_valid_command_train_configs,
    saliency_cache_status,
    saliency_cache_command,
    SALIENCY_CACHE_MIN_VERSION,
    select_error_metric,
    summarize_experiment,
    storage_roots_preflight_status,
    TINY_IMAGENET_XLA4_SPECS,
    TINY_IMAGENET_XLA4_PROTOCOL_ID,
    tiny_imagenet_data_status,
    TrainArgValidationError,
    train_arg_value,
    train_args_for_method,
    train_args_use_non_batch_saliency,
    train_args_validation_status,
    training_command,
    validate_tiny_xla4_protocol,
    xla_env_preflight_status,
)
from allthemix.cli.build_saliency_cache import CACHE_BUILDER_VERSION
from allthemix.cli.train import load_config


def _cache_metadata(method: str, count: int = 3, shape: list[int] | None = None, **overrides):
    metadata = {
        "builder_version": CACHE_BUILDER_VERSION,
        "method": method,
        "dataset": "tinyimagenet",
        "recipe": "openmixup",
        "transform_profile": "openmixup",
        "image_size": 64,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "base_transform": "tensor_normalize_only",
        "blur_kernel": 7,
        "count": count,
        "shape": shape or [count, 64, 64],
        "dtype": "float32",
        "raw_unit_images": True,
        "minmax_normalized": True,
        "allow_gradient_fallback": False,
    }
    metadata.update(overrides)
    return metadata


def _write_run_metadata(metrics_path: Path, raw_config: dict, **config_overrides):
    resolved = expected_resume_config(raw_config)
    resolved.update(config_overrides)
    (metrics_path.parent / "config.json").write_text(json.dumps({"resolved": resolved}, indent=2, sort_keys=True))


class SummarizeTests(unittest.TestCase):
    def test_saliency_cache_min_version_tracks_cache_builder_version(self):
        self.assertEqual(SALIENCY_CACHE_MIN_VERSION, CACHE_BUILDER_VERSION)

    def test_filter_specs_by_method_accepts_keys_and_aliases(self):
        selected = filter_specs_by_method(TINY_IMAGENET_XLA4_SPECS, ["ERM", "guided-sr", "resize_mix"])

        self.assertEqual([spec.method_key for spec in selected], ["baseline", "resizemix", "guided_sr"])

    def test_filter_specs_by_method_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown method filter"):
            filter_specs_by_method(TINY_IMAGENET_XLA4_SPECS, ["not_a_method"])

    def test_select_error_metric_prefers_final_test_in_auto_mode(self):
        rows = [
            {"phase": "train_val", "best_top1_error": "0.334300", "val_top1": "66.57"},
            {"phase": "final_test", "test_top1_accuracy": "0.594400", "best_top1_error": "0.334300"},
        ]

        error, source = select_error_metric(rows, mode="auto")

        self.assertAlmostEqual(error, 0.4056)
        self.assertEqual(source, "test_top1_accuracy")

    def test_select_error_metric_falls_back_to_best(self):
        rows = [{"phase": "train_val", "best_top1": "66.57"}]

        error, source = select_error_metric(rows, mode="auto")

        self.assertAlmostEqual(error, 0.3343)
        self.assertEqual(source, "best_top1")

    def test_select_error_metric_supports_openmixup_last10_median(self):
        rows = [
            {"epoch": str(epoch), "phase": "train_val", "val_top1": str(50 + epoch)}
            for epoch in range(1, 13)
        ]

        error, source = select_error_metric(rows, mode="last10_median")

        self.assertAlmostEqual(error, 1.0 - 0.575)
        self.assertEqual(source, "last10_median:val_top1")

    def test_select_error_metric_requires_ten_epochs_for_last10_median(self):
        rows = [
            {"epoch": str(epoch), "phase": "train_val", "val_top1": str(50 + epoch)}
            for epoch in range(192, 200)
        ]

        error, source = select_error_metric(rows, mode="last10_median")

        self.assertIsNone(error)
        self.assertEqual(source, "incomplete_last10_median")

    def test_select_error_metric_uses_latest_duplicate_epoch_for_last10_median(self):
        rows = [
            {"epoch": str(epoch), "phase": "train_val", "val_top1": "10.0"}
            for epoch in range(191, 201)
        ]
        rows.extend(
            {"epoch": str(epoch), "phase": "train_val", "val_top1": "90.0"}
            for epoch in range(191, 201)
        )

        error, source = select_error_metric(rows, mode="last10_median")

        self.assertAlmostEqual(error, 0.10)
        self.assertEqual(source, "last10_median:val_top1")

    def test_tiny_imagenet_data_status_accepts_original_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch.dict(
                "allthemix.cli.summarize.TINY_IMAGENET_EXPECTED_COUNTS",
                {"train": 0, "val": 0},
                clear=True,
            ):
                status, path, detail = tiny_imagenet_data_status(root, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(status, "ok")
        self.assertEqual(path, tiny_root)
        self.assertIn("original layout", detail)
        self.assertIn("train_images=0/0", detail)

    def test_tiny_imagenet_data_status_prefers_class_folder_layout_with_wnids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train" / "n00000001").mkdir(parents=True)
            (tiny_root / "val" / "n00000001").mkdir(parents=True)
            (tiny_root / "train" / "n00000001" / "train.JPEG").write_bytes(b"fake")
            (tiny_root / "val" / "n00000001" / "val.JPEG").write_bytes(b"fake")
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch.dict(
                "allthemix.cli.summarize.TINY_IMAGENET_EXPECTED_COUNTS",
                {"train": 1, "val": 1},
                clear=True,
            ):
                status, path, detail = tiny_imagenet_data_status(root, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(status, "ok")
        self.assertEqual(path, tiny_root)
        self.assertIn("ImageFolder train/val layout", detail)
        self.assertIn("train_images=1/1", detail)

    def test_tiny_imagenet_data_status_accepts_nested_train_imagefolder_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train" / "n00000001" / "images").mkdir(parents=True)
            (tiny_root / "val" / "n00000001").mkdir(parents=True)
            (tiny_root / "train" / "n00000001" / "images" / "train.JPEG").write_bytes(b"fake")
            (tiny_root / "val" / "n00000001" / "val.JPEG").write_bytes(b"fake")

            with patch.dict(
                "allthemix.cli.summarize.TINY_IMAGENET_EXPECTED_COUNTS",
                {"train": 1, "val": 1},
                clear=True,
            ):
                status, path, detail = tiny_imagenet_data_status(root, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(status, "ok")
        self.assertEqual(path, tiny_root)
        self.assertIn("ImageFolder train/val layout", detail)
        self.assertIn("train_images=1/1", detail)

    def test_tiny_imagenet_data_status_rejects_incomplete_image_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train" / "n00000001" / "images").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "train" / "n00000001" / "images" / "train.JPEG").write_bytes(b"fake")
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            status, path, detail = tiny_imagenet_data_status(root, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(status, "incomplete")
        self.assertEqual(path, tiny_root)
        self.assertIn("train_images=1/100000", detail)
        self.assertIn("val_images=0/10000", detail)

    def test_tiny_imagenet_data_status_rejects_empty_class_folder_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train" / "n00000001").mkdir(parents=True)
            (tiny_root / "val" / "n00000001").mkdir(parents=True)

            status, path, detail = tiny_imagenet_data_status(root, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(status, "invalid")
        self.assertEqual(path, tiny_root)
        self.assertIn("missing Tiny-ImageNet", detail)

    def test_tiny_imagenet_data_status_reports_missing_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            status, path, detail = tiny_imagenet_data_status(root, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(status, "missing")
        self.assertEqual(path, root / "data" / "tiny-imagenet-200")
        self.assertIn("expected", detail)
        self.assertIn("bash scripts/download_tiny_imagenet.sh --data-dir ./data", detail)

    def test_tiny_imagenet_data_status_quotes_absolute_download_hint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data with spaces"

            status, path, detail = tiny_imagenet_data_status(root, {"dataset": "tiny_imagenet", "data_dir": str(data_dir)})

        self.assertEqual(status, "missing")
        self.assertEqual(path, data_dir / "tiny-imagenet-200")
        self.assertIn("bash scripts/download_tiny_imagenet.sh --data-dir", detail)
        self.assertIn(shlex.quote(str(data_dir)), detail)

    def test_render_preflight_allows_missing_cache_but_requires_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                root / "data/tiny_imagenet_train_saliency.npy",
            )

            cache_dir = root / "cache"
            with patch.dict(
                "allthemix.cli.summarize.TINY_IMAGENET_EXPECTED_COUNTS",
                {"train": 0, "val": 0},
                clear=True,
            ):
                rendered = render_preflight(root, [row], [], extra_args=["--saliency-dir", str(cache_dir)])
                blocked = preflight_has_blockers(root, [], extra_args=["--saliency-dir", str(cache_dir)])
                blocked_with_missing_run = preflight_has_blockers(
                    root,
                    [],
                    rows=[row],
                    extra_args=["--saliency-dir", str(cache_dir)],
                )

        self.assertIn("| ready_to_launch | ok |", rendered)
        self.assertIn("| saliency_caches | will_build |", rendered)
        self.assertIn((cache_dir / "tiny_imagenet_train_saliency.npy").as_posix(), rendered)
        self.assertFalse(blocked)
        self.assertFalse(blocked_with_missing_run)

    def test_render_preflight_honors_data_dir_train_arg_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            override_data_dir = root / "mnt" / "tiny"
            tiny_root = override_data_dir / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch.dict(
                "allthemix.cli.summarize.TINY_IMAGENET_EXPECTED_COUNTS",
                {"train": 0, "val": 0},
                clear=True,
            ):
                rendered = render_preflight(root, [], [], extra_args=["--data-dir", str(override_data_dir)])
                blocked = preflight_has_blockers(root, [], extra_args=["--data-dir", str(override_data_dir)])

        self.assertIn("| ready_to_launch | ok |", rendered)
        self.assertIn(tiny_root.as_posix(), rendered)
        self.assertFalse(blocked)

    def test_xla_env_preflight_status_skips_non_xla_devices(self):
        status, detail = xla_env_preflight_status(device="cpu")

        self.assertEqual(status, "skipped")
        self.assertIn("device=cpu", detail)

    def test_xla_env_preflight_status_forwards_required_venv_name(self):
        captured = {}

        def fake_build_checks(args):
            captured["require_venv_name"] = args.require_venv_name
            return [
                SimpleNamespace(name="python", ok=True),
                SimpleNamespace(name="venv", ok=False),
                SimpleNamespace(name="torch_xla", ok=True),
            ]

        with patch("allthemix.cli.verify_xla_env.build_checks", side_effect=fake_build_checks):
            status, detail = xla_env_preflight_status(require_tpu=True, require_venv_name=".venvxla")

        self.assertEqual(captured["require_venv_name"], ".venvxla")
        self.assertEqual(status, "invalid")
        self.assertIn("venv=fail", detail)
        self.assertIn("--require-venv-name .venvxla", detail)

    def test_xla_env_preflight_status_forwards_expected_tpu_devices(self):
        captured = {}

        def fake_build_checks(args):
            captured["require_tpu"] = args.require_tpu
            captured["skip_device_check"] = args.skip_device_check
            captured["expected_tpu_devices"] = args.expected_tpu_devices
            return [
                SimpleNamespace(name="python", ok=True),
                SimpleNamespace(name="tpu_devices", ok=False),
            ]

        with patch("allthemix.cli.verify_xla_env.build_checks", side_effect=fake_build_checks):
            status, detail = xla_env_preflight_status(require_tpu=True, expected_tpu_devices=4)

        self.assertTrue(captured["require_tpu"])
        self.assertFalse(captured["skip_device_check"])
        self.assertEqual(captured["expected_tpu_devices"], 4)
        self.assertEqual(status, "invalid")
        self.assertIn("--expected-tpu-devices 4", detail)

    def test_render_preflight_can_block_on_xla_env_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch("allthemix.cli.summarize.xla_env_preflight_status", return_value=("invalid", "torch_xla=fail")):
                rendered = render_preflight(root, [], [], check_env=True, require_tpu_env=True)
                blocked = preflight_has_blockers(root, [], check_env=True, require_tpu_env=True)

        self.assertIn("| ready_to_launch | blocked |", rendered)
        self.assertIn("xla_env=invalid", rendered)
        self.assertIn("| xla_env | invalid | torch_xla=fail |", rendered)
        self.assertTrue(blocked)

    def test_render_preflight_requires_visible_tpu_count_matching_num_cores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch("allthemix.cli.summarize.xla_env_preflight_status", return_value=("ok", "env=ok")) as env_check:
                render_preflight(root, [], [], check_env=True, require_tpu_env=True, num_cores=4)
                preflight_has_blockers(root, [], check_env=True, require_tpu_env=True, num_cores=4)

        for call in env_check.call_args_list:
            with self.subTest(call=call):
                self.assertEqual(call.kwargs["expected_tpu_devices"], 4)

    def test_render_preflight_allows_ok_xla_env_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with (
                patch("allthemix.cli.summarize.xla_env_preflight_status", return_value=("ok", "python=ok; torch_xla=ok")),
                patch.dict(
                    "allthemix.cli.summarize.TINY_IMAGENET_EXPECTED_COUNTS",
                    {"train": 0, "val": 0},
                    clear=True,
                ),
            ):
                rendered = render_preflight(root, [], [], check_env=True)
                blocked = preflight_has_blockers(root, [], check_env=True)

        self.assertIn("| ready_to_launch | ok |", rendered)
        self.assertIn("| xla_env | ok | python=ok; torch_xla=ok |", rendered)
        self.assertFalse(blocked)

    def test_render_preflight_skips_opencv_env_check_for_non_batch_saliency_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch("allthemix.cli.summarize.xla_env_preflight_status", return_value=("ok", "env=ok")) as env_check:
                render_preflight(
                    root,
                    [],
                    [],
                    check_env=True,
                    extra_args=["--saliency-source", "gradient"],
                )

        self.assertTrue(env_check.call_args.kwargs["skip_opencv_check"])

    def test_render_preflight_keeps_opencv_env_check_when_last_saliency_override_is_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch("allthemix.cli.summarize.xla_env_preflight_status", return_value=("ok", "env=ok")) as env_check:
                render_preflight(
                    root,
                    [],
                    [],
                    check_env=True,
                    extra_args=["--saliency-source", "gradient", "--saliency-source", "batch"],
                )

        self.assertFalse(env_check.call_args.kwargs["skip_opencv_check"])

    def test_disk_space_preflight_status_checks_threshold(self):
        usage = SimpleNamespace(total=20 * 1024**3, used=18 * 1024**3, free=2 * 1024**3)

        with patch("allthemix.cli.summarize.shutil.disk_usage", return_value=usage):
            ok_status, ok_detail = disk_space_preflight_status(Path("/repo"), min_free_gb=1)
            low_status, low_detail = disk_space_preflight_status(Path("/repo"), min_free_gb=5)

        self.assertEqual(ok_status, "ok")
        self.assertIn("free=2.0 GiB", ok_detail)
        self.assertEqual(low_status, "invalid")
        self.assertIn("min=5.0 GiB", low_detail)

    def test_disk_space_preflight_status_rejects_negative_threshold(self):
        status, detail = disk_space_preflight_status(Path("."), min_free_gb=-1)

        self.assertEqual(status, "invalid")
        self.assertIn("min_free_gb must be >= 0", detail)

    def test_render_preflight_can_block_on_low_disk_space(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch("allthemix.cli.summarize.disk_space_preflight_status_for_paths", return_value=("invalid", "free=1.0 GiB; min=10.0 GiB")):
                rendered = render_preflight(root, [], [], min_free_gb=10)
                blocked = preflight_has_blockers(root, [], min_free_gb=10)

        self.assertIn("| ready_to_launch | blocked |", rendered)
        self.assertIn("disk_space=invalid", rendered)
        self.assertIn("| disk_space | invalid | free=1.0 GiB; min=10.0 GiB |", rendered)
        self.assertTrue(blocked)

    def test_collect_preflight_disk_paths_includes_output_checkpoint_data_and_saliency_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "checkpoint_dir: ./checkpoints",
                        "run_name: saliency_run",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliency_run/metrics.csv"),
                None,
                "missing_file",
                "missing",
            )
            paths = collect_preflight_disk_paths(
                root,
                [row],
                extra_args=[
                    "--data-dir",
                    "/mnt/tiny",
                    "--checkpoint-dir",
                    "debug_checkpoints",
                    "--saliency-dir",
                    "/mnt/cache",
                ],
            )

        self.assertEqual(paths["repo"], root)
        self.assertEqual(paths["SaliencyMix_outputs"], root / "outputs" / "saliency_run")
        self.assertEqual(paths["SaliencyMix_checkpoints"], root / "debug_checkpoints" / "saliency_run")
        self.assertEqual(paths["SaliencyMix_data"], root / Path("/mnt/tiny"))
        self.assertEqual(paths["SaliencyMix_saliency_cache"], root / Path("/mnt/cache"))

    def test_collect_preflight_disk_paths_filters_saliency_overrides_per_method(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            saliency_config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / baseline_config_path.parent).mkdir(parents=True)
            (root / baseline_config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "data_dir: ./data",
                    ]
                )
            )
            (root / saliency_config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            rows = [
                ExperimentSummary(
                    ExperimentSpec("Baseline", "ERM", "baseline", baseline_config_path),
                    Path("outputs/baseline/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
                ExperimentSummary(
                    ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", saliency_config_path),
                    Path("outputs/saliencymix/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
            ]

            paths = collect_preflight_disk_paths(
                root,
                rows,
                extra_args=["--saliency-source", "batch", "--saliency-dir", "/mnt/cache"],
            )

        self.assertNotIn("ERM_saliency_cache", paths)
        self.assertEqual(paths["SaliencyMix_saliency_cache"], root / Path("/mnt/cache"))

    def test_collect_preflight_storage_roots_includes_output_checkpoint_data_and_cache_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "checkpoint_dir: ./checkpoints",
                        "run_name: saliency_run",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliency_run/metrics.csv"),
                None,
                "missing_file",
                "missing",
            )
            paths = collect_preflight_storage_roots(
                root,
                [row],
                extra_args=["--checkpoint-dir", "debug_checkpoints", "--saliency-dir", "cache"],
            )

        self.assertEqual(paths["SaliencyMix_output_dir"], root / "outputs")
        self.assertEqual(paths["SaliencyMix_checkpoint_dir"], root / "debug_checkpoints")
        self.assertEqual(paths["SaliencyMix_data_dir"], root / "data")
        self.assertEqual(paths["SaliencyMix_saliency_dir"], root / "cache")

    def test_collect_preflight_storage_roots_accepts_equals_style_train_args(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "checkpoint_dir: ./checkpoints",
                        "run_name: saliency_run",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliency_run/metrics.csv"),
                None,
                "missing_file",
                "missing",
            )

            paths = collect_preflight_storage_roots(
                root,
                [row],
                extra_args=[
                    "--checkpoint-dir=debug_checkpoints",
                    "--data-dir=/mnt/tiny",
                    "--saliency-dir=/mnt/cache",
                ],
            )

        self.assertEqual(train_arg_value(["--data-dir=/mnt/tiny"], "--data-dir"), "/mnt/tiny")
        self.assertEqual(paths["SaliencyMix_checkpoint_dir"], root / "debug_checkpoints")
        self.assertEqual(paths["SaliencyMix_data_dir"], root / Path("/mnt/tiny"))
        self.assertEqual(paths["SaliencyMix_saliency_dir"], root / Path("/mnt/cache"))

    def test_collect_preflight_storage_roots_filters_saliency_overrides_per_method(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            saliency_config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / baseline_config_path.parent).mkdir(parents=True)
            (root / baseline_config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "data_dir: ./data",
                    ]
                )
            )
            (root / saliency_config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            rows = [
                ExperimentSummary(
                    ExperimentSpec("Baseline", "ERM", "baseline", baseline_config_path),
                    Path("outputs/baseline/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
                ExperimentSummary(
                    ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", saliency_config_path),
                    Path("outputs/saliencymix/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
            ]

            paths = collect_preflight_storage_roots(
                root,
                rows,
                extra_args=["--saliency-source", "batch", "--saliency-dir", "cache"],
            )

        self.assertNotIn("ERM_saliency_dir", paths)
        self.assertEqual(paths["SaliencyMix_saliency_dir"], root / "cache")

    def test_storage_roots_preflight_status_can_require_existing_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "output_dir: ./outputs",
                        "checkpoint_dir: ./checkpoints",
                        "run_name: baseline_run",
                        "data_dir: ./data",
                    ]
                )
            )
            row = ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", config_path),
                Path("outputs/baseline_run/metrics.csv"),
                None,
                "missing_file",
                "missing",
            )

            allowed_status, allowed_detail = storage_roots_preflight_status(root, [row], require_existing=False)
            required_status, required_detail = storage_roots_preflight_status(root, [row], require_existing=True)

        self.assertEqual(allowed_status, "ok")
        self.assertIn("creatable roots may be created", allowed_detail)
        self.assertEqual(required_status, "invalid")
        self.assertIn("ERM_data_dir", required_detail)
        self.assertNotIn("missing_required=ERM_output_dir", required_detail)

    def test_storage_roots_preflight_allows_creatable_local_outputs_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / "data").mkdir()
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "output_dir: ./outputs",
                        "checkpoint_dir: ./checkpoints",
                        "run_name: baseline_run",
                        "data_dir: ./data",
                    ]
                )
            )
            row = ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", config_path),
                Path("outputs/baseline_run/metrics.csv"),
                None,
                "missing_file",
                "missing",
            )

            status, detail = storage_roots_preflight_status(root, [row], require_existing=True)

        self.assertEqual(status, "ok")
        self.assertIn("missing_creatable=", detail)
        self.assertIn("ERM_output_dir", detail)
        self.assertIn("ERM_checkpoint_dir", detail)

    def test_render_preflight_can_block_on_missing_required_storage_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "data_dir: ./data",
                        "output_dir: ./outputs",
                        "checkpoint_dir: ./checkpoints",
                    ]
                )
            )

            rendered = render_preflight(root, [], [], require_existing_storage_roots=True)
            blocked = preflight_has_blockers(root, [], require_existing_storage_roots=True)

        self.assertIn("| ready_to_launch | blocked |", rendered)
        self.assertIn("storage_roots=invalid", rendered)
        self.assertIn("| storage_roots | invalid |", rendered)
        self.assertTrue(blocked)

    def test_disk_space_preflight_status_for_paths_reports_low_mount_label(self):
        usage = SimpleNamespace(total=20 * 1024**3, used=18 * 1024**3, free=2 * 1024**3)

        with patch("allthemix.cli.summarize.shutil.disk_usage", return_value=usage):
            status, detail = disk_space_preflight_status_for_paths(
                {"repo": Path("/repo"), "cache": Path("/mnt/cache")},
                min_free_gb=5,
            )

        self.assertEqual(status, "invalid")
        self.assertIn("below_min=", detail)
        self.assertIn("repo", detail)

    def test_preflight_has_blockers_accepts_legacy_positional_extra_args(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            override_data_dir = root / "mnt" / "tiny"
            tiny_root = override_data_dir / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            with patch.dict(
                "allthemix.cli.summarize.TINY_IMAGENET_EXPECTED_COUNTS",
                {"train": 0, "val": 0},
                clear=True,
            ):
                blocked = preflight_has_blockers(root, [], ["--data-dir", str(override_data_dir)])

        self.assertFalse(blocked)

    def test_train_arg_value_uses_last_override(self):
        value = train_arg_value(
            [
                "--data-dir",
                "/mnt/default",
                "--data-dir=/mnt/equals",
                "--data-dir",
                "/mnt/final",
            ],
            "--data-dir",
        )

        self.assertEqual(value, "/mnt/final")

    def test_render_preflight_blocks_invalid_train_args(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            rendered = render_preflight(root, [], [], extra_args=["--data-dir"])
            blocked = preflight_has_blockers(root, [], extra_args=["--data-dir"])

        self.assertIn("| ready_to_launch | blocked |", rendered)
        self.assertIn("| train_args | invalid |", rendered)
        self.assertIn("expected one argument", rendered)
        self.assertTrue(blocked)

    def test_render_preflight_blocks_train_args_that_fail_resolved_config_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            rendered = render_preflight(root, [], [], extra_args=["--learning-rate", "nan"])
            blocked = preflight_has_blockers(root, [], extra_args=["--learning-rate", "nan"])

        self.assertIn("| ready_to_launch | blocked |", rendered)
        self.assertIn("| train_args | invalid |", rendered)
        self.assertIn("lr must be finite", rendered)
        self.assertTrue(blocked)

    def test_render_preflight_validates_train_args_against_each_selected_method_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            guided_config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            (root / baseline_config_path.parent).mkdir(parents=True)
            (root / baseline_config_path).write_text("dataset: tiny_imagenet\nmethod: baseline\ndata_dir: ./data\n")
            (root / guided_config_path).write_text("dataset: tiny_imagenet\nmethod: guided_sr\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            rows = [
                ExperimentSummary(
                    ExperimentSpec("Baseline", "ERM", "baseline", baseline_config_path),
                    Path("outputs/baseline/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
                ExperimentSummary(
                    ExperimentSpec("MixDA", "Guided-SR", "guided_sr", guided_config_path),
                    Path("outputs/guided_sr/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
            ]

            rendered = render_preflight(root, rows, [], extra_args=["--guidedmixup-blur-kernel", "4"])
            blocked = preflight_has_blockers(
                root,
                [],
                rows=rows,
                extra_args=["--guidedmixup-blur-kernel", "4"],
            )

        self.assertIn("| ready_to_launch | blocked |", rendered)
        self.assertIn("| train_args | invalid |", rendered)
        self.assertIn("Guided-SR", rendered)
        self.assertIn("guidedmixup_blur_kernel", rendered)
        self.assertTrue(blocked)

    def test_command_train_config_validation_uses_each_selected_method_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            guided_config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            (root / baseline_config_path.parent).mkdir(parents=True)
            (root / baseline_config_path).write_text("dataset: tiny_imagenet\nmethod: baseline\ndata_dir: ./data\n")
            (root / guided_config_path).write_text("dataset: tiny_imagenet\nmethod: guided_sr\ndata_dir: ./data\n")
            rows = [
                ExperimentSummary(
                    ExperimentSpec("Baseline", "ERM", "baseline", baseline_config_path),
                    Path("outputs/baseline/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
                ExperimentSummary(
                    ExperimentSpec("MixDA", "Guided-SR", "guided_sr", guided_config_path),
                    Path("outputs/guided_sr/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
            ]

            require_valid_command_train_configs(root, rows, ["--log-interval", "1"])
            with self.assertRaisesRegex(TrainArgValidationError, "Guided-SR"):
                require_valid_command_train_configs(root, rows, ["--guidedmixup-blur-kernel", "4"])

    def test_command_train_config_validation_uses_only_rows_that_will_generate_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            guided_config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            (root / baseline_config_path.parent).mkdir(parents=True)
            (root / baseline_config_path).write_text("dataset: tiny_imagenet\nmethod: baseline\ndata_dir: ./data\n")
            (root / guided_config_path).write_text("dataset: tiny_imagenet\nmethod: guided_sr\ndata_dir: ./data\n")
            rows = [
                ExperimentSummary(
                    ExperimentSpec("Baseline", "ERM", "baseline", baseline_config_path),
                    Path("outputs/baseline/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                ),
                ExperimentSummary(
                    ExperimentSpec("MixDA", "Guided-SR", "guided_sr", guided_config_path),
                    Path("outputs/guided_sr/metrics.csv"),
                    0.4,
                    "test_top1_error",
                    "ok",
                ),
            ]

            pending_rows = command_rows_for_generation(rows)
            all_rows = command_rows_for_generation(rows, include_complete=True)
            require_valid_command_train_configs(root, pending_rows, ["--guidedmixup-blur-kernel", "4"])
            with self.assertRaisesRegex(TrainArgValidationError, "Guided-SR"):
                require_valid_command_train_configs(root, all_rows, ["--guidedmixup-blur-kernel", "4"])

    def test_command_train_config_validation_uses_specs_when_rows_are_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guided_config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            (root / guided_config_path.parent).mkdir(parents=True)
            (root / guided_config_path).write_text("dataset: tiny_imagenet\nmethod: guided_sr\ndata_dir: ./data\n")
            specs = [ExperimentSpec("MixDA", "Guided-SR", "guided_sr", guided_config_path)]

            with self.assertRaisesRegex(TrainArgValidationError, "Guided-SR"):
                require_valid_command_train_configs(
                    root,
                    [],
                    ["--guidedmixup-blur-kernel", "4"],
                    specs=specs,
                )

    def test_render_preflight_blocks_xla_launch_size_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")

            rendered = render_preflight(root, [], [], device="xla", num_cores=8)
            blocked = preflight_has_blockers(root, [], device="xla", num_cores=8)

        self.assertIn("| ready_to_launch | blocked |", rendered)
        self.assertIn("| launch_args | invalid |", rendered)
        self.assertIn("expects --num-cores 4", rendered)
        self.assertTrue(blocked)

    def test_render_preflight_skips_cache_for_non_batch_saliency_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                root / "data/tiny_imagenet_train_saliency.npy",
            )

            rendered = render_preflight(root, [row], [], extra_args=["--saliency-source", "gradient"])

        self.assertIn("| saliency_caches | skipped |", rendered)
        self.assertIn("cache build is not required", rendered)

    def test_render_preflight_skips_cache_for_eval_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                root / "data/tiny_imagenet_train_saliency.npy",
                best_checkpoint_path=root / "checkpoints/saliencymix/best.pt",
            )

            rendered = render_preflight(root, [row], [], extra_args=["--eval-only"])

        self.assertIn("| saliency_caches | skipped |", rendered)
        self.assertIn("eval-only rows with best checkpoints", rendered)

    def test_render_preflight_checks_cache_for_eval_only_rows_without_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                root / "data/tiny_imagenet_train_saliency.npy",
            )

            rendered = render_preflight(root, [row], [], extra_args=["--eval-only"])

        self.assertIn("| saliency_caches | will_build |", rendered)

    def test_render_preflight_skips_cache_for_eval_only_with_user_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                root / "data/tiny_imagenet_train_saliency.npy",
            )

            rendered = render_preflight(
                root,
                [row],
                [],
                extra_args=["--eval-only", "--checkpoint", "/mnt/checkpoints/saliencymix_best.pt"],
            )

        self.assertIn("| saliency_caches | skipped |", rendered)
        self.assertIn("user checkpoint", rendered)
        self.assertIn("| best_checkpoints | ok |", rendered)
        self.assertIn("checkpoint=user-provided", rendered)

    def test_render_preflight_reports_eval_only_checkpoint_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text("dataset: tiny_imagenet\ndata_dir: ./data\n")
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            baseline = ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", config_path),
                Path("outputs/baseline/metrics.csv"),
                None,
                "missing_file",
                "missing",
                best_checkpoint_path=root / "checkpoints/baseline/best.pt",
            )
            fmix = ExperimentSummary(
                ExperimentSpec("MixDA", "FMix", "fmix", config_path),
                Path("outputs/fmix/metrics.csv"),
                None,
                "missing_file",
                "missing",
            )

            rendered = render_preflight(root, [baseline, fmix], [], extra_args=["--eval-only"])

        self.assertIn("| best_checkpoints | mixed |", rendered)
        self.assertIn("eval_only=1; full_train=1", rendered)
        self.assertIn("best=ERM", rendered)
        self.assertIn("missing_best=FMix", rendered)

    def test_render_preflight_rechecks_cache_after_saliency_dir_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            default_cache = root / "data" / "tiny_imagenet_train_saliency.npy"
            np.save(default_cache, np.ones((1, 64, 64), dtype=np.float32))
            default_cache.with_suffix(default_cache.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", count=1, shape=[1, 64, 64]))
            )
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                default_cache,
            )
            override_cache_dir = root / "override-cache"

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 1}):
                rendered = render_preflight(root, [row], [], extra_args=["--saliency-dir", str(override_cache_dir)])

        self.assertIn("| saliency_caches | will_build |", rendered)
        self.assertIn((override_cache_dir / "tiny_imagenet_train_saliency.npy").as_posix(), rendered)

    def test_render_preflight_rechecks_cache_after_data_dir_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            default_cache = root / "data" / "tiny_imagenet_train_saliency.npy"
            np.save(default_cache, np.ones((1, 64, 64), dtype=np.float32))
            default_cache.with_suffix(default_cache.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", count=1, shape=[1, 64, 64]))
            )
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                default_cache,
            )
            override_data_dir = root / "override-data"

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 1}):
                rendered = render_preflight(root, [row], [], extra_args=["--data-dir", str(override_data_dir)])

        self.assertIn("| saliency_caches | will_build |", rendered)
        self.assertIn((override_data_dir / "tiny_imagenet_train_saliency.npy").as_posix(), rendered)

    def test_render_preflight_relocates_relative_saliency_path_with_saliency_dir_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                root / "data/tiny_imagenet_train_saliency.npy",
            )
            override_cache_dir = root / "override-cache"

            rendered = render_preflight(
                root,
                [row],
                [],
                extra_args=["--saliency-dir", str(override_cache_dir), "--saliency-path", "maps.npy"],
            )

        self.assertIn("| saliency_caches | will_build |", rendered)
        self.assertIn((override_cache_dir / "maps.npy").as_posix(), rendered)

    def test_render_preflight_rechecks_cache_after_recipe_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "recipe: openmixup",
                        "method: saliencymix",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            cache_path = root / "data" / "tiny_imagenet_train_saliency.npy"
            np.save(cache_path, np.ones((1, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", count=1, shape=[1, 64, 64], recipe="openmixup"))
            )
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                cache_path,
            )

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 1}):
                rendered = render_preflight(root, [row], [], extra_args=["--recipe", "official"])

        self.assertIn("| saliency_caches | will_rebuild |", rendered)
        self.assertIn("SaliencyMix:will_rebuild", rendered)

    def test_render_preflight_rebuilds_nonfinite_saliency_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "recipe: openmixup",
                        "data_dir: ./data",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            tiny_root = root / "data" / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir()
            (tiny_root / "wnids.txt").write_text("n00000001\n")
            cache_path = root / "data" / "tiny_imagenet_train_saliency.npy"
            maps = np.ones((1, 64, 64), dtype=np.float32)
            maps[0, 0, 0] = np.inf
            np.save(cache_path, maps)
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", count=1, shape=[1, 64, 64]))
            )
            row = ExperimentSummary(
                ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path),
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                cache_path,
            )

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 1}):
                rendered = render_preflight(root, [row], [], extra_args=[])

        self.assertIn("| saliency_caches | will_rebuild |", rendered)
        self.assertIn("SaliencyMix:will_rebuild", rendered)

    def test_summarize_experiment_reads_config_run_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,best_top1_error,val_top1_error,test_top1_error",
                        "200,train_val,0.399100,0.399100,",
                        "200,final_test,0.399100,,0.405600",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "ok")
        self.assertAlmostEqual(summary.error, 0.4056)
        self.assertEqual(summary.metric_source, "test_top1_error")

    def test_summarize_experiment_requires_run_metadata_when_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "run_metadata_required: true",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,best_top1_error,val_top1_error,test_top1_error",
                        "200,train_val,0.399100,0.399100,",
                        "200,final_test,0.399100,,0.405600",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "missing_run_metadata")
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & --", render_latex_table([summary]))

    def test_summarize_experiment_accepts_compatible_run_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "run_metadata_required: true",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,best_top1_error,val_top1_error,test_top1_error",
                        "200,train_val,0.399100,0.399100,",
                        "200,final_test,0.399100,,0.405600",
                    ]
                )
            )
            raw_config = load_config(str(root / config_path))
            _write_run_metadata(metrics_path, raw_config)
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "ok")
        self.assertAlmostEqual(summary.error, 0.4056)

    def test_summarize_experiment_rejects_incompatible_run_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "epochs: 200",
                        "guidedmixup_alpha: 1.0",
                        "guidedmixup_prob: 0.5",
                        "guidedmixup_condition: greedy",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                        "final_test: true",
                        "run_metadata_required: true",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_guided_sr_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,best_top1_error,test_top1_error",
                        "200,final_test,0.399100,0.405600",
                    ]
                )
            )
            raw_config = load_config(str(root / config_path))
            _write_run_metadata(metrics_path, raw_config, method_prob=1.0, guidedmixup_condition="random")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            cache_path.parent.mkdir(parents=True)
            np.save(cache_path, np.ones((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(json.dumps(_cache_metadata("spectral_residual")))
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "incompatible_run_config")
        self.assertIn("Guided-SR & 4.31 & 23.34 & 12.34 & --", render_latex_table([summary]))

    def test_summarize_experiment_marks_smoke_metrics_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: baseline",
                        "epochs: 200",
                        "batch_size: 32",
                        "learning_rate: 0.1",
                        "momentum: 0.9",
                        "weight_decay: 0.0005",
                        "lr_schedule: step",
                        "lr_decay_epochs: [150, 180]",
                        "basic_aug: false",
                        "aug_recipe: tiny_openmixup",
                        "validation_split: 0.1",
                        "eval_on_test_each_epoch: false",
                        "final_test: true",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,best_top1_error,test_top1_error",
                        "1,final_test,0.750000,0.760000",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            raw_config = load_config(str(root / config_path))
            torch.save({"epoch": 1, "config": expected_resume_config(raw_config)}, checkpoint_path)
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "incomplete")
        self.assertAlmostEqual(summary.error, 0.76)
        self.assertEqual(summary.metric_source, "test_top1_error")
        self.assertEqual(summary.resume_checkpoint_path, checkpoint_path)

    def test_summarize_experiment_requires_final_epoch_eval_row_for_complete_default_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,test_top1_error",
                        "200,final_test,0.405600",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "incomplete")
        self.assertAlmostEqual(summary.error, 0.4056)
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & --", render_latex_table([summary]))

    def test_summarize_experiment_requires_final_epoch_final_test_row_for_complete_default_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,test_top1_error",
                        "200,train_val,0.399100,",
                        "1,final_test,,0.405600",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "missing_final_test")
        self.assertAlmostEqual(summary.error, 0.4056)
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & --", render_latex_table([summary]))

    def test_summarize_experiment_accepts_eval_only_refresh_shape_with_eval_and_final_test_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,test_top1_error",
                        "200,eval,0.399100,",
                        "200,final_test,,0.405600",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "ok")
        self.assertAlmostEqual(summary.error, 0.4056)

    def test_summarize_experiment_carries_final_test_checkpoint_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,test_top1_error,final_test_checkpoint,final_test_checkpoint_source",
                        "200,eval,0.399100,,,",
                        "200,final_test,,0.405600,best,memory",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "ok")
        self.assertEqual(summary.final_test_checkpoint, "best")
        self.assertEqual(summary.final_test_checkpoint_source, "memory")

    def test_summarize_experiment_requires_best_final_test_checkpoint_when_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "final_test_checkpoint: best",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,test_top1_error",
                        "200,eval,0.399100,",
                        "200,final_test,,0.405600",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)
            csv_rows = list(csv.DictReader(io.StringIO(render_csv([summary]))))

        self.assertEqual(summary.status, "missing_final_test_checkpoint")
        self.assertAlmostEqual(summary.error, 0.4056)
        self.assertEqual(csv_rows[0]["tiny_imagenet_top1_error"], "--")
        self.assertEqual(csv_rows[0]["candidate_top1_error"], "40.56")

    def test_summarize_experiment_rejects_last_final_test_checkpoint_when_best_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "final_test_checkpoint: best",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,test_top1_error,final_test_checkpoint,final_test_checkpoint_source",
                        "200,eval,0.399100,,,",
                        "200,final_test,,0.405600,last,current",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "wrong_final_test_checkpoint")
        self.assertEqual(summary.final_test_checkpoint, "last")
        self.assertEqual(summary.final_test_checkpoint_source, "current")

    def test_summarize_experiment_rejects_current_source_for_best_final_test_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "final_test: true",
                        "final_test_checkpoint: best",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,test_top1_error,final_test_checkpoint,final_test_checkpoint_source",
                        "200,eval,0.399100,,,",
                        "200,final_test,,0.405600,best,current",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "missing_final_test_checkpoint_source")
        self.assertEqual(summary.final_test_checkpoint, "best")
        self.assertEqual(summary.final_test_checkpoint_source, "current")

    def test_summarize_experiment_does_not_resume_finished_checkpoint_without_last10_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            raw_config = load_config(str(root / config_path))
            torch.save({"epoch": 200, "config": expected_resume_config(raw_config)}, checkpoint_path)
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "missing")
        self.assertIsNone(summary.resume_checkpoint_path)

    def test_summarize_experiment_resumes_checkpoint_when_future_epochs_cover_last10(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: baseline",
                        "epochs: 200",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            raw_config = load_config(str(root / config_path))
            torch.save({"epoch": 190, "config": expected_resume_config(raw_config)}, checkpoint_path)
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "missing")
        self.assertEqual(summary.resume_checkpoint_path, checkpoint_path)

    def test_summarize_experiment_does_not_resume_incompatible_smoke_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: saliencymix",
                        "epochs: 200",
                        "batch_size: 32",
                        "learning_rate: 0.1",
                        "momentum: 0.9",
                        "weight_decay: 0.0005",
                        "lr_schedule: step",
                        "lr_decay_epochs: [150, 180]",
                        "saliencymix_alpha: 1.0",
                        "saliencymix_prob: 0.5",
                        "saliencymix_no_repeat: true",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                        "basic_aug: false",
                        "sal_basic_aug: false",
                        "sal_aug_recipe: tiny_openmixup",
                        "cross_device_shuffle: true",
                        "validation_split: 0.1",
                        "eval_on_test_each_epoch: false",
                        "final_test: true",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_saliencymix_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,best_top1_error,test_top1_error",
                        "1,final_test,0.750000,0.760000",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_saliencymix_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            smoke_config = expected_resume_config(load_config(str(root / config_path)))
            smoke_config["epochs"] = 1
            smoke_config["saliency_source"] = "gradient"
            torch.save({"epoch": 1, "config": smoke_config}, checkpoint_path)
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)
                compatible = checkpoint_resume_compatible(checkpoint_path, load_config(str(root / config_path)))

        self.assertIsNone(summary.resume_checkpoint_path)
        self.assertFalse(compatible)

    def test_summarize_experiment_does_not_resume_checkpoint_with_different_augmentation_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: baseline",
                        "epochs: 200",
                        "batch_size: 32",
                        "basic_aug: false",
                        "aug_recipe: tiny_openmixup",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_config = expected_resume_config(load_config(str(root / config_path)))
            checkpoint_config["use_basic_augmentation"] = True
            checkpoint_config["aug_recipe"] = None
            torch.save({"epoch": 100, "config": checkpoint_config}, checkpoint_path)
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)
            compatible = checkpoint_resume_compatible(checkpoint_path, load_config(str(root / config_path)))

        self.assertIsNone(summary.resume_checkpoint_path)
        self.assertFalse(compatible)

    def test_checkpoint_resume_compatible_allows_saliency_cache_path_only_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: saliencymix",
                        "epochs: 200",
                        "batch_size: 32",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                    ]
                )
            )
            raw_config = load_config(str(root / config_path))
            checkpoint_config = expected_resume_config(raw_config)
            checkpoint_config["saliency_dir"] = "/mnt/tiny"
            checkpoint_config["saliency_path"] = "/mnt/tiny/tiny_imagenet_train_saliency.npy"
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_saliencymix_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"epoch": 100, "config": checkpoint_config}, checkpoint_path)

            compatible = checkpoint_resume_compatible(checkpoint_path, raw_config)

        self.assertTrue(compatible)

    def test_checkpoint_resume_compatible_rejects_missing_model_impl_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_config = {
                "dataset": "tiny_imagenet",
                "model": "preact_resnet18",
                "method": "baseline",
                "epochs": 200,
                "batch_size": 32,
                "checkpoint_dir": "./checkpoints",
                "output_dir": "./outputs",
                "run_name": "tiny_imagenet_preact_resnet18_baseline_xla4",
            }
            checkpoint_config = expected_resume_config(raw_config)
            checkpoint_config.pop("model_impl_version")
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"epoch": 100, "config": checkpoint_config}, checkpoint_path)

            compatible = checkpoint_resume_compatible(checkpoint_path, raw_config)

        self.assertFalse(compatible)

    def test_checkpoint_resume_compatible_accepts_canonical_model_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_config = {
                "dataset": "tiny_imagenet",
                "model": "preact_resnet18",
                "method": "baseline",
                "epochs": 200,
                "batch_size": 32,
                "checkpoint_dir": "./checkpoints",
                "output_dir": "./outputs",
                "run_name": "tiny_imagenet_preact_resnet18_baseline_xla4",
            }
            checkpoint_config = expected_resume_config(raw_config)
            checkpoint_config["model"] = "preact-resnet18"
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"epoch": 100, "config": checkpoint_config}, checkpoint_path)

            compatible = checkpoint_resume_compatible(checkpoint_path, raw_config)

        self.assertTrue(compatible)

    def test_checkpoint_resume_compatible_accepts_dataset_and_method_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_config = {
                "dataset": "tiny_imagenet",
                "model": "preact_resnet18",
                "method": "guided_sr",
                "epochs": 200,
                "batch_size": 32,
                "checkpoint_dir": "./checkpoints",
                "output_dir": "./outputs",
                "run_name": "tiny_imagenet_preact_resnet18_guided_sr_xla4",
            }
            checkpoint_config = expected_resume_config(raw_config)
            checkpoint_config["dataset"] = "tiny_imagenet"
            checkpoint_config["method"] = "guidedmixup"
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_guided_sr_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"epoch": 100, "config": checkpoint_config}, checkpoint_path)

            compatible = checkpoint_resume_compatible(checkpoint_path, raw_config)

        self.assertTrue(compatible)

    def test_checkpoint_resume_compatible_parses_boolean_strings_strictly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_config = {
                "dataset": "tiny_imagenet",
                "model": "preact_resnet18",
                "method": "baseline",
                "epochs": 200,
                "batch_size": 32,
                "final_test": True,
                "run_metadata_required": True,
                "checkpoint_dir": "./checkpoints",
                "output_dir": "./outputs",
                "run_name": "tiny_imagenet_preact_resnet18_baseline_xla4",
            }
            checkpoint_config = expected_resume_config(raw_config)
            checkpoint_config["run_metadata_required"] = "true"
            checkpoint_config["final_test"] = "false"
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"epoch": 100, "config": checkpoint_config}, checkpoint_path)

            compatible = checkpoint_resume_compatible(checkpoint_path, raw_config)

        self.assertFalse(compatible)

    def test_summarize_experiment_rejects_incompatible_best_checkpoint_for_eval_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: saliencymix",
                        "epochs: 200",
                        "batch_size: 32",
                        "learning_rate: 0.1",
                        "momentum: 0.9",
                        "weight_decay: 0.0005",
                        "lr_schedule: step",
                        "lr_decay_epochs: [150, 180]",
                        "saliencymix_alpha: 1.0",
                        "saliencymix_prob: 0.5",
                        "saliencymix_no_repeat: true",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                        "basic_aug: false",
                        "sal_basic_aug: false",
                        "sal_aug_recipe: tiny_openmixup",
                        "cross_device_shuffle: true",
                        "validation_split: 0.1",
                        "eval_on_test_each_epoch: false",
                        "final_test: true",
                        "final_test_checkpoint: best",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_saliencymix_xla4/best.pt"
            checkpoint_path.parent.mkdir(parents=True)
            smoke_config = expected_resume_config(load_config(str(root / config_path)))
            smoke_config["epochs"] = 1
            smoke_config["saliency_source"] = "gradient"
            torch.save({"epoch": 1, "config": smoke_config}, checkpoint_path)
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)
                commands = render_commands([summary], extra_args=["--eval-only"]).splitlines()

        self.assertIsNone(summary.best_checkpoint_path)
        self.assertEqual(commands[0], saliency_cache_command(spec))
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[1])
        self.assertNotIn("--eval-only", commands[1])

    def test_summarize_experiment_rejects_corrupt_best_checkpoint_sidecar_for_eval_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: baseline",
                        "epochs: 200",
                        "batch_size: 32",
                        "learning_rate: 0.1",
                        "momentum: 0.9",
                        "weight_decay: 0.0005",
                        "lr_schedule: step",
                        "lr_decay_epochs: [150, 180]",
                        "basic_aug: false",
                        "aug_recipe: tiny_openmixup",
                        "validation_split: 0.1",
                        "eval_on_test_each_epoch: false",
                        "final_test: true",
                        "final_test_checkpoint: best",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/best.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"epoch": 200, "config": expected_resume_config(load_config(str(root / config_path)))}, checkpoint_path)
            checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").write_text("{bad-json")
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)
            commands = render_commands([summary], extra_args=["--eval-only"]).splitlines()

        self.assertIsNone(summary.best_checkpoint_path)
        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands[0])
        self.assertNotIn("--eval-only", commands[0])

    def test_summarize_experiment_rejects_legacy_best_checkpoint_without_config_for_eval_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: baseline",
                        "epochs: 200",
                        "batch_size: 32",
                        "learning_rate: 0.1",
                        "momentum: 0.9",
                        "weight_decay: 0.0005",
                        "lr_schedule: step",
                        "lr_decay_epochs: [150, 180]",
                        "basic_aug: false",
                        "aug_recipe: tiny_openmixup",
                        "validation_split: 0.1",
                        "eval_on_test_each_epoch: false",
                        "final_test: true",
                        "final_test_checkpoint: best",
                        "checkpoint_dir: ./checkpoints",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            checkpoint_path = root / "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/best.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save({"epoch": 200}, checkpoint_path)
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)
            commands = render_commands([summary], extra_args=["--eval-only"]).splitlines()

        self.assertIsNone(summary.best_checkpoint_path)
        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands[0])
        self.assertNotIn("--eval-only", commands[0])
        self.assertNotIn("--checkpoint", commands[0])

    def test_summarize_experiment_requires_final_test_when_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "epochs: 200",
                        "final_test: true",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,best_top1_error,val_top1",
                        "200,train_val,0.334300,66.57",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "missing_final_test")
        self.assertAlmostEqual(summary.error, 0.3343)
        self.assertEqual(summary.metric_source, "best_top1_error")
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & --", render_latex_table([summary]))

    def test_summarize_experiment_rejects_final_test_from_wrong_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "epochs: 200",
                        "final_test: true",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,val_top1,test_top1_error,best_top1_error",
                        "1,final_test,,0.760000,",
                        "200,train_val,66.57,,0.334300",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)
            csv_rows = list(csv.DictReader(io.StringIO(render_csv([summary]))))

        self.assertEqual(summary.status, "missing_final_test")
        self.assertAlmostEqual(summary.error, 0.76)
        self.assertEqual(summary.metric_source, "test_top1_error")
        self.assertEqual(csv_rows[0]["tiny_imagenet_top1_error"], "--")
        self.assertEqual(csv_rows[0]["candidate_top1_error"], "76.00")
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & --", render_latex_table([summary]))

    def test_summarize_experiment_respects_string_false_final_test_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "epochs: 200",
                        'final_test: "false"',
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,val_top1",
                        "200,train_val,66.57",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "ok")
        self.assertAlmostEqual(summary.error, 0.3343)

    def test_summarize_experiment_requires_ten_rows_for_openmixup_last10(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "epochs: 200",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,val_top1",
                        "200,train_val,66.57",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec, metric_mode="last10_median")

        self.assertEqual(summary.status, "missing")
        self.assertIsNone(summary.error)
        self.assertEqual(summary.metric_source, "incomplete_last10_median")
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & --", render_latex_table([summary]))

    def test_summarize_experiment_requires_final_ten_epochs_for_openmixup_last10(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "epochs: 200",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            rows = ["epoch,phase,val_top1"]
            rows.extend(f"{epoch},train_val,{60.0 + epoch / 1000:.3f}" for epoch in range(1, 10))
            rows.append("200,train_val,64.000")
            metrics_path.write_text("\n".join(rows))
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path)

            summary = summarize_experiment(root, spec, metric_mode="last10_median")

        self.assertEqual(summary.status, "incomplete")
        self.assertIsNotNone(summary.error)
        self.assertEqual(summary.metric_source, "last10_median:val_top1")
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & --", render_latex_table([summary]))

    def test_render_latex_places_tiny_imagenet_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("config.yaml")
            (root / config_path).write_text("output_dir: ./outputs\nrun_name: run\n")
            metrics_path = root / "outputs/run/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text("epoch,phase,best_top1_error\n1,train_val,0.334300\n")
            summary = summarize_experiment(root, ExperimentSpec("MixDA", "FMix", "fmix", config_path))

            rendered = render_latex([summary])

        self.assertEqual(format_error_percent(summary.error), "33.43")
        self.assertIn("MixDA & FMix & -- & -- & -- & 33.43 & -- & -- \\\\", rendered)

    def test_render_latex_table_fills_tiny_column_and_preserves_existing_values(self):
        rows = [
            ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", Path("baseline.yaml")),
                Path("outputs/baseline/metrics.csv"),
                0.4056,
                "test_top1_error",
                "ok",
            ),
            ExperimentSummary(
                ExperimentSpec("MixDA", "FMix", "fmix", Path("fmix.yaml")),
                Path("outputs/fmix/metrics.csv"),
                0.3343,
                "best_top1_error",
                "ok",
            ),
        ]

        rendered = render_latex_table(rows)

        self.assertIn("Type & Method & CIFAR-10 & CIFAR-100 & STL-10 & Tiny-ImageNet & Cars196 & CUB", rendered)
        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & 40.56 & 21.35 & -- \\\\", rendered)
        self.assertIn(r"MixDA & FMix & 3.70 & 20.71 & 11.99 & \textbf{33.43} & -- & -- \\", rendered)
        self.assertIn(r"MixDA & ResizeMix & \textbf{3.53} & \textbf{20.50}", rendered)
        self.assertIn(r"Baseline$^\dagger$ & ERM (split) & 5.22 & 25.19 & -- & -- & -- & -- \\", rendered)
        self.assertTrue(rendered.endswith(r"\bottomrule"))

    def test_render_latex_table_bolds_all_complete_tiny_best_ties(self):
        rows = [
            ExperimentSummary(
                ExperimentSpec("MixDA", "MixUp", "mixup", Path("mixup.yaml")),
                Path("outputs/mixup/metrics.csv"),
                0.3343,
                "test_top1_error",
                "ok",
            ),
            ExperimentSummary(
                ExperimentSpec("MixDA", "FMix", "fmix", Path("fmix.yaml")),
                Path("outputs/fmix/metrics.csv"),
                np.float64(0.3343),
                "test_top1_error",
                "ok",
            ),
            ExperimentSummary(
                ExperimentSpec("MixDA", "CutMix", "cutmix", Path("cutmix.yaml")),
                Path("outputs/cutmix/metrics.csv"),
                0.4056,
                "test_top1_error",
                "ok",
            ),
        ]

        rendered = render_latex_table(rows)

        self.assertIn(r"MixDA & MixUp & 4.11 & 21.65 & \textbf{10.79} & \textbf{33.43}", rendered)
        self.assertIn(r"MixDA & FMix & 3.70 & 20.71 & 11.99 & \textbf{33.43}", rendered)
        self.assertIn("MixDA & CutMix & 3.62 & 21.07 & 11.69 & 40.56", rendered)

    def test_render_latex_table_ignores_incomplete_metrics(self):
        rows = [
            ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", Path("baseline.yaml")),
                Path("outputs/baseline/metrics.csv"),
                0.76,
                "test_top1_error",
                "incomplete",
            )
        ]

        rendered = render_latex_table(rows)

        self.assertIn("Baseline & ERM & 4.94 & 24.17 & 14.31 & -- & 21.35 & -- \\\\", rendered)
        self.assertNotIn("76.00", rendered)

    def test_render_csv_keeps_table_value_safe_for_incomplete_metrics(self):
        rows = [
            ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", Path("baseline.yaml")),
                Path("outputs/baseline/metrics.csv"),
                0.76,
                "test_top1_error",
                "incomplete",
                prerequisite_path=Path("data/cache.npy"),
                resume_checkpoint_path=Path("checkpoints/baseline/last.pt"),
                best_checkpoint_path=Path("checkpoints/baseline/best.pt"),
                final_test_checkpoint="best",
                final_test_checkpoint_source="memory",
            )
        ]

        rendered = render_csv(rows)
        csv_rows = list(csv.DictReader(io.StringIO(rendered)))

        self.assertIn(
            "protocol_id,type,method,method_key,tiny_imagenet_top1_error,candidate_top1_error,"
            "metric_source,status,prerequisite_status,config_path,metrics_path,"
            "resume_checkpoint_path,best_checkpoint_path,final_test_checkpoint,"
            "final_test_checkpoint_source,prerequisite_path",
            rendered,
        )
        self.assertEqual(csv_rows[0]["protocol_id"], TINY_IMAGENET_XLA4_PROTOCOL_ID)
        self.assertEqual(csv_rows[0]["tiny_imagenet_top1_error"], "--")
        self.assertEqual(csv_rows[0]["candidate_top1_error"], "76.00")
        self.assertEqual(csv_rows[0]["resume_checkpoint_path"], "checkpoints/baseline/last.pt")
        self.assertEqual(csv_rows[0]["best_checkpoint_path"], "checkpoints/baseline/best.pt")
        self.assertEqual(csv_rows[0]["final_test_checkpoint"], "best")
        self.assertEqual(csv_rows[0]["final_test_checkpoint_source"], "memory")
        self.assertEqual(csv_rows[0]["prerequisite_path"], "data/cache.npy")

    def test_render_json_keeps_safe_table_value_and_candidate_metric(self):
        rows = [
            ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", Path("baseline.yaml")),
                Path("outputs/baseline/metrics.csv"),
                0.76,
                "test_top1_error",
                "incomplete",
            ),
            ExperimentSummary(
                ExperimentSpec("MixDA", "FMix", "fmix", Path("fmix.yaml")),
                Path("outputs/fmix/metrics.csv"),
                0.3343,
                "test_top1_error",
                "ok",
                final_test_checkpoint="best",
                final_test_checkpoint_source="checkpoints/fmix/best.pt",
            ),
        ]

        payload = json.loads(render_json(rows))

        self.assertEqual(payload["preset"], "tiny-imagenet-xla4")
        self.assertEqual(payload["protocol_id"], TINY_IMAGENET_XLA4_PROTOCOL_ID)
        self.assertEqual(payload["protocol"]["epochs"], 200)
        self.assertIn("400 epochs", payload["protocol"]["openmixup_reference"])
        self.assertEqual(payload["metric_mode"], "auto")
        self.assertEqual(payload["best_complete_method_keys"], ["fmix"])
        self.assertEqual(payload["rows"][0]["tiny_imagenet_top1_error"], "--")
        self.assertIsNone(payload["rows"][0]["tiny_imagenet_top1_error_fraction"])
        self.assertEqual(payload["rows"][0]["candidate_top1_error"], "76.00")
        self.assertEqual(payload["rows"][0]["candidate_top1_error_fraction"], 0.76)
        self.assertEqual(payload["rows"][1]["tiny_imagenet_top1_error"], "33.43")
        self.assertTrue(payload["rows"][1]["is_best_complete_tiny_imagenet"])
        self.assertEqual(payload["rows"][1]["final_test_checkpoint"], "best")
        self.assertEqual(payload["rows"][1]["final_test_checkpoint_source"], "checkpoints/fmix/best.pt")

        last10_payload = json.loads(render_json(rows, metric_mode="last10_median"))
        self.assertEqual(last10_payload["metric_mode"], "last10_median")

    def test_render_markdown_keeps_table_value_safe_for_invalid_cache(self):
        rows = [
            ExperimentSummary(
                ExperimentSpec("MixDA", "Guided-SR", "guided_sr", Path("guided_sr.yaml")),
                Path("outputs/guided_sr/metrics.csv"),
                0.4056,
                "test_top1_error",
                "invalid_cache",
                "invalid_cache",
            )
        ]

        rendered = render_markdown(rows)

        self.assertIn("| MixDA | Guided-SR | -- | 40.56 | invalid_cache | test_top1_error |", rendered)

    def test_training_command_uses_script_and_xla_defaults(self):
        spec = ExperimentSpec(
            "MixDA",
            "FMix",
            "fmix",
            Path("configs/tiny_imagenet/preact_resnet18/fmix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_fmix_xla4.sh"),
        )

        command = training_command(spec, extra_args=["--epochs", "1"])

        self.assertEqual(
            command,
            "bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_fmix_xla4.sh "
            "--device xla --num-cores 4 --num-workers 0 --epochs 1",
        )

    def test_render_commands_skips_complete_runs_by_default(self):
        complete = ExperimentSummary(
            ExperimentSpec("Baseline", "ERM", "baseline", Path("baseline.yaml")),
            Path("outputs/baseline/metrics.csv"),
            0.4,
            "test_top1_error",
            "ok",
        )
        missing = ExperimentSummary(
            ExperimentSpec(
                "MixDA",
                "MixUp",
                "mixup",
                Path("mixup.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_mixup_xla4.sh"),
            ),
            Path("outputs/mixup/metrics.csv"),
            None,
            "missing_file",
            "missing",
        )

        commands = render_commands([complete, missing])

        self.assertNotIn("baseline", commands)
        self.assertIn("run_tiny_imagenet_preact_resnet18_mixup_xla4.sh", commands)

    def test_render_commands_reruns_incomplete_runs(self):
        incomplete = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("baseline.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            0.76,
            "test_top1_error",
            "incomplete",
        )

        commands = render_commands([incomplete])

        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands)

    def test_render_commands_include_complete_eval_only_uses_best_checkpoint(self):
        complete = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("baseline.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            0.4056,
            "test_top1_error",
            "ok",
            best_checkpoint_path=Path("checkpoints/baseline/best.pt"),
        )

        commands = render_commands([complete], include_complete=True, extra_args=["--eval-only"]).splitlines()

        self.assertEqual(len(commands), 1)
        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands[0])
        self.assertIn("--eval-only", commands[0])
        self.assertIn("--checkpoint checkpoints/baseline/best.pt", commands[0])

    def test_render_commands_checkpoint_dir_override_disables_auto_eval_only_best_checkpoint(self):
        complete = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("baseline.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            0.4056,
            "test_top1_error",
            "ok",
            best_checkpoint_path=Path("checkpoints/baseline/best.pt"),
        )

        commands = render_commands(
            [complete],
            include_complete=True,
            extra_args=["--eval-only", "--checkpoint-dir", "/tmp/debug_checkpoints"],
        ).splitlines()

        self.assertEqual(len(commands), 1)
        self.assertIn("--checkpoint-dir /tmp/debug_checkpoints", commands[0])
        self.assertNotIn("--eval-only", commands[0])
        self.assertNotIn("--checkpoint checkpoints/baseline/best.pt", commands[0])

    def test_render_commands_resumes_from_last_checkpoint_when_available(self):
        incomplete = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("baseline.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            0.76,
            "test_top1_error",
            "incomplete",
            resume_checkpoint_path=Path("checkpoints/baseline/last.pt"),
        )

        commands = render_commands([incomplete])
        manual_commands = render_commands([incomplete], extra_args=["--checkpoint", "manual.pt"])
        checkpoint_dir_commands = render_commands(
            [incomplete],
            extra_args=["--checkpoint-dir", "/tmp/debug_checkpoints"],
        )

        self.assertIn("--checkpoint checkpoints/baseline/last.pt", commands)
        self.assertIn("--checkpoint manual.pt", manual_commands)
        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", manual_commands)
        self.assertIn("--checkpoint-dir /tmp/debug_checkpoints", checkpoint_dir_commands)
        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", checkpoint_dir_commands)

    def test_generated_commands_quote_paths_with_spaces(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )

        train_command = training_command(
            spec,
            extra_args=["--data-dir", "/mnt/tiny data"],
            checkpoint=Path("/mnt/check points/last.pt"),
        )
        cache_command = saliency_cache_command(
            spec,
            extra_args=["--data-dir", "/mnt/tiny data", "--saliency-dir", "/mnt/cache maps"],
        )

        self.assertIn("'/mnt/tiny data'", train_command)
        self.assertIn("'/mnt/check points/last.pt'", train_command)
        self.assertEqual(
            shlex.split(train_command)[-4:],
            ["--checkpoint", "/mnt/check points/last.pt", "--data-dir", "/mnt/tiny data"],
        )
        self.assertIn("'/mnt/tiny data'", cache_command)
        self.assertIn("'/mnt/cache maps'", cache_command)
        self.assertIn("/mnt/cache maps", shlex.split(cache_command))

    def test_render_commands_disables_auto_resume_when_train_args_override_protocol(self):
        incomplete = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("baseline.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            0.76,
            "test_top1_error",
            "incomplete",
            resume_checkpoint_path=Path("checkpoints/baseline/last.pt"),
        )

        smoke_commands = render_commands(
            [incomplete],
            extra_args=["--epochs", "1", "--max-train-steps=20", "--saliency-source", "gradient"],
        )
        logging_commands = render_commands([incomplete], extra_args=["--log-interval", "1"])
        data_dir_commands = render_commands([incomplete], extra_args=["--data-dir", "/mnt/data"])
        saliency_dir_commands = render_commands([incomplete], extra_args=["--saliency-dir", "/mnt/cache"])
        saliency_path_commands = render_commands([incomplete], extra_args=["--saliency-path", "/mnt/cache/maps.npy"])
        seed_commands = render_commands([incomplete], extra_args=["--seed", "7"])
        max_eval_alias_commands = render_commands([incomplete], extra_args=["--max-eval-steps", "5"])
        lr_alias_commands = render_commands([incomplete], extra_args=["--learning-rate", "0.2"])
        scheduler_alias_commands = render_commands([incomplete], extra_args=["--lr-schedule", "cosine"])
        milestones_alias_commands = render_commands([incomplete], extra_args=["--lr-decay-epochs", "100", "150"])

        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", smoke_commands)
        self.assertIn("--epochs 1 --max-train-steps=20", smoke_commands)
        self.assertNotIn("--saliency-source gradient", smoke_commands)
        self.assertIn("--checkpoint checkpoints/baseline/last.pt", logging_commands)
        self.assertIn("--checkpoint checkpoints/baseline/last.pt", data_dir_commands)
        self.assertIn("--data-dir /mnt/data", data_dir_commands)
        self.assertIn("--checkpoint checkpoints/baseline/last.pt", saliency_dir_commands)
        self.assertNotIn("--saliency-dir /mnt/cache", saliency_dir_commands)
        self.assertIn("--checkpoint checkpoints/baseline/last.pt", saliency_path_commands)
        self.assertNotIn("--saliency-path /mnt/cache/maps.npy", saliency_path_commands)
        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", seed_commands)
        self.assertIn("--seed 7", seed_commands)
        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", max_eval_alias_commands)
        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", lr_alias_commands)
        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", scheduler_alias_commands)
        self.assertNotIn("--checkpoint checkpoints/baseline/last.pt", milestones_alias_commands)

    def test_render_commands_filters_guided_sr_only_override_per_method(self):
        baseline = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("baseline.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            None,
            "missing_file",
            "missing",
            resume_checkpoint_path=Path("checkpoints/baseline/last.pt"),
        )
        guided_sr = ExperimentSummary(
            ExperimentSpec(
                "MixDA",
                "Guided-SR",
                "guided_sr",
                Path("guided_sr.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
            ),
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            resume_checkpoint_path=Path("checkpoints/guided_sr/last.pt"),
        )

        commands = render_commands([baseline, guided_sr], extra_args=["--guidedmixup-blur-kernel", "5"]).splitlines()

        self.assertEqual(len(commands), 2)
        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands[0])
        self.assertIn("--checkpoint checkpoints/baseline/last.pt", commands[0])
        self.assertNotIn("--guidedmixup-blur-kernel", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[1])
        self.assertIn("--guidedmixup-blur-kernel 5", commands[1])
        self.assertNotIn("--checkpoint checkpoints/guided_sr/last.pt", commands[1])

    def test_render_commands_filters_fmix_only_override_per_method(self):
        baseline = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("baseline.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            None,
            "missing_file",
            "missing",
            resume_checkpoint_path=Path("checkpoints/baseline/last.pt"),
        )
        fmix = ExperimentSummary(
            ExperimentSpec(
                "MixDA",
                "FMix",
                "fmix",
                Path("fmix.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_fmix_xla4.sh"),
            ),
            Path("outputs/fmix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            resume_checkpoint_path=Path("checkpoints/fmix/last.pt"),
        )

        commands = render_commands(
            [baseline, fmix],
            extra_args=["--reformulate", "--decay-power", "2.0"],
        ).splitlines()

        self.assertEqual(len(commands), 2)
        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands[0])
        self.assertIn("--checkpoint checkpoints/baseline/last.pt", commands[0])
        self.assertNotIn("--reformulate", commands[0])
        self.assertNotIn("--decay-power", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_fmix_xla4.sh", commands[1])
        self.assertIn("--reformulate --decay-power 2.0", commands[1])
        self.assertNotIn("--checkpoint checkpoints/fmix/last.pt", commands[1])

    def test_render_commands_emits_missing_saliencymix_cache_first(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row]).splitlines()

        self.assertEqual(commands[0], saliency_cache_command(spec))
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[1])

    def test_render_commands_forwards_seed_to_saliency_cache_command(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--seed", "7"]).splitlines()

        self.assertIn("--seed 7", commands[0])
        self.assertIn("--seed 7", commands[1])

    def test_render_commands_skips_missing_cache_for_eval_only_with_best_checkpoint(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
            best_checkpoint_path=Path("checkpoints/saliencymix/best.pt"),
        )

        commands = render_commands([row], extra_args=["--eval-only"]).splitlines()

        self.assertEqual(len(commands), 1)
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[0])
        self.assertIn("--checkpoint checkpoints/saliencymix/best.pt", commands[0])
        self.assertIn("--eval-only", commands[0])
        self.assertNotIn("build_tiny_imagenet_saliencymix_cache.sh", commands[0])

    def test_render_commands_falls_back_to_full_training_when_eval_only_best_is_missing(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--eval-only"]).splitlines()

        self.assertEqual(commands[0], saliency_cache_command(spec))
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[1])
        self.assertNotIn("--eval-only", commands[1])

    def test_render_commands_uses_eval_only_with_user_checkpoint_when_default_best_is_missing(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands(
            [row],
            extra_args=["--eval-only", "--checkpoint", "/mnt/checkpoints/saliencymix_best.pt"],
        ).splitlines()

        self.assertEqual(len(commands), 1)
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[0])
        self.assertIn("--eval-only", commands[0])
        self.assertIn("--checkpoint /mnt/checkpoints/saliencymix_best.pt", commands[0])
        self.assertNotIn("build_tiny_imagenet_saliencymix_cache.sh", commands[0])

    def test_render_commands_rejects_manual_checkpoint_for_multiple_methods(self):
        baseline = ExperimentSummary(
            ExperimentSpec(
                "Baseline",
                "ERM",
                "baseline",
                Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
            ),
            Path("outputs/baseline/metrics.csv"),
            None,
            "missing_file",
            "missing",
        )
        saliencymix = ExperimentSummary(
            ExperimentSpec(
                "MixDA",
                "SaliencyMix",
                "saliencymix",
                Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
            ),
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        with self.assertRaisesRegex(TrainArgValidationError, "one selected method"):
            render_commands(
                [baseline, saliencymix],
                extra_args=["--eval-only", "--checkpoint", "/mnt/checkpoints/best.pt"],
            )

    def test_cache_args_from_train_args_keeps_cache_safe_path_overrides(self):
        args = cache_args_from_train_args(
            [
                "--data-dir",
                "/mnt/tiny",
                "--dataset",
                "cifar10",
                "--epochs",
                "1",
                "--saliency-path=/mnt/cache/guided.npy",
                "--saliency-source",
                "gradient",
                "--recipe=openmixup",
            ]
        )

        self.assertEqual(
            args,
            [
                "--data-dir",
                "/mnt/tiny",
                "--output",
                "/mnt/cache/guided.npy",
                "--recipe",
                "openmixup",
            ],
        )

    def test_cache_args_from_train_args_maps_guided_sr_blur_kernel(self):
        args = cache_args_from_train_args(
            [
                "--guidedmixup-blur-kernel",
                "5",
                "--guidedmixup-blur-kernel=9",
            ],
            method_key="guided_sr",
        )

        self.assertEqual(args, ["--blur-kernel", "5", "--blur-kernel", "9"])

    def test_cache_args_from_train_args_filters_guided_sr_blur_kernel_for_saliencymix(self):
        args = cache_args_from_train_args(
            [
                "--data-dir",
                "/mnt/tiny",
                "--guidedmixup-blur-kernel",
                "5",
                "--guidedmixup-blur-kernel=9",
            ],
            method_key="saliencymix",
        )

        self.assertEqual(args, ["--data-dir", "/mnt/tiny"])

    def test_train_args_for_method_filters_guided_sr_only_overrides(self):
        args = [
            "--epochs",
            "1",
            "--guidedmixup-blur-kernel",
            "5",
            "--guidedmixup-condition=greedy",
        ]

        self.assertEqual(train_args_for_method(args, "baseline"), ["--epochs", "1"])
        self.assertEqual(train_args_for_method(args, "guided_sr"), args)

    def test_train_args_for_method_filters_fmix_only_overrides_without_swallowing_next_arg(self):
        args = [
            "--reformulate",
            "--epochs",
            "1",
            "--decay-power",
            "3.0",
            "--max-soft=0.1",
            "--alpha",
            "1.0",
        ]

        self.assertEqual(train_args_for_method(args, "baseline"), ["--epochs", "1"])
        self.assertEqual(train_args_for_method(args, "fmix"), args)
        self.assertEqual(
            train_args_for_method(args, "mixup"),
            ["--epochs", "1", "--alpha", "1.0"],
        )

    def test_train_args_for_method_filters_saliency_only_overrides(self):
        args = [
            "--epochs",
            "1",
            "--saliency-source",
            "gradient",
            "--saliency-dir=/mnt/cache",
            "--sal-aug-recipe",
            "tiny_openmixup",
        ]

        self.assertEqual(train_args_for_method(args, "baseline"), ["--epochs", "1"])
        self.assertEqual(train_args_for_method(args, "saliencymix"), args)
        self.assertEqual(train_args_for_method(args, "guided_sr"), args)

    def test_train_arg_value_uses_last_argparse_style_value(self):
        self.assertEqual(
            train_arg_value(["--recipe", "official", "--recipe=openmixup"], "--recipe"),
            "openmixup",
        )
        self.assertEqual(
            train_arg_value(["--data-dir", "/old", "--not-target=value", "--data-dir", "/new"], "--data-dir"),
            "/new",
        )
        self.assertIsNone(train_arg_value(["--not-target=value"], "--data-dir"))

    def test_train_args_use_non_batch_saliency_uses_last_value(self):
        self.assertFalse(
            train_args_use_non_batch_saliency(
                ["--saliency-source", "gradient", "--saliency-source", "batch"]
            )
        )
        self.assertTrue(
            train_args_use_non_batch_saliency(
                ["--saliency-source=batch", "--saliency-source=gradient"]
            )
        )

    def test_train_arg_validation_rejects_unknown_or_incomplete_args(self):
        status, detail = train_args_validation_status(["--data-dir"])
        self.assertEqual(status, "invalid")
        self.assertIn("expected one argument", detail)

        status, detail = train_args_validation_status(["--not-a-train-flag"])
        self.assertEqual(status, "invalid")
        self.assertIn("unrecognized arguments", detail)

        status, detail = train_args_validation_status(["--max-train-steps", "0"])
        self.assertEqual(status, "invalid")
        self.assertIn("step limit must be positive", detail)

    def test_train_arg_validation_rejects_generated_command_overrides(self):
        status, detail = train_args_validation_status(["--config", "other.yaml"])
        self.assertEqual(status, "invalid")
        self.assertIn("--config is reserved", detail)

        status, detail = train_args_validation_status(["--device=xla"])
        self.assertEqual(status, "invalid")
        self.assertIn("--device is reserved", detail)

        status, detail = train_args_validation_status(["--dataset", "cifar10"])
        self.assertEqual(status, "invalid")
        self.assertIn("--dataset is reserved", detail)

        status, detail = train_args_validation_status(["--output-dir", "/tmp/other-outputs"])
        self.assertEqual(status, "invalid")
        self.assertIn("--output-dir is reserved", detail)

    def test_generated_launch_validation_rejects_non_xla4_tpu_core_count(self):
        status, detail = generated_launch_validation_status("xla", 8, 0)
        self.assertEqual(status, "invalid")
        self.assertIn("expects --num-cores 4", detail)

        status, detail = generated_launch_validation_status("cpu", 8, 0)
        self.assertEqual(status, "ok")
        self.assertIn("device=cpu", detail)

    def test_render_commands_rejects_invalid_train_args_before_emitting_commands(self):
        spec = ExperimentSpec(
            "Baseline",
            "ERM",
            "baseline",
            Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/baseline/metrics.csv"),
            None,
            "missing_file",
            "missing",
        )

        with self.assertRaisesRegex(TrainArgValidationError, "expected one argument"):
            render_commands([row], extra_args=["--data-dir"])

    def test_render_commands_rejects_invalid_xla_launch_args_before_emitting_commands(self):
        spec = ExperimentSpec(
            "Baseline",
            "ERM",
            "baseline",
            Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/baseline/metrics.csv"),
            None,
            "missing_file",
            "missing",
        )

        with self.assertRaisesRegex(TrainArgValidationError, "invalid generated command launch"):
            render_commands([row], device="xla", num_cores=8)

    def test_render_commands_forwards_data_dir_to_missing_cache_build(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--data-dir", "/mnt/tiny"]).splitlines()

        self.assertIn("--data-dir /mnt/tiny", commands[0])
        self.assertIn("--data-dir /mnt/tiny", commands[1])

    def test_render_commands_builds_cache_when_data_dir_override_moves_cache_location(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "ok",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--data-dir", "/mnt/tiny"]).splitlines()

        self.assertIn("build_tiny_imagenet_saliencymix_cache.sh", commands[0])
        self.assertIn("--data-dir /mnt/tiny", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[1])
        self.assertIn("--data-dir /mnt/tiny", commands[1])

    def test_render_commands_forwards_num_workers_to_missing_cache_build(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row], num_workers=3).splitlines()

        self.assertIn("--num-workers 3", commands[0])
        self.assertIn("--num-workers 3", commands[1])

    def test_render_commands_builds_cache_when_saliency_dir_override_changes_location(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "ok",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--saliency-dir", "/mnt/cache"]).splitlines()

        self.assertIn("build_tiny_imagenet_saliencymix_cache.sh", commands[0])
        self.assertIn("--saliency-dir /mnt/cache", commands[0])
        self.assertNotIn("--overwrite", commands[0])
        self.assertIn("--saliency-dir /mnt/cache", commands[1])

    def test_saliency_cache_command_uses_script_config_once(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )

        command = saliency_cache_command(spec, num_workers=2)

        self.assertEqual(
            command,
            "bash scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh --num-workers 2",
        )
        self.assertNotIn("--config", command)

    def test_saliency_cache_command_filters_guided_sr_blur_kernel_for_saliencymix(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )

        command = saliency_cache_command(spec, extra_args=["--guidedmixup-blur-kernel", "5"])

        self.assertIn("build_tiny_imagenet_saliencymix_cache.sh", command)
        self.assertNotIn("--blur-kernel", command)

    def test_render_commands_maps_saliency_path_to_cache_output(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_guided_sr_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--saliency-path", "/mnt/cache/guided.npy"]).splitlines()

        self.assertIn("--output /mnt/cache/guided.npy", commands[0])
        self.assertIn("--saliency-path /mnt/cache/guided.npy", commands[1])

    def test_render_commands_relocates_relative_saliency_path_cache_output(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_guided_sr_saliency.npy"),
        )

        commands = render_commands(
            [row],
            extra_args=["--saliency-dir", "/mnt/cache", "--saliency-path", "guided.npy"],
        ).splitlines()

        self.assertIn("--saliency-dir /mnt/cache", commands[0])
        self.assertIn("--output /mnt/cache/guided.npy", commands[0])
        self.assertIn("--saliency-dir /mnt/cache", commands[1])
        self.assertIn("--saliency-path guided.npy", commands[1])

    def test_render_commands_overwrites_incomplete_cache(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "incomplete_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row]).splitlines()

        self.assertEqual(commands[0], saliency_cache_command(spec, overwrite=True))
        self.assertIn("--overwrite", commands[0])

    def test_render_commands_emits_missing_guided_sr_cache_first(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_guided_sr_saliency.npy"),
        )

        commands = render_commands([row]).splitlines()

        self.assertEqual(commands[0], saliency_cache_command(spec))
        self.assertIn("build_tiny_imagenet_guided_sr_cache.sh", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[1])

    def test_render_commands_skips_cache_when_train_args_override_saliency_source(self):
        spec = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/saliencymix/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "missing_cache",
            Path("data/tiny_imagenet_train_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--saliency-source", "gradient"]).splitlines()

        self.assertEqual(len(commands), 1)
        self.assertNotIn("build_tiny_imagenet_saliencymix_cache.sh", commands[0])
        self.assertIn("--saliency-source gradient", commands[0])

    def test_render_commands_skips_all_saliency_caches_for_smoke_gradient_override(self):
        saliencymix = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        guided_sr = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        rows = [
            ExperimentSummary(
                saliencymix,
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                Path("data/tiny_imagenet_train_saliency.npy"),
            ),
            ExperimentSummary(
                guided_sr,
                Path("outputs/guided_sr/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "missing_cache",
                Path("data/tiny_imagenet_train_guided_sr_saliency.npy"),
            ),
        ]

        commands = render_commands(rows, extra_args=["--saliency-source", "gradient"]).splitlines()

        self.assertEqual(len(commands), 2)
        self.assertNotIn("build_tiny_imagenet_saliencymix_cache.sh", "\n".join(commands))
        self.assertNotIn("build_tiny_imagenet_guided_sr_cache.sh", "\n".join(commands))
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[1])
        for command in commands:
            self.assertIn("--saliency-source gradient", command)

    def test_render_commands_builds_guided_sr_cache_when_batch_source_is_requested(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "ok",
            None,
        )

        commands = render_commands([row], extra_args=["--saliency-source", "batch"]).splitlines()

        self.assertEqual(len(commands), 2)
        self.assertIn("build_tiny_imagenet_guided_sr_cache.sh", commands[0])
        self.assertIn("--output data/tiny_imagenet_train_guided_sr_saliency.npy", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[1])
        self.assertIn("--saliency-source batch", commands[1])

    def test_render_commands_guided_sr_batch_cache_path_follows_saliency_dir_override(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "ok",
            None,
        )

        commands = render_commands(
            [row],
            extra_args=["--saliency-source", "batch", "--saliency-dir", "/mnt/cache"],
        ).splitlines()

        self.assertIn("--output /mnt/cache/tiny_imagenet_train_guided_sr_saliency.npy", commands[0])
        self.assertIn("--saliency-dir /mnt/cache", commands[0])
        self.assertIn("--saliency-dir /mnt/cache", commands[1])

    def test_render_commands_for_full_xla4_suite_keeps_all_table_methods(self):
        rows = []
        for spec in TINY_IMAGENET_XLA4_SPECS:
            uses_cache = spec.method_key in {"saliencymix", "guided_sr"}
            rows.append(
                ExperimentSummary(
                    spec,
                    Path(f"outputs/tiny_imagenet_preact_resnet18_{spec.method_key}_xla4/metrics.csv"),
                    None,
                    "missing_file",
                    "missing_file",
                    "missing_cache" if uses_cache else "ok",
                    Path(f"data/tiny_imagenet_train_{spec.method_key}_saliency.npy") if uses_cache else None,
                )
            )

        commands = render_commands(rows, device="xla", num_cores=4, num_workers=0).splitlines()
        train_commands = [command for command in commands if "run_tiny_imagenet_preact_resnet18_" in command]
        cache_commands = [command for command in commands if "build_tiny_imagenet_" in command]

        self.assertEqual(len(train_commands), len(TINY_IMAGENET_XLA4_SPECS))
        self.assertEqual(len(cache_commands), 2)
        self.assertIn("build_tiny_imagenet_saliencymix_cache.sh", commands[5])
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[6])
        self.assertIn("build_tiny_imagenet_guided_sr_cache.sh", commands[7])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[8])
        self.assertIn("run_tiny_imagenet_preact_resnet18_catchupmix_xla4.sh", commands[9])

        for spec in TINY_IMAGENET_XLA4_SPECS:
            with self.subTest(method=spec.method_key):
                matches = [command for command in train_commands if f"_{spec.method_key}_xla4.sh" in command]
                self.assertEqual(len(matches), 1)
                self.assertIn("--device xla", matches[0])
                self.assertIn("--num-cores 4", matches[0])
                self.assertIn("--num-workers 0", matches[0])
                script = Path(f"scripts/experiment_run/run_tiny_imagenet_preact_resnet18_{spec.method_key}_xla4.sh")
                self.assertIn(f"--config {spec.config_path.as_posix()}", script.read_text())

    def test_summarize_experiment_marks_missing_saliencymix_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "missing")
        self.assertEqual(summary.prerequisite_status, "missing_cache")
        self.assertEqual(summary.prerequisite_path.name, "tiny_imagenet_train_saliency.npy")

    def test_summarize_experiment_applies_saliency_cache_path_train_arg_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            summary = summarize_experiment(
                root,
                spec,
                extra_args=["--saliency-dir", "/mnt/cache", "--saliency-path", "maps.npy"],
            )
            csv_rows = list(csv.DictReader(io.StringIO(render_csv([summary]))))
            payload = json.loads(render_json([summary]))

        expected_suffix = "/mnt/cache/maps.npy"
        self.assertEqual(summary.prerequisite_status, "missing_cache")
        self.assertTrue(summary.prerequisite_path.as_posix().endswith(expected_suffix))
        self.assertTrue(csv_rows[0]["prerequisite_path"].endswith(expected_suffix))
        self.assertTrue(payload["rows"][0]["prerequisite_path"].endswith(expected_suffix))

    def test_summarize_experiment_keeps_formal_cache_gate_under_semantic_train_arg_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "epochs: 200",
                        "final_test: true",
                        "run_metadata_required: true",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_saliencymix_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,test_top1_error",
                        "200,eval,0.399100,",
                        "200,final_test,,0.405600",
                    ]
                )
            )
            raw_config = load_config(str(root / config_path))
            _write_run_metadata(metrics_path, raw_config)
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            summary = summarize_experiment(root, spec, extra_args=["--saliency-source", "gradient"])
            csv_rows = list(csv.DictReader(io.StringIO(render_csv([summary]))))

        self.assertEqual(summary.prerequisite_status, "missing_cache")
        self.assertEqual(summary.status, "missing_cache")
        self.assertEqual(csv_rows[0]["tiny_imagenet_top1_error"], "--")
        self.assertEqual(csv_rows[0]["candidate_top1_error"], "40.56")

    def test_summarize_experiment_marks_incomplete_saliencymix_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "incomplete_cache")
        self.assertEqual(summary.prerequisite_path.name, "tiny_imagenet_train_saliency.npy")

    def test_summarize_experiment_requires_saliencymix_cache_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_accepts_saliencymix_opencv_cache_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv"))
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "ok")

    def test_summarize_experiment_accepts_canonical_tinyimagenet_cache_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tinyimagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(json.dumps(_cache_metadata("opencv")))
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "ok")
        self.assertEqual(summary.prerequisite_path.name, "tinyimagenet_train_saliency.npy")

    def test_saliency_cache_status_closes_mmap_after_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(json.dumps(_cache_metadata("opencv")))
            raw_config = {
                "dataset": "tiny_imagenet",
                "method": "saliencymix",
                "saliency_source": "batch",
                "saliency_dir": str(cache_path.parent),
            }

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                status = saliency_cache_status(cache_path, raw_config)

            self.assertEqual(status, "ok")
            cache_path.unlink()
            self.assertFalse(cache_path.exists())

    def test_summarize_experiment_rejects_old_saliencymix_cache_builder_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", builder_version=CACHE_BUILDER_VERSION - 1))
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_rejects_saliencymix_wrong_recipe_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", recipe="official", transform_profile="official"))
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_rejects_saliencymix_wrong_normalization_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", mean=[0.5, 0.5, 0.5]))
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_rejects_saliencymix_gradient_fallback_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", allow_gradient_fallback=True))
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_rejects_saliencymix_missing_fallback_policy_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            metadata = _cache_metadata("opencv")
            metadata.pop("allow_gradient_fallback")
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(json.dumps(metadata))
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_requires_guided_sr_cache_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_guided_sr_online_ignores_stale_cache_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: spectral_residual",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.metric_source, "missing_file")
        self.assertEqual(summary.status, "missing")
        self.assertEqual(summary.prerequisite_status, "ok")
        self.assertIsNone(summary.prerequisite_path)

    def test_summarize_experiment_accepts_guided_sr_cache_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("spectral_residual"))
            )
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "ok")

    def test_summarize_experiment_rejects_guided_sr_wrong_blur_kernel_cache_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "guidedmixup_blur_kernel: 7",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("spectral_residual", blur_kernel=5))
            )
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_render_commands_rebuilds_guided_sr_wrong_blur_kernel_cache(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "invalid_cache",
            "invalid_cache",
            Path("data/tiny_imagenet_train_guided_sr_saliency.npy"),
        )

        commands = render_commands([row]).splitlines()

        self.assertEqual(commands[0], saliency_cache_command(spec, overwrite=True))
        self.assertIn("build_tiny_imagenet_guided_sr_cache.sh", commands[0])
        self.assertIn("--overwrite", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[1])

    def test_render_commands_rebuilds_guided_sr_cache_for_blur_kernel_override(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "ok",
            Path("data/tiny_imagenet_train_guided_sr_saliency.npy"),
        )

        commands = render_commands([row], extra_args=["--guidedmixup-blur-kernel", "5"]).splitlines()

        self.assertIn("build_tiny_imagenet_guided_sr_cache.sh", commands[0])
        self.assertIn("--overwrite", commands[0])
        self.assertIn("--blur-kernel 5", commands[0])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[1])
        self.assertIn("--guidedmixup-blur-kernel 5", commands[1])

    def test_render_commands_guided_sr_online_blur_override_does_not_build_cache(self):
        spec = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        row = ExperimentSummary(
            spec,
            Path("outputs/guided_sr/metrics.csv"),
            None,
            "missing_file",
            "missing",
            "ok",
            None,
        )

        commands = render_commands([row], extra_args=["--guidedmixup-blur-kernel", "5"]).splitlines()

        self.assertEqual(len(commands), 1)
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[0])
        self.assertIn("--guidedmixup-blur-kernel 5", commands[0])
        self.assertNotIn("build_tiny_imagenet_guided_sr_cache.sh", commands[0])

    def test_render_commands_does_not_rebuild_saliencymix_cache_for_guided_sr_blur_override(self):
        saliencymix = ExperimentSpec(
            "MixDA",
            "SaliencyMix",
            "saliencymix",
            Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh"),
        )
        guided_sr = ExperimentSpec(
            "MixDA",
            "Guided-SR",
            "guided_sr",
            Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml"),
            Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
        )
        rows = [
            ExperimentSummary(
                saliencymix,
                Path("outputs/saliencymix/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                Path("data/tiny_imagenet_train_saliency.npy"),
            ),
            ExperimentSummary(
                guided_sr,
                Path("outputs/guided_sr/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                Path("data/tiny_imagenet_train_guided_sr_saliency.npy"),
            ),
        ]

        commands = render_commands(rows, extra_args=["--guidedmixup-blur-kernel", "5"]).splitlines()
        joined = "\n".join(commands)

        self.assertEqual(len(commands), 3)
        self.assertIn("run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh", commands[0])
        self.assertNotIn("build_tiny_imagenet_saliencymix_cache.sh", joined)
        self.assertIn("build_tiny_imagenet_guided_sr_cache.sh", commands[1])
        self.assertIn("--overwrite", commands[1])
        self.assertIn("--blur-kernel 5", commands[1])
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[2])

    def test_render_preflight_rebuilds_guided_sr_cache_for_blur_kernel_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "guidedmixup_blur_kernel: 7",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("spectral_residual", blur_kernel=7))
            )
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)
            row = ExperimentSummary(
                spec,
                Path("outputs/guided_sr/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                cache_path,
            )

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                rendered = render_preflight(
                    root,
                    [row],
                    [],
                    extra_args=["--guidedmixup-blur-kernel", "5"],
                )

        self.assertIn("| saliency_caches | will_rebuild | Guided-SR:will_rebuild:", rendered)

    def test_render_preflight_builds_guided_sr_cache_for_batch_source_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: spectral_residual",
                    ]
                )
            )
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)
            row = ExperimentSummary(
                spec,
                Path("outputs/guided_sr/metrics.csv"),
                None,
                "missing_file",
                "missing",
                "ok",
                None,
            )

            rendered = render_preflight(root, [row], [], extra_args=["--saliency-source", "batch"])

        self.assertIn("| saliency_caches | will_build | Guided-SR:will_build:", rendered)
        self.assertIn("tiny_imagenet_train_guided_sr_saliency.npy", rendered)

    def test_render_preflight_keeps_saliencymix_cache_ok_for_guided_sr_blur_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir(parents=True)
            saliency_config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            guided_config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            (root / saliency_config_path.parent).mkdir(parents=True)
            (root / saliency_config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            (root / guided_config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "guidedmixup_blur_kernel: 7",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                    ]
                )
            )
            saliency_cache = data_dir / "tiny_imagenet_train_saliency.npy"
            guided_cache = data_dir / "tiny_imagenet_train_guided_sr_saliency.npy"
            np.save(saliency_cache, np.zeros((3, 64, 64), dtype=np.float32))
            np.save(guided_cache, np.zeros((3, 64, 64), dtype=np.float32))
            saliency_cache.with_suffix(saliency_cache.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv"))
            )
            guided_cache.with_suffix(guided_cache.suffix + ".json").write_text(
                json.dumps(_cache_metadata("spectral_residual", blur_kernel=7))
            )
            rows = [
                ExperimentSummary(
                    ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", saliency_config_path),
                    Path("outputs/saliencymix/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                    "ok",
                    saliency_cache,
                ),
                ExperimentSummary(
                    ExperimentSpec("MixDA", "Guided-SR", "guided_sr", guided_config_path),
                    Path("outputs/guided_sr/metrics.csv"),
                    None,
                    "missing_file",
                    "missing",
                    "ok",
                    guided_cache,
                ),
            ]

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                rendered = render_preflight(
                    root,
                    rows,
                    [],
                    extra_args=["--guidedmixup-blur-kernel", "5"],
                )

        self.assertIn("| saliency_caches | will_rebuild |", rendered)
        self.assertIn("SaliencyMix:ok:", rendered)
        self.assertIn("Guided-SR:will_rebuild:", rendered)

    def test_summarize_experiment_rejects_guided_sr_gradient_cache_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("gradient"))
            )
            spec = ExperimentSpec("MixDA", "Guided-SR", "guided_sr", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_rejects_stale_cache_metadata_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", count=4, shape=[4, 64, 64]))
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_rejects_stale_cache_metadata_dtype(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("opencv", dtype="float16"))
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_does_not_complete_with_invalid_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_guided_sr_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: guided_sr",
                        "epochs: 200",
                        "final_test: true",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_guided_sr_xla4",
                        "saliency_source: batch",
                        "saliency_path: ./data/tiny_imagenet_train_guided_sr_saliency.npy",
                    ]
                )
            )
            metrics_path = root / "outputs/tiny_imagenet_preact_resnet18_guided_sr_xla4/metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,test_top1_error",
                        "200,final_test,0.405600",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 64, 64), dtype=np.float32))
            cache_path.with_suffix(cache_path.suffix + ".json").write_text(
                json.dumps(_cache_metadata("gradient"))
            )
            spec = ExperimentSpec(
                "MixDA",
                "Guided-SR",
                "guided_sr",
                config_path,
                Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh"),
            )

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.status, "invalid_cache")
        self.assertEqual(summary.prerequisite_status, "invalid_cache")
        self.assertAlmostEqual(summary.error, 0.4056)
        self.assertIn("Guided-SR & 4.31 & 23.34 & 12.34 & --", render_latex_table([summary]))
        commands = render_commands([summary]).splitlines()
        self.assertEqual(commands[0], saliency_cache_command(spec, overwrite=True))
        self.assertIn("run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh", commands[1])

    def test_summarize_experiment_rejects_wrong_saliency_cache_spatial_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 1, 1), dtype=np.float32))
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_summarize_experiment_rejects_multichannel_saliency_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            cache_path = root / "data/tiny_imagenet_train_saliency.npy"
            (root / config_path.parent).mkdir(parents=True)
            cache_path.parent.mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "method: saliencymix",
                        "output_dir: ./outputs",
                        "run_name: tiny_imagenet_preact_resnet18_saliencymix_xla4",
                        "saliency_source: batch",
                        "saliency_dir: ./data",
                    ]
                )
            )
            np.save(cache_path, np.zeros((3, 3, 64, 64), dtype=np.float32))
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path)

            with patch.dict("allthemix.cli.summarize.EXPECTED_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
                summary = summarize_experiment(root, spec)

        self.assertEqual(summary.prerequisite_status, "invalid_cache")

    def test_render_status_includes_missing_and_ok(self):
        rows = [
            ExperimentSummary(
                ExperimentSpec("Baseline", "ERM", "baseline", Path("baseline.yaml")),
                Path("outputs/baseline/metrics.csv"),
                0.4,
                "test_top1_error",
                "ok",
                final_test_checkpoint="best",
                final_test_checkpoint_source="memory",
            ),
            ExperimentSummary(
                ExperimentSpec("MixDA", "MixUp", "mixup", Path("mixup.yaml")),
                Path("outputs/mixup/metrics.csv"),
                None,
                "missing_file",
                "missing",
            ),
        ]

        status = render_status(rows)

        self.assertIn("| ERM | ok | ok | 40.00 |", status)
        self.assertIn("| ERM | ok | ok | 40.00 | test_top1_error | memory |", status)
        self.assertIn("| MixUp | missing | ok | -- |", status)

    def test_validate_tiny_xla4_protocol_accepts_current_configs(self):
        issues = validate_tiny_xla4_protocol(Path("."))

        self.assertEqual(render_protocol(issues), "tiny-imagenet-xla4 protocol: ok")

    def test_validate_tiny_xla4_protocol_reports_field_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            script_path = Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh")
            (root / config_path.parent).mkdir(parents=True)
            (root / script_path.parent).mkdir(parents=True)
            (root / script_path).write_text("#!/usr/bin/env bash\n")
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "model: preact_resnet18",
                        "method: baseline",
                        "batch_size: 64",
                        "global_batch_size: 128",
                        "epochs: 200",
                        "max_train_steps: -1",
                        "max_eval_steps: -1",
                        "validation_split: 0.1",
                        "eval_on_test_each_epoch: false",
                        "final_test: true",
                        "learning_rate: 0.1",
                        "momentum: 0.9",
                        "weight_decay: 0.0005",
                        "lr_schedule: step",
                        "min_learning_rate: 0.0",
                        "lr_decay_epochs: [150, 180]",
                        "lr_decay_rate: 0.1",
                        "basic_aug: true",
                        "save_csv: true",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        "run_name: tiny_imagenet_preact_resnet18_baseline_xla4",
                        "save_checkpoint: true",
                        "checkpoint_dir: ./checkpoints",
                        "save_best_only: true",
                        "distributed: true",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path, script_path)

            issues = validate_tiny_xla4_protocol(root, [spec])

        rendered = render_protocol(issues)
        self.assertIn(
            "| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | batch_size | 32 | 64 |",
            rendered,
        )

    def test_validate_tiny_xla4_protocol_rejects_stateful_resume_and_missing_timing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            script_path = Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh")
            (root / config_path.parent).mkdir(parents=True)
            (root / script_path.parent).mkdir(parents=True)
            config_text = (
                Path(config_path)
                .read_text()
                .replace('resume_checkpoint: ""', "resume_checkpoint: ./checkpoints/smoke/best.pt")
                .replace("log_time: true", "log_time: false")
            )
            (root / config_path).write_text(config_text)
            (root / script_path).write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "python -m allthemix.cli.train \\",
                        f"  --config {config_path.as_posix()}",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path, script_path)

            issues = validate_tiny_xla4_protocol(root, [spec])

        rendered = render_protocol(issues)
        self.assertIn(
            "| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | resume_checkpoint | '' | './checkpoints/smoke/best.pt' |",
            rendered,
        )
        self.assertIn(
            "| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | log_time | True | False |",
            rendered,
        )

    def test_validate_tiny_xla4_protocol_reports_script_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            script_path = Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh")
            (root / config_path.parent).mkdir(parents=True)
            (root / script_path.parent).mkdir(parents=True)
            (root / config_path).write_text(Path(config_path).read_text())
            (root / script_path).write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "python -m allthemix.cli.train \\",
                        "  --config configs/tiny_imagenet/preact_resnet18/mixup_xla4.yaml",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path, script_path)

            issues = validate_tiny_xla4_protocol(root, [spec])

        rendered = render_protocol(issues)
        self.assertIn("| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | script_config |", rendered)

    def test_validate_tiny_xla4_protocol_reports_missing_xla_script_guard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            script_path = Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh")
            (root / config_path.parent).mkdir(parents=True)
            (root / script_path.parent).mkdir(parents=True)
            (root / config_path).write_text(Path(config_path).read_text())
            (root / script_path).write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "python -m allthemix.cli.train \\",
                        f"  --config {config_path.as_posix()}",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path, script_path)

            issues = validate_tiny_xla4_protocol(root, [spec])

        rendered = render_protocol(issues)
        self.assertIn(
            "| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | script_tpu_env_guard | True | False |",
            rendered,
        )
        self.assertIn(
            "| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | script_pjrt_device_default | 'TPU' | 'missing' |",
            rendered,
        )

    def test_validate_tiny_xla4_protocol_rejects_saliencymix_gradient_cache_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            script_path = Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh")
            cache_script_path = Path("scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh")
            (root / config_path.parent).mkdir(parents=True)
            (root / script_path.parent).mkdir(parents=True)
            (root / cache_script_path.parent).mkdir(parents=True, exist_ok=True)
            (root / config_path).write_text(Path(config_path).read_text())
            (root / script_path).write_text(Path(script_path).read_text())
            (root / cache_script_path).write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        'source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"',
                        "python -m allthemix.cli.build_saliency_cache \\",
                        f"  --config {config_path.as_posix()} \\",
                        "  --method gradient --allow-gradient-fallback",
                    ]
                )
            )
            spec = ExperimentSpec("MixDA", "SaliencyMix", "saliencymix", config_path, script_path)

            issues = validate_tiny_xla4_protocol(root, [spec])

        rendered = render_protocol(issues)
        self.assertIn(
            "| saliencymix | configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml | cache_script_strict_opencv | 'no gradient fallback' | 'fallback enabled' |",
            rendered,
        )

    def test_validate_tiny_xla4_protocol_checks_resolved_recipe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            script_path = Path("scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh")
            (root / config_path.parent).mkdir(parents=True)
            (root / script_path.parent).mkdir(parents=True)
            config_text = Path(config_path).read_text().replace(
                "data_dir: ./data\n",
                "data_dir: ./data\nrecipe: official\n",
            )
            (root / config_path).write_text(config_text)
            (root / script_path).write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "python -m allthemix.cli.train \\",
                        f"  --config {config_path.as_posix()}",
                    ]
                )
            )
            spec = ExperimentSpec("Baseline", "ERM", "baseline", config_path, script_path)

            issues = validate_tiny_xla4_protocol(root, [spec])

        rendered = render_protocol(issues)
        self.assertIn("| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | resolved.recipe |", rendered)
        self.assertIn("| baseline | configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml | resolved.transform_profile |", rendered)


if __name__ == "__main__":
    unittest.main()
