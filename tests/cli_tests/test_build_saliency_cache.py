import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
from io import StringIO
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from allthemix.cli.build_saliency_cache import build_saliency_maps_from_dataset, main, output_path_from_args, parse_args
from allthemix.cli.build_saliency_cache import (
    CACHE_BUILDER_VERSION,
    blur_kernel_from_config,
    validate_saliency_cache_dataset_length,
    _backup_cache_path,
    _compute_maps,
    _existing_cache_matches_request,
    _is_suspicious_saliency_map,
    _numpy_gradient_saliency_map,
    _temporary_cache_path,
    _tensor_to_unit_images,
    _tensor_to_uint8_images,
)
from allthemix.cli.train import make_data_loader_generator, seed_worker


class BuildSaliencyCacheTests(unittest.TestCase):
    def _metadata_context(self):
        return {
            "dataset": "tinyimagenet",
            "recipe": "openmixup",
            "transform_profile": "openmixup",
            "image_size": 64,
            "base_transform": "tensor_normalize_only",
        }

    def test_build_saliency_maps_from_dataset_writes_nhw_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(3, 3, 8, 8), torch.arange(3))
            output_path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"

            with redirect_stdout(StringIO()):
                saved_path = build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 1, images.size(-2), images.size(-1)),
                    metadata_context=self._metadata_context(),
                )

            saliency_maps = np.load(saved_path)
            metadata = json.loads(saved_path.with_suffix(saved_path.suffix + ".json").read_text())

        self.assertEqual(tuple(saliency_maps.shape), (3, 8, 8))
        self.assertEqual(saliency_maps.dtype, np.float32)
        self.assertTrue(np.allclose(saliency_maps, 1.0))
        self.assertEqual(int(metadata["builder_version"]), CACHE_BUILDER_VERSION)
        self.assertEqual(metadata["dataset"], "tinyimagenet")
        self.assertEqual(metadata["recipe"], "openmixup")
        self.assertEqual(metadata["transform_profile"], "openmixup")
        self.assertEqual(metadata["image_size"], 64)
        self.assertEqual(metadata["base_transform"], "tensor_normalize_only")
        self.assertEqual(metadata["blur_kernel"], 7)
        self.assertTrue(metadata["raw_unit_images"])
        self.assertTrue(metadata["minmax_normalized"])

    def test_existing_current_cache_is_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                        "allow_gradient_fallback": False,
                    }
                )
            )

            with redirect_stdout(StringIO()):
                saved_path = build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=False,
                    log_interval=0,
                    saliency_fn=lambda images: torch.zeros(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

        self.assertEqual(saved_path, output_path)

    def test_existing_current_cache_reuse_does_not_iterate_dataset(self):
        class ExplodingDataset:
            def __len__(self):
                return 3

            def __getitem__(self, index):
                raise AssertionError(f"cache reuse should not read dataset index {index}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                        "allow_gradient_fallback": False,
                    }
                )
            )

            with redirect_stdout(StringIO()):
                saved_path = build_saliency_maps_from_dataset(
                    ExplodingDataset(),
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=False,
                    log_interval=0,
                    saliency_fn=lambda images: torch.zeros(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

        self.assertEqual(saved_path, output_path)

    def test_existing_short_cache_requires_overwrite_for_full_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(5, 3, 4, 4), torch.arange(5))
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                        "allow_gradient_fallback": False,
                    }
                )
            )

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=False,
                    log_interval=0,
                    saliency_fn=lambda images: torch.zeros(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

    def test_existing_spectral_residual_cache_requires_matching_blur_kernel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "spectral_residual",
                        "blur_kernel": 5,
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                        "allow_gradient_fallback": False,
                    }
                )
            )

            matches = _existing_cache_matches_request(
                output_path,
                method="spectral_residual",
                dtype="float32",
                metadata_context={**self._metadata_context(), "blur_kernel": 7},
            )

        self.assertFalse(matches)

    def test_existing_cache_requires_actual_array_dtype_to_match_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float16))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                        "allow_gradient_fallback": False,
                    }
                )
            )

            matches = _existing_cache_matches_request(
                output_path,
                method="opencv",
                dtype="float32",
                metadata_context=self._metadata_context(),
                allow_gradient_fallback=False,
            )

        self.assertFalse(matches)

    def test_existing_cache_check_closes_mmap_after_reuse_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                        "allow_gradient_fallback": False,
                    }
                )
            )

            matches = _existing_cache_matches_request(
                output_path,
                method="opencv",
                dtype="float32",
                metadata_context=self._metadata_context(),
                allow_gradient_fallback=False,
            )

            self.assertTrue(matches)
            output_path.unlink()
            self.assertFalse(output_path.exists())

    def test_existing_stale_cache_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION - 1,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                    }
                )
            )

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=False,
                    log_interval=0,
                    saliency_fn=lambda images: torch.zeros(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

    def test_existing_gradient_fallback_cache_requires_overwrite_for_strict_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                        "allow_gradient_fallback": True,
                    }
                )
            )

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=False,
                    log_interval=0,
                    saliency_fn=lambda images: torch.zeros(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

    def test_existing_missing_gradient_fallback_policy_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))
            output_path = Path(temp_dir) / "cache.npy"
            np.save(output_path, np.ones((3, 4, 4), dtype=np.float32))
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                    }
                )
            )

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=False,
                    log_interval=0,
                    saliency_fn=lambda images: torch.zeros(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

    def test_existing_nonfinite_cache_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))
            output_path = Path(temp_dir) / "cache.npy"
            saliency_maps = np.ones((3, 4, 4), dtype=np.float32)
            saliency_maps[0, 0, 0] = np.nan
            np.save(output_path, saliency_maps)
            output_path.with_suffix(output_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        **self._metadata_context(),
                        "builder_version": CACHE_BUILDER_VERSION,
                        "method": "opencv",
                        "count": 3,
                        "shape": [3, 4, 4],
                        "dtype": "float32",
                        "raw_unit_images": True,
                        "minmax_normalized": True,
                    }
                )
            )

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=False,
                    log_interval=0,
                    saliency_fn=lambda images: torch.zeros(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

    def test_build_saliency_maps_honors_limit_and_float16(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(5, 3, 4, 4), torch.arange(5))
            output_path = Path(temp_dir) / "cache.npy"

            with redirect_stdout(StringIO()):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=4,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    limit=3,
                    dtype="float16",
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                )

            saliency_maps = np.load(output_path)

        self.assertEqual(tuple(saliency_maps.shape), (3, 4, 4))
        self.assertEqual(saliency_maps.dtype, np.float16)

    def test_build_saliency_maps_atomic_write_does_not_append_numpy_suffix_to_temp_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache"

            with redirect_stdout(StringIO()):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

            saliency_maps = np.load(output_path)
            metadata_path = output_path.with_suffix(output_path.suffix + ".json")
            temporary_paths = list(Path(temp_dir).glob("*.tmp*"))
            metadata_exists = metadata_path.exists()
            appended_numpy_path_exists = Path(str(output_path) + ".npy").exists()

        self.assertEqual(tuple(saliency_maps.shape), (2, 4, 4))
        self.assertTrue(metadata_exists)
        self.assertEqual(temporary_paths, [])
        self.assertFalse(appended_numpy_path_exists)

    def test_build_saliency_maps_keeps_existing_cache_when_atomic_metadata_write_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"
            metadata_path = output_path.with_suffix(output_path.suffix + ".json")
            old_maps = np.full((2, 4, 4), 3.0, dtype=np.float32)
            np.save(output_path, old_maps)
            metadata_path.write_text(json.dumps({"old": True}) + "\n")

            original_write_text = Path.write_text

            def fail_temp_metadata_write(path, *args, **kwargs):
                if Path(path).name == _temporary_cache_path(metadata_path).name:
                    raise OSError("simulated metadata write failure")
                return original_write_text(path, *args, **kwargs)

            with (
                patch.object(Path, "write_text", fail_temp_metadata_write),
                self.assertRaisesRegex(OSError, "simulated metadata write failure"),
                redirect_stdout(StringIO()),
            ):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

            reloaded_maps = np.load(output_path)
            metadata_text = metadata_path.read_text()
            temp_paths = [
                _temporary_cache_path(output_path),
                _temporary_cache_path(metadata_path),
            ]

        np.testing.assert_array_equal(reloaded_maps, old_maps)
        self.assertEqual(json.loads(metadata_text), {"old": True})
        for temp_path in temp_paths:
            self.assertFalse(temp_path.exists())

    def test_build_saliency_maps_rolls_back_existing_cache_when_atomic_metadata_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"
            metadata_path = output_path.with_suffix(output_path.suffix + ".json")
            old_maps = np.full((2, 4, 4), 5.0, dtype=np.float32)
            np.save(output_path, old_maps)
            metadata_path.write_text(json.dumps({"old": True}) + "\n")

            original_replace = Path.replace
            temp_metadata_path = _temporary_cache_path(metadata_path)

            def fail_temp_metadata_replace(path, target):
                if Path(path).name == temp_metadata_path.name and Path(target) == metadata_path:
                    raise OSError("simulated metadata replace failure")
                return original_replace(path, target)

            with (
                patch.object(Path, "replace", fail_temp_metadata_replace),
                self.assertRaisesRegex(OSError, "simulated metadata replace failure"),
                redirect_stdout(StringIO()),
            ):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

            reloaded_maps = np.load(output_path)
            metadata_text = metadata_path.read_text()
            temp_paths = [
                _temporary_cache_path(output_path),
                _temporary_cache_path(metadata_path),
                _backup_cache_path(output_path),
                _backup_cache_path(metadata_path),
            ]

        np.testing.assert_array_equal(reloaded_maps, old_maps)
        self.assertEqual(json.loads(metadata_text), {"old": True})
        for temp_path in temp_paths:
            self.assertFalse(temp_path.exists())

    def test_build_saliency_maps_recovers_interrupted_cache_backup_before_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"
            metadata_path = output_path.with_suffix(output_path.suffix + ".json")
            backup_data_path = _backup_cache_path(output_path)
            backup_metadata_path = _backup_cache_path(metadata_path)
            old_maps = np.full((2, 4, 4), 7.0, dtype=np.float32)
            with backup_data_path.open("wb") as handle:
                np.save(handle, old_maps)
            backup_metadata_path.write_text(json.dumps({"old": True}) + "\n")

            original_write_text = Path.write_text

            def fail_temp_metadata_write(path, *args, **kwargs):
                if Path(path).name == _temporary_cache_path(metadata_path).name:
                    raise OSError("simulated metadata write failure after backup recovery")
                return original_write_text(path, *args, **kwargs)

            with (
                patch.object(Path, "write_text", fail_temp_metadata_write),
                self.assertRaisesRegex(OSError, "backup recovery"),
                redirect_stdout(StringIO()),
            ):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

            reloaded_maps = np.load(output_path)
            metadata_text = metadata_path.read_text()
            temp_paths = [
                _temporary_cache_path(output_path),
                _temporary_cache_path(metadata_path),
                backup_data_path,
                backup_metadata_path,
            ]

        np.testing.assert_array_equal(reloaded_maps, old_maps)
        self.assertEqual(json.loads(metadata_text), {"old": True})
        for temp_path in temp_paths:
            self.assertFalse(temp_path.exists())

    def test_build_saliency_maps_recovers_partial_data_backup_before_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"
            metadata_path = output_path.with_suffix(output_path.suffix + ".json")
            backup_data_path = _backup_cache_path(output_path)
            old_maps = np.full((2, 4, 4), 9.0, dtype=np.float32)
            with backup_data_path.open("wb") as handle:
                np.save(handle, old_maps)
            metadata_path.write_text(json.dumps({"old": True}) + "\n")

            original_write_text = Path.write_text

            def fail_temp_metadata_write(path, *args, **kwargs):
                if Path(path).name == _temporary_cache_path(metadata_path).name:
                    raise OSError("simulated metadata write failure after partial data backup recovery")
                return original_write_text(path, *args, **kwargs)

            with (
                patch.object(Path, "write_text", fail_temp_metadata_write),
                self.assertRaisesRegex(OSError, "partial data backup recovery"),
                redirect_stdout(StringIO()),
            ):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

            reloaded_maps = np.load(output_path)
            metadata_text = metadata_path.read_text()
            temp_paths = [
                _temporary_cache_path(output_path),
                _temporary_cache_path(metadata_path),
                backup_data_path,
                _backup_cache_path(metadata_path),
            ]

        np.testing.assert_array_equal(reloaded_maps, old_maps)
        self.assertEqual(json.loads(metadata_text), {"old": True})
        for temp_path in temp_paths:
            self.assertFalse(temp_path.exists())

    def test_build_saliency_maps_recovers_partial_metadata_backup_before_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"
            metadata_path = output_path.with_suffix(output_path.suffix + ".json")
            backup_metadata_path = _backup_cache_path(metadata_path)
            old_maps = np.full((2, 4, 4), 11.0, dtype=np.float32)
            np.save(output_path, old_maps)
            backup_metadata_path.write_text(json.dumps({"old": True}) + "\n")

            original_write_text = Path.write_text

            def fail_temp_metadata_write(path, *args, **kwargs):
                if Path(path).name == _temporary_cache_path(metadata_path).name:
                    raise OSError("simulated metadata write failure after partial metadata backup recovery")
                return original_write_text(path, *args, **kwargs)

            with (
                patch.object(Path, "write_text", fail_temp_metadata_write),
                self.assertRaisesRegex(OSError, "partial metadata backup recovery"),
                redirect_stdout(StringIO()),
            ):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=0,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                    metadata_context=self._metadata_context(),
                )

            reloaded_maps = np.load(output_path)
            metadata_text = metadata_path.read_text()
            temp_paths = [
                _temporary_cache_path(output_path),
                _temporary_cache_path(metadata_path),
                _backup_cache_path(output_path),
                backup_metadata_path,
            ]

        np.testing.assert_array_equal(reloaded_maps, old_maps)
        self.assertEqual(json.loads(metadata_text), {"old": True})
        for temp_path in temp_paths:
            self.assertFalse(temp_path.exists())

    def test_build_saliency_maps_configures_seeded_dataloader(self):
        captured_kwargs = {}

        class CapturingDataLoader:
            def __init__(self, dataset, **kwargs):
                del dataset
                captured_kwargs.update(kwargs)

            def __iter__(self):
                return iter([(torch.zeros(2, 3, 4, 4), torch.arange(2))])

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"

            with (
                patch("allthemix.cli.build_saliency_cache.DataLoader", CapturingDataLoader),
                redirect_stdout(StringIO()),
            ):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    num_workers=1,
                    seed=7,
                    device="cpu",
                    overwrite=True,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                )

        self.assertIs(captured_kwargs["worker_init_fn"], seed_worker)
        self.assertTrue(
            torch.equal(
                torch.randperm(20, generator=captured_kwargs["generator"]),
                torch.randperm(20, generator=make_data_loader_generator(7, offset=30_000)),
            )
        )

    def test_build_saliency_maps_rejects_invalid_limits_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"

            with self.assertRaisesRegex(ValueError, "limit must be positive"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    limit=0,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                )

        self.assertFalse(output_path.exists())

    def test_build_saliency_maps_rejects_invalid_batch_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))

            with self.assertRaisesRegex(ValueError, "batch_size"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=Path(temp_dir) / "cache.npy",
                    batch_size=0,
                    log_interval=0,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                )

    def test_parse_args_rejects_even_blur_kernel(self):
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parse_args(["--blur-kernel", "4"])

    def test_blur_kernel_defaults_to_config_when_cli_is_omitted(self):
        self.assertEqual(blur_kernel_from_config({"guidedmixup_blur_kernel": 9}, None), 9)
        self.assertEqual(blur_kernel_from_config({"guidedmixup_blur_kernel": 9}, 7), 7)

        with self.assertRaisesRegex(ValueError, "positive odd"):
            blur_kernel_from_config({"guidedmixup_blur_kernel": 4}, None)

    def test_validate_saliency_cache_dataset_length_rejects_incomplete_tiny_train_split(self):
        dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))

        with self.assertRaisesRegex(ValueError, "complete train split"):
            validate_saliency_cache_dataset_length("tiny_imagenet", dataset)

    def test_validate_saliency_cache_dataset_length_allows_matching_tiny_train_split(self):
        dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))

        with patch.dict("allthemix.cli.build_saliency_cache.EXPECTED_CACHE_TRAIN_EXAMPLES", {"tinyimagenet": 3}):
            validate_saliency_cache_dataset_length("tinyimagenet", dataset)

    def test_validate_saliency_cache_dataset_length_skips_datasets_without_fixed_cache_count(self):
        dataset = TensorDataset(torch.zeros(3, 3, 4, 4), torch.arange(3))

        validate_saliency_cache_dataset_length("imagenet_a", dataset)

    def test_parse_args_rejects_invalid_runtime_counts(self):
        for args in [
            ["--batch-size", "0"],
            ["--num-workers", "-1"],
            ["--limit", "0"],
            ["--log-interval", "-1"],
        ]:
            with self.subTest(args=args), self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                parse_args(args)

    def test_parse_args_defaults_to_formal_opencv_cache_without_fallback(self):
        args = parse_args([])

        self.assertEqual(args.method, "opencv")
        self.assertFalse(args.allow_gradient_fallback)

    def test_parse_args_accepts_saliency_path_alias_for_output(self):
        args = parse_args(["--saliency-path", "/mnt/cache/maps.npy", "--allow-gradient-fallback"])

        self.assertEqual(args.output, "/mnt/cache/maps.npy")
        self.assertTrue(args.allow_gradient_fallback)

    def test_output_path_from_args_relocates_relative_explicit_output_with_saliency_dir_override(self):
        args = parse_args(["--saliency-dir", "/mnt/cache", "--saliency-path", "maps.npy"])

        path = output_path_from_args(args, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(path, Path("/mnt/cache/maps.npy"))

    def test_build_saliency_maps_records_gradient_fallback_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(1, 3, 4, 4), torch.zeros(1, dtype=torch.long))
            output_path = Path(temp_dir) / "cache.npy"

            with redirect_stdout(StringIO()):
                saved_path = build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=1,
                    log_interval=0,
                    allow_gradient_fallback=True,
                    saliency_fn=lambda images: torch.ones(images.size(0), 4, 4),
                )

            metadata = json.loads(saved_path.with_suffix(saved_path.suffix + ".json").read_text())

        self.assertTrue(metadata["allow_gradient_fallback"])

    def test_build_saliency_maps_rejects_nonfinite_outputs_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
            output_path = Path(temp_dir) / "cache.npy"

            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                build_saliency_maps_from_dataset(
                    dataset,
                    output_path=output_path,
                    batch_size=2,
                    log_interval=0,
                    saliency_fn=lambda images: torch.full((images.size(0), 4, 4), float("nan")),
                )

        self.assertFalse(output_path.exists())

    def test_output_path_from_args_preserves_tiny_underscore_config_name(self):
        args = type(
            "Args",
            (),
            {
                "output": None,
                "dataset": "tiny_imagenet",
                "saliency_dir": None,
                "data_dir": "./data",
            },
        )()

        path = output_path_from_args(args, {"dataset": "tiny_imagenet", "data_dir": "./data"})

        self.assertEqual(path, Path("data/tiny_imagenet_train_saliency.npy"))

    def test_output_path_from_args_relocates_relative_saliency_path_with_saliency_dir_override(self):
        args = type(
            "Args",
            (),
            {
                "output": None,
                "dataset": "tiny_imagenet",
                "saliency_dir": "/mnt/cache",
                "data_dir": "./data",
            },
        )()

        path = output_path_from_args(
            args,
            {
                "dataset": "tiny_imagenet",
                "data_dir": "./data",
                "saliency_path": "./data/tiny_imagenet_train_guided_sr_saliency.npy",
            },
        )

        self.assertEqual(path, Path("/mnt/cache/tiny_imagenet_train_guided_sr_saliency.npy"))

    def test_output_path_from_args_uses_data_dir_override_for_default_saliency_dir(self):
        args = type(
            "Args",
            (),
            {
                "output": None,
                "dataset": "tiny_imagenet",
                "saliency_dir": None,
                "data_dir": "/mnt/tiny",
            },
        )()

        path = output_path_from_args(
            args,
            {
                "dataset": "tiny_imagenet",
                "data_dir": "./data",
                "saliency_dir": "./data",
            },
        )

        self.assertEqual(path, Path("/mnt/tiny/tiny_imagenet_train_saliency.npy"))

    def test_output_path_from_args_relocates_relative_config_saliency_path_with_data_dir_override(self):
        args = type(
            "Args",
            (),
            {
                "output": None,
                "dataset": "tiny_imagenet",
                "saliency_dir": None,
                "data_dir": "/mnt/tiny",
            },
        )()

        path = output_path_from_args(
            args,
            {
                "dataset": "tiny_imagenet",
                "data_dir": "./data",
                "saliency_path": "./data/tiny_imagenet_train_guided_sr_saliency.npy",
            },
        )

        self.assertEqual(path, Path("/mnt/tiny/tiny_imagenet_train_guided_sr_saliency.npy"))

    def test_tensor_to_uint8_images_denormalizes_with_dataset_stats(self):
        images = torch.zeros(1, 3, 2, 2)

        uint8_images = _tensor_to_uint8_images(images, mean=(0.5, 0.25, 0.0), std=(0.1, 0.1, 0.1))

        self.assertEqual(tuple(uint8_images.shape), (1, 2, 2, 3))
        self.assertEqual(int(uint8_images[0, 0, 0, 0]), 128)
        self.assertEqual(int(uint8_images[0, 0, 0, 1]), 64)
        self.assertEqual(int(uint8_images[0, 0, 0, 2]), 0)

    def test_tensor_to_unit_images_denormalizes_on_original_device(self):
        images = torch.zeros(1, 3, 2, 2)

        unit_images = _tensor_to_unit_images(images, mean=(0.5, 0.25, 0.0), std=(0.1, 0.1, 0.1))

        self.assertEqual(unit_images.device, images.device)
        torch.testing.assert_close(unit_images[0, :, 0, 0], torch.tensor([0.5, 0.25, 0.0]))

    def test_opencv_cache_method_rejects_missing_cv2_by_default(self):
        images = torch.zeros(2, 3, 8, 8)

        with patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "OpenCV saliency backend is not installed"):
                _compute_maps(images, method="opencv", blur_kernel=7, mean=None, std=None)

    def test_opencv_cache_method_falls_back_to_gradient_only_when_allowed(self):
        images = torch.zeros(2, 3, 8, 8)
        images[:, :, :, 4:] = 1.0

        with patch.dict("sys.modules", {"cv2": None}):
            maps = _compute_maps(
                images,
                method="opencv",
                blur_kernel=7,
                mean=None,
                std=None,
                allow_gradient_fallback=True,
            )
        expected = torch.from_numpy(
            np.stack([_numpy_gradient_saliency_map(np.asarray(image)) for image in _tensor_to_uint8_images(images, None, None)])[
                :, None
            ]
        )

        self.assertEqual(tuple(maps.shape), (2, 1, 8, 8))
        torch.testing.assert_close(maps.cpu(), expected.float())

    def test_spectral_residual_cache_uses_denormalized_unit_images_and_normalizes_maps(self):
        images = torch.zeros(2, 3, 4, 4)
        captured = {}

        def fake_spectral_residual(unit_images, blur_kernel):
            captured["mean"] = float(unit_images.mean())
            base = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
            return base.repeat(unit_images.size(0), 1, 1, 1) + 2.0

        with patch("allthemix.cli.build_saliency_cache.compute_spectral_residual_saliency_maps", fake_spectral_residual):
            maps = _compute_maps(
                images,
                method="spectral_residual",
                blur_kernel=7,
                mean=(0.5, 0.25, 0.0),
                std=(0.1, 0.1, 0.1),
            )

        self.assertAlmostEqual(captured["mean"], 0.25)
        self.assertEqual(float(maps.min()), 0.0)
        self.assertEqual(float(maps.max()), 1.0)

    def test_suspicious_saliency_rejects_constant_maps(self):
        self.assertTrue(_is_suspicious_saliency_map(np.zeros((4, 4), dtype=np.float32)))

    def test_main_builds_cache_from_explicit_no_aug_dataset_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": "tiny_imagenet",
                        "recipe": "openmixup",
                        "data_dir": "./data",
                    }
                )
            )
            fake_train_set = TensorDataset(torch.zeros(1, 3, 64, 64), torch.zeros(1, dtype=torch.long))

            with (
                patch.object(sys, "argv", ["build_saliency_cache", "--config", str(config_path), "--num-workers", "0"]),
                patch(
                    "allthemix.cli.build_saliency_cache.build_datasets",
                    return_value=(fake_train_set, object()),
                ) as build_datasets,
                patch.dict("allthemix.cli.build_saliency_cache.EXPECTED_CACHE_TRAIN_EXAMPLES", {"tinyimagenet": 1}),
                patch("allthemix.cli.build_saliency_cache.build_saliency_maps_from_dataset"),
            ):
                main()

        build_datasets.assert_called_once()
        _, kwargs = build_datasets.call_args
        self.assertFalse(kwargs["use_basic_augmentation"])
        self.assertEqual(kwargs["augmentation_recipe"], "none")

    def test_main_forwards_data_dir_and_recipe_overrides_to_dataset_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": "tiny_imagenet",
                        "recipe": "official",
                        "data_dir": "./data",
                    }
                )
            )
            fake_train_set = TensorDataset(torch.zeros(1, 3, 64, 64), torch.zeros(1, dtype=torch.long))

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "build_saliency_cache",
                        "--config",
                        str(config_path),
                        "--recipe",
                        "openmixup",
                        "--data-dir",
                        "/mnt/tiny",
                        "--blur-kernel",
                        "9",
                        "--seed",
                        "7",
                        "--num-workers",
                        "0",
                    ],
                ),
                patch(
                    "allthemix.cli.build_saliency_cache.build_datasets",
                    return_value=(fake_train_set, object()),
                ) as build_datasets,
                patch.dict("allthemix.cli.build_saliency_cache.EXPECTED_CACHE_TRAIN_EXAMPLES", {"tinyimagenet": 1}),
                patch("allthemix.cli.build_saliency_cache.build_saliency_maps_from_dataset") as build_cache,
            ):
                main()

        _, dataset_kwargs = build_datasets.call_args
        self.assertEqual(dataset_kwargs["data_dir"], "/mnt/tiny")
        build_args, build_kwargs = build_cache.call_args
        self.assertIs(build_args[0], fake_train_set)
        self.assertEqual(build_kwargs["seed"], 7)
        self.assertEqual(build_kwargs["blur_kernel"], 9)
        self.assertEqual(build_kwargs["metadata_context"]["blur_kernel"], 9)
        self.assertEqual(build_kwargs["metadata_context"]["recipe"], "openmixup")
        self.assertEqual(build_kwargs["metadata_context"]["transform_profile"], "openmixup")


if __name__ == "__main__":
    unittest.main()
