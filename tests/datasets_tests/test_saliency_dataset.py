import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
from unittest.mock import patch

from allthemix.data.saliency_dataset import (
    SaliencyMapDataset,
    attach_train_saliency_maps,
    load_train_saliency_maps,
    resolve_train_saliency_path,
    saliency_array_is_finite,
    saliency_path_candidates,
)


class SaliencyDatasetTests(unittest.TestCase):
    def test_resolve_train_saliency_path_accepts_tiny_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"
            np.save(path, np.zeros((2, 64, 64), dtype=np.float32))

            resolved = resolve_train_saliency_path("tinyimagenet", temp_dir)

        self.assertEqual(resolved.name, "tiny_imagenet_train_saliency.npy")

    def test_saliency_path_candidates_preserve_requested_tiny_alias_first(self):
        candidates = saliency_path_candidates("tiny_imagenet", "./data")

        self.assertEqual(candidates[0], Path("data/tiny_imagenet_train_saliency.npy"))
        self.assertIn(Path("data/tinyimagenet_train_saliency.npy"), candidates)

    def test_resolve_train_saliency_path_accepts_canonical_cache_from_tiny_underscore_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tinyimagenet_train_saliency.npy"
            np.save(path, np.zeros((2, 64, 64), dtype=np.float32))

            resolved = resolve_train_saliency_path("tiny_imagenet", temp_dir)

        self.assertEqual(resolved.name, "tinyimagenet_train_saliency.npy")

    def test_saliency_map_dataset_returns_image_label_saliency(self):
        base = TensorDataset(torch.zeros(2, 3, 8, 8), torch.arange(2))
        saliency_maps = np.ones((2, 4, 4), dtype=np.float32)

        dataset = SaliencyMapDataset(base, saliency_maps)
        image, label, saliency_map = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 8, 8))
        self.assertEqual(int(label), 0)
        self.assertEqual(tuple(saliency_map.shape), (1, 8, 8))

    def test_saliency_map_dataset_applies_paired_horizontal_flip(self):
        image = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(1, 3, 2, 3)
        saliency_maps = np.arange(2 * 3, dtype=np.float32).reshape(1, 2, 3)
        base = TensorDataset(image, torch.zeros(1, dtype=torch.long))
        dataset = SaliencyMapDataset(base, saliency_maps, saliency_augmentation_recipe="hflip")

        with patch("torch.rand", return_value=torch.tensor(0.0)):
            flipped_image, _, flipped_saliency = dataset[0]

        torch.testing.assert_close(flipped_image, torch.flip(image[0], dims=(-1,)))
        torch.testing.assert_close(flipped_saliency, torch.flip(torch.from_numpy(saliency_maps[0]).unsqueeze(0), dims=(-1,)))

    def test_saliency_map_dataset_tiny_openmixup_resizes_pair(self):
        base = TensorDataset(torch.zeros(1, 3, 80, 72), torch.zeros(1, dtype=torch.long))
        saliency_maps = np.zeros((1, 80, 72), dtype=np.float32)
        dataset = SaliencyMapDataset(
            base,
            saliency_maps,
            saliency_augmentation_recipe="tiny_openmixup",
            image_size=64,
        )

        image, _, saliency_map = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(tuple(saliency_map.shape), (1, 64, 64))

    def test_saliency_map_dataset_normalizes_image_after_pair_path(self):
        base = TensorDataset(torch.full((1, 3, 4, 4), 0.5), torch.zeros(1, dtype=torch.long))
        saliency_maps = np.zeros((1, 4, 4), dtype=np.float32)
        dataset = SaliencyMapDataset(
            base,
            saliency_maps,
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.25, 0.25, 0.25),
        )

        image, _, _ = dataset[0]

        torch.testing.assert_close(image, torch.zeros_like(image))

    def test_attach_train_saliency_maps_checks_length(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cifar10_train_saliency.npy"
            np.save(path, np.zeros((1, 8, 8), dtype=np.float32))
            base = TensorDataset(torch.zeros(2, 3, 8, 8), torch.arange(2))

            with self.assertRaisesRegex(ValueError, "Number of saliency maps"):
                attach_train_saliency_maps(base, "cifar10", temp_dir)

    def test_load_train_saliency_maps_rejects_nonfinite_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"
            maps = np.zeros((2, 8, 8), dtype=np.float32)
            maps[0, 0, 0] = np.nan
            np.save(path, maps)

            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                load_train_saliency_maps("tinyimagenet", temp_dir)

    def test_load_train_saliency_maps_can_skip_full_finite_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"
            maps = np.zeros((2, 8, 8), dtype=np.float32)
            maps[0, 0, 0] = np.nan
            np.save(path, maps)

            saliency_maps = load_train_saliency_maps("tinyimagenet", temp_dir, validate_finite=False)
            try:
                self.assertIsInstance(saliency_maps, np.memmap)
                self.assertEqual(tuple(saliency_maps.shape), (2, 8, 8))
            finally:
                saliency_maps._mmap.close()

    def test_saliency_map_dataset_rejects_nonfinite_sample(self):
        base = TensorDataset(torch.zeros(2, 3, 8, 8), torch.arange(2))
        saliency_maps = np.zeros((2, 8, 8), dtype=np.float32)
        saliency_maps[1, 0, 0] = np.inf
        dataset = SaliencyMapDataset(base, saliency_maps)

        with self.assertRaisesRegex(ValueError, "index 1 contains NaN or infinite"):
            dataset[1]

    def test_saliency_array_is_finite_scans_in_chunks(self):
        maps = np.zeros((5, 2, 2), dtype=np.float32)
        self.assertTrue(saliency_array_is_finite(maps, chunk_size=2))

        maps[4, 0, 0] = np.nan
        self.assertFalse(saliency_array_is_finite(maps, chunk_size=2))

    def test_load_train_saliency_maps_uses_mmap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"
            np.save(path, np.zeros((2, 8, 8), dtype=np.float32))

            saliency_maps = load_train_saliency_maps("tinyimagenet", temp_dir)
            try:
                self.assertIsInstance(saliency_maps, np.memmap)
                self.assertEqual(tuple(saliency_maps.shape), (2, 8, 8))
            finally:
                saliency_maps._mmap.close()

    def test_saliency_map_dataset_copies_mmap_sample_as_float32_tensor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"
            np.save(path, np.ones((2, 4, 4), dtype=np.float16))
            saliency_maps = load_train_saliency_maps("tinyimagenet", temp_dir)
            try:
                base = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
                dataset = SaliencyMapDataset(base, saliency_maps)

                _, _, saliency_map = dataset[0]
            finally:
                saliency_maps._mmap.close()

        self.assertEqual(saliency_map.dtype, torch.float32)
        self.assertEqual(tuple(saliency_map.shape), (1, 4, 4))

    def test_subset_uses_original_indices_for_saliency_cache_alignment(self):
        base = TensorDataset(torch.zeros(4, 3, 2, 2), torch.arange(4))
        saliency_maps = np.stack([np.full((2, 2), index, dtype=np.float32) for index in range(4)])
        dataset = SaliencyMapDataset(base, saliency_maps)
        subset = Subset(dataset, [2, 0])

        _, label, saliency_map = subset[0]

        self.assertEqual(int(label), 2)
        self.assertTrue(torch.all(saliency_map == 2))


if __name__ == "__main__":
    unittest.main()
