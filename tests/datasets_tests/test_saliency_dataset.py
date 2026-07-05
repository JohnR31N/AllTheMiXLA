import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from allthemix.data.saliency_dataset import (
    SaliencyMapDataset,
    attach_train_saliency_maps,
    resolve_train_saliency_path,
)


class SaliencyDatasetTests(unittest.TestCase):
    def test_resolve_train_saliency_path_accepts_tiny_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tiny_imagenet_train_saliency.npy"
            np.save(path, np.zeros((2, 64, 64), dtype=np.float32))

            resolved = resolve_train_saliency_path("tinyimagenet", temp_dir)

        self.assertEqual(resolved.name, "tiny_imagenet_train_saliency.npy")

    def test_saliency_map_dataset_returns_image_label_saliency(self):
        base = TensorDataset(torch.zeros(2, 3, 8, 8), torch.arange(2))
        saliency_maps = np.ones((2, 4, 4), dtype=np.float32)

        dataset = SaliencyMapDataset(base, saliency_maps)
        image, label, saliency_map = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 8, 8))
        self.assertEqual(int(label), 0)
        self.assertEqual(tuple(saliency_map.shape), (1, 8, 8))

    def test_attach_train_saliency_maps_checks_length(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cifar10_train_saliency.npy"
            np.save(path, np.zeros((1, 8, 8), dtype=np.float32))
            base = TensorDataset(torch.zeros(2, 3, 8, 8), torch.arange(2))

            with self.assertRaisesRegex(ValueError, "Number of saliency maps"):
                attach_train_saliency_maps(base, "cifar10", temp_dir)


if __name__ == "__main__":
    unittest.main()
