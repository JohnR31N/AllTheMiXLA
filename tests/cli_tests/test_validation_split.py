import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from allthemix.cli.train import SequentialDistributedEvalSampler, apply_validation_split
from allthemix.data.saliency_dataset import SaliencyMapDataset


class ValidationSplitTests(unittest.TestCase):
    def test_sequential_distributed_eval_sampler_shards_without_padding_duplicates(self):
        dataset = TensorDataset(torch.arange(10), torch.arange(10))

        shards = [
            list(SequentialDistributedEvalSampler(dataset, num_replicas=4, rank=rank))
            for rank in range(4)
        ]

        self.assertEqual(shards, [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]])
        self.assertEqual([len(shard) for shard in shards], [3, 3, 2, 2])
        self.assertEqual(sorted(index for shard in shards for index in shard), list(range(10)))

    def test_sequential_distributed_eval_sampler_rejects_invalid_rank_or_replicas(self):
        dataset = TensorDataset(torch.arange(3), torch.arange(3))

        with self.assertRaisesRegex(ValueError, "num_replicas"):
            SequentialDistributedEvalSampler(dataset, num_replicas=0, rank=0)
        with self.assertRaisesRegex(ValueError, "rank"):
            SequentialDistributedEvalSampler(dataset, num_replicas=2, rank=2)

    def test_validation_split_uses_eval_train_view(self):
        train_set = TensorDataset(torch.arange(10), torch.arange(10))
        eval_train_set = TensorDataset(torch.arange(10) + 100, torch.arange(10))
        original_test_set = TensorDataset(torch.arange(4), torch.arange(4))
        preset = object()
        recipe = type("Recipe", (), {"transform_profile": "openmixup"})()
        config = {
            "validation_split": 0.2,
            "data_dir": "./data",
            "download": False,
        }

        with patch("allthemix.cli.train.build_datasets", return_value=(eval_train_set, object())) as build_datasets:
            split_train, split_val, test_set = apply_validation_split(
                train_set,
                original_test_set,
                preset,
                recipe,
                config,
                seed=0,
            )

        build_datasets.assert_called_once_with(
            preset,
            "openmixup",
            data_dir="./data",
            download=False,
            use_basic_augmentation=False,
            augmentation_recipe="none",
            normalize_train=True,
        )
        self.assertIs(split_train.dataset, train_set)
        self.assertIs(split_val.dataset, eval_train_set)
        self.assertIs(test_set, original_test_set)
        self.assertEqual(len(split_train), 8)
        self.assertEqual(len(split_val), 2)

    def test_validation_split_preserves_attached_saliency_original_indices(self):
        images = torch.zeros(10, 3, 4, 4)
        labels = torch.arange(10)
        saliency_maps = np.stack([np.full((4, 4), index, dtype=np.float32) for index in range(10)])
        train_set = SaliencyMapDataset(TensorDataset(images, labels), saliency_maps)
        eval_train_set = TensorDataset(torch.arange(10) + 100, labels)
        original_test_set = TensorDataset(torch.arange(4), torch.arange(4))
        preset = object()
        recipe = type("Recipe", (), {"transform_profile": "openmixup"})()
        config = {
            "validation_split": 0.2,
            "data_dir": "./data",
            "download": False,
        }

        with patch("allthemix.cli.train.build_datasets", return_value=(eval_train_set, object())):
            split_train, split_val, test_set = apply_validation_split(
                train_set,
                original_test_set,
                preset,
                recipe,
                config,
                seed=0,
            )

        self.assertIs(split_train.dataset, train_set)
        self.assertIs(split_val.dataset, eval_train_set)
        self.assertIs(test_set, original_test_set)
        for subset_position in range(len(split_train)):
            _, label, saliency_map = split_train[subset_position]
            self.assertTrue(torch.all(saliency_map == int(label)))
        val_sample = split_val[0]
        self.assertEqual(len(val_sample), 2)
        self.assertGreaterEqual(int(val_sample[0]), 100)

    def test_validation_split_disabled_keeps_original_val_set(self):
        train_set = TensorDataset(torch.arange(3), torch.arange(3))
        val_set = TensorDataset(torch.arange(2), torch.arange(2))

        split_train, split_val, test_set = apply_validation_split(
            train_set,
            val_set,
            object(),
            object(),
            {"validation_split": 0.0},
            seed=0,
        )

        self.assertIs(split_train, train_set)
        self.assertIs(split_val, val_set)
        self.assertIsNone(test_set)

    def test_validation_split_rejects_mismatched_eval_train_view_length(self):
        train_set = TensorDataset(torch.arange(10), torch.arange(10))
        eval_train_set = TensorDataset(torch.arange(9), torch.arange(9))
        preset = object()
        recipe = type("Recipe", (), {"transform_profile": "openmixup"})()
        config = {
            "validation_split": 0.2,
            "data_dir": "./data",
            "download": False,
        }

        with patch("allthemix.cli.train.build_datasets", return_value=(eval_train_set, object())):
            with self.assertRaisesRegex(ValueError, "same length"):
                apply_validation_split(
                    train_set,
                    TensorDataset(torch.arange(4), torch.arange(4)),
                    preset,
                    recipe,
                    config,
                    seed=0,
                )

    def test_validation_split_rejects_empty_train_subset(self):
        train_set = TensorDataset(torch.arange(1), torch.arange(1))
        eval_train_set = TensorDataset(torch.arange(1), torch.arange(1))
        preset = object()
        recipe = type("Recipe", (), {"transform_profile": "openmixup"})()
        config = {
            "validation_split": 0.5,
            "data_dir": "./data",
            "download": False,
        }

        with patch("allthemix.cli.train.build_datasets", return_value=(eval_train_set, object())):
            with self.assertRaisesRegex(ValueError, "leaves no training examples"):
                apply_validation_split(
                    train_set,
                    TensorDataset(torch.arange(1), torch.arange(1)),
                    preset,
                    recipe,
                    config,
                    seed=0,
                )


if __name__ == "__main__":
    unittest.main()
