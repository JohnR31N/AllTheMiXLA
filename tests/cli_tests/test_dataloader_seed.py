import argparse
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from allthemix.cli.train import derive_seed, make_data_loader_generator, parse_seed_arg, seed_worker, validate_seed


class DataLoaderSeedTests(unittest.TestCase):
    def test_make_data_loader_generator_is_reproducible(self):
        first = make_data_loader_generator(7, rank=2, offset=10)
        second = make_data_loader_generator(7, rank=2, offset=10)
        different_rank = make_data_loader_generator(7, rank=3, offset=10)

        self.assertTrue(
            torch.equal(
                torch.randperm(20, generator=first),
                torch.randperm(20, generator=second),
            )
        )
        self.assertFalse(
            torch.equal(
                torch.randperm(20, generator=make_data_loader_generator(7, rank=2, offset=10)),
                torch.randperm(20, generator=different_rank),
            )
        )

    def test_validate_seed_rejects_negative_and_too_large_values(self):
        self.assertEqual(validate_seed(0), 0)
        self.assertEqual(validate_seed(2**32 - 1), 2**32 - 1)

        with self.assertRaisesRegex(ValueError, "--seed"):
            validate_seed(-1)
        with self.assertRaisesRegex(ValueError, "--seed"):
            validate_seed(2**32)

    def test_parse_seed_arg_reports_argparse_errors(self):
        self.assertEqual(parse_seed_arg("7"), 7)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seed_arg("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seed_arg("not-an-int")

    def test_derive_seed_wraps_rank_offsets_into_numpy_range(self):
        self.assertEqual(derive_seed(2**32 - 1, rank=1, offset=1), 1_000_003)

    def test_seed_worker_seeds_python_and_numpy_from_torch_initial_seed(self):
        worker_seed = 123456789 % 2**32

        with patch("allthemix.cli.train.torch.initial_seed", return_value=123456789):
            seed_worker(4)
        python_value = random.random()
        numpy_value = float(np.random.random())

        random.seed(worker_seed)
        expected_python = random.random()
        np.random.seed(worker_seed)
        expected_numpy = float(np.random.random())

        self.assertEqual(python_value, expected_python)
        self.assertEqual(numpy_value, expected_numpy)


if __name__ == "__main__":
    unittest.main()
