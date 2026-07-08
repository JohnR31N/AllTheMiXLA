import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from allthemix.methods.fmix import FMix, fmix_cross_entropy, sample_mask

class FMixTests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)

    def test_sample_mask_shape_and_values(self):
        lam, mask = sample_mask(alpha=1.0, decay_power=3.0, shape=(32, 32))
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)
        self.assertEqual(mask.shape, (1, 32, 32))
        self.assertTrue(np.isin(mask, [0.0, 1.0]).all())

    def test_sample_mask_mean_tracks_official_lambda(self):
        lam, mask = sample_mask(alpha=1.0, decay_power=3.0, shape=(16, 16), max_soft=0.0)

        self.assertLessEqual(abs(float(mask.mean()) - lam), 1.0 / mask.size)

    def test_fmix_batch_and_loss(self):
        mixer = FMix(alpha=1.0, decay_power=3.0, size=(32, 32))
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        result = mixer(images, targets)

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(result.targets_a.shape, targets.shape)
        self.assertEqual(result.targets_b.shape, targets.shape)
        self.assertEqual(result.mask.shape, (1, 32, 32))
        torch.testing.assert_close(torch.as_tensor(result.lam), result.mask.mean())

        logits = torch.randn(4, 10, requires_grad=True)
        loss = fmix_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_fmix_lambda_matches_actual_mask_area(self):
        mixer = FMix(alpha=1.0, decay_power=3.0, size=(4, 4))
        images = torch.zeros(2, 3, 4, 4)
        targets = torch.tensor([0, 1])
        mask = np.zeros((1, 4, 4), dtype=np.float32)
        mask[:, :2, :] = 1.0

        with patch("allthemix.methods.fmix.sample_mask", return_value=(0.99, mask)):
            result = mixer(images, targets)

        torch.testing.assert_close(torch.as_tensor(result.lam), torch.tensor(0.5))
        torch.testing.assert_close(torch.as_tensor(result.lam), result.mask.mean())

    def test_fmix_cross_entropy_accepts_tensor_lambda(self):
        logits = torch.randn(2, 3, requires_grad=True)
        targets_a = torch.tensor([0, 1])
        targets_b = torch.tensor([1, 2])
        lam = torch.tensor(0.25)

        loss = fmix_cross_entropy(logits, targets_a, targets_b, lam)
        loss.backward()

        self.assertIsNotNone(logits.grad)

    def test_fmix_cross_entropy_matches_per_sample_tensor_lambda(self):
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.5, 0.5]], requires_grad=True)
        targets_a = torch.tensor([0, 1])
        targets_b = torch.tensor([2, 0])
        lam = torch.tensor([0.25, 0.75])

        loss = fmix_cross_entropy(logits, targets_a, targets_b, lam)
        expected = (
            F.cross_entropy(logits, targets_a, reduction="none") * lam
            + F.cross_entropy(logits, targets_b, reduction="none") * (1.0 - lam)
        ).mean()

        torch.testing.assert_close(loss, expected)

    def test_fmix_cross_entropy_reformulate_uses_original_targets(self):
        logits = torch.tensor([[0.0, 2.0], [1.0, -1.0]], requires_grad=True)
        targets_a = torch.tensor([1, 0])
        targets_b = torch.tensor([0, 1])

        loss = fmix_cross_entropy(logits, targets_a, targets_b, lam=torch.tensor(0.0), reformulate=True)

        torch.testing.assert_close(loss, F.cross_entropy(logits, targets_a))

    def test_uses_external_partner_batch(self):
        mixer = FMix(alpha=1.0, decay_power=3.0, size=(4, 4))
        images = torch.zeros(2, 3, 4, 4)
        partners = torch.ones(2, 3, 4, 4)
        targets = torch.tensor([0, 1])
        partner_targets = torch.tensor([2, 3])
        index = torch.tensor([5, 6])

        result = mixer(images, targets, partners, partner_targets, index)

        self.assertEqual(result.images.shape, images.shape)
        self.assertTrue(torch.equal(result.targets_b, partner_targets))
        self.assertTrue(torch.equal(result.index, index))

    def test_rejects_mismatched_targets_and_partners(self):
        mixer = FMix(alpha=1.0, decay_power=3.0, size=(4, 4))
        images = torch.zeros(2, 3, 4, 4)

        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            mixer(images, torch.tensor([0]))

        with self.assertRaisesRegex(ValueError, "partner images must match"):
            mixer(
                images,
                torch.tensor([0, 1]),
                partner_images=torch.ones(2, 3, 5, 4),
                partner_targets=torch.tensor([2, 3]),
                index=torch.tensor([0, 1]),
            )

    def test_no_repeat_avoids_self_pairs(self):
        mixer = FMix(alpha=1.0, decay_power=3.0, size=(4, 4), no_repeat=True)
        images = torch.randn(8, 3, 4, 4)
        targets = torch.arange(8)

        result = mixer(images, targets)

        self.assertTrue(torch.all(result.index.cpu() != torch.arange(8)))

    def test_no_repeat_rejects_singleton_batch(self):
        mixer = FMix(alpha=1.0, decay_power=3.0, size=(4, 4), no_repeat=True)
        images = torch.randn(1, 3, 4, 4)
        targets = torch.tensor([0])

        with self.assertRaisesRegex(ValueError, "no_repeat"):
            mixer(images, targets)


if __name__ == "__main__":
    unittest.main()
