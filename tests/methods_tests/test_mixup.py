import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from allthemix.methods.mixup import MixUp, mixup_cross_entropy, sample_lam


class MixUpTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        torch.manual_seed(7)

    def test_sample_lam_range(self):
        lam = sample_lam(alpha=1.0)

        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)

    def test_mixup_batch_and_loss(self):
        mixer = MixUp(alpha=1.0)
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        result = mixer(images, targets)

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(result.targets_a.shape, targets.shape)
        self.assertEqual(result.targets_b.shape, targets.shape)
        self.assertEqual(result.index.shape, targets.shape)

        logits = torch.randn(4, 10, requires_grad=True)
        loss = mixup_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_alpha_zero_keeps_original_images(self):
        mixer = MixUp(alpha=0.0)
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        result = mixer(images, targets)

        self.assertTrue(torch.equal(result.images, images))
        self.assertEqual(result.lam, 1.0)

    def test_uses_external_partner_batch(self):
        mixer = MixUp(alpha=0.0)
        images = torch.zeros(2, 3, 4, 4)
        partners = torch.ones(2, 3, 4, 4)
        targets = torch.tensor([0, 1])
        partner_targets = torch.tensor([2, 3])
        index = torch.tensor([5, 6])

        result = mixer(images, targets, partners, partner_targets, index)

        self.assertTrue(torch.equal(result.targets_b, partner_targets))
        self.assertTrue(torch.equal(result.index, index))

    def test_mixup_images_match_convex_combination_with_external_partners(self):
        mixer = MixUp(alpha=1.0)
        images = torch.zeros(2, 1, 2, 2)
        images[0].fill_(1.0)
        images[1].fill_(2.0)
        partner_images = torch.zeros_like(images)
        partner_images[0].fill_(5.0)
        partner_images[1].fill_(9.0)
        targets = torch.tensor([0, 1])
        partner_targets = torch.tensor([10, 11])

        with patch("allthemix.methods.mixup.sample_lam", return_value=0.25):
            result = mixer(
                images,
                targets,
                partner_images=partner_images,
                partner_targets=partner_targets,
                index=torch.tensor([7, 8]),
            )

        expected_images = 0.25 * images + 0.75 * partner_images
        torch.testing.assert_close(result.images, expected_images)
        self.assertEqual(result.lam, 0.25)
        torch.testing.assert_close(result.targets_b, partner_targets)

    def test_mixup_cross_entropy_matches_per_sample_tensor_lambda(self):
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.5, 0.5]], requires_grad=True)
        targets_a = torch.tensor([0, 1])
        targets_b = torch.tensor([2, 0])
        lam = torch.tensor([0.25, 0.75])

        loss = mixup_cross_entropy(logits, targets_a, targets_b, lam)
        expected = (
            F.cross_entropy(logits, targets_a, reduction="none") * lam
            + F.cross_entropy(logits, targets_b, reduction="none") * (1.0 - lam)
        ).mean()

        torch.testing.assert_close(loss, expected)

    def test_rejects_mismatched_targets_and_partners(self):
        mixer = MixUp(alpha=1.0)
        images = torch.zeros(2, 3, 4, 4)
        targets = torch.tensor([0])

        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            mixer(images, targets)

        with self.assertRaisesRegex(ValueError, "partner images must match"):
            mixer(
                images,
                torch.tensor([0, 1]),
                partner_images=torch.ones(3, 3, 4, 4),
                partner_targets=torch.tensor([2, 3, 4]),
                index=torch.tensor([0, 1]),
            )

        with self.assertRaisesRegex(ValueError, "partner index"):
            mixer(
                images,
                torch.tensor([0, 1]),
                partner_images=torch.ones_like(images),
                partner_targets=torch.tensor([2, 3]),
                index=torch.tensor([[0, 1]]),
            )

    def test_no_repeat_avoids_self_pairs(self):
        mixer = MixUp(alpha=1.0, no_repeat=True)
        images = torch.randn(8, 3, 4, 4)
        targets = torch.arange(8)

        result = mixer(images, targets)

        self.assertTrue(torch.all(result.index.cpu() != torch.arange(8)))

    def test_no_repeat_rejects_singleton_batch(self):
        mixer = MixUp(alpha=1.0, no_repeat=True)
        images = torch.randn(1, 3, 4, 4)
        targets = torch.tensor([0])

        with self.assertRaisesRegex(ValueError, "no_repeat"):
            mixer(images, targets)


if __name__ == "__main__":
    unittest.main()
