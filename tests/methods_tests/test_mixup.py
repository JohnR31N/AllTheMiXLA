import unittest

import numpy as np
import torch

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


if __name__ == "__main__":
    unittest.main()
