import random
import unittest

import numpy as np
import torch

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

    def test_fmix_batch_and_loss(self):
        mixer = FMix(alpha=1.0, decay_power=3.0, size=(32, 32))
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        result = mixer(images, targets)

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(result.targets_a.shape, targets.shape)
        self.assertEqual(result.targets_b.shape, targets.shape)
        self.assertEqual(result.mask.shape, (1, 32, 32))

        logits = torch.randn(4, 10, requires_grad=True)
        loss = fmix_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)
        loss.backward()
        self.assertIsNotNone(logits.grad)

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


if __name__ == "__main__":
    unittest.main()
