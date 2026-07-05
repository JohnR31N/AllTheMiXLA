import unittest

import torch

from allthemix.methods.guided_sr import (
    GuidedSR,
    build_pairing,
    compute_spectral_residual_saliency_maps,
    guided_sr,
    guidedmixup_from_saliency,
    normalize_saliency_maps,
)
from allthemix.methods.mixup import mixup_cross_entropy


def _toy_batch(batch_size=4, height=8, width=8, channels=3):
    images = torch.arange(batch_size * channels * height * width, dtype=torch.float32).reshape(
        batch_size,
        channels,
        height,
        width,
    )
    images = images / images.max()
    targets = torch.arange(batch_size, dtype=torch.long)
    saliency_maps = torch.zeros(batch_size, 1, height, width)
    for index in range(batch_size):
        saliency_maps[index, 0, index % height, (index * 2 + 1) % width] = 1.0
        saliency_maps[index, 0, (index + 1) % height, (index * 3 + 2) % width] = 0.5
    return images, targets, saliency_maps


class GuidedSRTests(unittest.TestCase):
    def test_normalize_saliency_maps_sum_to_one(self):
        saliency_maps = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[-1.0, 0.0], [0.0, 3.0]]]])

        normalized = normalize_saliency_maps(saliency_maps)

        self.assertEqual(tuple(normalized.shape), (2, 1, 2, 2))
        torch.testing.assert_close(normalized.sum(dim=(1, 2, 3)), torch.ones(2))
        self.assertGreaterEqual(float(normalized.min()), 0.0)

    def test_guidedmixup_from_saliency_returns_per_sample_lambda(self):
        images, targets, saliency_maps = _toy_batch(batch_size=4, height=4, width=4)

        result = guidedmixup_from_saliency(
            images,
            targets,
            saliency_maps,
            blur_kernel=3,
            condition="random",
        )

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(result.targets_a.shape, targets.shape)
        self.assertEqual(result.targets_b.shape, targets.shape)
        self.assertEqual(tuple(result.lam.shape), (4,))
        self.assertTrue(torch.isfinite(result.images).all())
        self.assertTrue(torch.isfinite(result.lam).all())

    def test_guided_sr_matches_online_saliency_path(self):
        images, targets, _ = _toy_batch(batch_size=4, height=8, width=8)
        mixer = GuidedSR(alpha=1.0, blur_kernel=3, condition="random")

        result = mixer(images, targets)
        saliency_maps = compute_spectral_residual_saliency_maps(images, blur_kernel=3)

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(tuple(saliency_maps.shape), (4, 1, 8, 8))
        self.assertGreaterEqual(float(saliency_maps.min()), 0.0)

    def test_guided_sr_prob_zero_returns_clean_batch(self):
        images, targets, _ = _toy_batch(batch_size=4, height=8, width=8)

        result = guided_sr(images, targets, prob=0.0, blur_kernel=3)

        torch.testing.assert_close(result.images, images)
        torch.testing.assert_close(result.targets_a, targets)
        torch.testing.assert_close(result.targets_b, targets)
        torch.testing.assert_close(result.lam, torch.ones_like(result.lam))

    def test_greedy_pairing_is_valid_nonself_permutation(self):
        _, _, saliency_maps = _toy_batch(batch_size=4, height=4, width=4)

        permutation = build_pairing(normalize_saliency_maps(saliency_maps), condition="greedy")

        self.assertEqual(sorted(permutation.tolist()), [0, 1, 2, 3])
        self.assertTrue(torch.all(permutation != torch.arange(4)))

    def test_mixup_loss_accepts_per_sample_lambda(self):
        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        targets_a = torch.tensor([0, 1])
        targets_b = torch.tensor([1, 0])
        lam = torch.tensor([0.25, 0.75])

        loss = mixup_cross_entropy(logits, targets_a, targets_b, lam)

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
