import unittest
from unittest.mock import patch

import torch

from allthemix.methods.guided_sr import (
    GuidedSR,
    build_pairing,
    compute_spectral_residual_saliency_maps,
    denormalize_images_for_saliency,
    gaussian_blur_2d_single_channel,
    guided_sr,
    guidedmixup_from_saliency,
    minmax_normalize_saliency_maps,
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
    def test_gaussian_blur_uses_replicate_padding_like_allthemix_sr(self):
        saliency_maps = torch.zeros(1, 1, 3, 3)
        saliency_maps[:, :, 0, 0] = 1.0

        blurred = gaussian_blur_2d_single_channel(saliency_maps, kernel_size=3, sigma=3.0)

        self.assertGreater(float(blurred[0, 0, 0, 0]), float(blurred[0, 0, 1, 1]))
        self.assertGreater(float(blurred[0, 0, 0, 0]), float(blurred[0, 0, 2, 2]))

    def test_normalize_saliency_maps_sum_to_one(self):
        saliency_maps = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[-1.0, 0.0], [0.0, 3.0]]]])

        normalized = normalize_saliency_maps(saliency_maps)

        self.assertEqual(tuple(normalized.shape), (2, 1, 2, 2))
        torch.testing.assert_close(normalized.sum(dim=(1, 2, 3)), torch.ones(2))
        self.assertGreaterEqual(float(normalized.min()), 0.0)

    def test_minmax_normalize_saliency_maps_matches_reference_preprocessor(self):
        saliency_maps = torch.tensor([[[[2.0, 4.0], [6.0, 10.0]]], [[[5.0, 5.0], [5.0, 5.0]]]])

        normalized = minmax_normalize_saliency_maps(saliency_maps)

        torch.testing.assert_close(normalized[0].amin(), torch.tensor(0.0))
        torch.testing.assert_close(normalized[0].amax(), torch.tensor(1.0))
        torch.testing.assert_close(normalized[1], torch.zeros_like(normalized[1]))

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

    def test_guidedmixup_from_saliency_accepts_cross_device_partner_maps(self):
        images, targets, saliency_maps = _toy_batch(batch_size=4, height=4, width=4)
        partner_images = torch.flip(images, dims=(0,))
        partner_targets = torch.flip(targets, dims=(0,))
        partner_saliency_maps = torch.flip(saliency_maps, dims=(0,))
        partner_index = torch.tensor([7, 6, 5, 4], dtype=torch.long)

        result = guidedmixup_from_saliency(
            images,
            targets,
            saliency_maps,
            blur_kernel=1,
            condition="random",
            partner_images=partner_images,
            partner_targets=partner_targets,
            partner_saliency_maps=partner_saliency_maps,
            index=partner_index,
        )

        torch.testing.assert_close(result.targets_a, targets)
        torch.testing.assert_close(result.targets_b, partner_targets)
        torch.testing.assert_close(result.index, partner_index)
        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(tuple(result.lam.shape), (4,))

    def test_guidedmixup_images_and_lambda_match_mask_formula(self):
        images, targets, saliency_maps = _toy_batch(batch_size=3, height=4, width=4)
        partner_images = torch.flip(images, dims=(0,))
        partner_targets = torch.flip(targets, dims=(0,))
        partner_saliency_maps = torch.flip(saliency_maps, dims=(0,))

        result = guidedmixup_from_saliency(
            images,
            targets,
            saliency_maps,
            blur_kernel=1,
            condition="random",
            partner_images=partner_images,
            partner_targets=partner_targets,
            partner_saliency_maps=partner_saliency_maps,
            index=torch.tensor([5, 4, 3], dtype=torch.long),
        )

        expected_images = result.mask.to(dtype=images.dtype) * images + (1.0 - result.mask.to(dtype=images.dtype)) * partner_images
        torch.testing.assert_close(result.images, expected_images)
        torch.testing.assert_close(result.lam, result.mask.mean(dim=(1, 2, 3)).to(dtype=images.dtype))

    def test_guidedmixup_from_saliency_rejects_mismatched_external_partner_index(self):
        images, targets, saliency_maps = _toy_batch(batch_size=4, height=4, width=4)

        with self.assertRaisesRegex(ValueError, "partner index"):
            guidedmixup_from_saliency(
                images,
                targets,
                saliency_maps,
                blur_kernel=1,
                condition="random",
                partner_images=torch.flip(images, dims=(0,)),
                partner_targets=torch.flip(targets, dims=(0,)),
                partner_saliency_maps=torch.flip(saliency_maps, dims=(0,)),
                index=torch.arange(5, dtype=torch.long),
            )

    def test_guided_sr_matches_online_saliency_path(self):
        images, targets, _ = _toy_batch(batch_size=4, height=8, width=8)
        mixer = GuidedSR(alpha=1.0, blur_kernel=3, condition="random")

        result = mixer(images, targets)
        saliency_maps = compute_spectral_residual_saliency_maps(images, blur_kernel=3)
        per_sample_min = saliency_maps.amin(dim=(1, 2, 3))
        per_sample_max = saliency_maps.amax(dim=(1, 2, 3))

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(tuple(saliency_maps.shape), (4, 1, 8, 8))
        self.assertGreaterEqual(float(saliency_maps.min()), 0.0)
        self.assertLessEqual(float(saliency_maps.max()), 1.0)
        torch.testing.assert_close(per_sample_min, torch.zeros_like(per_sample_min), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(per_sample_max, torch.ones_like(per_sample_max), atol=1e-5, rtol=0.0)

    def test_denormalize_images_for_saliency_restores_unit_image(self):
        unit_images = torch.rand(2, 3, 4, 4)
        mean = (0.5, 0.25, 0.1)
        std = (0.2, 0.5, 0.4)
        normalized = (unit_images - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

        restored = denormalize_images_for_saliency(normalized, mean, std)

        torch.testing.assert_close(restored, unit_images)

    def test_guided_sr_online_saliency_uses_denormalized_images(self):
        unit_images = torch.rand(2, 3, 4, 4)
        targets = torch.arange(2, dtype=torch.long)
        mean = (0.5, 0.25, 0.1)
        std = (0.2, 0.5, 0.4)
        normalized = (unit_images - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        captured = {}

        def fake_saliency(images, blur_kernel):
            captured["images"] = images.detach().clone()
            captured["blur_kernel"] = blur_kernel
            return torch.ones(images.size(0), 1, images.size(-2), images.size(-1))

        with patch("allthemix.methods.guided_sr.compute_spectral_residual_saliency_maps", fake_saliency):
            GuidedSR(
                alpha=1.0,
                blur_kernel=3,
                condition="random",
                saliency_mean=mean,
                saliency_std=std,
            )(normalized, targets)

        torch.testing.assert_close(captured["images"], unit_images)
        self.assertEqual(captured["blur_kernel"], 3)

    def test_guided_sr_prob_zero_returns_clean_batch(self):
        images, targets, _ = _toy_batch(batch_size=4, height=8, width=8)

        result = guided_sr(images, targets, prob=0.0, blur_kernel=3)

        torch.testing.assert_close(result.images, images)
        torch.testing.assert_close(result.targets_a, targets)
        torch.testing.assert_close(result.targets_b, targets)
        torch.testing.assert_close(result.lam, torch.ones_like(result.lam))

    def test_guided_sr_functional_wrapper_uses_supplied_saliency_maps(self):
        images, targets, saliency_maps = _toy_batch(batch_size=4, height=4, width=4)

        with patch(
            "allthemix.methods.guided_sr.compute_spectral_residual_saliency_maps",
            side_effect=AssertionError("cached saliency maps should be used"),
        ):
            result = guided_sr(
                images,
                targets,
                prob=1.0,
                blur_kernel=1,
                condition="random",
                saliency_maps=saliency_maps,
            )

        torch.testing.assert_close(result.saliency_maps, normalize_saliency_maps(saliency_maps))

    def test_alpha_is_interface_compatible_but_not_used_by_guided_sr_formula(self):
        images, targets, saliency_maps = _toy_batch(batch_size=4, height=4, width=4)

        result_a = GuidedSR(alpha=0.5, blur_kernel=1, condition="greedy")(
            images,
            targets,
            saliency_maps=saliency_maps,
        )
        result_b = GuidedSR(alpha=2.0, blur_kernel=1, condition="greedy")(
            images,
            targets,
            saliency_maps=saliency_maps,
        )

        torch.testing.assert_close(result_a.images, result_b.images)
        torch.testing.assert_close(result_a.lam, result_b.lam)
        torch.testing.assert_close(result_a.index, result_b.index)

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
