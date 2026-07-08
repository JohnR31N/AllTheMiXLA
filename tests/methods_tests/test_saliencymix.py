import unittest
from unittest.mock import patch

import torch

from allthemix.methods.saliencymix import (
    SaliencyMix,
    build_saliency_box_mask,
    compute_gradient_saliency_maps,
    saliencymix,
)


class SaliencyMixTests(unittest.TestCase):
    def test_build_saliency_box_mask_uses_peak_as_center(self):
        saliency_map = torch.zeros(4, 4)
        saliency_map[2, 2] = 10.0

        mask, patch_area = build_saliency_box_mask(
            saliency_map=saliency_map,
            cut_width=2,
            cut_height=2,
            image_height=4,
            image_width=4,
        )

        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        self.assertEqual(int(patch_area.item()), 4)
        self.assertTrue(mask[0, 0, 1, 1])
        self.assertTrue(mask[0, 0, 2, 2])
        self.assertFalse(mask[0, 0, 0, 0])

    def test_saliencymix_uses_batch_saliency_maps(self):
        images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)
        saliency_maps = torch.zeros(4, 1, 8, 8)
        saliency_maps[:, :, 4, 4] = 1.0

        result = SaliencyMix(alpha=1.0, saliency_source="batch")(
            images,
            targets,
            saliency_maps=saliency_maps,
        )

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(result.targets_a.shape, targets.shape)
        self.assertEqual(result.targets_b.shape, targets.shape)
        self.assertTrue(torch.isfinite(torch.as_tensor(result.lam)))

    def test_saliencymix_uses_first_partner_saliency_for_shared_patch(self):
        images = torch.zeros(2, 1, 6, 6)
        targets = torch.arange(2, dtype=torch.long)
        saliency_maps = torch.zeros(2, 1, 6, 6)
        partner_images = torch.zeros_like(images)
        partner_images[0].fill_(5.0)
        partner_images[1].fill_(9.0)
        partner_targets = torch.tensor([10, 11], dtype=torch.long)
        partner_saliency_maps = torch.zeros(2, 1, 6, 6)
        partner_saliency_maps[0, :, 1, 1] = 1.0
        partner_saliency_maps[1, :, 4, 4] = 1.0
        index = torch.tensor([1, 0], dtype=torch.long)

        with patch("allthemix.methods.saliencymix.sample_lam", return_value=0.25):
            result = SaliencyMix(alpha=1.0, saliency_source="batch")(
                images,
                targets,
                saliency_maps=saliency_maps,
                partner_images=partner_images,
                partner_targets=partner_targets,
                partner_saliency_maps=partner_saliency_maps,
                index=index,
            )

        self.assertTrue(result.mask[0, 0, 0, 0])
        self.assertFalse(result.mask[0, 0, 4, 4])
        torch.testing.assert_close(result.images[:, :, 0, 0], partner_images[:, :, 0, 0])
        torch.testing.assert_close(result.targets_b, partner_targets)

    def test_saliencymix_rejects_mismatched_external_partner_batch(self):
        images = torch.zeros(2, 1, 6, 6)
        targets = torch.arange(2, dtype=torch.long)
        saliency_maps = torch.zeros(2, 1, 6, 6)
        partner_images = torch.zeros(3, 1, 6, 6)
        partner_targets = torch.arange(3, dtype=torch.long)
        partner_saliency_maps = torch.zeros(3, 1, 6, 6)
        index = torch.arange(3, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "partner images must match"):
            SaliencyMix(alpha=1.0, saliency_source="batch")(
                images,
                targets,
                saliency_maps=saliency_maps,
                partner_images=partner_images,
                partner_targets=partner_targets,
                partner_saliency_maps=partner_saliency_maps,
                index=index,
            )

        with self.assertRaisesRegex(ValueError, "partner index"):
            SaliencyMix(alpha=1.0, saliency_source="batch")(
                images,
                targets,
                saliency_maps=saliency_maps,
                partner_images=torch.zeros_like(images),
                partner_targets=targets,
                partner_saliency_maps=torch.zeros_like(saliency_maps),
                index=torch.arange(3, dtype=torch.long),
            )

    def test_saliencymix_lambda_matches_actual_mask_area(self):
        images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)
        saliency_maps = torch.zeros(4, 1, 8, 8)
        saliency_maps[:, :, 4, 4] = 1.0

        result = SaliencyMix(alpha=1.0, saliency_source="batch")(
            images,
            targets,
            saliency_maps=saliency_maps,
        )
        patch_area = float(result.mask.to(dtype=torch.float32).sum().item())
        expected_lam = 1.0 - patch_area / float(images.size(-2) * images.size(-1))

        self.assertAlmostEqual(float(result.lam), expected_lam)

    def test_saliencymix_images_match_shared_mask_formula(self):
        images = torch.zeros(2, 1, 6, 6)
        images[0].fill_(1.0)
        images[1].fill_(2.0)
        partner_images = torch.zeros_like(images)
        partner_images[0].fill_(5.0)
        partner_images[1].fill_(9.0)
        targets = torch.tensor([0, 1], dtype=torch.long)
        partner_targets = torch.tensor([10, 11], dtype=torch.long)
        saliency_maps = torch.zeros(2, 1, 6, 6)
        partner_saliency_maps = torch.zeros(2, 1, 6, 6)
        partner_saliency_maps[0, :, 3, 3] = 1.0

        with patch("allthemix.methods.saliencymix.sample_lam", return_value=0.25):
            result = SaliencyMix(alpha=1.0, saliency_source="batch")(
                images,
                targets,
                saliency_maps=saliency_maps,
                partner_images=partner_images,
                partner_targets=partner_targets,
                partner_saliency_maps=partner_saliency_maps,
                index=torch.tensor([7, 8], dtype=torch.long),
            )

        expected_images = torch.where(result.mask, partner_images, images)
        torch.testing.assert_close(result.images, expected_images)
        torch.testing.assert_close(result.targets_b, partner_targets)

    def test_saliencymix_batch_source_requires_maps(self):
        images = torch.ones(2, 3, 8, 8)
        targets = torch.arange(2, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "requires saliency maps"):
            SaliencyMix(alpha=1.0, saliency_source="batch")(images, targets)

    def test_saliencymix_spectral_residual_fallback_runs(self):
        images = torch.rand(2, 3, 8, 8)
        targets = torch.arange(2, dtype=torch.long)

        result = SaliencyMix(alpha=1.0, saliency_source="spectral_residual", blur_kernel=3)(images, targets)

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(tuple(result.saliency_maps.shape), (2, 1, 8, 8))

    def test_saliencymix_online_saliency_uses_denormalized_images(self):
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

        with patch("allthemix.methods.saliencymix.compute_spectral_residual_saliency_maps", fake_saliency):
            SaliencyMix(
                alpha=1.0,
                saliency_source="spectral_residual",
                blur_kernel=3,
                saliency_mean=mean,
                saliency_std=std,
            )(normalized, targets)

        torch.testing.assert_close(captured["images"], unit_images)
        self.assertEqual(captured["blur_kernel"], 3)

    def test_saliencymix_gradient_fallback_runs_without_fft(self):
        images = torch.rand(2, 3, 8, 8)
        targets = torch.arange(2, dtype=torch.long)

        saliency_maps = compute_gradient_saliency_maps(images)
        result = SaliencyMix(alpha=1.0, saliency_source="gradient")(images, targets)

        self.assertEqual(tuple(saliency_maps.shape), (2, 1, 8, 8))
        self.assertEqual(result.images.shape, images.shape)
        self.assertGreaterEqual(float(saliency_maps.min()), 0.0)
        self.assertLessEqual(float(saliency_maps.max()), 1.0)

    def test_saliencymix_no_repeat_avoids_self_pairs(self):
        images = torch.rand(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)
        saliency_maps = torch.zeros(4, 1, 8, 8)
        saliency_maps[:, :, 4, 4] = 1.0

        result = SaliencyMix(alpha=1.0, saliency_source="batch", no_repeat=True)(
            images,
            targets,
            saliency_maps=saliency_maps,
        )

        self.assertTrue(torch.all(result.index.cpu() != torch.arange(4)))

    def test_saliencymix_no_repeat_rejects_singleton_batch(self):
        images = torch.rand(1, 3, 8, 8)
        targets = torch.zeros(1, dtype=torch.long)
        saliency_maps = torch.zeros(1, 1, 8, 8)

        with self.assertRaisesRegex(ValueError, "no_repeat"):
            SaliencyMix(alpha=1.0, saliency_source="batch", no_repeat=True)(
                images,
                targets,
                saliency_maps=saliency_maps,
            )

    def test_saliencymix_prob_zero_returns_clean_batch(self):
        images = torch.rand(2, 3, 8, 8)
        targets = torch.arange(2, dtype=torch.long)

        result = saliencymix(images, targets, prob=0.0)

        torch.testing.assert_close(result.images, images)
        torch.testing.assert_close(result.targets_a, targets)
        torch.testing.assert_close(result.targets_b, targets)
        self.assertEqual(result.lam, 1.0)


if __name__ == "__main__":
    unittest.main()
