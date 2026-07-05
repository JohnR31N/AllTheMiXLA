import unittest

import torch

from allthemix.methods.saliencymix import SaliencyMix, build_saliency_box_mask, saliencymix


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
