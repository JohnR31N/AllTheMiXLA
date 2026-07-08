import unittest
from unittest.mock import patch

import torch

from allthemix.methods.mixup import mixup_cross_entropy
from allthemix.methods.resizemix import ResizeMix, resize_source_to_box_nearest


class ResizeMixTests(unittest.TestCase):
    def test_resize_source_to_box_nearest_builds_mask(self):
        source_images = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)

        resized_source, mask = resize_source_to_box_nearest(source_images, x1=1, y1=1, x2=3, y2=3)

        self.assertEqual(resized_source.shape, source_images.shape)
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        self.assertTrue(mask[0, 0, 1, 1])
        self.assertTrue(mask[0, 0, 2, 2])
        self.assertFalse(mask[0, 0, 0, 0])

    def test_resizemix_batch_and_loss(self):
        images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)

        result = ResizeMix(scope_min=0.1, scope_max=0.8)(images, targets)
        logits = torch.randn(4, 4)
        loss = mixup_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(result.targets_a.shape, targets.shape)
        self.assertEqual(result.targets_b.shape, targets.shape)
        self.assertGreaterEqual(result.lam, 0.0)
        self.assertLessEqual(result.lam, 1.0)
        self.assertTrue(torch.isfinite(loss))

    def test_resizemix_lambda_matches_actual_mask_area(self):
        images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)

        result = ResizeMix(scope_min=0.1, scope_max=0.8)(images, targets)
        pasted_area = float(result.mask.to(dtype=torch.float32).sum().item())
        expected_lam = 1.0 - pasted_area / float(images.size(-2) * images.size(-1))

        self.assertAlmostEqual(result.lam, expected_lam)

    def test_resizemix_images_match_resized_source_mask_formula(self):
        images = torch.zeros(2, 1, 6, 6)
        images[0].fill_(1.0)
        images[1].fill_(2.0)
        partner_images = torch.arange(2 * 1 * 6 * 6, dtype=torch.float32).reshape(2, 1, 6, 6)
        targets = torch.tensor([0, 1], dtype=torch.long)
        partner_targets = torch.tensor([10, 11], dtype=torch.long)
        mixer = ResizeMix(scope_min=0.1, scope_max=0.8)

        with (
            patch.object(mixer, "sample_resize_ratio", return_value=0.5),
            patch("allthemix.methods.resizemix.random.randrange", side_effect=[3, 3]),
        ):
            result = mixer(
                images,
                targets,
                partner_images=partner_images,
                partner_targets=partner_targets,
                index=torch.tensor([7, 8], dtype=torch.long),
            )

        resized_source, expected_mask = resize_source_to_box_nearest(partner_images, x1=2, y1=2, x2=4, y2=4)
        torch.testing.assert_close(result.mask, expected_mask)
        torch.testing.assert_close(result.images, torch.where(result.mask, resized_source, images))
        torch.testing.assert_close(result.targets_b, partner_targets)

    def test_resizemix_rejects_mismatched_targets_and_partners(self):
        images = torch.ones(2, 3, 8, 8)

        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            ResizeMix()(images, torch.zeros(1, dtype=torch.long))

        with self.assertRaisesRegex(ValueError, "partner index"):
            ResizeMix()(
                images,
                torch.zeros(2, dtype=torch.long),
                partner_images=torch.ones_like(images),
                partner_targets=torch.ones(2, dtype=torch.long),
                index=torch.tensor([0]),
            )

    def test_no_repeat_avoids_self_pairs(self):
        images = torch.arange(8 * 3 * 8 * 8, dtype=torch.float32).reshape(8, 3, 8, 8)
        targets = torch.arange(8, dtype=torch.long)

        result = ResizeMix(scope_min=0.1, scope_max=0.8, no_repeat=True)(images, targets)

        self.assertTrue(torch.all(result.index != torch.arange(8)))

    def test_resizemix_rejects_bad_scope(self):
        with self.assertRaisesRegex(ValueError, "resizemix_scope_min"):
            ResizeMix(scope_min=0.8, scope_max=0.1)

    def test_use_alpha_samples_beta_when_inside_scope(self):
        mixer = ResizeMix(scope_min=0.1, scope_max=0.8, alpha=1.0, use_alpha=True)

        with patch("allthemix.methods.resizemix.np.random.beta", return_value=0.5):
            self.assertEqual(mixer.sample_resize_ratio(), 0.5)

    def test_use_alpha_falls_back_to_uniform_when_beta_is_outside_scope(self):
        mixer = ResizeMix(scope_min=0.1, scope_max=0.8, alpha=1.0, use_alpha=True)

        with patch("allthemix.methods.resizemix.np.random.beta", return_value=0.95):
            with patch("allthemix.methods.resizemix.random.uniform", return_value=0.25):
                self.assertEqual(mixer.sample_resize_ratio(), 0.25)

    def test_no_repeat_rejects_singleton_batch(self):
        images = torch.zeros(1, 3, 8, 8)
        targets = torch.zeros(1, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "no_repeat"):
            ResizeMix(no_repeat=True)(images, targets)


if __name__ == "__main__":
    unittest.main()
