import unittest
from unittest.mock import patch

import torch

from allthemix.methods.cutmix import CutMix, box_mask, no_repeat_permutation
from allthemix.methods.mixup import mixup_cross_entropy


class CutMixTests(unittest.TestCase):
    def test_box_mask_selects_rectangle(self):
        mask = box_mask(4, 4, x1=1, y1=1, x2=3, y2=3, device=torch.device("cpu"))

        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        self.assertTrue(mask[0, 0, 1, 1])
        self.assertTrue(mask[0, 0, 2, 2])
        self.assertFalse(mask[0, 0, 0, 0])

    def test_no_repeat_permutation_has_no_fixed_points(self):
        permutation = no_repeat_permutation(8, torch.device("cpu"))

        self.assertEqual(sorted(permutation.tolist()), list(range(8)))
        self.assertTrue(torch.all(permutation != torch.arange(8)))

    def test_cutmix_batch_and_loss(self):
        images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)

        result = CutMix(alpha=1.0, no_repeat=True)(images, targets)
        logits = torch.randn(4, 4)
        loss = mixup_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)

        self.assertEqual(result.images.shape, images.shape)
        self.assertEqual(result.targets_a.shape, targets.shape)
        self.assertEqual(result.targets_b.shape, targets.shape)
        self.assertGreaterEqual(result.lam, 0.0)
        self.assertLessEqual(result.lam, 1.0)
        self.assertTrue(torch.isfinite(loss))

    def test_cutmix_lambda_matches_actual_mask_area(self):
        images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)

        result = CutMix(alpha=1.0)(images, targets)
        patch_area = float(result.mask.to(dtype=torch.float32).sum().item())
        expected_lam = 1.0 - patch_area / float(images.size(-2) * images.size(-1))

        self.assertAlmostEqual(result.lam, expected_lam)

    def test_cutmix_images_match_mask_formula_with_external_partners(self):
        images = torch.zeros(2, 1, 6, 6)
        images[0].fill_(1.0)
        images[1].fill_(2.0)
        partner_images = torch.zeros_like(images)
        partner_images[0].fill_(5.0)
        partner_images[1].fill_(9.0)
        targets = torch.tensor([0, 1], dtype=torch.long)
        partner_targets = torch.tensor([10, 11], dtype=torch.long)

        with (
            patch("allthemix.methods.cutmix.sample_lam", return_value=0.5),
            patch("allthemix.methods.cutmix.build_random_box", return_value=(1, 2, 4, 5)),
        ):
            result = CutMix(alpha=1.0)(
                images,
                targets,
                partner_images=partner_images,
                partner_targets=partner_targets,
                index=torch.tensor([7, 8], dtype=torch.long),
            )

        expected_images = torch.where(result.mask, partner_images, images)
        torch.testing.assert_close(result.images, expected_images)
        torch.testing.assert_close(result.targets_b, partner_targets)

    def test_cutmix_rejects_mismatched_targets_and_partners(self):
        images = torch.ones(2, 3, 8, 8)

        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            CutMix(alpha=1.0)(images, torch.zeros(1, dtype=torch.long))

        with self.assertRaisesRegex(ValueError, "partner images must match"):
            CutMix(alpha=1.0)(
                images,
                torch.zeros(2, dtype=torch.long),
                partner_images=torch.ones(2, 1, 8, 8),
                partner_targets=torch.ones(2, dtype=torch.long),
                index=torch.tensor([0, 1]),
            )

    def test_cutmix_no_repeat_rejects_single_item_batch(self):
        images = torch.ones(1, 3, 8, 8)
        targets = torch.zeros(1, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "no_repeat"):
            CutMix(alpha=1.0, no_repeat=True)(images, targets)


if __name__ == "__main__":
    unittest.main()
