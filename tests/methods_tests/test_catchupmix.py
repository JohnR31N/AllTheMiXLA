import unittest
from unittest.mock import patch

import torch

from allthemix.methods.catchupmix import CatchUpMix, catchup_mix_features, make_catchup_mix_feature_hook
from allthemix.methods.mixup import mixup_cross_entropy
from allthemix.networks import build_model


class CatchUpMixTests(unittest.TestCase):
    def test_catchup_mix_features_preserves_shape(self):
        features = torch.arange(4 * 6 * 3 * 3, dtype=torch.float32).reshape(4, 6, 3, 3)
        index = torch.tensor([1, 2, 3, 0])

        mixed = catchup_mix_features(features, lam=0.5, index=index)

        self.assertEqual(mixed.shape, features.shape)
        self.assertTrue(torch.isfinite(mixed).all())

    def test_catchup_mix_features_selects_channels_by_relative_influence(self):
        features = torch.tensor(
            [
                [[[[10.0]], [[4.0]], [[2.0]], [[1.0]]]],
                [[[[1.0]], [[2.0]], [[4.0]], [[10.0]]]],
            ]
        ).squeeze(1)
        index = torch.tensor([1, 0])

        mixed = catchup_mix_features(features, lam=0.5, index=index)
        expected = torch.stack(
            [
                torch.tensor([1.0, 2.0, 2.0, 1.0]).view(4, 1, 1),
                torch.tensor([1.0, 2.0, 2.0, 1.0]).view(4, 1, 1),
            ]
        )

        torch.testing.assert_close(mixed, expected)

    def test_feature_hook_only_applies_selected_layer(self):
        features = torch.randn(4, 6, 3, 3)
        index = torch.tensor([1, 2, 3, 0])
        hook = make_catchup_mix_feature_hook(layer=2, lam=0.5, index=index)

        untouched = hook(features, 1)
        mixed = hook(features, 2)

        torch.testing.assert_close(untouched, features)
        self.assertEqual(mixed.shape, features.shape)

    def test_catchupmix_output_supports_loss(self):
        images = torch.randn(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)
        mixer = CatchUpMix(alpha=1.0, cutmix_alpha=1.0, num_feature_layers=5)

        result = mixer(images, targets)
        logits = torch.randn(4, 4)
        loss = mixup_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)

        self.assertEqual(result.images.shape, images.shape)
        self.assertIn(result.layer, range(6))
        self.assertTrue(torch.isfinite(loss))

    def test_catchupmix_feature_hook_runs_through_preact_classifier(self):
        images = torch.randn(4, 3, 32, 32)
        targets = torch.arange(4, dtype=torch.long)
        mixer = CatchUpMix(alpha=1.0, cutmix_alpha=1.0, num_feature_layers=5)
        model = build_model("preact_resnet18", num_classes=4)

        with patch("allthemix.methods.catchupmix.random.randint", return_value=3):
            result = mixer(images, targets)
        logits = model(result.images, feature_hook=result.feature_hook)
        loss = mixup_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)

        self.assertEqual(result.layer, 3)
        self.assertEqual(tuple(logits.shape), (4, 4))
        self.assertTrue(torch.isfinite(loss))

    def test_catchupmix_all_preact_feature_layers_are_hookable(self):
        images = torch.randn(4, 3, 32, 32)
        targets = torch.arange(4, dtype=torch.long)
        mixer = CatchUpMix(alpha=1.0, cutmix_alpha=1.0, num_feature_layers=5)
        model = build_model("preact_resnet18", num_classes=4)

        for layer in range(1, 6):
            with self.subTest(layer=layer):
                with patch("allthemix.methods.catchupmix.random.randint", return_value=layer):
                    result = mixer(images, targets)
                logits = model(result.images, feature_hook=result.feature_hook)
                loss = mixup_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)

                self.assertEqual(result.layer, layer)
                self.assertIsNotNone(result.feature_hook)
                self.assertEqual(tuple(logits.shape), (4, 4))
                self.assertTrue(torch.isfinite(loss))

    def test_catchupmix_layer_zero_uses_input_cutmix_without_feature_hook(self):
        images = torch.randn(4, 3, 32, 32)
        targets = torch.arange(4, dtype=torch.long)
        mixer = CatchUpMix(alpha=1.0, cutmix_alpha=1.0, num_feature_layers=5)
        model = build_model("preact_resnet18", num_classes=4)

        with patch("allthemix.methods.catchupmix.random.randint", return_value=0):
            result = mixer(images, targets)
        logits = model(result.images, feature_hook=result.feature_hook)
        loss = mixup_cross_entropy(logits, result.targets_a, result.targets_b, result.lam)

        self.assertEqual(result.layer, 0)
        self.assertIsNone(result.feature_hook)
        self.assertEqual(tuple(logits.shape), (4, 4))
        self.assertTrue(torch.isfinite(loss))

    def test_feature_level_rejects_external_partner_images(self):
        images = torch.randn(4, 3, 8, 8)
        partner_images = torch.randn(4, 3, 8, 8)
        targets = torch.arange(4, dtype=torch.long)
        partner_targets = torch.arange(4, dtype=torch.long).flip(0)
        index = torch.tensor([3, 2, 1, 0])
        mixer = CatchUpMix(alpha=1.0, cutmix_alpha=1.0, num_feature_layers=5)

        with patch("allthemix.methods.catchupmix.random.randint", return_value=2):
            with self.assertRaisesRegex(ValueError, "feature-level mixing does not support external partner_images"):
                mixer(
                    images,
                    targets,
                    partner_images=partner_images,
                    partner_targets=partner_targets,
                    index=index,
                )


if __name__ == "__main__":
    unittest.main()
