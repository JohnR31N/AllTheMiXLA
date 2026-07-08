import unittest
from pathlib import Path

import torch

from allthemix.networks import build_model, canonical_model_name, model_impl_version
from allthemix.networks.backbones import PreActResNetBackbone, preact_resnet18_backbone, torch_resnet101_backbone
from allthemix.networks.backbones.preact_resnet import PreActBasicBlock
from allthemix.networks.classifiers import ImageClassifier
from allthemix.networks.heads import LinearHead


class ModelTests(unittest.TestCase):
    def test_network_components_live_in_sibling_packages(self):
        root = Path("allthemix/networks")

        self.assertTrue((root / "backbones" / "preact_resnet.py").exists())
        self.assertTrue((root / "classifiers" / "image_classifier.py").exists())
        self.assertTrue((root / "heads" / "linear_head.py").exists())
        self.assertFalse((root / "preact_resnet.py").exists())

    def test_preact_resnet18_backbone_forward(self):
        model = preact_resnet18_backbone()
        features = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(features.shape, (2, 512))

    def test_preact_basic_block_identity_shortcut_preserves_input(self):
        block = PreActBasicBlock(in_planes=16, planes=16, stride=1)
        block.eval()
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
        images = torch.randn(2, 16, 8, 8) - 0.5

        with torch.no_grad():
            output = block(images)

        torch.testing.assert_close(output, images)

    def test_linear_head_forward(self):
        head = LinearHead(in_features=512, num_classes=10)
        logits = head(torch.randn(2, 512))
        self.assertEqual(logits.shape, (2, 10))

    def test_torch_resnet101_backbone_forward(self):
        model = torch_resnet101_backbone()
        model.eval()
        with torch.no_grad():
            features = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(features.shape, (1, 2048))

    def test_build_model_forward(self):
        model = build_model("preact_resnet18", num_classes=10)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(logits.shape, (2, 10))

    def test_build_model_composes_backbone_and_head(self):
        model = build_model("preact_resnet18", num_classes=10)

        self.assertIsInstance(model, ImageClassifier)
        self.assertIsInstance(model.backbone, PreActResNetBackbone)
        self.assertIsInstance(model.head, LinearHead)

    def test_build_model_forward_accepts_feature_hook(self):
        model = build_model("preact_resnet18", num_classes=10)
        seen_layers = []

        def feature_hook(features, layer_index):
            seen_layers.append(layer_index)
            return features

        logits = model(torch.randn(2, 3, 32, 32), feature_hook=feature_hook)

        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(seen_layers, [1, 2, 3, 4, 5])

    def test_build_torch_resnet101_model_forward(self):
        model = build_model("torch_resnet101", num_classes=1000)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(logits.shape, (1, 1000))

    def test_model_aliases_have_canonical_metadata_names(self):
        self.assertEqual(canonical_model_name("preact-resnet18"), "preact_resnet18")
        self.assertEqual(canonical_model_name("torchvision_resnet101"), "torch_resnet101")
        self.assertEqual(canonical_model_name("resnet101"), "torch_resnet101")
        self.assertEqual(model_impl_version("preact-resnet18"), 2)
        self.assertEqual(model_impl_version("torchvision_resnet101"), 1)


if __name__ == "__main__":
    unittest.main()
