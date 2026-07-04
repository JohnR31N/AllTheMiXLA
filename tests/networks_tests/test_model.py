import unittest

import torch

from allthemix.networks import build_model
from allthemix.networks.backbones import preact_resnet18_backbone, torch_resnet101_backbone
from allthemix.networks.heads import LinearHead


class ModelTests(unittest.TestCase):
    def test_preact_resnet18_backbone_forward(self):
        model = preact_resnet18_backbone()
        features = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(features.shape, (2, 512))

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

    def test_build_torch_resnet101_model_forward(self):
        model = build_model("torch_resnet101", num_classes=1000)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(logits.shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
