import unittest

import torch

from allthemix.networks import build_model
from allthemix.networks.heads import LinearHead
from allthemix.networks.nn import preact_resnet18_nn


class ModelTests(unittest.TestCase):
    def test_preact_resnet18_nn_forward(self):
        model = preact_resnet18_nn()
        features = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(features.shape, (2, 512))

    def test_linear_head_forward(self):
        head = LinearHead(in_features=512, num_classes=10)
        logits = head(torch.randn(2, 512))
        self.assertEqual(logits.shape, (2, 10))

    def test_build_model_forward(self):
        model = build_model("preact_resnet18", num_classes=10)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(logits.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
