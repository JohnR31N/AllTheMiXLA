import unittest

import torch

from allthemix.cli.train import build_batch_mixer, prepare_state_dict_for_model, reduce_logits_for_dataset
from allthemix.methods import FMix, MixUp
from allthemix.networks import build_model


class ImageNetAEvalTests(unittest.TestCase):
    def test_reduce_imagenet_logits_to_imagenet_a_classes(self):
        logits = torch.arange(1000.0).reshape(1, 1000)
        reduced = reduce_logits_for_dataset(logits, "imagenet_a")

        self.assertEqual(reduced.shape, (1, 200))
        self.assertEqual(float(reduced[0, 0]), 6.0)
        self.assertEqual(float(reduced[0, -1]), 988.0)

    def test_keep_already_reduced_imagenet_a_logits(self):
        logits = torch.randn(2, 200)
        reduced = reduce_logits_for_dataset(logits, "imagenet_a")

        self.assertIs(reduced, logits)

    def test_prepare_torchvision_resnet_state_for_split_classifier(self):
        model = build_model("torch_resnet101", num_classes=1000)
        state_dict = {
            "conv1.weight": model.backbone.conv1.weight.detach().clone(),
            "bn1.weight": model.backbone.bn1.weight.detach().clone(),
            "bn1.bias": model.backbone.bn1.bias.detach().clone(),
            "bn1.running_mean": model.backbone.bn1.running_mean.detach().clone(),
            "bn1.running_var": model.backbone.bn1.running_var.detach().clone(),
            "fc.weight": model.head.fc.weight.detach().clone(),
            "fc.bias": model.head.fc.bias.detach().clone(),
        }

        mapped = prepare_state_dict_for_model(state_dict, model)

        self.assertIn("backbone.conv1.weight", mapped)
        self.assertIn("head.fc.weight", mapped)

    def test_build_batch_mixer_dispatches_methods(self):
        fmix = build_batch_mixer(
            {
                "method": "fmix",
                "decay_power": 3.0,
                "alpha": 1.0,
                "image_size": 32,
                "max_soft": 0.0,
                "reformulate": False,
            }
        )
        mixup = build_batch_mixer({"method": "mixup", "alpha": 1.0})

        self.assertIsInstance(fmix, FMix)
        self.assertIsInstance(mixup, MixUp)
        self.assertIsNone(build_batch_mixer({"method": "none"}))


if __name__ == "__main__":
    unittest.main()
