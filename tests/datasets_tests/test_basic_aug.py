import unittest

import torch
from PIL import Image
from torchvision import transforms

from allthemix.cli.presets import get_dataset_preset
from allthemix.data.preprocessors import build_eval_preprocess, build_train_preprocess


class BasicAugTests(unittest.TestCase):
    def test_cifar_train_preprocess_has_basic_aug(self):
        preset = get_dataset_preset("cifar10")
        preprocess = build_train_preprocess(preset, "openmixup", use_basic_augmentation=True)

        self.assertIsInstance(preprocess.transforms[0], transforms.RandomCrop)
        self.assertIsInstance(preprocess.transforms[1], transforms.RandomHorizontalFlip)

        image = Image.new("RGB", (32, 32))
        output = preprocess(image)
        self.assertEqual(output.shape, (3, 32, 32))
        self.assertIsInstance(output, torch.Tensor)

    def test_eval_preprocess_has_no_random_basic_aug(self):
        preset = get_dataset_preset("cifar10")
        preprocess = build_eval_preprocess(preset)

        self.assertEqual(len(preprocess.transforms), 2)
        self.assertIsInstance(preprocess.transforms[0], transforms.ToTensor)
        self.assertIsInstance(preprocess.transforms[1], transforms.Normalize)

    def test_imagenet_a_eval_preprocess_matches_official_shape(self):
        preset = get_dataset_preset("imagenet_a")
        preprocess = build_eval_preprocess(preset)

        self.assertIsInstance(preprocess.transforms[0], transforms.Resize)
        self.assertIsInstance(preprocess.transforms[1], transforms.CenterCrop)

        image = Image.new("RGB", (320, 260))
        output = preprocess(image)
        self.assertEqual(output.shape, (3, 224, 224))
        self.assertIsInstance(output, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
