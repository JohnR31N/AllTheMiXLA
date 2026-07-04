import tempfile
import unittest
from pathlib import Path

from PIL import Image

from allthemix.cli.presets import get_dataset_preset
from allthemix.data.datasets import IMAGENET_A_INDICES_IN_1K, IMAGENET_A_NUM_CLASSES
from allthemix.data.pipeline import build_datasets


class ImageNetATests(unittest.TestCase):
    def test_official_index_mapping_has_200_classes(self):
        self.assertEqual(IMAGENET_A_NUM_CLASSES, 200)
        self.assertEqual(len(IMAGENET_A_INDICES_IN_1K), 200)
        self.assertEqual(IMAGENET_A_INDICES_IN_1K[0], 6)
        self.assertEqual(IMAGENET_A_INDICES_IN_1K[-1], 988)

    def test_build_datasets_returns_eval_only_imagefolder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "imagenet-a"
            for class_name in ("n01498041", "n01531178"):
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                Image.new("RGB", (300, 260)).save(class_dir / "sample.jpg")

            preset = get_dataset_preset("imagenet_a")
            train_set, val_set = build_datasets(
                preset,
                "imagenet_a",
                data_dir=temp_dir,
                use_basic_augmentation=False,
            )

            self.assertIsNone(train_set)
            self.assertEqual(len(val_set), 2)
            image, target = val_set[0]
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertIn(target, {0, 1})


if __name__ == "__main__":
    unittest.main()
