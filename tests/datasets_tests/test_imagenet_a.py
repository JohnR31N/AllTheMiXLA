import tempfile
import unittest
from pathlib import Path

from PIL import Image

from allthemix.cli.presets import get_dataset_preset
from allthemix.data.datasets import IMAGENET_A_INDICES_IN_1K, IMAGENET_A_NUM_CLASSES, IMAGENET_A_WNID_TO_REDUCED_INDEX
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
            self.assertEqual(val_set.classes[0], "n01498041")

    def test_build_datasets_remaps_targets_to_official_reduced_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "imagenet-a"
            for class_name in ("n01498041", "n02133161"):
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                Image.new("RGB", (300, 260)).save(class_dir / "sample.jpg")

            preset = get_dataset_preset("imagenet_a")
            _, val_set = build_datasets(
                preset,
                "imagenet_a",
                data_dir=temp_dir,
                use_basic_augmentation=False,
            )

            targets_by_wnid = {Path(path).parent.name: target for path, target in val_set.samples}

        self.assertEqual(targets_by_wnid["n01498041"], IMAGENET_A_WNID_TO_REDUCED_INDEX["n01498041"])
        self.assertEqual(targets_by_wnid["n02133161"], IMAGENET_A_WNID_TO_REDUCED_INDEX["n02133161"])
        self.assertNotEqual(IMAGENET_A_WNID_TO_REDUCED_INDEX["n02133161"], 1)

    def test_build_datasets_rejects_non_imagenet_a_class_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "imagenet-a"
            for class_name in ("n01498041", "n01440764"):
                class_dir = root / class_name
                class_dir.mkdir(parents=True)
                Image.new("RGB", (300, 260)).save(class_dir / "sample.jpg")

            preset = get_dataset_preset("imagenet_a")
            with self.assertRaisesRegex(ValueError, "outside the official 200"):
                build_datasets(
                    preset,
                    "imagenet_a",
                    data_dir=temp_dir,
                    use_basic_augmentation=False,
                )


if __name__ == "__main__":
    unittest.main()
