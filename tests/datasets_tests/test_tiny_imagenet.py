import tempfile
import unittest
from pathlib import Path

from allthemix.cli.presets import get_dataset_preset
from allthemix.data.datasets import TinyImageNet
from allthemix.data.pipeline import build_datasets


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image bytes")


class TinyImageNetTests(unittest.TestCase):
    def test_original_train_layout_uses_images_subdirectory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "wnids.txt").write_text("n00000001\n")
            _touch(root / "train" / "n00000001" / "images" / "a.JPEG")
            _touch(root / "train" / "n00000001" / "n00000001_boxes.txt")

            dataset = TinyImageNet(root, train=True)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.samples[0][0].name, "a.JPEG")
        self.assertEqual(dataset.samples[0][1], 0)

    def test_class_foldered_train_and_val_layout_with_wnids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "wnids.txt").write_text("n00000001\nn00000002\n")
            _touch(root / "train" / "n00000001" / "a.JPEG")
            _touch(root / "train" / "n00000002" / "b.jpg")
            _touch(root / "val" / "n00000002" / "v.JPEG")

            train_set = TinyImageNet(root, train=True)
            val_set = TinyImageNet(root, train=False)

        self.assertEqual([sample[1] for sample in train_set.samples], [0, 1])
        self.assertEqual(len(val_set), 1)
        self.assertEqual(val_set.samples[0][0].name, "v.JPEG")
        self.assertEqual(val_set.samples[0][1], 1)

    def test_annotation_val_layout_still_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "wnids.txt").write_text("n00000001\nn00000002\n")
            _touch(root / "train" / "n00000001" / "images" / "a.JPEG")
            _touch(root / "val" / "images" / "val_0.JPEG")
            (root / "val" / "val_annotations.txt").write_text("val_0.JPEG\tn00000002\t0\t0\t1\t1\n")
            _touch(root / "val" / "n00000001" / "foldered.JPEG")

            val_set = TinyImageNet(root, train=False)

        self.assertEqual(len(val_set), 1)
        self.assertEqual(val_set.samples[0][0].name, "val_0.JPEG")
        self.assertEqual(val_set.samples[0][1], 1)

    def test_pipeline_class_folder_layout_reuses_train_class_mapping_for_val_subset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch(root / "train" / "n00000001" / "a.JPEG")
            _touch(root / "train" / "n00000002" / "b.JPEG")
            _touch(root / "val" / "n00000002" / "v.JPEG")

            train_set, val_set = build_datasets(
                get_dataset_preset("tiny_imagenet"),
                "openmixup",
                data_dir=root,
                use_basic_augmentation=False,
            )

        self.assertIsInstance(train_set, TinyImageNet)
        self.assertIsInstance(val_set, TinyImageNet)
        self.assertEqual(train_set.class_to_idx, {"n00000001": 0, "n00000002": 1})
        self.assertEqual(val_set.class_to_idx, train_set.class_to_idx)
        self.assertEqual(val_set.samples[0][0].name, "v.JPEG")
        self.assertEqual(val_set.samples[0][1], 1)


if __name__ == "__main__":
    unittest.main()
