"""Tiny-ImageNet dataset reader."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class TinyImageNet(Dataset):
    """Tiny-ImageNet reader for the original tiny-imagenet-200 layout."""

    def __init__(self, root: str | Path, train: bool = True, transform=None) -> None:
        self.root = Path(root)
        self.train = bool(train)
        self.transform = transform
        self.class_to_idx = self._parse_classes()
        self.samples = self._parse_train() if train else self._parse_val()

    def _parse_classes(self) -> dict[str, int]:
        wnids_path = self.root / "wnids.txt"
        if wnids_path.exists():
            return {
                line.strip().split("\t")[0]: idx
                for idx, line in enumerate(wnids_path.read_text().splitlines())
                if line.strip()
            }

        train_root = self.root / "train"
        if train_root.exists():
            classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
            return {name: idx for idx, name in enumerate(classes)}

        raise FileNotFoundError(
            f"Could not find Tiny-ImageNet classes under {self.root}. "
            "Expected wnids.txt or train/<wnid>/ directories."
        )

    def _parse_train(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name, label in self.class_to_idx.items():
            images_path = self.root / "train" / class_name / "images"
            if not images_path.exists():
                continue
            for image_path in sorted(images_path.iterdir()):
                if image_path.is_file():
                    samples.append((image_path, label))
        if not samples:
            raise FileNotFoundError(f"No Tiny-ImageNet training images found under {self.root / 'train'}")
        return samples

    def _parse_val(self) -> list[tuple[Path, int]]:
        annotations = self.root / "val" / "val_annotations.txt"
        if not annotations.exists():
            raise FileNotFoundError(
                f"Could not find {annotations}. If your validation set is already "
                "class-foldered, use torchvision ImageFolder layout instead."
            )

        samples: list[tuple[Path, int]] = []
        for line in annotations.read_text().splitlines():
            image, class_name, *_ = line.split("\t")
            label = self.class_to_idx[class_name]
            samples.append((self.root / "val" / "images" / image, label))
        return samples

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = default_loader(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.samples)
