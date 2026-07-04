"""Dataset construction and preprocessing pipelines."""

from __future__ import annotations

from pathlib import Path

from torchvision import datasets

from allthemix.cli.presets import DatasetPreset
from allthemix.data.datasets import TinyImageNet
from allthemix.data.preprocessors import build_preprocess_pair


def _cifar_root(data_dir: str | Path, dataset: str) -> Path:
    return Path(data_dir) / dataset


def _tiny_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    candidates = [
        root,
        root / "tiny-imagenet-200",
        root / "TinyImageNet",
        root / "tiny_imagenet",
    ]
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "val").exists():
            return candidate
    return root / "tiny-imagenet-200"


def _has_wnid_class_dirs(path: Path) -> bool:
    if not path.exists():
        return False
    return any(child.is_dir() and child.name.startswith("n") and child.name[1:].isdigit() for child in path.iterdir())


def _imagenet_a_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    named_candidates = [
        root / "imagenet-a",
        root / "imagenet_a",
        root / "ImageNet-A",
    ]
    for candidate in named_candidates:
        for nested in (candidate / "val", candidate):
            if _has_wnid_class_dirs(nested):
                return nested

    for candidate in (root, root / "val"):
        if _has_wnid_class_dirs(candidate):
            return candidate

    return root / "imagenet-a"


def build_datasets(
    preset: DatasetPreset,
    recipe_profile: str,
    data_dir: str | Path,
    download: bool = False,
    use_basic_augmentation: bool = True,
):
    train_transform, val_transform = build_preprocess_pair(
        preset,
        recipe_profile,
        use_basic_augmentation,
    )

    if preset.name == "cifar10":
        root = _cifar_root(data_dir, "cifar10")
        train_set = datasets.CIFAR10(root=str(root), train=True, transform=train_transform, download=download)
        val_set = datasets.CIFAR10(root=str(root), train=False, transform=val_transform, download=download)
        return train_set, val_set

    if preset.name == "cifar100":
        root = _cifar_root(data_dir, "cifar100")
        train_set = datasets.CIFAR100(root=str(root), train=True, transform=train_transform, download=download)
        val_set = datasets.CIFAR100(root=str(root), train=False, transform=val_transform, download=download)
        return train_set, val_set

    if preset.name == "tinyimagenet":
        root = _tiny_root(data_dir)
        class_folder_val = root / "val"
        if (root / "wnids.txt").exists() or (root / "val" / "val_annotations.txt").exists():
            train_set = TinyImageNet(root, train=True, transform=train_transform)
            val_set = TinyImageNet(root, train=False, transform=val_transform)
            return train_set, val_set

        if (
            (root / "train").exists()
            and class_folder_val.exists()
            and any(path.is_dir() for path in class_folder_val.iterdir())
        ):
            train_set = datasets.ImageFolder(str(root / "train"), transform=train_transform)
            val_set = datasets.ImageFolder(str(root / "val"), transform=val_transform)
            return train_set, val_set

        raise FileNotFoundError(
            f"Tiny-ImageNet not found at {root}. Expected original layout "
            "with wnids.txt/val_annotations.txt or ImageFolder train/val directories."
        )

    if preset.name == "imagenet_a":
        root = _imagenet_a_root(data_dir)
        if not _has_wnid_class_dirs(root):
            raise FileNotFoundError(
                f"ImageNet-A not found at {root}. Expected ImageFolder class directories "
                "such as imagenet-a/n01498041/*.jpg."
            )
        val_set = datasets.ImageFolder(str(root), transform=val_transform)
        return None, val_set

    raise ValueError(f"Unsupported dataset: {preset.name}")
