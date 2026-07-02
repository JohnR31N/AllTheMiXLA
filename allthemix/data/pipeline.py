"""Dataset construction and preprocessing pipelines."""

from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from allthemix.cli.presets import DatasetPreset
from allthemix.data.datasets import TinyImageNet


def _normalize(preset: DatasetPreset) -> transforms.Normalize:
    return transforms.Normalize(preset.mean, preset.std)


def build_transforms(preset: DatasetPreset, recipe_profile: str, augment: bool = True):
    normalize = _normalize(preset)
    base = [transforms.ToTensor(), normalize]

    if preset.name in {"cifar10", "cifar100"}:
        padding_mode = "reflect" if preset.name == "cifar100" and recipe_profile == "openmixup" else "constant"
        train_aug = [
            transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
            transforms.RandomHorizontalFlip(),
        ]
    elif preset.name == "tinyimagenet":
        if recipe_profile == "openmixup":
            train_aug = [
                transforms.RandomResizedCrop(64, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            train_aug = [transforms.RandomHorizontalFlip()]
    else:
        raise ValueError(f"Unsupported dataset: {preset.name}")

    train_transform = transforms.Compose((train_aug if augment else []) + base)
    val_transform = transforms.Compose(base)
    return train_transform, val_transform


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


def build_datasets(
    preset: DatasetPreset,
    recipe_profile: str,
    data_dir: str | Path,
    download: bool = False,
    augment: bool = True,
):
    train_transform, val_transform = build_transforms(preset, recipe_profile, augment)

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

    raise ValueError(f"Unsupported dataset: {preset.name}")
