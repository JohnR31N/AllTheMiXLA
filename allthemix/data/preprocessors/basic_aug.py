"""Sample-level basic augmentation and normalization."""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from allthemix.cli.presets import DatasetPreset


def _basic_aug_steps(preset: DatasetPreset, recipe_profile: str) -> list:
    if preset.name in {"cifar10", "cifar100"}:
        padding_mode = "reflect" if preset.name == "cifar100" and recipe_profile == "openmixup" else "constant"
        return [
            transforms.RandomCrop(preset.image_size, padding=4, padding_mode=padding_mode),
            transforms.RandomHorizontalFlip(),
        ]

    if preset.name == "tinyimagenet":
        if recipe_profile == "openmixup":
            return [
                transforms.RandomResizedCrop(
                    preset.image_size,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        return [transforms.RandomHorizontalFlip()]

    if preset.name == "imagenet_a":
        return [
            transforms.RandomResizedCrop(preset.image_size),
            transforms.RandomHorizontalFlip(),
        ]

    raise ValueError(f"Unsupported dataset: {preset.name}")


def _tensor_steps(preset: DatasetPreset) -> list:
    return [
        transforms.ToTensor(),
        transforms.Normalize(preset.mean, preset.std),
    ]


def build_train_preprocess(
    preset: DatasetPreset,
    recipe_profile: str,
    use_basic_augmentation: bool = True,
):
    steps = []
    if use_basic_augmentation:
        steps.extend(_basic_aug_steps(preset, recipe_profile))
    steps.extend(_tensor_steps(preset))
    return transforms.Compose(steps)


def build_eval_preprocess(preset: DatasetPreset):
    steps = []
    if preset.name == "imagenet_a":
        steps.extend(
            [
                transforms.Resize(256),
                transforms.CenterCrop(preset.image_size),
            ]
        )
    steps.extend(_tensor_steps(preset))
    return transforms.Compose(steps)


def build_preprocess_pair(
    preset: DatasetPreset,
    recipe_profile: str,
    use_basic_augmentation: bool = True,
):
    return (
        build_train_preprocess(preset, recipe_profile, use_basic_augmentation),
        build_eval_preprocess(preset),
    )
