"""Sample-level basic augmentation and normalization."""

from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from allthemix.cli.presets import DatasetPreset


TRAIN_AUGMENTATION_RECIPES = {
    "none",
    "basic",
    "hflip",
    "horizontal_flip",
    "imagenet",
    "tiny_official",
    "tiny_openmixup",
}


def resolve_augmentation_recipe(
    use_basic_augmentation: bool,
    augmentation_recipe: str | None = None,
) -> str:
    recipe = str(augmentation_recipe or "").lower()
    if not recipe:
        return "basic" if use_basic_augmentation else "none"
    if recipe not in TRAIN_AUGMENTATION_RECIPES:
        raise ValueError(f"Unsupported aug_recipe: {augmentation_recipe}")
    return recipe


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


def _recipe_aug_steps(preset: DatasetPreset, recipe: str) -> list:
    if recipe == "none":
        return []
    if recipe in {"hflip", "horizontal_flip", "tiny_official"}:
        return [transforms.RandomHorizontalFlip()]
    if recipe == "tiny_openmixup":
        return [
            transforms.RandomResizedCrop(
                preset.image_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    if recipe == "imagenet":
        return [
            transforms.RandomResizedCrop(preset.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ]
    if recipe == "basic":
        return [
            transforms.RandomCrop(preset.image_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
    raise ValueError(f"Unsupported aug_recipe: {recipe}")


def _tensor_steps(preset: DatasetPreset, normalize: bool = True) -> list:
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(preset.mean, preset.std))
    return steps


def build_train_preprocess(
    preset: DatasetPreset,
    recipe_profile: str,
    use_basic_augmentation: bool = True,
    augmentation_recipe: str | None = None,
    normalize: bool = True,
):
    steps = []
    if augmentation_recipe is not None:
        steps.extend(_recipe_aug_steps(preset, resolve_augmentation_recipe(use_basic_augmentation, augmentation_recipe)))
    elif use_basic_augmentation:
        steps.extend(_basic_aug_steps(preset, recipe_profile))
    steps.extend(_tensor_steps(preset, normalize=normalize))
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
    augmentation_recipe: str | None = None,
    normalize_train: bool = True,
):
    return (
        build_train_preprocess(
            preset,
            recipe_profile,
            use_basic_augmentation,
            augmentation_recipe,
            normalize=normalize_train,
        ),
        build_eval_preprocess(preset),
    )
