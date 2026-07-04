"""Dataset and training presets for FMix reproduction runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    num_classes: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class RecipePreset:
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    scheduler: str
    milestones: tuple[int, ...]
    alpha: float
    decay_power: float
    max_soft: float
    transform_profile: str


DATASETS: dict[str, DatasetPreset] = {
    "cifar10": DatasetPreset(
        name="cifar10",
        num_classes=10,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
    "cifar100": DatasetPreset(
        name="cifar100",
        num_classes=100,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
    "tinyimagenet": DatasetPreset(
        name="tinyimagenet",
        num_classes=200,
        image_size=64,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "imagenet_a": DatasetPreset(
        name="imagenet_a",
        num_classes=1000,
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


OPENMIXUP_RECIPES: dict[str, RecipePreset] = {
    "cifar10": RecipePreset(
        epochs=400,
        batch_size=100,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler="cosine",
        milestones=(),
        alpha=0.2,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="openmixup",
    ),
    "cifar100": RecipePreset(
        epochs=400,
        batch_size=100,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler="cosine",
        milestones=(),
        alpha=0.2,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="openmixup",
    ),
    "tinyimagenet": RecipePreset(
        epochs=400,
        batch_size=100,
        lr=0.2,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler="cosine",
        milestones=(),
        alpha=1.0,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="openmixup",
    ),
    "imagenet_a": RecipePreset(
        epochs=0,
        batch_size=100,
        lr=0.0,
        momentum=0.9,
        weight_decay=0.0,
        scheduler="cosine",
        milestones=(),
        alpha=1.0,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="imagenet_a",
    ),
}


OFFICIAL_RECIPES: dict[str, RecipePreset] = {
    "cifar10": RecipePreset(
        epochs=200,
        batch_size=128,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler="multistep",
        milestones=(100, 150),
        alpha=1.0,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="official",
    ),
    "cifar100": RecipePreset(
        epochs=200,
        batch_size=128,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler="multistep",
        milestones=(100, 150),
        alpha=1.0,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="official",
    ),
    "tinyimagenet": RecipePreset(
        epochs=200,
        batch_size=128,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler="multistep",
        milestones=(150, 180),
        alpha=1.0,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="official",
    ),
    "imagenet_a": RecipePreset(
        epochs=0,
        batch_size=100,
        lr=0.0,
        momentum=0.9,
        weight_decay=0.0,
        scheduler="cosine",
        milestones=(),
        alpha=1.0,
        decay_power=3.0,
        max_soft=0.0,
        transform_profile="imagenet_a",
    ),
}


RECIPES = {
    "openmixup": OPENMIXUP_RECIPES,
    "official": OFFICIAL_RECIPES,
}


def get_dataset_preset(dataset: str) -> DatasetPreset:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {dataset!r}; valid choices: {valid}") from exc


def get_recipe_preset(dataset: str, recipe: str) -> RecipePreset:
    try:
        return RECIPES[recipe][dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown recipe {recipe!r}; valid choices: {valid}") from exc


def preset_dict(dataset: str, recipe: str) -> dict[str, object]:
    return {
        "dataset": asdict(get_dataset_preset(dataset)),
        "recipe": asdict(get_recipe_preset(dataset, recipe)),
    }
