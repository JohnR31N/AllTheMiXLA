"""Dataset helpers for cached SaliencyMix maps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
import torchvision.transforms.functional as TF


DATASET_SALIENCY_ALIASES = {
    "tinyimagenet": ("tinyimagenet", "tiny_imagenet", "tiny-imagenet"),
}
SALIENCY_AUGMENTATION_RECIPES = {
    "none",
    "basic",
    "hflip",
    "horizontal_flip",
    "imagenet",
    "tiny_official",
    "tiny_openmixup",
}


def _saliency_dataset_names(dataset_name: str) -> tuple[str, ...]:
    requested = str(dataset_name).lower()
    canonical = "tinyimagenet" if requested in {"tinyimagenet", "tiny_imagenet", "tiny-imagenet"} else requested
    names = [requested]
    names.extend(DATASET_SALIENCY_ALIASES.get(canonical, (canonical,)))
    return tuple(dict.fromkeys(names))


def saliency_path_candidates(dataset_name: str, saliency_dir: str | Path, saliency_path: str | Path | None = None) -> list[Path]:
    if saliency_path:
        return [Path(saliency_path)]
    names: Sequence[str] = _saliency_dataset_names(dataset_name)
    root = Path(saliency_dir)
    return [root / f"{name}_train_saliency.npy" for name in names]


def resolve_train_saliency_path(
    dataset_name: str,
    saliency_dir: str | Path,
    saliency_path: str | Path | None = None,
) -> Path:
    candidates = saliency_path_candidates(dataset_name, saliency_dir, saliency_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find precomputed train saliency maps. Tried:\n"
        f"{formatted}\n"
        "Generate a cache or use saliency_source: gradient for a fast TPU smoke test."
    )


def saliency_array_is_finite(saliency_maps: np.ndarray, chunk_size: int = 8192) -> bool:
    """Check a saliency cache without materializing a full-size boolean mask."""

    if saliency_maps.ndim == 0:
        return bool(np.isfinite(saliency_maps).all())
    num_maps = int(saliency_maps.shape[0])
    chunk_size = max(1, int(chunk_size))
    for start in range(0, num_maps, chunk_size):
        if not np.isfinite(saliency_maps[start : start + chunk_size]).all():
            return False
    return True


def load_train_saliency_maps(
    dataset_name: str,
    saliency_dir: str | Path,
    saliency_path: str | Path | None = None,
    validate_finite: bool = True,
) -> np.ndarray:
    path = resolve_train_saliency_path(dataset_name, saliency_dir, saliency_path)
    saliency_maps = np.load(path, mmap_mode="r")
    if saliency_maps.ndim not in (3, 4):
        raise ValueError(f"saliency maps must have shape N,H,W or N,1,H,W, got {saliency_maps.shape}.")
    if not np.issubdtype(saliency_maps.dtype, np.number):
        raise ValueError(f"saliency maps at {path} must be numeric, got dtype={saliency_maps.dtype}.")
    if validate_finite and not saliency_array_is_finite(saliency_maps):
        raise ValueError(f"saliency maps at {path} contain NaN or infinite values.")
    return saliency_maps


def resolve_saliency_augmentation_recipe(
    use_sal_basic_augmentation: bool = False,
    saliency_augmentation_recipe: str | None = None,
) -> str:
    recipe = str(saliency_augmentation_recipe or "").lower()
    if not recipe:
        recipe = "basic" if use_sal_basic_augmentation else "none"
    if recipe not in SALIENCY_AUGMENTATION_RECIPES:
        raise ValueError(f"Unsupported sal_aug_recipe: {saliency_augmentation_recipe}")
    return recipe


def _resize_saliency_to_image(saliency_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if tuple(saliency_map.shape[-2:]) == tuple(image.shape[-2:]):
        return saliency_map.float()
    return F.interpolate(
        saliency_map.unsqueeze(0).float(),
        size=tuple(image.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _normalize_tensor_image(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError(f"expected CHW tensor image before normalization, got shape {tuple(image.shape)}.")
    if image.size(0) != len(mean) or image.size(0) != len(std):
        raise ValueError(
            "normalization stats must match image channels: "
            f"channels={image.size(0)}, mean={len(mean)}, std={len(std)}."
        )
    mean_tensor = torch.as_tensor(mean, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.as_tensor(std, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor


def _random_crop_with_padding_pair(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    image_size: int,
    padding: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect").squeeze(0)
    saliency_map = F.pad(saliency_map.unsqueeze(0), (padding, padding, padding, padding), mode="reflect").squeeze(0)
    max_top = int(image.size(-2)) - int(image_size)
    max_left = int(image.size(-1)) - int(image_size)
    top = int(torch.randint(0, max(max_top, 0) + 1, ()).item())
    left = int(torch.randint(0, max(max_left, 0) + 1, ()).item())
    return (
        TF.crop(image, top, left, int(image_size), int(image_size)),
        TF.crop(saliency_map, top, left, int(image_size), int(image_size)),
    )


def _random_resized_crop_pair(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    image_size: int,
    image_interpolation: InterpolationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    top, left, height, width = RandomResizedCrop.get_params(
        image,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
    )
    return (
        TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            (int(image_size), int(image_size)),
            interpolation=image_interpolation,
            antialias=True,
        ),
        TF.resized_crop(
            saliency_map,
            top,
            left,
            height,
            width,
            (int(image_size), int(image_size)),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    )


def _random_horizontal_flip_pair(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool((torch.rand(()) < 0.5).item()):
        return TF.hflip(image), TF.hflip(saliency_map)
    return image, saliency_map


def apply_paired_saliency_augmentation(
    image: torch.Tensor,
    saliency_map: torch.Tensor,
    image_size: int,
    use_sal_basic_augmentation: bool = False,
    saliency_augmentation_recipe: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same spatial augmentation to an image tensor and saliency map."""

    recipe = resolve_saliency_augmentation_recipe(use_sal_basic_augmentation, saliency_augmentation_recipe)
    if recipe == "none":
        return image, saliency_map.float()
    if not isinstance(image, torch.Tensor):
        raise TypeError("paired saliency augmentation expects the base dataset to return tensor images.")

    saliency_map = _resize_saliency_to_image(saliency_map, image)
    if recipe == "basic":
        image, saliency_map = _random_crop_with_padding_pair(image, saliency_map, image_size=int(image_size))
        return _random_horizontal_flip_pair(image, saliency_map)
    if recipe in {"hflip", "horizontal_flip", "tiny_official"}:
        return _random_horizontal_flip_pair(image, saliency_map)
    if recipe == "tiny_openmixup":
        image, saliency_map = _random_resized_crop_pair(
            image,
            saliency_map,
            image_size=int(image_size),
            image_interpolation=InterpolationMode.BICUBIC,
        )
        return _random_horizontal_flip_pair(image, saliency_map)

    image, saliency_map = _random_resized_crop_pair(
        image,
        saliency_map,
        image_size=int(image_size),
        image_interpolation=InterpolationMode.BILINEAR,
    )
    return _random_horizontal_flip_pair(image, saliency_map)


class SaliencyMapDataset(Dataset):
    """Attach cached saliency maps to a base image classification dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        saliency_maps: np.ndarray,
        use_sal_basic_augmentation: bool = False,
        saliency_augmentation_recipe: str | None = None,
        image_size: int | None = None,
        normalization_mean: Sequence[float] | None = None,
        normalization_std: Sequence[float] | None = None,
        validate_sample_finite: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.saliency_maps = saliency_maps
        self.saliency_augmentation_recipe = resolve_saliency_augmentation_recipe(
            use_sal_basic_augmentation,
            saliency_augmentation_recipe,
        )
        self.image_size = image_size
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("normalization_mean and normalization_std must be provided together.")
        self.normalization_mean = tuple(float(value) for value in normalization_mean) if normalization_mean is not None else None
        self.normalization_std = tuple(float(value) for value in normalization_std) if normalization_std is not None else None
        self.validate_sample_finite = bool(validate_sample_finite)
        if len(base_dataset) != int(saliency_maps.shape[0]):
            raise ValueError(
                "Number of saliency maps does not match dataset length: "
                f"maps={saliency_maps.shape[0]}, dataset={len(base_dataset)}."
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError("base dataset must return at least (image, label).")
        image, label = sample[:2]
        saliency_array = np.asarray(self.saliency_maps[index], dtype=np.float32).copy()
        saliency_map = torch.from_numpy(saliency_array)
        if saliency_map.dim() == 2:
            saliency_map = saliency_map.unsqueeze(0)
        elif saliency_map.dim() == 3 and saliency_map.shape[-1] == 1:
            saliency_map = saliency_map.permute(2, 0, 1).contiguous()
        elif saliency_map.dim() == 3 and saliency_map.shape[0] != 1:
            saliency_map = saliency_map.mean(dim=0, keepdim=True)
        if self.validate_sample_finite and not torch.isfinite(saliency_map).all():
            raise ValueError(f"saliency map at index {index} contains NaN or infinite values.")

        if isinstance(image, torch.Tensor):
            saliency_map = _resize_saliency_to_image(saliency_map, image)
            if self.saliency_augmentation_recipe != "none":
                image_size = int(self.image_size or image.shape[-1])
                image, saliency_map = apply_paired_saliency_augmentation(
                    image,
                    saliency_map,
                    image_size=image_size,
                    saliency_augmentation_recipe=self.saliency_augmentation_recipe,
                )
            if self.normalization_mean is not None and self.normalization_std is not None:
                image = _normalize_tensor_image(image, self.normalization_mean, self.normalization_std)
        elif self.saliency_augmentation_recipe != "none":
            raise TypeError("paired saliency augmentation expects the base dataset to return tensor images.")

        return image, label, saliency_map.float()


def attach_train_saliency_maps(
    train_set: Dataset,
    dataset_name: str,
    saliency_dir: str | Path,
    saliency_path: str | Path | None = None,
    use_sal_basic_augmentation: bool = False,
    saliency_augmentation_recipe: str | None = None,
    image_size: int | None = None,
    normalization_mean: Sequence[float] | None = None,
    normalization_std: Sequence[float] | None = None,
    validate_finite: bool = True,
    validate_sample_finite: bool = True,
) -> SaliencyMapDataset:
    saliency_maps = load_train_saliency_maps(
        dataset_name,
        saliency_dir,
        saliency_path,
        validate_finite=validate_finite,
    )
    return SaliencyMapDataset(
        train_set,
        saliency_maps,
        use_sal_basic_augmentation=use_sal_basic_augmentation,
        saliency_augmentation_recipe=saliency_augmentation_recipe,
        image_size=image_size,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        validate_sample_finite=validate_sample_finite,
    )


__all__ = [
    "SaliencyMapDataset",
    "apply_paired_saliency_augmentation",
    "attach_train_saliency_maps",
    "load_train_saliency_maps",
    "resolve_saliency_augmentation_recipe",
    "resolve_train_saliency_path",
    "saliency_array_is_finite",
    "saliency_path_candidates",
]
