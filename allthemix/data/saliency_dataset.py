"""Dataset helpers for cached SaliencyMix maps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DATASET_SALIENCY_ALIASES = {
    "tinyimagenet": ("tinyimagenet", "tiny_imagenet", "tiny-imagenet"),
}


def saliency_path_candidates(dataset_name: str, saliency_dir: str | Path, saliency_path: str | Path | None = None) -> list[Path]:
    if saliency_path:
        return [Path(saliency_path)]
    names: Sequence[str] = DATASET_SALIENCY_ALIASES.get(dataset_name, (dataset_name,))
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


def load_train_saliency_maps(
    dataset_name: str,
    saliency_dir: str | Path,
    saliency_path: str | Path | None = None,
) -> np.ndarray:
    path = resolve_train_saliency_path(dataset_name, saliency_dir, saliency_path)
    saliency_maps = np.load(path).astype(np.float32)
    if saliency_maps.ndim not in (3, 4):
        raise ValueError(f"saliency maps must have shape N,H,W or N,1,H,W, got {saliency_maps.shape}.")
    return saliency_maps


class SaliencyMapDataset(Dataset):
    """Attach cached saliency maps to a base image classification dataset."""

    def __init__(self, base_dataset: Dataset, saliency_maps: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.saliency_maps = saliency_maps
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
        saliency_map = torch.from_numpy(self.saliency_maps[index])
        if saliency_map.dim() == 2:
            saliency_map = saliency_map.unsqueeze(0)
        elif saliency_map.dim() == 3 and saliency_map.shape[-1] == 1:
            saliency_map = saliency_map.permute(2, 0, 1).contiguous()
        elif saliency_map.dim() == 3 and saliency_map.shape[0] != 1:
            saliency_map = saliency_map.mean(dim=0, keepdim=True)

        if isinstance(image, torch.Tensor) and tuple(saliency_map.shape[-2:]) != tuple(image.shape[-2:]):
            saliency_map = F.interpolate(
                saliency_map.unsqueeze(0).float(),
                size=tuple(image.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return image, label, saliency_map.float()


def attach_train_saliency_maps(
    train_set: Dataset,
    dataset_name: str,
    saliency_dir: str | Path,
    saliency_path: str | Path | None = None,
) -> SaliencyMapDataset:
    saliency_maps = load_train_saliency_maps(dataset_name, saliency_dir, saliency_path)
    return SaliencyMapDataset(train_set, saliency_maps)


__all__ = [
    "SaliencyMapDataset",
    "attach_train_saliency_maps",
    "load_train_saliency_maps",
    "resolve_train_saliency_path",
    "saliency_path_candidates",
]
