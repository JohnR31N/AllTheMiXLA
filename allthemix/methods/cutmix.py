"""CutMix augmentation in PyTorch/XLA-friendly form."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True)
class CutMixResult:
    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float
    index: torch.Tensor
    mask: torch.Tensor


def sample_lam(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def validate_nchw_images(images: torch.Tensor, method_name: str) -> None:
    if images.dim() != 4:
        raise ValueError(f"{method_name} expects NCHW images, got shape {tuple(images.shape)}")


def validate_targets(images: torch.Tensor, targets: torch.Tensor, method_name: str) -> None:
    if targets.dim() != 1 or targets.size(0) != images.size(0):
        raise ValueError(f"{method_name} image/label batch mismatch.")


def validate_partner_batch(
    images: torch.Tensor,
    partner_images: torch.Tensor,
    partner_targets: torch.Tensor,
    index: torch.Tensor,
    method_name: str,
) -> None:
    validate_nchw_images(partner_images, f"{method_name} partner")
    if partner_images.shape != images.shape:
        raise ValueError(
            f"{method_name} partner images must match image batch shape: "
            f"images={tuple(images.shape)}, partners={tuple(partner_images.shape)}"
        )
    validate_targets(partner_images, partner_targets, f"{method_name} partner")
    if index.dim() != 1 or index.size(0) != images.size(0):
        raise ValueError(
            f"{method_name} partner index must be a 1D tensor with one entry per image: "
            f"index={tuple(index.shape)}, images batch={images.size(0)}"
        )


def no_repeat_permutation(batch_size: int, device: torch.device) -> torch.Tensor:
    """Create a permutation without fixed points when batch size allows it."""

    if batch_size <= 1:
        return torch.arange(batch_size, device=device)
    order = torch.randperm(batch_size, device=device)
    shifted_order = torch.roll(order, shifts=1, dims=0)
    permutation = torch.empty(batch_size, device=device, dtype=torch.long)
    permutation.scatter_(0, order, shifted_order)
    return permutation


def build_random_box(
    lam: float,
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    cut_ratio = float(np.sqrt(max(0.0, 1.0 - lam)))
    cut_width = int(image_width * cut_ratio)
    cut_height = int(image_height * cut_ratio)
    center_x = random.randrange(image_width)
    center_y = random.randrange(image_height)

    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, image_width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, image_height)
    return x1, y1, x2, y2


def box_mask(
    image_height: int,
    image_width: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    device: torch.device,
) -> torch.Tensor:
    y_positions = torch.arange(image_height, device=device)[:, None]
    x_positions = torch.arange(image_width, device=device)[None, :]
    mask = (y_positions >= y1) & (y_positions < y2) & (x_positions >= x1) & (x_positions < x2)
    return mask[None, None, :, :]


class CutMix:
    """Apply CutMix by pasting a paired image rectangle."""

    def __init__(self, alpha: float = 1.0, no_repeat: bool = False) -> None:
        if alpha <= 0:
            raise ValueError(f"cutmix_alpha must be positive, got {alpha}.")
        self.alpha = float(alpha)
        self.no_repeat = bool(no_repeat)

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        partner_images: torch.Tensor | None = None,
        partner_targets: torch.Tensor | None = None,
        index: torch.Tensor | None = None,
    ) -> CutMixResult:
        validate_nchw_images(images, "CutMix")
        validate_targets(images, targets, "CutMix")
        if self.no_repeat and images.size(0) <= 1:
            raise ValueError("CutMix no_repeat requires batch size > 1.")

        if partner_images is None:
            index = no_repeat_permutation(images.size(0), images.device) if self.no_repeat else torch.randperm(images.size(0), device=images.device)
            partner_images = images[index]
            partner_targets = targets[index]
        elif partner_targets is None or index is None:
            raise ValueError("partner_targets and index are required when partner_images is provided.")
        else:
            validate_partner_batch(images, partner_images, partner_targets, index, "CutMix")

        lam = sample_lam(self.alpha)
        image_height, image_width = int(images.size(-2)), int(images.size(-1))
        x1, y1, x2, y2 = build_random_box(lam, image_height, image_width)
        mask = box_mask(image_height, image_width, x1, y1, x2, y2, images.device)
        mixed = torch.where(mask, partner_images, images)

        patch_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        adjusted_lam = 1.0 - float(patch_area) / float(image_height * image_width)
        return CutMixResult(
            images=mixed,
            targets_a=targets,
            targets_b=partner_targets,
            lam=adjusted_lam,
            index=index,
            mask=mask,
        )
