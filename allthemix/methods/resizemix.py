"""ResizeMix augmentation in PyTorch/XLA-friendly form."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

from allthemix.methods.cutmix import no_repeat_permutation, validate_nchw_images, validate_partner_batch, validate_targets


@dataclass(frozen=True)
class ResizeMixResult:
    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float
    index: torch.Tensor
    mask: torch.Tensor


def resize_source_to_box_nearest(
    source_images: torch.Tensor,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize each NCHW source image into a sampled paste box with nearest pixels."""

    if source_images.dim() != 4:
        raise ValueError(f"ResizeMix expects NCHW images, got shape {tuple(source_images.shape)}")
    image_height, image_width = int(source_images.size(-2)), int(source_images.size(-1))
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)

    y_positions = torch.arange(image_height, device=source_images.device)
    x_positions = torch.arange(image_width, device=source_images.device)
    grid_y = y_positions[:, None]
    grid_x = x_positions[None, :]

    inside_y = (grid_y >= y1) & (grid_y < y2)
    inside_x = (grid_x >= x1) & (grid_x < x2)
    mask = (inside_y & inside_x)[None, None, :, :]

    relative_y = grid_y - y1
    relative_x = grid_x - x1
    source_y = torch.floor(relative_y.float() * image_height / float(box_height)).long().clamp(0, image_height - 1)
    source_x = torch.floor(relative_x.float() * image_width / float(box_width)).long().clamp(0, image_width - 1)

    resized_source_full = source_images[:, :, source_y, source_x]
    return resized_source_full, mask


class ResizeMix:
    """Apply ResizeMix by shrinking a paired image into a random paste box."""

    def __init__(
        self,
        scope_min: float = 0.1,
        scope_max: float = 0.8,
        alpha: float = 1.0,
        use_alpha: bool = False,
        no_repeat: bool = False,
    ) -> None:
        if not (0.0 < scope_min <= scope_max <= 1.0):
            raise ValueError(
                "ResizeMix scope must satisfy 0 < resizemix_scope_min <= "
                f"resizemix_scope_max <= 1, got {scope_min}, {scope_max}."
            )
        if alpha <= 0:
            raise ValueError(f"resizemix_alpha must be positive, got {alpha}.")
        self.scope_min = float(scope_min)
        self.scope_max = float(scope_max)
        self.alpha = float(alpha)
        self.use_alpha = bool(use_alpha)
        self.no_repeat = bool(no_repeat)

    def sample_resize_ratio(self) -> float:
        if self.use_alpha:
            tao = float(np.random.beta(self.alpha, self.alpha))
            if self.scope_min <= tao <= self.scope_max:
                return tao
        return random.uniform(self.scope_min, self.scope_max)

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        partner_images: torch.Tensor | None = None,
        partner_targets: torch.Tensor | None = None,
        index: torch.Tensor | None = None,
    ) -> ResizeMixResult:
        validate_nchw_images(images, "ResizeMix")
        validate_targets(images, targets, "ResizeMix")
        if self.no_repeat and images.size(0) <= 1:
            raise ValueError("ResizeMix no_repeat requires batch size > 1.")

        if partner_images is None:
            index = no_repeat_permutation(images.size(0), images.device) if self.no_repeat else torch.randperm(images.size(0), device=images.device)
            partner_images = images[index]
            partner_targets = targets[index]
        elif partner_targets is None or index is None:
            raise ValueError("partner_targets and index are required when partner_images is provided.")
        else:
            validate_partner_batch(images, partner_images, partner_targets, index, "ResizeMix")

        image_height, image_width = int(images.size(-2)), int(images.size(-1))
        tao = self.sample_resize_ratio()
        cut_width = max(int(image_width * tao), 1)
        cut_height = max(int(image_height * tao), 1)
        center_x = random.randrange(image_width)
        center_y = random.randrange(image_height)

        x1 = max(center_x - cut_width // 2, 0)
        x2 = min(max(center_x + cut_width // 2, x1 + 1), image_width)
        y1 = max(center_y - cut_height // 2, 0)
        y2 = min(max(center_y + cut_height // 2, y1 + 1), image_height)

        resized_source_full, mask = resize_source_to_box_nearest(partner_images, x1=x1, y1=y1, x2=x2, y2=y2)
        mixed = torch.where(mask, resized_source_full, images)
        pasted_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        adjusted_lam = 1.0 - float(pasted_area) / float(image_height * image_width)

        return ResizeMixResult(
            images=mixed,
            targets_a=targets,
            targets_b=partner_targets,
            lam=adjusted_lam,
            index=index,
            mask=mask,
        )
