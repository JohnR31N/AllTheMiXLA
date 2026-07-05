"""SaliencyMix augmentation in PyTorch.

This follows the AllTheMix SaliencyMix implementation: a single saliency-
centered CutMix rectangle is selected from the first shuffled sample and
applied to the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

from allthemix.methods.guided_sr import (
    _validate_nchw_images,
    _validate_positive,
    _validate_probability,
    _validate_targets,
    compute_spectral_residual_saliency_maps,
    ensure_nchw_saliency_maps,
    rgb_to_grayscale,
)


@dataclass(frozen=True)
class SaliencyMixResult:
    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float | torch.Tensor
    index: torch.Tensor
    mask: torch.Tensor
    saliency_maps: torch.Tensor


def sample_lam(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def normalize_saliency_map_batch(saliency_maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    saliency_maps = saliency_maps.to(dtype=torch.float32)
    saliency_maps = saliency_maps - saliency_maps.amin(dim=(-2, -1), keepdim=True)
    scale = saliency_maps.amax(dim=(-2, -1), keepdim=True)
    return saliency_maps / (scale + eps)


def compute_gradient_saliency_maps(images: torch.Tensor) -> torch.Tensor:
    """Compute a cheap edge-gradient saliency map without FFT."""

    _validate_nchw_images(images, "SaliencyMix")
    gray = rgb_to_grayscale(images)
    dx = torch.nn.functional.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (1, 0, 0, 0))
    dy = torch.nn.functional.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 1, 0))
    return normalize_saliency_map_batch(dx + dy)


def build_saliency_box_mask(
    saliency_map: torch.Tensor,
    cut_width: int,
    cut_height: int,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the shared SaliencyMix rectangle mask around the saliency peak."""

    if saliency_map.dim() == 3:
        saliency_map = saliency_map.squeeze(0)
    flat_index = torch.argmax(saliency_map.reshape(-1))
    center_y = torch.div(flat_index, image_width, rounding_mode="floor")
    center_x = torch.remainder(flat_index, image_width)

    cut_width_tensor = torch.as_tensor(cut_width, device=saliency_map.device, dtype=torch.long)
    cut_height_tensor = torch.as_tensor(cut_height, device=saliency_map.device, dtype=torch.long)
    width_tensor = torch.as_tensor(image_width, device=saliency_map.device, dtype=torch.long)
    height_tensor = torch.as_tensor(image_height, device=saliency_map.device, dtype=torch.long)

    x1 = torch.clamp(center_x - cut_width_tensor // 2, min=0, max=image_width)
    x2 = torch.clamp(center_x + cut_width_tensor // 2, min=0, max=image_width)
    y1 = torch.clamp(center_y - cut_height_tensor // 2, min=0, max=image_height)
    y2 = torch.clamp(center_y + cut_height_tensor // 2, min=0, max=image_height)

    y_positions = torch.arange(image_height, device=saliency_map.device)[:, None]
    x_positions = torch.arange(image_width, device=saliency_map.device)[None, :]
    spatial_mask = (y_positions >= y1) & (y_positions < y2) & (x_positions >= x1) & (x_positions < x2)
    patch_area = (x2 - x1) * (y2 - y1)
    patch_area = torch.clamp(patch_area, min=0)
    patch_area = torch.minimum(patch_area, height_tensor * width_tensor)
    return spatial_mask[None, None, :, :], patch_area


class SaliencyMix:
    """Apply official-style SaliencyMix to a mini-batch.

    Args:
        alpha: Beta distribution alpha for retained area.
        saliency_source: ``"batch"`` requires saliency maps from the dataloader.
            ``"spectral_residual"`` computes online maps when none are passed.
        blur_kernel: Kernel used by the online spectral-residual fallback.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        saliency_source: str = "spectral_residual",
        blur_kernel: int = 7,
    ) -> None:
        _validate_positive("saliencymix_alpha", float(alpha))
        self.alpha = float(alpha)
        self.saliency_source = str(saliency_source).lower()
        self.blur_kernel = int(blur_kernel)

    def _resolve_saliency_maps(
        self,
        images: torch.Tensor,
        saliency_maps: torch.Tensor | None,
    ) -> torch.Tensor:
        if saliency_maps is not None:
            return ensure_nchw_saliency_maps(saliency_maps, images)
        if self.saliency_source in {"gradient", "grad"}:
            return compute_gradient_saliency_maps(images)
        if self.saliency_source in {"spectral_residual", "guided_sr", "sr", "online"}:
            return compute_spectral_residual_saliency_maps(images, blur_kernel=self.blur_kernel)
        raise ValueError(
            "SaliencyMix requires saliency maps when saliency_source='batch'. "
            "Pass batch item (images, labels, saliency_maps) or set saliency_source: gradient/spectral_residual."
        )

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        saliency_maps: torch.Tensor | None = None,
        partner_images: torch.Tensor | None = None,
        partner_targets: torch.Tensor | None = None,
        partner_saliency_maps: torch.Tensor | None = None,
        index: torch.Tensor | None = None,
    ) -> SaliencyMixResult:
        _validate_nchw_images(images, "SaliencyMix")
        _validate_targets(images, targets, "SaliencyMix")

        saliency_maps = self._resolve_saliency_maps(images, saliency_maps)
        if partner_images is None:
            index = torch.randperm(images.size(0), device=images.device)
            partner_images = images[index]
            partner_targets = targets[index]
            partner_saliency_maps = saliency_maps[index]
        elif partner_targets is None or index is None:
            raise ValueError("partner_targets and index are required when partner_images is provided.")
        elif partner_saliency_maps is None:
            partner_saliency_maps = self._resolve_saliency_maps(partner_images, None)
        else:
            partner_saliency_maps = ensure_nchw_saliency_maps(partner_saliency_maps, partner_images)

        lam = sample_lam(self.alpha)
        image_height, image_width = int(images.size(-2)), int(images.size(-1))
        cut_ratio = float(np.sqrt(1.0 - lam))
        cut_width = int(image_width * cut_ratio)
        cut_height = int(image_height * cut_ratio)
        saliency_map_for_bbox = partner_saliency_maps[0]

        mask, patch_area = build_saliency_box_mask(
            saliency_map=saliency_map_for_bbox,
            cut_width=cut_width,
            cut_height=cut_height,
            image_height=image_height,
            image_width=image_width,
        )
        mask = mask.to(device=images.device)
        mixed = torch.where(mask, partner_images, images)
        adjusted_lam = 1.0 - patch_area.to(dtype=images.dtype) / float(image_height * image_width)

        return SaliencyMixResult(
            images=mixed,
            targets_a=targets,
            targets_b=partner_targets,
            lam=adjusted_lam,
            index=index,
            mask=mask,
            saliency_maps=saliency_maps,
        )


def saliencymix(
    images: torch.Tensor,
    targets: torch.Tensor,
    saliency_maps: torch.Tensor | None = None,
    alpha: float = 1.0,
    prob: float = 0.5,
    saliency_source: str = "spectral_residual",
    blur_kernel: int = 7,
) -> SaliencyMixResult:
    """Functional SaliencyMix wrapper with batch-level probability."""

    _validate_probability("saliencymix_prob", float(prob))
    if random.random() >= float(prob):
        clean_saliency = torch.zeros(
            (images.size(0), 1, images.size(-2), images.size(-1)),
            device=images.device,
            dtype=torch.float32,
        )
        index = torch.arange(images.size(0), device=images.device)
        mask = torch.zeros_like(clean_saliency, dtype=torch.bool)
        return SaliencyMixResult(images, targets, targets, 1.0, index, mask, clean_saliency)
    return SaliencyMix(alpha=alpha, saliency_source=saliency_source, blur_kernel=blur_kernel)(
        images,
        targets,
        saliency_maps=saliency_maps,
    )
