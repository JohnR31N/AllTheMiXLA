"""GuidedMixup / Guided-SR augmentations in PyTorch.

The implementation mirrors the JAX AllTheMix Guided-SR path:
online spectral-residual saliency maps are computed from the input images,
then samples are paired either randomly or by the greedy one-cycle saliency
distance rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F


INF = 1e9


@dataclass(frozen=True)
class GuidedSRResult:
    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: torch.Tensor
    index: torch.Tensor
    saliency_maps: torch.Tensor
    mask: torch.Tensor


def _validate_nchw_images(images: torch.Tensor, method_name: str) -> None:
    if images.dim() != 4:
        raise ValueError(f"{method_name} expects NCHW images, got shape {tuple(images.shape)}")


def _validate_targets(images: torch.Tensor, targets: torch.Tensor, method_name: str) -> None:
    if targets.dim() != 1 or targets.size(0) != images.size(0):
        raise ValueError(
            f"{method_name} image/label batch mismatch: "
            f"images batch={images.size(0)}, labels shape={tuple(targets.shape)}"
        )


def _validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_odd_positive_int(name: str, value: int) -> None:
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}.")


def ensure_nchw_saliency_maps(saliency_maps: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Convert saliency maps to ``N,1,H,W`` and validate spatial shape."""

    if saliency_maps.dim() == 3:
        saliency_maps = saliency_maps.unsqueeze(1)
    elif saliency_maps.dim() == 4 and saliency_maps.size(1) != 1 and saliency_maps.size(-1) == 1:
        saliency_maps = saliency_maps.permute(0, 3, 1, 2).contiguous()
    elif saliency_maps.dim() != 4:
        raise ValueError(f"saliency maps must be 3D or 4D, got shape {tuple(saliency_maps.shape)}")

    if saliency_maps.size(0) != images.size(0) or saliency_maps.shape[-2:] != images.shape[-2:]:
        raise ValueError(
            "saliency maps must match image batch, height and width: "
            f"maps={tuple(saliency_maps.shape)}, images={tuple(images.shape)}"
        )

    if saliency_maps.size(1) != 1:
        saliency_maps = saliency_maps.mean(dim=1, keepdim=True)

    return saliency_maps.to(device=images.device, dtype=torch.float32)


def normalize_saliency_maps(saliency_maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize saliency maps into nonnegative per-sample distributions."""

    saliency_maps = torch.clamp(saliency_maps, min=0.0)
    saliency_sum = saliency_maps.sum(dim=(1, 2, 3), keepdim=True)
    return saliency_maps / (saliency_sum + eps)


def make_gaussian_kernel_1d(kernel_size: int, sigma: float = 3.0, device=None, dtype=None) -> torch.Tensor:
    _validate_odd_positive_int("guidedmixup_blur_kernel", int(kernel_size))
    if kernel_size <= 1:
        return torch.ones((1,), device=device, dtype=dtype or torch.float32)

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype or torch.float32)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur_2d_single_channel(
    saliency_maps: torch.Tensor,
    kernel_size: int,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Apply torchvision-style reflect-padded separable Gaussian blur."""

    _validate_odd_positive_int("guidedmixup_blur_kernel", int(kernel_size))
    if kernel_size <= 1:
        return saliency_maps

    kernel = make_gaussian_kernel_1d(
        int(kernel_size),
        sigma=sigma,
        device=saliency_maps.device,
        dtype=saliency_maps.dtype,
    )
    pad = int(kernel_size) // 2
    kernel_y = kernel.view(1, 1, -1, 1)
    kernel_x = kernel.view(1, 1, 1, -1)

    blurred = F.conv2d(F.pad(saliency_maps, (0, 0, pad, pad), mode="reflect"), kernel_y)
    return F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode="reflect"), kernel_x)


def mean_filter_2d_single_channel(values: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size <= 1:
        return values
    _validate_odd_positive_int("spectral_residual_kernel_size", int(kernel_size))

    pad = int(kernel_size) // 2
    kernel = torch.full(
        (1, 1, int(kernel_size), int(kernel_size)),
        1.0 / float(kernel_size * kernel_size),
        device=values.device,
        dtype=values.dtype,
    )
    return F.conv2d(F.pad(values, (pad, pad, pad, pad), mode="replicate"), kernel)


def rgb_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    if images.size(1) == 1:
        return images.to(dtype=torch.float32)
    if images.size(1) != 3:
        raise ValueError("Guided-SR expects images with either 1 or 3 channels.")

    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device, dtype=torch.float32).view(1, 3, 1, 1)
    return (images.to(dtype=torch.float32) * weights).sum(dim=1, keepdim=True)


def compute_spectral_residual_saliency_maps(
    images: torch.Tensor,
    blur_kernel: int = 7,
    blur_sigma: float = 3.0,
    spectral_kernel_size: int = 3,
    max_size: int = 128,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute Guided-SR spectral-residual saliency maps for NCHW images."""

    _validate_nchw_images(images, "Guided-SR")
    grayscale = rgb_to_grayscale(images)
    image_height, image_width = int(grayscale.size(-2)), int(grayscale.size(-1))
    needs_resize = max(image_height, image_width) > int(max_size)

    if needs_resize:
        grayscale = F.interpolate(grayscale, size=(int(max_size), int(max_size)), mode="bilinear", align_corners=False)

    frequency = torch.fft.fft2(grayscale, dim=(-2, -1))
    magnitude = torch.sqrt(frequency.real.square() + frequency.imag.square() + eps)
    log_magnitude = torch.log(magnitude)
    local_average = mean_filter_2d_single_channel(log_magnitude, kernel_size=int(spectral_kernel_size))
    residual_scale = torch.exp(log_magnitude - local_average)
    residual_frequency = frequency * (residual_scale / magnitude).to(dtype=frequency.dtype)
    saliency_response = torch.fft.ifft2(residual_frequency, dim=(-2, -1))
    saliency_maps = torch.abs(saliency_response)
    saliency_maps = gaussian_blur_2d_single_channel(saliency_maps, kernel_size=int(blur_kernel), sigma=blur_sigma)

    if needs_resize:
        saliency_maps = F.interpolate(saliency_maps, size=(image_height, image_width), mode="bilinear", align_corners=False)

    return torch.clamp(saliency_maps, min=0.0)


def compute_l2_distance_matrix(saliency_maps: torch.Tensor) -> torch.Tensor:
    flat_maps = saliency_maps.reshape(saliency_maps.size(0), -1)
    diff = flat_maps[:, None, :] - flat_maps[None, :, :]
    return torch.sqrt(diff.square().sum(dim=-1) + 1e-12)


def onecycle_cover(distance_matrix: torch.Tensor) -> torch.Tensor:
    """Greedy one-cycle pairing from the GuidedMixup reference implementation."""

    batch_size = int(distance_matrix.size(0))
    if batch_size <= 1:
        return torch.arange(batch_size, device=distance_matrix.device, dtype=torch.long)

    matrix = distance_matrix.clone()
    eye = torch.eye(batch_size, device=matrix.device, dtype=torch.bool)
    matrix = torch.where(eye, torch.full_like(matrix, -INF), matrix)

    max_idx = torch.argmax(matrix)
    row = torch.div(max_idx, batch_size, rounding_mode="floor").to(torch.long)
    col = torch.remainder(max_idx, batch_size).to(torch.long)
    first_row = row

    permutation = torch.zeros((batch_size,), device=matrix.device, dtype=torch.long)
    permutation = permutation.scatter(0, row.reshape(1), col.reshape(1))
    matrix = matrix.index_fill(1, row.reshape(1), -INF)

    for _ in range(batch_size - 2):
        row = col
        col = torch.argmax(matrix[row]).to(torch.long)
        permutation = permutation.scatter(0, row.reshape(1), col.reshape(1))
        matrix = matrix.index_fill(1, row.reshape(1), -INF)

    permutation = permutation.scatter(0, col.reshape(1), first_row.reshape(1))
    return permutation


def build_pairing(saliency_maps: torch.Tensor, condition: str = "greedy") -> torch.Tensor:
    condition = str(condition).lower()
    batch_size = int(saliency_maps.size(0))
    if condition == "random":
        return torch.randperm(batch_size, device=saliency_maps.device)
    if condition == "greedy":
        return onecycle_cover(compute_l2_distance_matrix(saliency_maps))
    raise ValueError(f"guidedmixup_condition must be one of: random, greedy. Got {condition}.")


def guidedmixup_from_saliency(
    images: torch.Tensor,
    targets: torch.Tensor,
    saliency_maps: torch.Tensor,
    blur_kernel: int = 7,
    condition: str = "greedy",
    eps: float = 1e-8,
) -> GuidedSRResult:
    _validate_nchw_images(images, "GuidedMixup")
    _validate_targets(images, targets, "GuidedMixup")
    _validate_odd_positive_int("guidedmixup_blur_kernel", int(blur_kernel))

    saliency_maps = ensure_nchw_saliency_maps(saliency_maps, images)
    saliency_maps = normalize_saliency_maps(saliency_maps, eps=eps)
    saliency_maps = gaussian_blur_2d_single_channel(saliency_maps, kernel_size=int(blur_kernel), sigma=3.0)
    saliency_maps = normalize_saliency_maps(saliency_maps, eps=eps)

    index = build_pairing(saliency_maps, condition=condition)
    paired_images = images[index]
    paired_targets = targets[index]
    paired_saliency_maps = saliency_maps[index]

    pixel_mask = saliency_maps / (saliency_maps + paired_saliency_maps + eps)
    guided_images = pixel_mask.to(dtype=images.dtype) * images + (1.0 - pixel_mask.to(dtype=images.dtype)) * paired_images
    lam = pixel_mask.mean(dim=(1, 2, 3)).to(dtype=images.dtype)

    return GuidedSRResult(
        images=guided_images,
        targets_a=targets,
        targets_b=paired_targets,
        lam=lam,
        index=index,
        saliency_maps=saliency_maps,
        mask=pixel_mask,
    )


class GuidedSR:
    """Apply Guided-SR with online spectral-residual saliency."""

    def __init__(
        self,
        alpha: float = 1.0,
        blur_kernel: int = 7,
        condition: str = "greedy",
        eps: float = 1e-8,
    ) -> None:
        _validate_positive("guidedmixup_alpha", float(alpha))
        _validate_odd_positive_int("guidedmixup_blur_kernel", int(blur_kernel))
        if str(condition).lower() not in {"random", "greedy"}:
            raise ValueError(f"guidedmixup_condition must be one of: random, greedy. Got {condition}.")
        self.alpha = float(alpha)
        self.blur_kernel = int(blur_kernel)
        self.condition = str(condition).lower()
        self.eps = float(eps)

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        saliency_maps: torch.Tensor | None = None,
        **_: object,
    ) -> GuidedSRResult:
        _validate_nchw_images(images, "Guided-SR")
        _validate_targets(images, targets, "Guided-SR")
        if saliency_maps is None:
            saliency_maps = compute_spectral_residual_saliency_maps(images, blur_kernel=self.blur_kernel)
        return guidedmixup_from_saliency(
            images=images,
            targets=targets,
            saliency_maps=saliency_maps,
            blur_kernel=self.blur_kernel,
            condition=self.condition,
            eps=self.eps,
        )


def guided_sr(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    prob: float = 1.0,
    blur_kernel: int = 7,
    condition: str = "greedy",
) -> GuidedSRResult:
    """Functional Guided-SR wrapper with batch-level probability."""

    _validate_probability("guidedmixup_prob", float(prob))
    if random.random() >= float(prob):
        saliency_maps = torch.zeros(
            (images.size(0), 1, images.size(-2), images.size(-1)),
            device=images.device,
            dtype=torch.float32,
        )
        lam = torch.ones((images.size(0),), device=images.device, dtype=images.dtype)
        index = torch.arange(images.size(0), device=images.device)
        mask = torch.ones_like(saliency_maps)
        return GuidedSRResult(images, targets, targets, lam, index, saliency_maps, mask)
    return GuidedSR(alpha=alpha, blur_kernel=blur_kernel, condition=condition)(images, targets)
