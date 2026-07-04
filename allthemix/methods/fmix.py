"""FMix augmentation with the official Fourier-mask sampling recipe.

The mask generator mirrors the public FMix implementation:
https://github.com/ecs-vlc/FMix/blob/master/fmix.py

The torch wrapper avoids CUDA-specific calls so the same code can run on CPU,
CUDA, or PyTorch/XLA devices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


ShapeLike = int | Sequence[int]


def _shape_tuple(shape: ShapeLike) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(v) for v in shape)


def fftfreqnd(h: int, w: int | None = None, z: int | None = None) -> np.ndarray:
    """Get bin values for a one-, two-, or three-dimensional FFT."""

    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)
        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(
    freqs: np.ndarray,
    decay_power: float,
    ch: int,
    h: int,
    w: int = 0,
    z: int = 0,
) -> np.ndarray:
    """Sample a Fourier image whose amplitudes decay as 1/f**decay_power."""

    min_freq = np.array([1.0 / max(w, h, z)])
    scale = np.ones(1) / (np.maximum(freqs, min_freq) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)
    scale = np.expand_dims(scale, -1)[None, :]
    return scale * param


def make_low_freq_image(decay: float, shape: ShapeLike, ch: int = 1) -> np.ndarray:
    """Sample a normalized low-frequency image from Fourier space."""

    shape = _shape_tuple(shape)
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    axes = tuple(range(-len(shape), 0))
    mask = np.real(np.fft.irfftn(spectrum, s=shape, axes=axes))

    if len(shape) == 1:
        mask = mask[:1, : shape[0]]
    if len(shape) == 2:
        mask = mask[:1, : shape[0], : shape[1]]
    if len(shape) == 3:
        mask = mask[:1, : shape[0], : shape[1], : shape[2]]

    mask = mask - mask.min()
    max_value = mask.max()
    if max_value > 0:
        mask = mask / max_value
    return mask


def sample_lam(alpha: float, reformulate: bool = False) -> float:
    """Sample lambda from the FMix beta distribution."""

    if alpha <= 0:
        return 1.0
    if reformulate:
        return float(np.random.beta(alpha + 1, alpha))
    return float(np.random.beta(alpha, alpha))


def binarise_mask(
    mask: np.ndarray,
    lam: float,
    in_shape: ShapeLike,
    max_soft: float = 0.0,
) -> np.ndarray:
    """Binarize a low-frequency image so its expected mean is lambda."""

    in_shape = _shape_tuple(in_shape)
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1 - lam):
        eff_soft = min(lam, 1 - lam)

    soft = int(mask.size * eff_soft)
    num_low = max(num - soft, 0)
    num_high = min(num + soft, mask.size)

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    if num_high > num_low:
        mask[idx[num_low:num_high]] = np.linspace(1, 0, num_high - num_low)

    return mask.reshape((1, *in_shape))


def sample_mask(
    alpha: float,
    decay_power: float,
    shape: ShapeLike,
    max_soft: float = 0.0,
    reformulate: bool = False,
) -> tuple[float, np.ndarray]:
    """Sample a lambda value and its corresponding FMix mask."""

    shape = _shape_tuple(shape)
    lam = sample_lam(alpha, reformulate)
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)
    return lam, mask


@dataclass(frozen=True)
class FMixResult:
    """Return bundle produced by :class:`FMix`."""

    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float
    index: torch.Tensor
    mask: torch.Tensor


class FMix:
    """Apply FMix to a mini-batch.

    Args:
        decay_power: Frequency decay power. Official/OpenMixup default: 3.
        alpha: Beta distribution alpha.
        size: Mask spatial size, usually ``(32, 32)`` or ``(64, 64)``.
        max_soft: Edge softening fraction. Official/OpenMixup default: 0.
        reformulate: Use the FMix paper's reformulated objective.
    """

    def __init__(
        self,
        decay_power: float = 3.0,
        alpha: float = 1.0,
        size: ShapeLike = (32, 32),
        max_soft: float = 0.0,
        reformulate: bool = False,
    ) -> None:
        self.decay_power = float(decay_power)
        self.alpha = float(alpha)
        self.size = _shape_tuple(size)
        self.max_soft = float(max_soft)
        self.reformulate = bool(reformulate)

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        partner_images: torch.Tensor | None = None,
        partner_targets: torch.Tensor | None = None,
        index: torch.Tensor | None = None,
    ) -> FMixResult:
        if images.dim() != 4:
            raise ValueError(f"FMix expects NCHW images, got shape {tuple(images.shape)}")

        lam, mask_np = sample_mask(
            self.alpha,
            self.decay_power,
            self.size,
            self.max_soft,
            self.reformulate,
        )
        if partner_images is None:
            index = torch.randperm(images.size(0)).to(images.device)
            partner_images = images[index]
            partner_targets = targets[index]
        elif partner_targets is None or index is None:
            raise ValueError("partner_targets and index are required when partner_images is provided.")

        mask = torch.from_numpy(mask_np).to(device=images.device, dtype=images.dtype)

        mixed = mask * images + (1 - mask) * partner_images
        return FMixResult(
            images=mixed,
            targets_a=targets,
            targets_b=partner_targets,
            lam=lam,
            index=index,
            mask=mask,
        )


def fmix_cross_entropy(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
    reformulate: bool = False,
) -> torch.Tensor:
    """FMix criterion from the official PyTorch binding."""

    if reformulate:
        return F.cross_entropy(logits, targets_a)
    return F.cross_entropy(logits, targets_a) * lam + F.cross_entropy(logits, targets_b) * (1.0 - lam)
