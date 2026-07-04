"""MixUp batch augmentation.

This follows the standard MixUp recipe: sample ``lam`` from
``Beta(alpha, alpha)``, linearly mix images, and train with the same convex
combination of the two labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def sample_lam(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


@dataclass(frozen=True)
class MixUpResult:
    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float
    index: torch.Tensor


class MixUp:
    """Apply MixUp to a mini-batch."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> MixUpResult:
        if images.dim() != 4:
            raise ValueError(f"MixUp expects NCHW images, got shape {tuple(images.shape)}")

        lam = sample_lam(self.alpha)
        index = torch.randperm(images.size(0)).to(images.device)
        mixed = lam * images + (1.0 - lam) * images[index]
        return MixUpResult(
            images=mixed,
            targets_a=targets,
            targets_b=targets[index],
            lam=lam,
            index=index,
        )


def mixup_cross_entropy(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return F.cross_entropy(logits, targets_a) * lam + F.cross_entropy(logits, targets_b) * (1.0 - lam)
