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

from allthemix.methods.cutmix import no_repeat_permutation, validate_nchw_images, validate_partner_batch, validate_targets


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

    def __init__(self, alpha: float = 1.0, no_repeat: bool = False) -> None:
        self.alpha = float(alpha)
        self.no_repeat = bool(no_repeat)

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        partner_images: torch.Tensor | None = None,
        partner_targets: torch.Tensor | None = None,
        index: torch.Tensor | None = None,
    ) -> MixUpResult:
        validate_nchw_images(images, "MixUp")
        validate_targets(images, targets, "MixUp")
        if self.no_repeat and images.size(0) <= 1:
            raise ValueError("MixUp no_repeat requires batch size > 1.")

        lam = sample_lam(self.alpha)
        if partner_images is None:
            index = (
                no_repeat_permutation(images.size(0), images.device)
                if self.no_repeat
                else torch.randperm(images.size(0), device=images.device)
            )
            partner_images = images[index]
            partner_targets = targets[index]
        elif partner_targets is None or index is None:
            raise ValueError("partner_targets and index are required when partner_images is provided.")
        else:
            validate_partner_batch(images, partner_images, partner_targets, index, "MixUp")

        mixed = lam * images + (1.0 - lam) * partner_images
        return MixUpResult(
            images=mixed,
            targets_a=targets,
            targets_b=partner_targets,
            lam=lam,
            index=index,
        )


def mixup_cross_entropy(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float | torch.Tensor,
) -> torch.Tensor:
    if isinstance(lam, torch.Tensor):
        lam_tensor = lam.to(device=logits.device, dtype=logits.dtype)
        loss_a = F.cross_entropy(logits, targets_a, reduction="none")
        loss_b = F.cross_entropy(logits, targets_b, reduction="none")
        if lam_tensor.dim() == 0:
            return (loss_a * lam_tensor + loss_b * (1.0 - lam_tensor)).mean()
        return (loss_a * lam_tensor.reshape(-1) + loss_b * (1.0 - lam_tensor.reshape(-1))).mean()
    return F.cross_entropy(logits, targets_a) * lam + F.cross_entropy(logits, targets_b) * (1.0 - lam)
