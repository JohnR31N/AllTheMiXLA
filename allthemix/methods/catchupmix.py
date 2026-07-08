"""CatchUpMix feature-level augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

import numpy as np
import torch

from allthemix.methods.cutmix import CutMix, no_repeat_permutation


FeatureHook = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class CatchUpMixResult:
    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float
    index: torch.Tensor
    layer: int
    feature_hook: FeatureHook | None = None


def sample_lam(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def normalize_filter_influence(influence: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return influence / (influence.sum(dim=-1, keepdim=True) + eps)


def catchup_mix_features(features: torch.Tensor, lam: float, index: torch.Tensor) -> torch.Tensor:
    """Mix feature channels according to relative filter influence."""

    if features.dim() != 4:
        raise ValueError("CatchUpMix feature mixing expects NCHW feature maps.")
    num_channels = int(features.size(1))
    target_features = features.index_select(0, index)

    source_influence = torch.sqrt(features.square().sum(dim=(2, 3)) + 1e-12)
    target_influence = torch.sqrt(target_features.square().sum(dim=(2, 3)) + 1e-12)
    relative_influence = normalize_filter_influence(source_influence) - normalize_filter_influence(target_influence)
    relative_influence = relative_influence.detach()

    channel_rank = torch.argsort(torch.argsort(relative_influence, dim=1), dim=1)
    num_source_channels = int(np.floor(float(lam) * float(num_channels)))
    num_source_channels = max(0, min(num_source_channels, num_channels))
    source_mask = channel_rank < num_source_channels
    source_mask = source_mask[:, :, None, None]
    return torch.where(source_mask, features, target_features)


def make_catchup_mix_feature_hook(layer: int, lam: float, index: torch.Tensor) -> FeatureHook:
    def feature_hook(features: torch.Tensor, layer_index: int) -> torch.Tensor:
        if int(layer_index) != int(layer):
            return features
        return catchup_mix_features(features=features, lam=lam, index=index)

    return feature_hook


class CatchUpMix:
    """Sample CatchUpMix metadata and apply input-level CutMix for layer 0."""

    def __init__(
        self,
        alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        num_feature_layers: int = 5,
        no_repeat: bool = False,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"catchupmix_alpha must be positive, got {alpha}.")
        if cutmix_alpha <= 0:
            raise ValueError(f"catchupmix_cutmix_alpha must be positive, got {cutmix_alpha}.")
        if int(num_feature_layers) <= 0:
            raise ValueError(f"catchupmix_num_layers must be positive, got {num_feature_layers}.")
        self.alpha = float(alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.num_feature_layers = int(num_feature_layers)
        self.no_repeat = bool(no_repeat)

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        partner_images: torch.Tensor | None = None,
        partner_targets: torch.Tensor | None = None,
        index: torch.Tensor | None = None,
    ) -> CatchUpMixResult:
        if images.dim() != 4:
            raise ValueError(f"CatchUpMix expects NCHW images, got shape {tuple(images.shape)}")
        if targets.dim() != 1 or targets.size(0) != images.size(0):
            raise ValueError("CatchUpMix image/label batch mismatch.")
        if self.no_repeat and images.size(0) <= 1:
            raise ValueError("CatchUpMix no_repeat requires batch size > 1.")

        layer = random.randint(0, self.num_feature_layers)
        if layer == 0:
            cutmix = CutMix(alpha=self.cutmix_alpha, no_repeat=self.no_repeat)
            cutmixed = cutmix(images, targets, partner_images=partner_images, partner_targets=partner_targets, index=index)
            return CatchUpMixResult(
                images=cutmixed.images,
                targets_a=cutmixed.targets_a,
                targets_b=cutmixed.targets_b,
                lam=float(cutmixed.lam),
                index=cutmixed.index,
                layer=layer,
                feature_hook=None,
            )

        if partner_images is not None:
            raise ValueError(
                "CatchUpMix feature-level mixing does not support external partner_images. "
                "Keep CatchUpMix cross_device_shuffle disabled, or use layer 0 CutMix-style mixing."
            )
        if index is None:
            index = no_repeat_permutation(images.size(0), images.device) if self.no_repeat else torch.randperm(images.size(0), device=images.device)
        if partner_targets is None:
            partner_targets = targets.index_select(0, index)

        lam = sample_lam(self.alpha)
        return CatchUpMixResult(
            images=images,
            targets_a=targets,
            targets_b=partner_targets,
            lam=lam,
            index=index,
            layer=layer,
            feature_hook=make_catchup_mix_feature_hook(layer=layer, lam=lam, index=index),
        )
