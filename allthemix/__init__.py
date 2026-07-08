"""Lightweight PyTorch/XLA reproductions for mix-based augmentations."""

from .methods import (
    CutMix,
    CutMixResult,
    CatchUpMix,
    CatchUpMixResult,
    FMix,
    FMixResult,
    GuidedSR,
    GuidedSRResult,
    MixUp,
    MixUpResult,
    ResizeMix,
    ResizeMixResult,
    SaliencyMix,
    SaliencyMixResult,
    fmix_cross_entropy,
    mixup_cross_entropy,
    sample_mask,
)
from .networks import build_model

__all__ = [
    "CatchUpMix",
    "CatchUpMixResult",
    "CutMix",
    "CutMixResult",
    "FMix",
    "FMixResult",
    "GuidedSR",
    "GuidedSRResult",
    "MixUp",
    "MixUpResult",
    "ResizeMix",
    "ResizeMixResult",
    "SaliencyMix",
    "SaliencyMixResult",
    "build_model",
    "fmix_cross_entropy",
    "mixup_cross_entropy",
    "sample_mask",
]
