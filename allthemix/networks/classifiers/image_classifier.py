"""Image classifier composition."""

from __future__ import annotations

import torch
from torch import nn


class ImageClassifier(nn.Module):
    """Compose a feature network and a classifier head."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits
