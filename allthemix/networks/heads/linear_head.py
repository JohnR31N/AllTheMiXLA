"""Classifier heads."""

from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    """Linear projection from feature vectors to class logits."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)
