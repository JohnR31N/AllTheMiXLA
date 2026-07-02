"""Compatibility exports for the old backbones import path."""

from allthemix.networks.builder import build_model
from allthemix.networks.nn import PreActBasicBlock, PreActResNetNN, preact_resnet18_nn


def preact_resnet18(num_classes: int, in_channels: int = 3):
    return build_model("preact_resnet18", num_classes=num_classes, in_channels=in_channels)


__all__ = ["PreActBasicBlock", "PreActResNetNN", "preact_resnet18", "preact_resnet18_nn"]
