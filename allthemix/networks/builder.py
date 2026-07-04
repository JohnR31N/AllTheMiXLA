"""Network builder."""

from __future__ import annotations

from torch import nn

from allthemix.networks.classifiers import ImageClassifier
from allthemix.networks.heads import LinearHead
from allthemix.networks.backbones import preact_resnet18_backbone, torch_resnet101_backbone


def normalize_model_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def build_model(name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    model_name = normalize_model_name(name)

    if model_name == "preact_resnet18":
        backbone = preact_resnet18_backbone(in_channels=in_channels)
        head = LinearHead(in_features=backbone.output_dim, num_classes=num_classes)
        return ImageClassifier(backbone=backbone, head=head)

    if model_name in {"resnet101", "torch_resnet101", "torchvision_resnet101"}:
        backbone = torch_resnet101_backbone(in_channels=in_channels)
        head = LinearHead(in_features=backbone.output_dim, num_classes=num_classes)
        return ImageClassifier(backbone=backbone, head=head)

    raise ValueError(f"Unsupported model: {name}")
