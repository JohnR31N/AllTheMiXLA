"""Network builder."""

from __future__ import annotations

from torch import nn

from allthemix.networks.classifiers import ImageClassifier
from allthemix.networks.heads import LinearHead
from allthemix.networks.nn import preact_resnet18_nn


def normalize_model_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def build_model(name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    model_name = normalize_model_name(name)

    if model_name == "preact_resnet18":
        nn_module = preact_resnet18_nn(in_channels=in_channels)
        head = LinearHead(in_features=nn_module.output_dim, num_classes=num_classes)
        return ImageClassifier(nn_module=nn_module, head=head)

    raise ValueError(f"Unsupported model: {name}")
