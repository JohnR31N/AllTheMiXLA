"""Torchvision ResNet backbones for ImageNet-scale evaluation."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _build_resnet(arch: str) -> nn.Module:
    try:
        return getattr(models, arch)(weights=None)
    except TypeError:
        return getattr(models, arch)(pretrained=False)


class TorchvisionResNetBackbone(nn.Module):
    """Torchvision ResNet feature extractor without the final fc layer."""

    def __init__(self, arch: str = "resnet101", in_channels: int = 3) -> None:
        super().__init__()
        model = _build_resnet(arch)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=False,
            )
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.output_dim = model.fc.in_features
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


def torch_resnet101_backbone(in_channels: int = 3) -> TorchvisionResNetBackbone:
    return TorchvisionResNetBackbone("resnet101", in_channels=in_channels)
