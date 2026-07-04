"""PreAct-ResNet backbones."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut: nn.Module = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + shortcut


class PreActResNetBackbone(nn.Module):
    """PreAct-ResNet feature extractor without a classifier head."""

    output_dim = 512

    def __init__(
        self,
        block_sizes: Sequence[int] = (2, 2, 2, 2),
        channels: Sequence[int] = (64, 128, 256, 512),
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_planes = int(channels[0])
        self.output_dim = int(channels[-1]) * PreActBasicBlock.expansion

        self.stem = _conv3x3(in_channels, int(channels[0]))
        self.layer1 = self._make_layer(int(channels[0]), int(block_sizes[0]), stride=1)
        self.layer2 = self._make_layer(int(channels[1]), int(block_sizes[1]), stride=2)
        self.layer3 = self._make_layer(int(channels[2]), int(block_sizes[2]), stride=2)
        self.layer4 = self._make_layer(int(channels[3]), int(block_sizes[3]), stride=2)
        self.bn = nn.BatchNorm2d(self.output_dim)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(PreActBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes * PreActBasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(x, 1)


def preact_resnet18_backbone(in_channels: int = 3) -> PreActResNetBackbone:
    return PreActResNetBackbone(block_sizes=(2, 2, 2, 2), in_channels=in_channels)
