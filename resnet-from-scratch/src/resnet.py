"""
ResNet implementation from scratch (He et al., CVPR 2016).

Supports ResNet-18, 34, 50, 101, 152 with both Basic and Bottleneck blocks
and CIFAR-scale variants (ResNet-20/32/44/56/110).
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Residual block with two 3x3 convolutions (used in ResNet-18/34)."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block: 1x1, 3x3, 1x1 convolutions (ResNet-50/101/152)."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        width = out_channels
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model supporting both ImageNet-scale and CIFAR-scale variants."""

    _ARCHITECTURES = {
        18:  (BasicBlock, [2, 2, 2, 2]),
        34:  (BasicBlock, [3, 4, 6, 3]),
        50:  (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
    }

    def __init__(self, depth: int = 18, num_classes: int = 10,
                 in_channels: int = 3, cifar_layers: list[int] | None = None):
        super().__init__()
        self._is_cifar = cifar_layers is not None

        if self._is_cifar:
            block = BasicBlock
            self.in_planes = 16
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_stage(block, 16, cifar_layers[0], stride=1)
            self.layer2 = self._make_stage(block, 32, cifar_layers[1], stride=2)
            self.layer3 = self._make_stage(block, 64, cifar_layers[2], stride=2)
            self._out_planes = 64 * block.expansion
        else:
            if depth not in self._ARCHITECTURES:
                raise ValueError(
                    f"Unsupported depth {depth}. "
                    f"Choose from: {list(self._ARCHITECTURES.keys())}"
                )
            block, layers = self._ARCHITECTURES[depth]
            self.in_planes = 64
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self._out_planes = 512 * block.expansion

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._out_planes, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, block, planes: int, blocks: int,
                    stride: int) -> nn.Sequential:
        """Build one stage for CIFAR-scale ResNet."""
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self._is_cifar:
            x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @classmethod
    def _build_cifar(cls, n: int, layers_per_stage: list[int],
                     num_classes: int = 10) -> "ResNet":
        """Build a small ResNet for CIFAR (per paper: 6n+2 total layers)."""
        return cls(depth=18, num_classes=num_classes,
                   cifar_layers=layers_per_stage)


def resnet18(**kwargs) -> ResNet:
    return ResNet(depth=18, **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(depth=34, **kwargs)


def resnet50(**kwargs) -> ResNet:
    return ResNet(depth=50, **kwargs)


def resnet101(**kwargs) -> ResNet:
    return ResNet(depth=101, **kwargs)


def resnet152(**kwargs) -> ResNet:
    return ResNet(depth=152, **kwargs)
