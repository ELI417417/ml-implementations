"""
MobileNetV2-SSD: lightweight object detector for edge deployment.

MobileNetV2 backbone (Howard et al., CVPR 2018) + SSD detection head
(Liu et al., ECCV 2016). Designed for CPU inference and ONNX export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round to a multiple of divisor (per MobileNetV2 paper)."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class InvertedResidual(nn.Module):
    """MobileNetV2 bottleneck block with inverted residual."""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int, expand_ratio: int):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 feature extractor — returns multi-scale feature maps."""

    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        input_channels = _make_divisible(32 * width_mult)

        # Config: (expand_ratio, channels, repeats, stride)
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
        )

        self.features = nn.ModuleList()
        in_c = input_channels
        self.output_channels = []

        for t, c, n, s in self.cfgs:
            out_c = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(in_c, out_c, stride, t)
                )
                in_c = out_c
            self.output_channels.append(out_c)

        # Return channels for selected layers (stride 16, 32)
        self.extra_layers = nn.ModuleList([
            # Conv13: 10x10 feature map
            nn.Sequential(
                nn.Conv2d(in_c, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU6(inplace=True),
            ),
            # Conv14_1 + Conv14_2: 5x5 feature map
            nn.Sequential(
                nn.Conv2d(256, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU6(inplace=True),
                nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU6(inplace=True),
            ),
            # Conv15_1 + Conv15_2: 3x3 feature map
            nn.Sequential(
                nn.Conv2d(256, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU6(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU6(inplace=True),
            ),
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        sources = []
        x = self.conv1(x)
        for i, feat in enumerate(self.features):
            x = feat(x)
            # Extract at strides 16, 32
            if i == len(self.features) - 1:
                sources.append(x)
        # Add extra layers
        for extra in self.extra_layers:
            x = extra(x)
            sources.append(x)
        return sources


class MobileNetSSD(nn.Module):
    """MobileNetV2-SSD object detector.

    Predicts bounding boxes and class scores across 6 feature map scales.
    """

    def __init__(self, num_classes: int = 21, width_mult: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetV2(width_mult)

        # Feature map output channels — need 6 for 6 scales
        # We compute in_channels from the actual backbone outputs
        self.num_priors = [4, 6, 6, 6, 4, 4]

        # Classification and regression heads will be created in forward
        # since we need to know the actual feature map channels
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        self._heads_initialized = False

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        sources = self.backbone(x)
        sources = sources[-5:]

        # Initialize heads on first forward pass (dynamic channel detection)
        if not self._heads_initialized:
            for i, src in enumerate(sources):
                in_c = src.shape[1]
                self.loc_layers.append(
                    nn.Conv2d(in_c, self.num_priors[i] * 4, 3, padding=1)
                )
                self.conf_layers.append(
                    nn.Conv2d(in_c, self.num_priors[i] * self.num_classes,
                              3, padding=1)
                )
            # Move heads to correct device
            self.loc_layers = self.loc_layers.to(x.device)
            self.conf_layers = self.conf_layers.to(x.device)
            self._heads_initialized = True

        locs, confs = [], []
        for i, src in enumerate(sources):
            locs.append(
                self.loc_layers[i](src).permute(0, 2, 3, 1).contiguous()
            )
            confs.append(
                self.conf_layers[i](src).permute(0, 2, 3, 1).contiguous()
            )

        batch_size = x.size(0)
        locs = torch.cat([o.view(batch_size, -1) for o in locs], 1)
        confs = torch.cat([o.view(batch_size, -1) for o in confs], 1)

        return (
            locs.view(batch_size, -1, 4),
            confs.view(batch_size, -1, self.num_classes),
        )
