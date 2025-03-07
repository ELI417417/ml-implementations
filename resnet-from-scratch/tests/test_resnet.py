"""
Tests for ResNet from scratch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from resnet import BasicBlock, Bottleneck, ResNet, resnet18, resnet50


class TestBasicBlock:
    def test_forward_shape(self):
        block = BasicBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


class TestResNet:
    @pytest.mark.parametrize("depth", [18, 34, 50, 101, 152])
    def test_forward(self, depth):
        model = ResNet(depth=depth, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_cifar_builder_resnet20(self):
        model = ResNet._build_cifar(3, [3, 3, 3], num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_cifar_builder_param_count(self):
        model = ResNet._build_cifar(3, [3, 3, 3])
        n_params = sum(p.numel() for p in model.parameters())
        assert 250_000 < n_params < 300_000


class TestBottleneck:
    def test_forward_shape(self):
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
        )
        block = Bottleneck(256, 128, downsample=downsample)
        x = torch.randn(2, 256, 16, 16)
        out = block(x)
        assert out.shape == (2, 512, 16, 16)
