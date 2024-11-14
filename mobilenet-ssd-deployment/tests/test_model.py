"""
Tests for MobileNet-SSD model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from model import MobileNetV2, MobileNetSSD, InvertedResidual


class TestInvertedResidual:
    def test_forward_shape(self):
        block = InvertedResidual(32, 32, stride=1, expand_ratio=6)
        x = torch.randn(2, 32, 56, 56)
        out = block(x)
        assert out.shape == (2, 32, 56, 56)

    def test_stride_downsample(self):
        block = InvertedResidual(32, 64, stride=2, expand_ratio=6)
        x = torch.randn(2, 32, 56, 56)
        out = block(x)
        assert out.shape == (2, 64, 28, 28)


class TestMobileNetV2:
    def test_forward(self):
        backbone = MobileNetV2(width_mult=1.0)
        x = torch.randn(2, 3, 300, 300)
        sources = backbone(x)
        assert len(sources) > 0


class TestMobileNetSSD:
    def test_forward_shape(self):
        model = MobileNetSSD(num_classes=21)
        x = torch.randn(2, 3, 300, 300)
        locs, confs = model(x)
        assert locs.dim() == 3 and locs.shape[-1] == 4  # (B, N, 4)
        assert confs.dim() == 3 and confs.shape[-1] == 21  # (B, N, 21)
