"""
Tests for DCGAN model components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from model import Discriminator, Generator, weights_init


class TestGenerator:
    def test_output_shape(self):
        batch_size, latent_dim = 16, 100
        model = Generator(latent_dim=latent_dim)
        z = torch.randn(batch_size, latent_dim, 1, 1)
        out = model(z)
        assert out.shape == (batch_size, 3, 64, 64)

    def test_output_range(self):
        model = Generator()
        z = torch.randn(4, 100, 1, 1)
        out = model(z)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_deterministic(self):
        model = Generator()
        model.eval()  # eval mode for deterministic output
        torch.manual_seed(42)
        z = torch.randn(1, 100, 1, 1)
        out1 = model(z)
        torch.manual_seed(42)
        z = torch.randn(1, 100, 1, 1)
        out2 = model(z)
        assert torch.allclose(out1, out2)


class TestDiscriminator:
    def test_output_shape(self):
        batch_size = 16
        model = Discriminator()
        x = torch.randn(batch_size, 3, 64, 64)
        out = model(x)
        assert out.shape == (batch_size, 1, 1, 1)

    def test_output_range(self):
        model = Discriminator()
        x = torch.randn(4, 3, 64, 64)
        out = torch.sigmoid(model(x))
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestWeightInit:
    def test_conv_weights_normal(self):
        conv = torch.nn.Conv2d(3, 64, 4, 2, 1)
        weights_init(conv)
        mean = conv.weight.data.mean()
        assert -0.1 < mean < 0.1, f"Mean {mean} not near zero"

    def test_batchnorm_weights_init(self):
        bn = torch.nn.BatchNorm2d(64)
        weights_init(bn)
        assert bn.weight.data.mean().item() == pytest.approx(1.0, abs=0.1)
