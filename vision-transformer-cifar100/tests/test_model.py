"""
Tests for Vision Transformer model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from model import (
    PatchEmbedding,
    VisionTransformer,
    vit_tiny,
    vit_small,
)


class TestPatchEmbedding:
    def test_output_shape(self):
        pe = PatchEmbedding(img_size=32, patch_size=4, embed_dim=192)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        assert out.shape == (2, 64, 192)  # (32/4)^2 = 64 patches

    def test_patch_count(self):
        for ps in [2, 4, 8]:
            pe = PatchEmbedding(img_size=32, patch_size=ps)
            assert pe.n_patches == (32 // ps) ** 2


class TestVisionTransformer:
    def test_forward_shape(self):
        model = vit_tiny(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)

    def test_vit_small_shape(self):
        model = vit_small(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)

    def test_gradient_flow(self):
        model = vit_tiny(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 10, (2,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                pytest.fail(f"No gradient for {name}")
