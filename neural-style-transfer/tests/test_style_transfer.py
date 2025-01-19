"""
Tests for neural style transfer components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pytest

from style_transfer import gram_matrix, total_variation
from model import VGGFeatureExtractor


class TestGramMatrix:
    def test_shape(self):
        x = torch.randn(2, 64, 32, 32)
        gram = gram_matrix(x)
        assert gram.shape == (2, 64, 64)

    def test_symmetric(self):
        x = torch.randn(1, 16, 8, 8)
        gram = gram_matrix(x)
        diff = (gram - gram.transpose(1, 2)).abs().max().item()
        assert diff < 1e-5

    def test_positive_semidefinite(self):
        x = torch.randn(1, 8, 4, 4)
        gram = gram_matrix(x).squeeze(0)
        eigvals = torch.linalg.eigvalsh(gram)
        assert (eigvals >= -1e-5).all()


class TestTotalVariation:
    def test_constant_image(self):
        x = torch.ones(1, 3, 64, 64)
        tv = total_variation(x).item()
        assert tv == 0.0

    def test_noisy_image(self):
        x = torch.randn(1, 3, 64, 64)
        tv = total_variation(x).item()
        assert tv > 0.0


class TestVGGFeatureExtractor:
    def test_content_layers(self):
        extractor = VGGFeatureExtractor()
        assert "conv4_2" in extractor.content_layers()

    def test_style_layers(self):
        extractor = VGGFeatureExtractor()
        assert len(extractor.style_layers()) == 5

    def test_forward_shapes(self):
        extractor = VGGFeatureExtractor()
        x = torch.randn(1, 3, 256, 256).to(extractor.device)
        features = extractor(x)
        for name, fmap in features.items():
            assert fmap.dim() == 4, f"{name} should be 4D"
            assert fmap.size(0) == 1, f"{name} batch size should be 1"
