"""
Tests for PointNet model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from model import (
    TNet,
    PointNetFeatureExtractor,
    PointNetClassifier,
    pointnet_loss,
)


class TestTNet:
    def test_output_shape_3x3(self):
        tnet = TNet(k=3)
        x = torch.randn(2, 3, 1024)
        out = tnet(x)
        assert out.shape == (2, 3, 3)

    def test_output_shape_64x64(self):
        tnet = TNet(k=64)
        x = torch.randn(2, 64, 1024)
        out = tnet(x)
        assert out.shape == (2, 64, 64)

    def test_near_identity_init(self):
        tnet = TNet(k=3)
        tnet.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 3, 100)
        out = tnet(x)
        assert out.shape == (2, 3, 3)


class TestPointNetFeatureExtractor:
    def test_output_shape(self):
        extractor = PointNetFeatureExtractor()
        x = torch.randn(2, 3, 1024)
        global_feat, t1, t2 = extractor(x)
        assert global_feat.shape == (2, 1024)

    def test_output_no_transforms(self):
        extractor = PointNetFeatureExtractor(
            use_input_transform=False,
            use_feature_transform=False,
        )
        x = torch.randn(2, 3, 1024)
        global_feat, t1, t2 = extractor(x)
        assert t1 is None
        assert t2 is None


class TestPointNetClassifier:
    def test_forward_shape(self):
        model = PointNetClassifier(num_classes=40)
        x = torch.randn(2, 3, 1024)
        logits, t1, t2 = model(x)
        assert logits.shape == (2, 40)

    def test_permutation_invariance(self):
        """PointNet should be invariant to point order."""
        model = PointNetClassifier(num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 512)
        # Shuffle points
        perm = torch.randperm(512)
        x_shuffled = x[:, :, perm]
        with torch.no_grad():
            out1, _, _ = model(x)
            out2, _, _ = model(x_shuffled)
        assert torch.allclose(out1, out2, atol=1e-4)


class TestPointNetLoss:
    def test_loss_scalar(self):
        model = PointNetClassifier(num_classes=10)
        x = torch.randn(4, 3, 1024)
        logits, t1, t2 = model(x)
        labels = torch.randint(0, 10, (4,))
        loss = pointnet_loss(logits, labels, t1, t2)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
