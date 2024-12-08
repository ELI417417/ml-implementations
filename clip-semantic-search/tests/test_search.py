"""
Tests for CLIP-based image search.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from model import CLIPEncoder
from search import ImageSearchEngine


class TestCLIPEncoder:
    @pytest.fixture(scope="class")
    @classmethod
    def encoder(cls):
        return CLIPEncoder()

    def test_init(self, encoder):
        assert encoder.embedding_dim == 512

    def test_encode_text_shape(self, encoder):
        texts = ["a cat sitting on a chair", "a dog running in a field"]
        embeddings = encoder.encode_text(texts)
        assert embeddings.shape == (2, 512)

    def test_encode_text_normalized(self, encoder):
        embeddings = encoder.encode_text(["hello world"])
        norms = embeddings.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_encode_images(self, encoder):
        images = [
            Image.new("RGB", (224, 224), color=(255, 0, 0)),
            Image.new("RGB", (224, 224), color=(0, 255, 0)),
        ]
        embeddings = encoder.encode_images(images)
        assert embeddings.shape == (2, 512)


class TestImageSearchEngine:
    def test_index_and_search(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(5):
            img = Image.new("RGB", (64, 64), color=(i * 50, 100, 200))
            img.save(img_dir / f"img_{i}.png")

        index_dir = tmp_path / "index"
        engine = ImageSearchEngine(index_dir=str(index_dir))

        count = engine.index_images(str(img_dir))
        assert count == 5

        results = engine.search("a blue image", top_k=3)
        assert len(results) == 3
        for r in results:
            assert -1.0 <= r.score <= 1.0

    def test_save_and_load(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img_dir / "a.png")

        index_dir = tmp_path / "index"
        engine1 = ImageSearchEngine(index_dir=str(index_dir))
        engine1.index_images(str(img_dir))

        engine2 = ImageSearchEngine(index_dir=str(index_dir))
        assert engine2._load_index()
        assert len(engine2.image_paths) == 1
