"""
Search engine: CLIP embeddings + FAISS nearest-neighbor search.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from PIL import Image

from model import CLIPEncoder


@dataclass
class SearchResult:
    path: str
    score: float


class ImageSearchEngine:
    """Builds a FAISS index from image embeddings and supports text search."""

    def __init__(self, encoder: Optional[CLIPEncoder] = None,
                 index_dir: str = "./index"):
        self.encoder = encoder or CLIPEncoder()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.IndexFlatIP | None = None
        self.image_paths: list[str] = []

    def index_images(self, image_dir: str, extensions=(".jpg", ".jpeg", ".png",
                     ".webp", ".bmp"), batch_size: int = 32) -> int:
        """Index all images in a directory."""
        image_dir = Path(image_dir)
        image_paths = []
        images = []

        for ext in extensions:
            for p in image_dir.rglob(f"*{ext}"):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    image_paths.append(str(p))
                except Exception:
                    continue

        if not images:
            print(f"No images found in {image_dir}")
            return 0

        print(f"Found {len(images)} images. Encoding...")
        embeddings = self.encoder.encode_images(images, batch_size=batch_size)

        self.image_paths = image_paths
        self._build_index(embeddings)
        self._save_index()
        return len(image_paths)

    def _build_index(self, embeddings: torch.Tensor) -> None:
        """Build a FAISS inner-product index from embeddings."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vectors
        self.index.add(embeddings.numpy().astype(np.float32))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search images by text query."""
        if self.index is None:
            if not self._load_index():
                raise RuntimeError("No index found. Run index_images() first.")

        query_vec = self.encoder.encode_text([query]).numpy().astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_paths) and idx >= 0:
                results.append(SearchResult(
                    path=self.image_paths[idx],
                    score=float(score),
                ))
        return results

    def _save_index(self) -> None:
        """Save index to disk."""
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "paths.json", "w", encoding="utf-8") as f:
            json.dump(self.image_paths, f)

    def _load_index(self) -> bool:
        """Load index from disk. Returns True if successful."""
        faiss_path = self.index_dir / "index.faiss"
        paths_path = self.index_dir / "paths.json"
        if not faiss_path.exists() or not paths_path.exists():
            return False
        self.index = faiss.read_index(str(faiss_path))
        with open(paths_path, "r", encoding="utf-8") as f:
            self.image_paths = json.load(f)
        return True
