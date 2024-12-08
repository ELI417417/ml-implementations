"""
CLIP model wrapper for text and image encoding.

Uses HuggingFace transformers for broad model compatibility.
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    """Encodes images and text into a shared embedding space using CLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image],
                      batch_size: int = 32) -> torch.Tensor:
        """Encode a list of PIL Images into normalized embeddings."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            # CLIPModel returns BaseModelOutputWithPooling; extract pooler_output
            embeddings = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text queries into normalized embeddings."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        embeddings = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        return nn.functional.normalize(embeddings, p=2, dim=-1).cpu()
