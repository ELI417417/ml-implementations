"""
Dataset utilities for DCGAN training.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_default_transforms(image_size: int = 64):
    """Get train/val transforms for DCGAN."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class ImageFolderDataset(Dataset):
    """Simple image folder dataset without requiring class subdirectories."""

    def __init__(self, root: str, image_size: int = 64,
                 extensions=(".jpg", ".jpeg", ".png")):
        self.root = Path(root)
        self.image_paths = [
            str(p) for p in self.root.rglob("*")
            if p.suffix.lower() in extensions
        ]
        if not self.image_paths:
            raise RuntimeError(f"No images found in {root}")
        print(f"Found {len(self.image_paths)} images in {root}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), 0  # label unused for GANs
