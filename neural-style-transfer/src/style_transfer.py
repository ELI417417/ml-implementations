"""
Neural Style Transfer algorithm — Gatys et al. (2015).

Optimizes an image to match the content of one image and the style of another.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from model import VGGFeatureExtractor


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix from feature maps: (C, H, W) -> (C, C)."""
    b, c, h, w = features.size()
    feats = features.view(b, c, h * w)
    gram = feats.bmm(feats.transpose(1, 2))
    return gram / (c * h * w)


def total_variation(img: torch.Tensor) -> torch.Tensor:
    """Total Variation loss for spatial smoothness."""
    return (
        torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        + torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    )


def load_image(path: str, size: int = 512, device: str = "cpu"
               ) -> torch.Tensor:
    """Load and preprocess an image for style transfer."""
    loader = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    tensor = loader(img).unsqueeze(0).to(device)
    # ImageNet normalization
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return norm(tensor)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor back to PIL Image."""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = tensor.clone().cpu()
    img = img * std + mean
    img = img.clamp(0, 1)
    img = img.squeeze(0)
    return transforms.ToPILImage()(img)


def style_transfer(
    content_path: str,
    style_path: str,
    output_path: Optional[str] = None,
    image_size: int = 512,
    num_steps: int = 300,
    content_weight: float = 1.0,
    style_weight: float = 1e6,
    tv_weight: float = 1e-6,
    lr: float = 0.05,
    log_interval: int = 50,
    progress_callback=None,
) -> Image.Image:
    """
    Run neural style transfer.

    Args:
        content_path: Path to content (photo) image.
        style_path: Path to style (artwork) image.
        output_path: Where to save the output image.
        image_size: Resize images to this size.
        num_steps: Number of optimization steps.
        content_weight: Weight of content loss.
        style_weight: Weight of style loss.
        tv_weight: Weight of total variation loss.
        lr: Learning rate for L-BFGS.
        log_interval: Print loss every N steps.
        progress_callback: Called with (step, content_loss, style_loss, tv_loss).

    Returns:
        PIL Image of the result.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load images
    content_img = load_image(content_path, image_size, device)
    style_img = load_image(style_path, image_size, device)

    # Initialize output with content image (converges faster than noise)
    output = content_img.clone().requires_grad_(True)

    # Build model
    extractor = VGGFeatureExtractor(device)

    # Extract target features
    with torch.no_grad():
        content_features = extractor(content_img)
        style_features = extractor(style_img)

    # Compute target Gram matrices for style
    style_grams = {
        layer: gram_matrix(style_features[layer])
        for layer in extractor.style_layers()
    }

    # Optimizer
    optimizer = optim.LBFGS([output], lr=lr)

    def closure():
        optimizer.zero_grad()
        output_features = extractor(output)

        # Content loss
        content_loss = 0.0
        for layer in extractor.content_layers():
            content_loss += F.mse_loss(
                output_features[layer], content_features[layer]
            )
        content_loss *= content_weight

        # Style loss
        style_loss = 0.0
        for layer in extractor.style_layers():
            output_gram = gram_matrix(output_features[layer])
            style_loss += F.mse_loss(output_gram, style_grams[layer])
        style_loss *= style_weight

        # Total variation loss
        tv_loss = tv_weight * total_variation(output)

        total_loss = content_loss + style_loss + tv_loss
        total_loss.backward()
        return total_loss

    print(f"Running {num_steps} optimization steps...")
    for step in range(1, num_steps + 1):
        loss = optimizer.step(closure)

        if step % log_interval == 0 or step == 1:
            with torch.no_grad():
                c_loss = content_weight * F.mse_loss(
                    extractor(output)[extractor.content_layers()[0]],
                    content_features[extractor.content_layers()[0]],
                ).item()
                s_loss = 0.0
                for layer in extractor.style_layers():
                    out_gram = gram_matrix(extractor(output)[layer])
                    s_loss += F.mse_loss(out_gram, style_grams[layer]).item()
                s_loss *= style_weight
                t_loss = tv_weight * total_variation(output).item()
            print(f"Step {step:4d} | "
                  f"content: {c_loss:8.1f} | "
                  f"style: {s_loss:10.1f} | "
                  f"tv: {t_loss:8.2f}")

            if progress_callback:
                progress_callback(step, c_loss, s_loss, t_loss)

    result = tensor_to_image(output)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"Saved to {output_path}")

    return result
