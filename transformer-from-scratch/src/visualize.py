"""
Multi-head attention visualization and analysis tools.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_heads(attn_weights: torch.Tensor,
                         tokens: list[str] | None = None,
                         output_path: str = "attention_heads.png",
                         n_heads: int = 8) -> None:
    """
    Plot all attention heads from a single layer.

    Args:
        attn_weights: Attention weights of shape (batch, n_heads, seq_len, seq_len).
        tokens: Optional list of token strings for axis labels.
        output_path: Where to save the figure.
        n_heads: Number of heads to display.
    """
    weights = attn_weights[0].detach().cpu().numpy()  # take first batch
    n_heads = min(n_heads, weights.shape[0])
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for i in range(n_heads):
        im = axes[i].imshow(weights[i], cmap="Blues", aspect="auto")
        axes[i].set_title(f"Head {i + 1}")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Multi-Head Attention Weights", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """Compute the average entropy of attention distributions."""
    weights = attn_weights.detach().cpu()
    # Avoid log(0)
    eps = 1e-9
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)
    return float(entropy.mean())
