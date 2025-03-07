"""
Visualization tools for ResNet training analysis.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def plot_training_curves(log_path: str | Path,
                         output_path: str | Path = "training_curves.png"
                         ) -> None:
    """Plot loss and accuracy curves from training log."""
    with open(log_path) as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    test_loss = [e["test_loss"] for e in log]
    train_acc = [e["train_acc"] for e in log]
    test_acc = [e["test_acc"] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train", color="#38bdf8")
    ax1.plot(epochs, test_loss, label="Test", color="#f472b6")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train", color="#38bdf8")
    ax2.plot(epochs, test_acc, label="Test", color="#f472b6")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("ResNet Training on CIFAR-10", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved curves to {output_path}")


def visualize_filters(model: nn.Module, layer_name: str = "conv1",
                      output_path: str = "filters.png") -> None:
    """Visualize the first-layer convolution filters."""
    weight = model.state_dict()[f"{layer_name}.weight"]
    # (out_channels, in_channels, kH, kW) -> grid
    out_c, in_c, kh, kw = weight.shape
    weight = weight.detach().cpu().numpy()
    # Normalize for display
    w_min, w_max = weight.min(), weight.max()
    weight = (weight - w_min) / (w_max - w_min + 1e-8)

    cols = 8
    rows = (out_c + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for i in range(out_c):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        # Show first input channel
        ax.imshow(weight[i, 0], cmap="gray")
        ax.axis("off")
    for i in range(out_c, rows * cols):
        axes[i // cols, i % cols].axis("off") if rows > 1 else axes[i].axis("off")

    plt.suptitle(f"Filters of {layer_name} ({out_c} channels, {kh}×{kw})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved filter visualization to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="output/training_log.json")
    parser.add_argument("--output", type=str, default="output/training_curves.png")
    args = parser.parse_args()
    plot_training_curves(args.log, args.output)
