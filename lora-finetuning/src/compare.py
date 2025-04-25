"""
Compare multiple LoRA fine-tuning runs side by side.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir: Path) -> dict:
    """Load config and training log from a run directory."""
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "training_log.json") as f:
        log = json.load(f)
    return {"config": config, "log": log, "dir": run_dir.name}


def compare_runs(runs_dir: str = "./runs/lora",
                 output_path: str = "comparison.png") -> None:
    """Compare training curves across multiple LoRA runs."""
    runs_dir = Path(runs_dir)
    runs = [load_run(d) for d in runs_dir.iterdir()
            if d.is_dir() and (d / "training_log.json").exists()]

    if not runs:
        print(f"No runs found in {runs_dir}")
        return

    # Sort by rank for consistent legend
    runs.sort(key=lambda r: r["config"]["rank"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(runs)))

    for run, color in zip(runs, colors):
        cfg = run["config"]
        log = run["log"]
        label = f"r={cfg['rank']}, α={cfg['alpha']}, {cfg['model'].split('/')[-1]}"

        epochs = [e["epoch"] for e in log]
        train_loss = [e["train_loss"] for e in log]
        val_loss = [e["val_loss"] for e in log]

        axes[0].plot(epochs, train_loss, color=color, label=label, linewidth=2)
        axes[1].plot(epochs, val_loss, color=color, label=label, linewidth=2)

        # Bar chart: final val loss
        axes[2].barh(label, val_loss[-1], color=color, height=0.6)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss by Configuration")
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Validation Loss by Configuration")
    axes[1].legend(fontsize=7.5, loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel("Final Val Loss")
    axes[2].set_title("Final Validation Loss Comparison")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Run':<30} {'Rank':>5} {'Alpha':>6} {'Final Val Loss':>15}")
    print("-" * 70)
    for run in runs:
        cfg = run["config"]
        final_val = run["log"][-1]["val_loss"]
        print(f"{run['dir']:<30} {cfg['rank']:>5} {cfg['alpha']:>6} "
              f"{final_val:>15.4f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="./runs/lora")
    parser.add_argument("--output", type=str, default="comparison.png")
    args = parser.parse_args()
    compare_runs(args.runs_dir, args.output)
