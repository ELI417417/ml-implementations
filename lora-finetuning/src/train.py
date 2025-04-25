"""
LoRA fine-tuning training loop.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data import load_alpaca_dataset, load_custom_dataset
from lora import LoRAConfig, count_trainable_params
from model import load_model_and_tokenizer, prepare_model_for_lora


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument("--model", type=str, default="google/gemma-2b")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", type=str, default="q_proj,v_proj",
                        help="Comma-separated target module names")
    parser.add_argument("--data", type=str, default="alpaca")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to custom JSON/JSONL dataset")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="./runs/lora")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output dir
    run_name = (f"lora_r{args.rank}_a{args.alpha}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        load_in_4bit=args.use_4bit,
        load_in_8bit=args.use_8bit,
    )

    # Inject LoRA
    lora_config = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    target_modules = [t.strip() for t in args.targets.split(",")]
    prepare_model_for_lora(model, lora_config, target_modules)

    # Load dataset
    if args.data_path:
        dataset = load_custom_dataset(
            args.data_path, tokenizer, args.max_length
        )
    else:
        dataset = load_alpaca_dataset(tokenizer, args.max_length)

    # Split train/val
    split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Optimizer: only trainable (LoRA) params
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Scheduler
    total_steps = (len(train_loader) // args.gradient_accumulation
                   * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float("inf")
    log_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += loss.item() * args.gradient_accumulation

                if global_step % args.log_steps == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item() * args.gradient_accumulation:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

        # Validation
        val_loss = evaluate(model, val_loader, device)
        avg_epoch_loss = epoch_loss / max(1, len(train_loader)
                                          // args.gradient_accumulation)
        print(f"Epoch {epoch} | Train loss: {avg_epoch_loss:.4f} | "
              f"Val loss: {val_loss:.4f}")

        log_history.append({
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "val_loss": val_loss,
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(out_dir / "best_model"))
            tokenizer.save_pretrained(str(out_dir / "best_model"))

    # Save final
    model.save_pretrained(str(out_dir / "final_model"))
    tokenizer.save_pretrained(str(out_dir / "final_model"))

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to {out_dir}")


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
    model.train()
    return total_loss / len(val_loader)


if __name__ == "__main__":
    train(parse_args())
