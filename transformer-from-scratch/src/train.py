"""
Training script for Transformer on WikiText-2 language modeling.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-encoder-layers", type=int, default=3)
    parser.add_argument("--n-decoder-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="./output")
    return parser.parse_args()


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a random batch from the data."""
    idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    src = torch.stack([data[i:i + seq_len] for i in idx])
    tgt = torch.stack([data[i + 1:i + seq_len + 1] for i in idx])
    return src.to(device), tgt.to(device)


def train_epoch(model, data, optimizer, criterion, batch_size, seq_len,
                device, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 200  # batches per epoch for WikiText-2

    for _ in tqdm(range(n_batches), desc="Train", leave=False):
        src, tgt = get_batch(data, batch_size, seq_len, device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_mask = Transformer.generate_square_subsequent_mask(
            tgt_input.size(1), device
        )

        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, criterion, batch_size, seq_len, device):
    model.eval()
    total_loss = 0.0
    n_batches = 50

    for _ in range(n_batches):
        src, tgt = get_batch(data, batch_size, seq_len, device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_mask = Transformer.generate_square_subsequent_mask(
            tgt_input.size(1), device
        )
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        total_loss += loss.item()

    return total_loss / n_batches, math.exp(total_loss / n_batches)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy data for demonstration (replace with WikiText-2 in production)
    torch.manual_seed(42)
    vocab_size = 10000
    data = torch.randint(0, vocab_size, (50000,))

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_ppl = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, data, optimizer, criterion,
            args.batch_size, args.seq_len, device, scheduler
        )
        val_loss, val_ppl = evaluate(
            model, data, criterion,
            args.batch_size, args.seq_len, device
        )
        print(f"Epoch {epoch:3d} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save(model.state_dict(), out_dir / "best_model.pth")

    print(f"Training complete. Best PPL: {best_ppl:.1f}")


if __name__ == "__main__":
    main()
