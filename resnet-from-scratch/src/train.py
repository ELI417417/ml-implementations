"""
Training loop for ResNet on CIFAR-10.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_cifar10
from resnet import ResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet20",
                        choices=["resnet20", "resnet32", "resnet44", "resnet56",
                                 "resnet110", "resnet18", "resnet34", "resnet50"],
                        help="Model variant")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    # CIFAR-scale ResNet variants (from the paper)
    cifar_configs = {
        "resnet20":  (3, [3, 3, 3]),
        "resnet32":  (3, [5, 5, 5]),
        "resnet44":  (3, [7, 7, 7]),
        "resnet56":  (3, [9, 9, 9]),
        "resnet110": (3, [18, 18, 18]),
    }
    if name in cifar_configs:
        from resnet import BasicBlock, ResNet
        # Small ResNet for CIFAR — manual construction for paper fidelity
        n, layers = cifar_configs[name]
        return ResNet._build_cifar(n, layers, num_classes)
    # Standard ImageNet-scale variants
    depth_map = {"resnet18": 18, "resnet34": 34, "resnet50": 50}
    return ResNet(depth=depth_map[name], num_classes=num_classes)


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10(args.batch_size)
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d} | lr {lr:.5f} | "
              f"Train Loss {train_loss:.3f} Acc {train_acc:.2f}% | "
              f"Test Loss {test_loss:.3f} Acc {test_acc:.2f}%")

        log.append({
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.2f}% — saved to {out_dir}")


if __name__ == "__main__":
    main()
