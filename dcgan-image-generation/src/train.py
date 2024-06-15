"""
Training loop for DCGAN on CelebA / custom image datasets.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Discriminator, Generator, weights_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCGAN")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Adam beta1")
    parser.add_argument("--latent-dim", type=int, default=100,
                        help="Latent vector dimension")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Input image size")
    parser.add_argument("--feature-maps", type=int, default=64,
                        help="Base feature map count")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory for outputs")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_dataloader(data_root: str, image_size: int, batch_size: int
                   ) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=2, drop_last=True)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(args.data_root, args.image_size,
                                args.batch_size)

    netG = Generator(args.latent_dim, args.feature_maps).to(device)
    netD = Discriminator(feature_maps=args.feature_maps).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr,
                            betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr,
                            betas=(args.beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    g_losses, d_losses = [], []

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for real, _ in pbar:
            batch_size = real.size(0)
            real = real.to(device)

            # --- Train Discriminator ---
            netD.zero_grad()
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(batch_size, args.latent_dim, 1, 1,
                                device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # --- Train Generator ---
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            g_losses.append(errG.item())
            d_losses.append(errD.item())

            pbar.set_postfix(D_loss=f"{errD.item():.4f}",
                             G_loss=f"{errG.item():.4f}")

        # Generate and save sample images
        with torch.no_grad():
            fake_images = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_images,
                          out_dir / f"fake_samples_epoch_{epoch:03d}.png",
                          normalize=True)

        if epoch % args.checkpoint_interval == 0:
            torch.save({
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict(),
                "g_losses": g_losses,
                "d_losses": d_losses,
            }, out_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    torch.save(netG.state_dict(), out_dir / "generator_final.pth")
    print(f"Training complete. Output saved to {out_dir}")


if __name__ == "__main__":
    train(parse_args())
