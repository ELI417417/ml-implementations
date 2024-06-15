"""
DCGAN Generator and Discriminator architectures.

Based on the architecture described in:
Radford et al., "Unsupervised Representation Learning with
Deep Convolutional Generative Adversarial Networks", ICLR 2016.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN Generator: maps latent vector z to a 64x64 RGB image."""

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64,
                 channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            # input: (latent_dim) x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0,
                               bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # state: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # state: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # state: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # state: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output: channels x 64 x 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator: classifies 64x64 images as real or fake."""

    def __init__(self, channels: int = 3, feature_maps: int = 64):
        super().__init__()

        self.main = nn.Sequential(
            # input: channels x 64 x 64
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (feature_maps) x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def weights_init(m: nn.Module):
    """Initialize weights from Normal(0, 0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
