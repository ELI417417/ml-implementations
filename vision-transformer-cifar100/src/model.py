"""
Vision Transformer (ViT) — Dosovitskiy et al., ICLR 2021.

Image → Patch Embedding → Transformer Encoder → Classifier Head
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, n_patches, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for ViT."""

    def __init__(self, embed_dim: int = 192, n_heads: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """ViT encoder block: Attention + MLP with residual connections."""

    def __init__(self, embed_dim: int = 192, n_heads: int = 3,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 100,
                 embed_dim: int = 192, depth: int = 12,
                 n_heads: int = 3, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def vit_tiny(num_classes: int = 100, **kwargs) -> VisionTransformer:
    """ViT-Tiny: 5.5M params — suitable for CIFAR."""
    return VisionTransformer(
        img_size=32, patch_size=4, embed_dim=192,
        depth=12, n_heads=3, num_classes=num_classes, **kwargs
    )


def vit_small(num_classes: int = 100, **kwargs) -> VisionTransformer:
    """ViT-Small: 22M params."""
    return VisionTransformer(
        img_size=32, patch_size=4, embed_dim=384,
        depth=12, n_heads=6, num_classes=num_classes, **kwargs
    )
