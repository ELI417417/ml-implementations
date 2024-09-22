# Vision Transformer (ViT) Classifier
## From-scratch implementation on CIFAR-100 — Dosovitskiy et al., ICLR 2021

A from-scratch PyTorch implementation of the Vision Transformer architecture. Trained on CIFAR-100 and compared against ResNet-50 baseline with the same parameter budget.

## Features

- Patch embedding with learnable position encoding
- Multi-head self-attention with class token
- Full ViT-Base, ViT-Tiny, and ViT-Small variants
- Comparison: ViT vs ResNet-50 on CIFAR-100
- Attention rollout visualization
- Impact of patch size ablation (2×2, 4×4, 8×8)

## Key Results (CIFAR-100, 200 epochs)

| Model | Params | Top-1 Acc | Top-5 Acc |
|-------|--------|-----------|-----------|
| ViT-Tiny (p=4) | 5.5M | 68.2% | 87.9% |
| ViT-Small (p=4) | 22M | 72.1% | 90.3% |
| ResNet-50 | 25M | 71.5% | 89.8% |

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --model vit-tiny --patch-size 4 --epochs 200
```

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
