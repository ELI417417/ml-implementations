# ResNet From Scratch
## Paper Reproduction: Deep Residual Learning for Image Recognition

A line-by-line reproduction of the ResNet architecture from the original paper by He et al. (CVPR 2016). Includes training on CIFAR-10 with detailed ablation studies and annotated code.

## Features

- ResNet-18, 34, 50, 101, 152 implementations from scratch (no torchvision)
- Residual block with identity and projection shortcuts
- Batch Normalization after each convolution (pre-activation variant)
- Training on CIFAR-10 with full logging
- Ablation studies: with/without skip connections, depth comparison
- Detailed Jupyter notebook with paper annotations
- Weight visualization and gradient flow analysis

## Key Findings

| Model | Params | CIFAR-10 Accuracy |
|-------|--------|-------------------|
| Plain-20 (no skip) | 0.27M | 86.2% |
| ResNet-20 | 0.27M | 91.4% |
| ResNet-56 | 0.85M | 93.0% |
| ResNet-110 | 1.73M | 93.5% |

Skip connections enable training deeper networks without degradation.

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --model resnet20 --epochs 200
# View results in notebook
jupyter notebook notebooks/paper_analysis.ipynb
```

## Project Structure

```
resnet-from-scratch/
├── src/
│   ├── resnet.py           # Full ResNet implementation
│   ├── train.py            # Training loop
│   ├── dataset.py          # CIFAR-10 data pipeline
│   └── visualize.py        # Weight/gradient visualization
├── tests/
│   └── test_resnet.py
├── notebooks/
│   └── paper_analysis.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- https://arxiv.org/abs/1512.03385
