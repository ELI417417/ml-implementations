# PointNet — 3D Point Cloud Classification
## From-scratch implementation of PointNet on ModelNet40

A PyTorch implementation of PointNet (Qi et al., CVPR 2017) for 3D point cloud classification. Trained on ModelNet40 with Open3D visualization of learned features.

## Features

- PointNet architecture with T-Net input transform
- Point cloud data augmentation (rotation, jitter, scaling)
- ModelNet40 data loader with .off file parsing
- Open3D visualization of critical point sets
- Ablation: with/without T-Net transform
- Feature space visualization (t-SNE of global features)
- Comparison: PointNet vs PointNet++ vs DGCNN

## Architecture

```
Input (N×3) → T-Net (3×3) → MLP(64,128,1024)
  → Max Pool (global) → MLP(512,256,K) → Output (K classes)
  └── Critical points (subset that defines the global shape)
```

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --epochs 200
```

## Project Structure

```
pointnet-3d-classification/
├── src/
│   ├── model.py          # PointNet architecture
│   ├── train.py          # Training loop on ModelNet40
│   ├── dataset.py        # ModelNet40 data loader
│   └── visualize.py      # Open3D visualization
├── tests/
│   └── test_model.py
├── notebooks/
│   └── critical_points.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017
- ModelNet40: https://modelnet.cs.princeton.edu/
