# ML Implementations

Monorepo of machine learning paper implementations and learning projects — built from scratch with PyTorch, tested, documented. Each subdirectory is a self-contained project.

## Paper Reproductions (8 papers)

| Paper | Venue | Directory | Key Components |
|-------|-------|-----------|----------------|
| Attention Is All You Need | NeurIPS 2017 | [transformer-from-scratch](transformer-from-scratch/) | Multi-head attention, positional encoding, encoder-decoder |
| Deep Residual Learning | CVPR 2016 | [resnet-from-scratch](resnet-from-scratch/) | BasicBlock, Bottleneck, ResNet-18/34/50/101/152 |
| DCGAN | ICLR 2016 | [dcgan-image-generation](dcgan-image-generation/) | Generator, discriminator, 64×64 face generation |
| Neural Style Transfer | 2015 | [neural-style-transfer](neural-style-transfer/) | VGG19, Gram matrix, L-BFGS optimization |
| ViT | ICLR 2021 | [vision-transformer-cifar100](vision-transformer-cifar100/) | Patch embedding, self-attention, class token |
| PointNet | CVPR 2017 | [pointnet-3d-classification](pointnet-3d-classification/) | T-Net, permutation invariance, ModelNet40 |
| LoRA | ICLR 2022 | [lora-finetuning](lora-finetuning/) | Low-rank adaptation, ablation study, weight merging |
| CLIP | ICML 2021 | [clip-semantic-search](clip-semantic-search/) | Vision-language embeddings, FAISS indexing |

## Application Projects (4)

| Project | Stack | Directory |
|---------|-------|-----------|
| YOLO Object Detection | ONNX Runtime, Flask, OpenCV | [yolo-object-detection](yolo-object-detection/) |
| MobileNet-SSD Study | PyTorch, ONNX | [mobilenet-ssd-deployment](mobilenet-ssd-deployment/) |
| Stable Diffusion Explorer | Diffusers, Streamlit | [stable-diffusion-explorer](stable-diffusion-explorer/) |
| Video Analytics Dashboard | OpenCV, Streamlit, Plotly | [realtime-video-analytics](realtime-video-analytics/) |

## Structure

```
ml-implementations/
├── dcgan-image-generation/      # 2024-06
├── transformer-from-scratch/    # 2024-08
├── vision-transformer-cifar100/ # 2024-09
├── yolo-object-detection/       # 2024-10
├── mobilenet-ssd-deployment/    # 2024-11
├── clip-semantic-search/        # 2024-12
├── neural-style-transfer/       # 2025-01
├── resnet-from-scratch/         # 2025-03
├── lora-finetuning/             # 2025-04
├── stable-diffusion-explorer/   # 2025-05
├── pointnet-3d-classification/  # 2025-06
└── realtime-video-analytics/    # 2025-07
```

## Running

Each project is self-contained:

```bash
cd <project-dir>
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Approach

Everything here is built from scratch for learning — I believe understanding comes from implementing. These are paper reproductions and learning projects, not production systems.

