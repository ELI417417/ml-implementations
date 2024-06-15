# ML Implementations

A monorepo of machine learning paper implementations and projects — built from scratch for learning. Each subdirectory is a self-contained project with its own README, source code, and tests.

## Projects

### Computer Vision

| Project | Paper | Description |
|---------|-------|-------------|
| [dcgan-image-generation](dcgan-image-generation/) | Radford et al., ICLR 2016 | DCGAN for 64×64 face generation |
| [resnet-from-scratch](resnet-from-scratch/) | He et al., CVPR 2016 | ResNet-18/34/50/101/152 + CIFAR variants |
| [vision-transformer-cifar100](vision-transformer-cifar100/) | Dosovitskiy et al., ICLR 2021 | ViT trained on CIFAR-100 |
| [neural-style-transfer](neural-style-transfer/) | Gatys et al., 2015 | Artistic style transfer with VGG19 |
| [yolo-object-detection](yolo-object-detection/) | Ultralytics YOLOv8 | Object detection web app with ONNX |
| [mobilenet-ssd-deployment](mobilenet-ssd-deployment/) | Howard et al., CVPR 2018 | Edge-optimized object detection |
| [realtime-video-analytics](realtime-video-analytics/) | — | Face detection + motion tracking dashboard |
| [pointnet-3d-classification](pointnet-3d-classification/) | Qi et al., CVPR 2017 | 3D point cloud classification |

### NLP & Multimodal

| Project | Paper | Description |
|---------|-------|-------------|
| [transformer-from-scratch](transformer-from-scratch/) | Vaswani et al., NeurIPS 2017 | Full Transformer implementation |
| [lora-finetuning](lora-finetuning/) | Hu et al., ICLR 2022 | LoRA ablation study for LLM fine-tuning |
| [clip-semantic-search](clip-semantic-search/) | Radford et al., ICML 2021 | CLIP + FAISS image search engine |
| [stable-diffusion-explorer](stable-diffusion-explorer/) | Rombach et al., CVPR 2022 | Diffusion model with attention visualization |

## Structure

```
ml-implementations/
├── dcgan-image-generation/     # 2024-06
├── transformer-from-scratch/   # 2024-08
├── vision-transformer-cifar100/# 2024-09
├── yolo-object-detection/      # 2024-10
├── mobilenet-ssd-deployment/   # 2024-11
├── clip-semantic-search/       # 2024-12
├── neural-style-transfer/      # 2025-01
├── resnet-from-scratch/        # 2025-03
├── lora-finetuning/            # 2025-04
├── stable-diffusion-explorer/  # 2025-05
├── pointnet-3d-classification/ # 2025-06
└── realtime-video-analytics/   # 2025-07
```

## Running Tests

Each project has its own test suite. Run from the project directory:

```bash
cd <project-dir>
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Approach

Every implementation in this repo is written from scratch for learning — I believe you don't understand something until you've built it yourself. These are not production systems; they are paper reproductions and learning projects.
