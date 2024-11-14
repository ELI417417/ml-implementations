# MobileNet-SSD Edge Deployment
## Lightweight object detection optimized for CPU & edge devices

A complete object detection pipeline using MobileNet-SSD, optimized for deployment on edge devices and CPU. Includes ONNX export, latency benchmarking, and comparison with YOLOv8-nano.

## Features

- MobileNetV2-SSD implementation in PyTorch
- Export to ONNX with dynamic batch support
- CPU latency benchmarking (OpenVINO, ONNX Runtime, PyTorch)
- Comparison: MobileNet-SSD vs YOLOv8n vs EfficientDet-Lite
- Edge deployment optimization: INT8 quantization, pruning
- Real-time webcam demo

## Benchmarks (Intel i7-13700K, 640×480)

| Model | Size | mAP@0.5 | CPU Latency | FPS |
|-------|------|---------|-------------|-----|
| MobileNet-SSD | 24 MB | 0.68 | 12 ms | 83 |
| YOLOv8n | 6 MB | 0.72 | 18 ms | 55 |
| EfficientDet-Lite0 | 16 MB | 0.66 | 25 ms | 40 |

## Quick Start

```bash
pip install -r requirements.txt
python src/export_onnx.py              # Convert to ONNX
python src/benchmark.py                # Run benchmarks
python src/demo.py --source webcam     # Live demo
```

## Project Structure

```
mobilenet-ssd-deployment/
├── src/
│   ├── model.py          # MobileNetV2-SSD architecture
│   ├── export_onnx.py    # ONNX export + optimization
│   ├── benchmark.py      # Multi-backend latency benchmark
│   ├── demo.py           # Real-time webcam demo
│   └── quantize.py       # INT8 post-training quantization
├── tests/
│   └── test_model.py
├── notebooks/
│   └── latency_study.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Howard et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016
