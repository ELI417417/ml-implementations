# LoRA Fine-tuning Experiment
## Low-Rank Adaptation of LLMs — hands-on comparison

A systematic fine-tuning experiment using LoRA (Low-Rank Adaptation) to adapt a pretrained language model to domain-specific tasks. Includes detailed comparisons of rank, alpha, and target module choices.

## Features

- LoRA fine-tuning of LLaMA-3 / Gemma 2B on Alpaca dataset
- Ablation studies: rank r ∈ {4, 8, 16, 32, 64}, alpha ∈ {8, 16, 32, 64}
- Target module comparison: Q only vs Q+K+V vs Q+K+V+O
- Parameter count and GPU memory benchmarks
- Side-by-side generation quality comparison
- Training loss curves and validation perplexity
- Merge LoRA weights into base model for inference
- Detailed Jupyter notebook with all experiments

## Key Findings

| Config | Trainable % | GPU Mem | Val PPL↓ | Time |
|--------|------------|---------|----------|------|
| Full FT | 100% | OOM | — | — |
| LoRA r=4 | 0.12% | 8.2 GB | 12.4 | 45m |
| LoRA r=8 | 0.24% | 8.8 GB | 11.2 | 52m |
| LoRA r=16 | 0.48% | 9.6 GB | 10.7 | 68m |
| LoRA r=32 | 0.95% | 11.2 GB | 10.3 | 95m |

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --rank 8 --alpha 16 --epochs 3
# Compare configs
python src/compare.py --runs-dir ./runs
```

## Project Structure

```
lora-finetuning/
├── src/
│   ├── train.py            # LoRA training script
│   ├── lora.py             # LoRA layer implementation
│   ├── model.py            # Model loading + LoRA injection
│   ├── data.py             # Dataset loading
│   ├── compare.py          # Compare training runs
│   └── generate.py         # Inference with LoRA
├── tests/
│   └── test_lora.py
├── notebooks/
│   └── ablation_study.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- https://arxiv.org/abs/2106.09685
