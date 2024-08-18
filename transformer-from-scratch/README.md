# Transformer From Scratch
## Paper Reproduction: Attention Is All You Need (Vaswani et al., NeurIPS 2017)

A minimal, readable PyTorch implementation of the original Transformer architecture. Trained on WikiText-2 for language modeling with full attention visualization.

## Features

- Multi-head scaled dot-product attention from scratch
- Positional encoding (sinusoidal + learned)
- Encoder-decoder architecture exactly as described in the paper
- Training on WikiText-2 with BPE tokenizer
- Attention weight visualization for every head
- Ablation: with/without residual, with/without layer norm
- Comparison: LSTM baseline vs Transformer

## Architecture

```
Input → Embedding + Positional Encoding
  → [Multi-Head Attention → Add & Norm → FFN → Add & Norm] × N
  → Linear → Softmax → Output
```

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --epochs 20 --batch-size 64
```

## Project Structure

```
transformer-from-scratch/
├── src/
│   ├── model.py          # Full Transformer implementation
│   ├── attention.py      # Multi-head attention
│   ├── train.py          # Training loop on WikiText-2
│   └── visualize.py      # Attention map visualization
├── tests/
│   └── test_model.py
├── notebooks/
│   └── attention_analysis.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- https://arxiv.org/abs/1706.03762
