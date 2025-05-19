# Stable Diffusion Explorer
## Interactive diffusion model visualization with attention map analysis

A Streamlit app for exploring Stable Diffusion inference. Visualize the denoising process step-by-step, inspect cross-attention maps between text tokens and generated pixels, and understand how text prompts guide image generation.

## Features

- Stable Diffusion v1.5 inference pipeline
- Step-by-step denoising visualization (progressive image refinement)
- Cross-attention map visualization — see which words influence which pixels
- Prompt interpolation (smooth transition between two prompts)
- Parameter controls: guidance scale, denoising steps, seed
- Per-token attention heatmap overlay on generated images
- Side-by-side comparison of different seeds/prompts

## How It Works

1. Text prompt → CLIP text encoder → text embeddings
2. Random noise → UNet denoiser (×50 steps) → latent image
3. Cross-attention layers map tokens → spatial regions
4. VAE decoder → final 512×512 image

## Quick Start

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Project Structure

```
stable-diffusion-explorer/
├── src/
│   ├── app.py              # Streamlit explorer UI
│   ├── pipeline.py         # Diffusion pipeline wrapper
│   └── attention.py        # Cross-attention extraction
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── prompt_experiments.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
