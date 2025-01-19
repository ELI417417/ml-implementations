# Neural Style Transfer
## Artistic style transfer with VGG19 and PyTorch

Apply the artistic style of famous paintings to your photos using a pretrained VGG19 network. Based on the method by Gatys et al. (2015), this project includes a Streamlit web app for interactive style transfer.

## Features

- Content + style image upload via Streamlit
- Pretrained VGG19 feature extractor
- Tunable content/style weight ratio
- Multiple style presets (Van Gogh, Monet, Picasso-style)
- Real-time progress and intermediate result preview
- High-resolution output with optional histogram matching
- Total Variation loss for spatial smoothness

## How It Works

1. Extract **content features** from higher VGG layers (conv4_2)
2. Extract **style features** via Gram matrices from multiple VGG layers
3. Optimize a **white noise image** to match both content and style features
4. Total Variation loss reduces high-frequency noise

## Quick Start

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Examples

| Content | Style | Result |
|---------|-------|--------|
| Photo | Starry Night | Stylized photo |

## Project Structure

```
neural-style-transfer/
├── src/
│   ├── app.py              # Streamlit web app
│   ├── style_transfer.py   # Core style transfer algorithm
│   └── model.py            # VGG19 feature extractor
├── tests/
│   └── test_style_transfer.py
├── notebooks/
│   └── experiments.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Gatys et al., "A Neural Algorithm of Artistic Style", 2015
- https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
