# DCGAN Image Generation
## Deep Convolutional Generative Adversarial Network for face generation

A PyTorch implementation of DCGAN trained on the CelebA dataset. Generate realistic 64x64 face images from random noise.

## Features

- DCGAN generator and discriminator from scratch
- Training on CelebA dataset with progressive image saving
- Streamlit web demo for interactive generation
- Checkpoint management and resume training
- Generated image gallery with interpolation

## Architecture

```
Generator: Noise(100) → ConvTranspose → BN → ReLU → ... → Tanh(64x64x3)
Discriminator: Image(64x64x3) → Conv → BN → LeakyReLU → ... → Sigmoid
```

## Quick Start

```bash
pip install -r requirements.txt
# Train
python src/train.py --epochs 50 --batch-size 128
# Streamlit demo
streamlit run src/app.py
```

## Results

After 50 epochs of training on CelebA:

![Generated samples](assets/samples.png)

## Project Structure

```
dcgan-image-generation/
├── src/
│   ├── model.py          # Generator & Discriminator
│   ├── train.py           # Training loop
│   ├── dataset.py         # Data loading & transforms
│   └── app.py             # Streamlit demo
├── tests/
│   └── test_model.py
├── notebooks/
│   └── training_log.ipynb
├── assets/
├── requirements.txt
└── README.md
```

## References

- Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs", ICLR 2016
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
