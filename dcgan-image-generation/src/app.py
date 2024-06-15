"""
Streamlit demo for DCGAN image generation.
"""

import io

import numpy as np
import streamlit as st
import torch
import torchvision.utils as vutils
from PIL import Image

from model import Generator


@st.cache_resource
def load_model(checkpoint_path: str, latent_dim: int = 100, device: str = "cpu"
               ) -> Generator:
    model = Generator(latent_dim=latent_dim)
    state = torch.load(checkpoint_path, map_location=device)
    if "netG" in state:
        state = state["netG"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def generate_images(generator: Generator, num_images: int, latent_dim: int,
                    seed: int | None, device: str) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake = generator(noise).cpu()
    return fake


def main():
    st.set_page_config(page_title="DCGAN Image Generator",
                       page_icon="🎨", layout="wide")

    st.title("🎨 DCGAN Face Generator")
    st.markdown(
        "Generate realistic human faces using a Deep Convolutional GAN "
        "trained on CelebA."
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Controls")
        num_images = st.slider("Number of images", 1, 16, 4)
        seed = st.number_input("Random seed (0 = random)", 0, 99999, 42)
        seed = seed if seed > 0 else None
        generate_btn = st.button("Generate", type="primary")

    with col2:
        st.subheader("Generated Faces")

        if generate_btn:
            with st.spinner("Generating..."):
                device = ("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    generator = load_model(
                        "output/generator_final.pth",
                        device=device
                    )
                    images = generate_images(generator, num_images, 100,
                                             seed, device)
                    grid = vutils.make_grid(images, nrow=4, normalize=True)
                    buf = io.BytesIO()
                    vutils.save_image(grid, buf, format="PNG")
                    st.image(buf, use_column_width=True)
                    st.success(f"Generated {num_images} images")
                except FileNotFoundError:
                    st.warning(
                        "No trained model found. Train first with: "
                        "`python src/train.py --epochs 50`"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("Trained on CelebA using DCGAN architecture (Radford et al., 2016)")


if __name__ == "__main__":
    main()
