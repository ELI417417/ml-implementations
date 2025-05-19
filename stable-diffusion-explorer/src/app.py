"""
Streamlit app for exploring Stable Diffusion with attention maps.
"""

import io

import numpy as np
import streamlit as st
from PIL import Image

from pipeline import SDExplorer


@st.cache_resource
def get_explorer() -> SDExplorer:
    return SDExplorer()


def main():
    st.set_page_config(
        page_title="Stable Diffusion Explorer",
        page_icon="🎨",
        layout="wide",
    )

    st.title("🎨 Stable Diffusion Explorer")
    st.markdown(
        "Explore how text prompts guide image generation through "
        "cross-attention visualization."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Generation Settings")
        prompt = st.text_area(
            "Prompt",
            value="a serene lake at sunset, mountains in the background, "
                  "digital painting",
            height=100,
        )
        negative_prompt = st.text_input(
            "Negative Prompt",
            value="blurry, low quality, distorted",
        )

        num_steps = st.slider("Denoising Steps", 10, 100, 50, 5)
        guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        seed = st.number_input("Seed", -1, 999999, 42)
        seed = seed if seed >= 0 else None

        show_attention = st.checkbox("Show Attention Maps", value=False)

        generate = st.button("Generate", type="primary",
                             use_container_width=True)

    with col2:
        if generate:
            with st.spinner("Generating image..."):
                try:
                    explorer = get_explorer()
                    image, attn_maps = explorer.generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_steps=num_steps,
                        guidance_scale=guidance,
                        seed=seed,
                        return_attention=show_attention,
                    )

                    st.image(image, caption=f"\"{prompt}\"",
                             use_column_width=True)

                    if attn_maps and show_attention:
                        st.subheader("Cross-Attention Maps")
                        st.caption(
                            "Showing the spatial influence of each token "
                            "from the last cross-attention layer"
                        )
                        # Show last layer's attention
                        last_map = attn_maps[-1]
                        if len(last_map.shape) == 4:
                            n_heads = min(4, last_map.shape[1])
                            cols = st.columns(n_heads)
                            for i, col in enumerate(cols):
                                with col:
                                    attn_img = (last_map[0, i, 0]
                                                .reshape(16, 16))
                                    fig_buf = io.BytesIO()
                                    Image.fromarray(
                                        (attn_img * 255).astype(np.uint8)
                                    ).save(fig_buf, format="PNG")
                                    st.image(
                                        fig_buf,
                                        caption=f"Head {i + 1}",
                                        use_column_width=True,
                                    )
                except Exception as e:
                    st.error(
                        f"Generation failed: {e}\n\n"
                        f"Note: Stable Diffusion requires significant GPU "
                        f"VRAM (8GB+ recommended)."
                    )

    st.markdown("---")
    st.caption(
        "Stable Diffusion v1.5 — Rombach et al., CVPR 2022. "
        "Cross-attention from Hertz et al., 2022."
    )


if __name__ == "__main__":
    main()
