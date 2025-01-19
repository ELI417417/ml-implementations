"""
Streamlit web app for Neural Style Transfer.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from style_transfer import style_transfer

STYLE_PRESETS = {
    "Van Gogh — Starry Night": "assets/starry_night.jpg",
    "Monet — Water Lilies": "assets/water_lilies.jpg",
    "Picasso — The Weeping Woman": "assets/picasso.jpg",
    "Munch — The Scream": "assets/the_scream.jpg",
    "Hokusai — The Great Wave": "assets/great_wave.jpg",
    "Custom upload": None,
}


def main():
    st.set_page_config(
        page_title="Neural Style Transfer",
        page_icon="🎨",
        layout="wide",
    )

    st.title("🎨 Neural Style Transfer")
    st.markdown(
        "Apply the artistic style of famous paintings to your photos, "
        "powered by VGG19 and the Gatys et al. algorithm."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader(
            "Upload your photo", type=["jpg", "jpeg", "png"], key="content"
        )
        if content_file:
            st.image(content_file, use_column_width=True)

    with col2:
        st.subheader("Style")
        style_choice = st.selectbox("Choose a style", list(STYLE_PRESETS.keys()))
        style_file = None
        if style_choice == "Custom upload":
            style_file = st.file_uploader(
                "Upload style image", type=["jpg", "jpeg", "png"], key="style"
            )
            if style_file:
                st.image(style_file, use_column_width=True)
        elif STYLE_PRESETS[style_choice]:
            preset_path = STYLE_PRESETS[style_choice]
            if Path(preset_path).exists():
                st.image(str(preset_path), use_column_width=True)
            else:
                st.info("(Preset image placeholder)")

    with col3:
        st.subheader("Controls")
        image_size = st.select_slider(
            "Image size", options=[256, 384, 512, 768], value=512
        )
        num_steps = st.slider("Optimization steps", 50, 500, 300, 50)
        content_weight = st.slider(
            "Content weight", 0.1, 10.0, 1.0, 0.1,
            help="Higher = preserve more photo details"
        )
        style_weight = st.slider(
            "Style weight", 1e4, 1e8, 1e6, 1e4, format="%.0e",
            help="Higher = stronger style application"
        )
        tv_weight = st.slider(
            "Smoothness (TV)", 0.0, 1e-4, 1e-6, 1e-6, format="%.1e",
            help="Reduces high-frequency noise"
        )

        run_btn = st.button("Run Style Transfer", type="primary",
                            disabled=not content_file)

    if run_btn and content_file:
        # Save uploaded content image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            content_path = f.name
            Image.open(content_file).save(content_path)

        # Get style image
        style_path = None
        if style_file:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as g:
                style_path = g.name
                Image.open(style_file).save(style_path)
        elif STYLE_PRESETS.get(style_choice):
            sp = STYLE_PRESETS[style_choice]
            if Path(sp).exists():
                style_path = sp

        if not style_path:
            st.error("No style image available.")
            st.stop()

        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(step, c_loss, s_loss, t_loss):
            pct = min(step / num_steps, 1.0)
            progress_bar.progress(pct)
            status.text(
                f"Step {step}/{num_steps} | "
                f"Content: {c_loss:.0f} | Style: {s_loss:.0f}"
            )

        with st.spinner(f"Optimizing for {num_steps} steps..."):
            try:
                result = style_transfer(
                    content_path=content_path,
                    style_path=style_path,
                    image_size=image_size,
                    num_steps=num_steps,
                    content_weight=content_weight,
                    style_weight=style_weight,
                    tv_weight=tv_weight,
                    log_interval=max(1, num_steps // 10),
                    progress_callback=update_progress,
                )
                st.subheader("Result")
                st.image(result, use_column_width=True)
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                st.download_button(
                    "Download Result",
                    data=buf.getvalue(),
                    file_name="styled_image.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("Gatys et al., 'A Neural Algorithm of Artistic Style', 2015")


if __name__ == "__main__":
    main()
