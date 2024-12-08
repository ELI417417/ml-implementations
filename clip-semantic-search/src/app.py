"""
Streamlit app for semantic image search.
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from search import ImageSearchEngine


@st.cache_resource
def get_engine(index_dir: str = "./index") -> ImageSearchEngine:
    engine = ImageSearchEngine(index_dir=index_dir)
    if not engine._load_index():
        pass  # Will be handled when user searches
    return engine


def main():
    st.set_page_config(
        page_title="CLIP Image Search",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Semantic Image Search")
    st.markdown(
        "Search your image collection using natural language. "
        "Powered by OpenAI CLIP + FAISS."
    )

    # Sidebar: index management
    with st.sidebar:
        st.subheader("Index Management")
        image_dir = st.text_input(
            "Image directory", value="./photos",
            help="Folder containing images to index"
        )
        index_dir = st.text_input(
            "Index directory", value="./index",
            help="Where to save the search index"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Build Index", use_container_width=True):
                with st.spinner("Encoding images..."):
                    engine = ImageSearchEngine(index_dir=index_dir)
                    count = engine.index_images(image_dir)
                    st.success(f"Indexed {count} images!")
                    st.cache_resource.clear()
        with col2:
            if st.button("Load Index", use_container_width=True):
                st.cache_resource.clear()
                st.success("Index reloaded!")

        st.divider()
        st.caption("CLIP ViT-B/32 · FAISS Inner Product Search")

    # Main area
    query = st.text_input(
        "Search query",
        placeholder="e.g., 'a dog playing in the snow', 'sunset over mountains'...",
    )
    top_k = st.slider("Results", 5, 50, 20, 5)

    if query:
        engine = get_engine(index_dir)
        if engine.index is None:
            st.warning(
                "No index found. Build an index first using the sidebar "
                "(point to a folder of images), or load an existing one."
            )
        else:
            with st.spinner(f"Searching for '{query}'..."):
                results = engine.search(query, top_k=top_k)

            st.success(
                f"Found {len(results)} results across "
                f"{len(engine.image_paths)} indexed images"
            )

            # Display as a grid of thumbnails
            cols = st.columns(4)
            for i, r in enumerate(results):
                col = cols[i % 4]
                with col:
                    try:
                        img = Image.open(r.path)
                        st.image(img, use_column_width=True,
                                 caption=f"Score: {r.score:.3f}")
                    except Exception:
                        st.error(f"Cannot load: {r.path}")

    elif Path("./demo_photos").exists():
        st.subheader("Sample Images")
        cols = st.columns(4)
        demo_dir = Path("./demo_photos")
        for i, p in enumerate(list(demo_dir.glob("*"))[:8]):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                with cols[i % 4]:
                    st.image(str(p), use_column_width=True)


if __name__ == "__main__":
    main()
