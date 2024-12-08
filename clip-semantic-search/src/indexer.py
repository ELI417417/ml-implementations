"""
CLI for building a CLIP-based image search index.
"""

import argparse
from pathlib import Path

from search import ImageSearchEngine


def main():
    parser = argparse.ArgumentParser(
        description="Build CLIP image search index"
    )
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory of images to index")
    parser.add_argument("--index-dir", type=str, default="./index",
                        help="Where to save the FAISS index")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Encoding batch size")
    args = parser.parse_args()

    if not Path(args.image_dir).exists():
        raise FileNotFoundError(f"Directory not found: {args.image_dir}")

    engine = ImageSearchEngine(index_dir=args.index_dir)
    count = engine.index_images(args.image_dir, batch_size=args.batch_size)
    print(f"Indexed {count} images. Index saved to {args.index_dir}")


if __name__ == "__main__":
    main()
