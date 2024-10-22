"""
Download YOLOv8n ONNX model from Ultralytics.
"""

import sys
from pathlib import Path

import requests

YOLOV8N_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
    "yolov8n.onnx"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.onnx"


def download_model(url: str = YOLOV8N_URL, dest: Path = MODEL_PATH) -> None:
    if dest.exists():
        print(f"Model already exists: {dest}")
        return

    print(f"Downloading YOLOv8n ONNX from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {downloaded}/{total} bytes ({pct}%)", end="")
    print(f"\nSaved to {dest}")


if __name__ == "__main__":
    download_model()
