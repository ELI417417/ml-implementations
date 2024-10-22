"""
Utility functions for YOLO post-processing and image handling.
"""

import numpy as np


def letterbox_resize(image: np.ndarray, target_size: tuple[int, int],
                     color: tuple[int, int, int] = (114, 114, 114)
                     ) -> tuple[np.ndarray, float, float, int, int]:
    """Resize with letterbox padding (preserves aspect ratio)."""
    import cv2
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (nw, nh))
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, scale, pad_x, pad_y


def compute_iou(box_a: tuple[int, int, int, int],
                box_b: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
