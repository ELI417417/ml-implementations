"""
YOLO object detector using ONNX Runtime.

Performs inference on preprocessed images and applies post-processing:
confidence thresholding, NMS, and coordinate rescaling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    """A single detection result."""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class YOLODetector:
    """YOLOv8 object detector backed by ONNX Runtime."""

    def __init__(self, model_path: str = "yolov8n.onnx",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Run `python src/download_model.py` first."
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        _, _, self.input_h, self.input_w = self.input_shape

        self.class_names = self._coco_classes()

    @staticmethod
    def _coco_classes() -> list[str]:
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush",
        ]

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Preprocess an image for YOLO inference.

        Args:
            image: BGR numpy array (H, W, 3) from cv2.imread.

        Returns:
            (input_tensor, scale_x, scale_y) where input_tensor is (1, 3, H, W)
            in CHW format normalized to [0, 1], and scales are the resize ratios.
        """
        h, w = image.shape[:2]
        scale_x = self.input_w / w
        scale_y = self.input_h / h

        import cv2
        resized = cv2.resize(image, (self.input_w, self.input_h))
        # HWC → CHW and normalize to [0, 1]
        tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, scale_x, scale_y

    def postprocess(self, output: np.ndarray, scale_x: float, scale_y: float,
                    orig_h: int, orig_w: int) -> list[Detection]:
        """
        Process raw YOLO output into Detection objects.

        YOLOv8 output shape: (1, 84, 8400) — 4 bbox coords + 80 class scores.
        """
        output = np.squeeze(output[0])  # (84, 8400)
        predictions = output.T          # (8400, 84)

        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold

        if not mask.any():
            return []

        filtered_boxes = predictions[mask, :4]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]

        # Convert cxcywh → xyxy
        boxes_xyxy = self._cxcywh_to_xyxy(filtered_boxes)

        # Rescale to original image dimensions
        boxes_xyxy[:, [0, 2]] /= scale_x
        boxes_xyxy[:, [1, 3]] /= scale_y

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # Apply NMS
        indices = self._nms(boxes_xyxy, filtered_scores, self.iou_threshold)

        detections = []
        for idx in indices:
            x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
            detections.append(Detection(
                class_name=self.class_names[filtered_class_ids[idx]],
                confidence=float(filtered_scores[idx]),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))
        return detections

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
        result = np.zeros_like(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return result

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray,
             iou_threshold: float) -> np.ndarray:
        """Non-maximum suppression — returns indices to keep."""
        if len(boxes) == 0:
            return np.array([], dtype=int)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return np.array(keep, dtype=int)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a BGR image."""
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        tensor, scale_x, scale_y = self.preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})
        return self.postprocess(output[0], scale_x, scale_y, h, w)
