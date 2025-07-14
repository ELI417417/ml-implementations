"""
Face and motion detection engine using OpenCV.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    """A single detection result."""
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    label: str


class FaceDetector:
    """Multi-method face detector: Haar Cascade + DNN fallback."""

    def __init__(self, method: str = "haar",
                 confidence_threshold: float = 0.5):
        self.method = method
        self.confidence_threshold = confidence_threshold

        # Load Haar cascade
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(haar_path)

        # DNN face detector (SSD-based)
        self.dnn_net = None
        if method == "dnn":
            self._init_dnn()

    def _init_dnn(self):
        """Load OpenCV DNN face detector (requires model files)."""
        model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "models/deploy.prototxt"
        if Path(model_file).exists() and Path(config_file).exists():
            self.dnn_net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect faces in a frame."""
        if self.method == "dnn" and self.dnn_net is not None:
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    def _detect_haar(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30),
        )
        return [
            Detection(bbox=tuple(f), confidence=1.0, label="face")
            for f in faces
        ]

    def _detect_dnn(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     [104, 117, 123], False, False)
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                results.append(Detection(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=float(confidence),
                    label="face",
                ))
        return results


class MotionDetector:
    """Detect motion via frame differencing."""

    def __init__(self, threshold: int = 25, min_area: int = 500,
                 blur_size: int = 21):
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self._prev_frame: np.ndarray | None = None
        self._motion_mask: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """Detect motion regions by comparing with the previous frame.

        Args:
            frame: Current BGR frame.

        Returns:
            (list of Detection, motion mask).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if self._prev_frame is None:
            self._prev_frame = gray
            self._motion_mask = np.zeros_like(gray)
            return [], self._motion_mask

        # Frame difference
        diff = cv2.absdiff(self._prev_frame, gray)
        _, thresh = cv2.threshold(diff, self.threshold, 255,
                                  cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        self._motion_mask = thresh

        # Find motion contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(Detection(
                    bbox=(x, y, w, h),
                    confidence=min(1.0, area / 10000),
                    label="motion",
                ))

        self._prev_frame = gray
        return detections, self._motion_mask

    def reset(self) -> None:
        """Reset the motion detector state."""
        self._prev_frame = None
        self._motion_mask = None

    @property
    def motion_mask(self) -> np.ndarray | None:
        return self._motion_mask

    @property
    def motion_level(self) -> float:
        """Fraction of frame with detected motion (0–1)."""
        if self._motion_mask is None:
            return 0.0
        return float(np.mean(self._motion_mask > 0))
