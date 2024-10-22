"""
Tests for YOLO detector and NMS.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from detector import YOLODetector


class TestNMS:
    def test_empty_boxes(self):
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        idx = YOLODetector._nms(boxes, scores, 0.5)
        assert len(idx) == 0

    def test_single_box(self):
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        idx = YOLODetector._nms(boxes, scores, 0.5)
        assert list(idx) == [0]

    def test_suppress_overlapping(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 48, 48],  # heavily overlaps box 0
        ])
        scores = np.array([0.9, 0.8])
        idx = YOLODetector._nms(boxes, scores, 0.5)
        assert list(idx) == [0]  # only higher-score box kept

    def test_keep_non_overlapping(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
        ])
        scores = np.array([0.9, 0.8])
        idx = YOLODetector._nms(boxes, scores, 0.5)
        assert len(idx) == 2


class TestCxcywhConversion:
    def test_conversion(self):
        boxes = np.array([[50, 50, 20, 30]])  # cx, cy, w, h
        result = YOLODetector._cxcywh_to_xyxy(boxes)
        expected = np.array([[40, 35, 60, 65]])  # x1, y1, x2, y2
        np.testing.assert_array_equal(result, expected)


class TestCOCOClasses:
    def test_class_count(self, detector):
        assert len(detector.class_names) == 80

    def test_first_class(self, detector):
        assert detector.class_names[0] == "person"


@pytest.fixture
def detector():
    """Skips if model not downloaded."""
    from pathlib import Path
    model = Path(__file__).resolve().parent.parent / "yolov8n.onnx"
    if not model.exists():
        pytest.skip("Model not downloaded — run download_model.py first")
    return YOLODetector(str(model))
