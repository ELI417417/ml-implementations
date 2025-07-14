"""
Tests for face and motion detectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import cv2

from detector import FaceDetector, MotionDetector


def test_import_ok():
    """Sanity: detectors can be imported."""
    assert FaceDetector is not None
    assert MotionDetector is not None


class TestMotionDetector:
    @pytest.fixture
    def detector(self):
        return MotionDetector(threshold=25, min_area=500)

    def test_first_frame_no_motion(self, detector):
        frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        detections, mask = detector.detect(frame)
        assert len(detections) == 0

    def test_identical_frames_no_motion(self, detector):
        frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        detector.detect(frame.copy())
        detections, mask = detector.detect(frame.copy())
        assert len(detections) == 0

    def test_different_frames_produces_motion(self, detector):
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.ones((240, 320, 3), dtype=np.uint8) * 255

        detector.detect(frame1)
        detections, mask = detector.detect(frame2)

        assert mask is not None
        assert mask.max() > 0

    def test_reset(self, detector):
        frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        detector.detect(frame)
        assert detector._prev_frame is not None
        detector.reset()
        assert detector._prev_frame is None
