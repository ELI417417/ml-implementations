"""
Video processing pipeline: reads frames, runs detections, produces annotated output.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Thread
from typing import Generator, Optional

import cv2
import numpy as np

from detector import Detection, FaceDetector, MotionDetector
from tracker import CentroidTracker


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame: np.ndarray
    annotated: np.ndarray
    face_detections: list[Detection]
    motion_detections: list[Detection]
    tracked_objects: int
    motion_level: float
    fps: float
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "face_count": len(self.face_detections),
            "motion_count": len(self.motion_detections),
            "tracked_objects": self.tracked_objects,
            "motion_level": round(self.motion_level, 4),
            "fps": round(self.fps, 1),
            "timestamp": self.timestamp,
        }


@dataclass
class PipelineStats:
    """Aggregated statistics from the pipeline."""
    total_frames: int = 0
    total_faces: int = 0
    total_motions: int = 0
    avg_fps: float = 0.0
    avg_motion_level: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=300))


class VideoPipeline:
    """Real-time video processing pipeline.

    Supports webcam, video file, and RTSP stream input.
    """

    def __init__(
        self,
        source: int | str = 0,
        enable_face_detection: bool = True,
        enable_motion_detection: bool = True,
        enable_tracking: bool = False,
        face_method: str = "haar",
        motion_threshold: int = 25,
        motion_min_area: int = 500,
        resize_width: int = 640,
        record_output: bool = False,
        output_path: str = "output.mp4",
    ):
        self.source = source
        self.resize_width = resize_width
        self.record_output = record_output
        self.output_path = output_path

        self.face_detector = FaceDetector(method=face_method) \
            if enable_face_detection else None
        self.motion_detector = MotionDetector(
            threshold=motion_threshold,
            min_area=motion_min_area,
        ) if enable_motion_detection else None
        self.tracker = CentroidTracker() if enable_tracking else None

        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self._running = False
        self.stats = PipelineStats()

        # Heatmap accumulation
        self.heatmap: np.ndarray | None = None

    def open(self) -> None:
        """Open the video source."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize heatmap
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        # Rewind if file
        if isinstance(self.source, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        if self.motion_detector:
            self.motion_detector.reset()

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None
                      ) -> FrameResult:
        """Process a single frame through the detection pipeline."""
        if timestamp is None:
            timestamp = time.time()

        # Resize for consistent processing
        if self.resize_width:
            h, w = frame.shape[:2]
            scale = self.resize_width / w
            frame = cv2.resize(frame, (self.resize_width, int(h * scale)))

        annotated = frame.copy()

        # Face detection
        face_detections = []
        if self.face_detector:
            face_detections = self.face_detector.detect(frame)
            for det in face_detections:
                x, y, w_, h_ = det.bbox
                cv2.rectangle(annotated, (x, y), (x + w_, y + h_),
                              (56, 189, 248), 2)
                cv2.putText(annotated, f"face {det.confidence:.2f}",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (56, 189, 248), 1)

        # Motion detection
        motion_detections = []
        if self.motion_detector:
            motion_detections, mask = self.motion_detector.detect(frame)
            for det in motion_detections:
                x, y, w_, h_ = det.bbox
                cv2.rectangle(annotated, (x, y), (x + w_, y + h_),
                              (74, 222, 128), 2)

            # Update heatmap
            if mask is not None and self.heatmap is not None:
                self.heatmap = cv2.addWeighted(
                    self.heatmap, 0.95, mask.astype(np.float32) / 255.0,
                    0.05, 0
                )

        # Tracking
        tracked = 0
        if self.tracker and face_detections:
            face_rects = [d.bbox for d in face_detections]
            tracks = self.tracker.update(face_rects)
            tracked = len(tracks)
            for tid, bbox in tracks.items():
                x, y, w_, h_ = bbox
                cv2.putText(annotated, f"ID:{tid}", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)

        # FPS overlay
        fps = self.stats.avg_fps if self.stats.avg_fps > 0 else 30.0
        cv2.putText(annotated, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        motion_level = (self.motion_detector.motion_level
                        if self.motion_detector else 0.0)

        result = FrameResult(
            frame=frame,
            annotated=annotated,
            face_detections=face_detections,
            motion_detections=motion_detections,
            tracked_objects=tracked,
            motion_level=motion_level,
            fps=fps,
            timestamp=timestamp,
        )

        # Update stats
        self.stats.total_frames += 1
        self.stats.total_faces += len(face_detections)
        self.stats.total_motions += len(motion_detections)
        self.stats.avg_motion_level = (
            0.9 * self.stats.avg_motion_level + 0.1 * motion_level
        )
        self.stats.history.append(result.to_dict())

        return result

    def stream(self) -> Generator[FrameResult, None, None]:
        """Generator yielding processed frames from the video source."""
        self.open()
        self._running = True

        frame_times = deque(maxlen=30)

        while self._running:
            start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            # Write output
            if self.writer is None and self.record_output:
                h, w = result.annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(
                    self.output_path, fourcc, 30.0, (w, h)
                )
            if self.writer:
                self.writer.write(result.annotated)

            # Track FPS
            elapsed = time.time() - start
            frame_times.append(elapsed)
            self.stats.avg_fps = 1.0 / (sum(frame_times) / len(frame_times))

            yield result

        self._running = False
        self.close()

    def stop(self) -> None:
        """Signal the pipeline to stop."""
        self._running = False

    def get_heatmap_image(self) -> np.ndarray | None:
        """Return a colorized version of the motion heatmap."""
        if self.heatmap is None:
            return None
        heatmap_norm = cv2.normalize(
            self.heatmap, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        return cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
