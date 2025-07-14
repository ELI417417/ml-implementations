"""
Analytics and statistics for the video pipeline.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class EventLog:
    """A timestamped detection event."""
    timestamp: float
    event_type: str
    details: dict


class AnalyticsCollector:
    """Collects and aggregates real-time analytics from the pipeline."""

    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.face_counts: deque[float] = deque(maxlen=window_size)
        self.motion_levels: deque[float] = deque(maxlen=window_size)
        self.fps_values: deque[float] = deque(maxlen=window_size)
        self.tracked_counts: deque[int] = deque(maxlen=window_size)
        self.events: deque[EventLog] = deque(maxlen=1000)

    def update(self, result_dict: dict) -> None:
        """Update metrics from a FrameResult dict."""
        self.face_counts.append(float(result_dict["face_count"]))
        self.motion_levels.append(float(result_dict["motion_level"]))
        self.fps_values.append(float(result_dict["fps"]))
        self.tracked_counts.append(int(result_dict["tracked_objects"]))

    def log_event(self, event_type: str, details: dict,
                  timestamp: float | None = None) -> None:
        import time
        self.events.append(EventLog(
            timestamp=timestamp or time.time(),
            event_type=event_type,
            details=details,
        ))

    def summary(self) -> dict:
        """Return current analytics summary."""
        return {
            "faces": {
                "current": self.face_counts[-1] if self.face_counts else 0,
                "avg_1min": np.mean(list(self.face_counts)[-60:]) if self.face_counts else 0,
                "max": np.max(list(self.face_counts)) if self.face_counts else 0,
            },
            "motion": {
                "current": self.motion_levels[-1] if self.motion_levels else 0,
                "avg_1min": np.mean(list(self.motion_levels)[-60:]) if self.motion_levels else 0,
                "max": np.max(list(self.motion_levels)) if self.motion_levels else 0,
            },
            "fps": {
                "current": self.fps_values[-1] if self.fps_values else 0,
                "avg": np.mean(list(self.fps_values)) if self.fps_values else 0,
            },
            "tracking": {
                "current": self.tracked_counts[-1] if self.tracked_counts else 0,
            },
            "total_events": len(self.events),
        }
