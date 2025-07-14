"""
Centroid-based object tracker.

Assigns unique IDs to detected objects and maintains tracks across frames
using Euclidean distance matching.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """A tracked object with a unique ID."""
    id: int
    centroids: list[tuple[float, float]] = field(default_factory=list)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    age: int = 0
    missed_frames: int = 0
    active: bool = True

    @property
    def centroid(self) -> tuple[float, float] | None:
        return self.centroids[-1] if self.centroids else None

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self.bboxes[-1] if self.bboxes else None

    def update(self, centroid: tuple[float, float],
               bbox: tuple[int, int, int, int]) -> None:
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.age += 1
        self.missed_frames = 0
        self.active = True

    def mark_missed(self) -> None:
        self.missed_frames += 1
        if self.missed_frames > 30:
            self.active = False


class CentroidTracker:
    """Tracks objects across frames using nearest-neighbor centroid matching."""

    def __init__(self, max_distance: float = 50.0,
                 max_missed_frames: int = 30):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 0
        self.tracks: dict[int, Track] = {}

    def update(self, rects: list[tuple[int, int, int, int]]
               ) -> dict[int, tuple[int, int, int, int]]:
        """Update tracks with new detections.

        Args:
            rects: List of bounding boxes as (x, y, w, h).

        Returns:
            Dict mapping track ID → bounding box for active tracks.
        """
        # Mark all existing tracks as missed
        for track in self.tracks.values():
            track.mark_missed()

        if not rects:
            # Clean up inactive tracks
            self.tracks = {
                tid: t for tid, t in self.tracks.items() if t.active
            }
            return {}

        # Compute centroids for new detections
        input_centroids = [
            (x + w / 2, y + h / 2) for (x, y, w, h) in rects
        ]

        # If no existing tracks, create new tracks for all
        if not self.tracks:
            for centroid, bbox in zip(input_centroids, rects):
                self._create_track(centroid, bbox)
            return self._active_track_bboxes()

        # Match: find nearest active track centroid to each input centroid
        active_ids = [tid for tid, t in self.tracks.items() if t.active]
        active_centroids = [self.tracks[tid].centroid for tid in active_ids]

        # Distance matrix
        distances = np.zeros((len(input_centroids), len(active_centroids)))
        for i, ic in enumerate(input_centroids):
            for j, ac in enumerate(active_centroids):
                if ac is not None:
                    distances[i, j] = np.sqrt(
                        (ic[0] - ac[0]) ** 2 + (ic[1] - ac[1]) ** 2
                    )
                else:
                    distances[i, j] = float("inf")

        # Greedy matching
        used_inputs = set()
        used_tracks = set()

        for i in range(len(input_centroids)):
            if not len(active_centroids):
                break
            j = np.argmin(distances[i])
            if distances[i, j] < self.max_distance:
                track_id = active_ids[j]
                self.tracks[track_id].update(input_centroids[i], rects[i])
                used_inputs.add(i)
                used_tracks.add(track_id)

        # Create new tracks for unmatched inputs
        for i in range(len(input_centroids)):
            if i not in used_inputs:
                self._create_track(input_centroids[i], rects[i])

        # Remove long-inactive tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t.active
        }

        return self._active_track_bboxes()

    def _create_track(self, centroid: tuple[float, float],
                      bbox: tuple[int, int, int, int]) -> int:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = Track(id=track_id)
        self.tracks[track_id].update(centroid, bbox)
        return track_id

    def _active_track_bboxes(self) -> dict[int, tuple[int, int, int, int]]:
        return {
            tid: t.bbox for tid, t in self.tracks.items()
            if t.active and t.bbox is not None
        }

    @property
    def active_count(self) -> int:
        return sum(1 for t in self.tracks.values() if t.active)
