"""
Pre-compute custom (virtual) landmarks and merge into extended coordinates array.

After MediaPipe extraction, extend_landmarks() computes all custom landmarks
and writes them at their logical-ID positions in the array, so downstream
code can reference them via plain integer indices.
"""
from __future__ import annotations

import numpy as np

from face_makeup.core.types import FaceLandmarks
from face_makeup.custom_landmark import (
    CUSTOM_LANDMARK_DEFS,
    MAX_LANDMARK_ID,
    NUM_MP_LANDMARKS,
)


def extend_landmarks(landmarks: FaceLandmarks) -> FaceLandmarks:
    """Compute custom landmarks and return extended coordinates.

    Writes each custom landmark at ``extended[def.logical_id]`` so the
    logical ID doubles as the array index.

    Args:
        landmarks: FaceLandmarks with ``points`` shaped ``(478, 2)``.

    Returns:
        FaceLandmarks with ``points`` shaped ``(N, 2)`` where
        ``N = max(MAX_LANDMARK_ID, 477) + 1``.
        Original 478 points are preserved at indices 0..477.
        Unused slots are zero-filled.
    """
    total = max(MAX_LANDMARK_ID, NUM_MP_LANDMARKS - 1) + 1
    extended = np.zeros((total, 2), dtype=np.float32)
    extended[:NUM_MP_LANDMARKS] = landmarks.points

    for d in CUSTOM_LANDMARK_DEFS:
        if d.kind == "mid":
            extended[d.logical_id] = np.mean(
                landmarks.points[list(d.base_indices)], axis=0
            )
        else:  # edge
            extended[d.logical_id] = landmarks.points[d.base_indices[0]]

    return FaceLandmarks(points=extended, points_3d=landmarks.points_3d)
