"""
Face detection using MediaPipe Face Detection.
"""
from __future__ import annotations

import numpy as np
import mediapipe as mp

from .types import FaceBBox


class FaceDetector:
    """Face detection using MediaPipe Face Detection."""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Args:
            min_detection_confidence: Detection confidence threshold (0..1).
        """
        self.min_detection_confidence = min_detection_confidence
        self.detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        self._closed = False

    def detect(self, image: np.ndarray) -> list[FaceBBox]:
        """
        Detect faces in an image.

        Args:
            image: RGB uint8 image (H, W, 3).

        Returns:
            list[FaceBBox]: List of detected face bounding boxes.

        Raises:
            ValueError: If image is invalid or no faces detected.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        results = self.detector.process(image)

        if not results.detections:
            raise ValueError("No face detected")

        h, w = image.shape[:2]
        bboxes = []

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coordinates to absolute and clamp to image bounds.
            # MediaPipe relative_bounding_box may exceed [0,1] on partial faces.
            # Clamp both edges first, then derive width/height from the clamped
            # edges — deriving width from the raw (unclamped) size would overstate
            # it whenever xmin/ymin was negative and got clamped up to 0.
            x0 = max(0, int(bbox.xmin * w))
            y0 = max(0, int(bbox.ymin * h))
            x1 = min(w, int((bbox.xmin + bbox.width) * w))
            y1 = min(h, int((bbox.ymin + bbox.height) * h))

            x, y = x0, y0
            width, height = x1 - x0, y1 - y0

            bboxes.append(FaceBBox(
                x=x,
                y=y,
                w=width,
                h=height,
                score=detection.score[0]
            ))

        return bboxes

    def close(self) -> None:
        """Release resources."""
        if getattr(self, '_closed', False):
            return
        if hasattr(self, 'detector'):
            self.detector.close()
        self._closed = True

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self):
        self.close()
