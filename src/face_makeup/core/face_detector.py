"""
顔検出
MediaPipe Face Detection を使用
"""
from typing import List
import numpy as np
import mediapipe as mp

from .types import FaceBBox


class FaceDetector:
    """MediaPipe Face Detection を使用した顔検出"""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Args:
            min_detection_confidence: 検出の信頼度閾値 (0..1)
        """
        self.min_detection_confidence = min_detection_confidence
        self.detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, image: np.ndarray) -> List[FaceBBox]:
        """
        Detect faces in an image.

        Args:
            image: RGB uint8 image (H, W, 3).

        Returns:
            List[FaceBBox]: List of detected face bounding boxes.

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

            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            bboxes.append(FaceBBox(
                x=x,
                y=y,
                w=width,
                h=height,
                score=detection.score[0]
            ))

        return bboxes

    def __del__(self):
        """リソース解放"""
        if hasattr(self, 'detector'):
            self.detector.close()
