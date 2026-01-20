"""
Facial landmark extraction using MediaPipe Face Mesh.

Extracts 478 landmarks when refine_landmarks=True.
"""
from typing import List

import numpy as np
import mediapipe as mp

from .types import FaceLandmarks


class LandmarkExtractor:
    """Facial landmark extraction using MediaPipe Face Mesh."""

    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            static_image_mode: Static image mode.
            max_num_faces: Maximum number of faces to detect.
            refine_landmarks: Enable landmark refinement (478 points).
            min_detection_confidence: Detection confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(self, image: np.ndarray) -> FaceLandmarks:
        """
        Extract facial landmarks.

        Args:
            image: RGB uint8 (H, W, 3).

        Returns:
            FaceLandmarks: Normalized coordinates (478, 2) and 3D coordinates (478, 3).

        Raises:
            ValueError: If no landmarks are detected.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            raise ValueError("No landmarks detected")

        # Get landmarks for the first face
        landmarks = results.multi_face_landmarks[0]

        # Extract normalized coordinates (x, y)
        points = np.array([[lm.x, lm.y] for lm in landmarks.landmark],
                          dtype=np.float32)

        # Extract normalized 3D coordinates (x, y, z)
        points_3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                             dtype=np.float32)

        return FaceLandmarks(points=points, points_3d=points_3d)

    def extract_all(self, image: np.ndarray) -> List[FaceLandmarks]:
        """
        Extract landmarks for all detected faces.

        Args:
            image: RGB uint8 image (H, W, 3).

        Returns:
            List[FaceLandmarks]: List of landmarks for each detected face.

        Raises:
            ValueError: If no landmarks detected.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            raise ValueError("No landmarks detected")

        landmarks_list = []

        for face_landmarks in results.multi_face_landmarks:
            # Extract normalized 2D coordinates (x, y)
            points = np.array(
                [[lm.x, lm.y] for lm in face_landmarks.landmark],
                dtype=np.float32
            )

            # Extract normalized 3D coordinates (x, y, z)
            points_3d = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                dtype=np.float32
            )

            landmarks_list.append(FaceLandmarks(points=points, points_3d=points_3d))

        return landmarks_list

    def __del__(self):
        """Release resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
