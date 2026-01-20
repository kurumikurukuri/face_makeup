"""
Facial landmark extraction using MediaPipe Face Mesh.

Extracts 478 landmarks when refine_landmarks=True.
"""
from typing import List

import numpy as np
import mediapipe as mp

from .types import FaceLandmarks


class LandmarkExtractor:
    """MediaPipe Face Mesh を使用した顔ランドマーク抽出"""

    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            static_image_mode: 静止画モード
            max_num_faces: 最大検出顔数
            refine_landmarks: ランドマークの精緻化（478点）
            min_detection_confidence: 検出の信頼度閾値
            min_tracking_confidence: トラッキングの信頼度閾値
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
        顔ランドマークを抽出

        Args:
            image: RGB uint8 (H, W, 3)

        Returns:
            FaceLandmarks: 正規化座標 (468, 2) と 3D座標 (468, 3)

        Raises:
            ValueError: ランドマークが検出されない場合
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            raise ValueError("No landmarks detected")

        # 最初の顔のランドマークを取得
        landmarks = results.multi_face_landmarks[0]

        # 正規化座標 (x, y) を抽出
        points = np.array([[lm.x, lm.y] for lm in landmarks.landmark],
                          dtype=np.float32)

        # 正規化3D座標 (x, y, z) を抽出
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
        """リソース解放"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
