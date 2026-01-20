"""
ステートレスなメイクアップ処理クラス。

モデルは一度だけロードされ、複数画像を処理可能。
スレッドセーフな設計で、複数スレッドから再利用できる。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path
import numpy as np
from PIL import Image

from src.face_makeup.core.face_detector import FaceDetector
from src.face_makeup.core.landmark_extractor import LandmarkExtractor
from src.face_makeup.core.segmenter import FaceSegmenter
from src.face_makeup.core.makeup_blender import MakeupBlender
from src.face_makeup.context import ImageContext
from src.face_makeup.face import Face

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """MakeupProcessor 用の設定クラス"""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True
    max_num_faces: int = 10


class MakeupProcessor:
    """
    ステートレスなメイクアッププロセッサ。
    モデルは一度だけ初期化され、複数画像を処理できる。

    使用例:
        >>> processor = MakeupProcessor()
        >>> ctx = processor.open_path("sample.jpg")
        >>> ctx.faces[0].lip(lip_params).cheek(cheek_params)
        >>> result = ctx.render()
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        プロセッサを初期化し、各種モデルをロードする。

        Args:
            config: プロセッサ設定。None の場合はデフォルト設定を使用。
        """
        self._config = config or ProcessorConfig()

        # 顔検出モデルの初期化
        self._detector = FaceDetector(
            min_detection_confidence=self._config.min_detection_confidence
        )

        # ランドマーク抽出モデルの初期化
        self._landmark_extractor = LandmarkExtractor(
            max_num_faces=self._config.max_num_faces,
            refine_landmarks=self._config.refine_landmarks,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence
        )

        # 描画処理で使用される各種コンポーネント
        self._segmenter = FaceSegmenter()
        self._blender = MakeupBlender()

        logger.info("MakeupProcessor initialized")

    @property
    def config(self) -> ProcessorConfig:
        """現在のプロセッサ設定を取得"""
        return self._config


    def open(self,image: np.ndarray) -> ImageContext:
        """
        ndarray 形式の画像を開き、メイクアップ処理用の ImageContext を生成する。

        顔検出およびランドマーク抽出を行う。

        Args:
            image: RGB形式の uint8 画像 (H, W, 3)

        Returns:
            ImageContext: 後続処理用のコンテキスト
        """
        self._validate_image(image)

        logger.debug(f"画像をオープン: shape={image.shape}")

        # 顔検出を実行
        bboxes = self._detector.detect(image)
        logger.debug(f"検出された顔の数: {len(bboxes)}")

        # 全顔分のランドマークを抽出
        landmarks_list = self._landmark_extractor.extract_all(image)
        logger.debug(f"ランドマーク抽出数: {len(landmarks_list)}")

        # bbox と landmarks の数が異なる場合は最小数を使用
        num_faces = min(len(bboxes), len(landmarks_list))

        if len(bboxes) != len(landmarks_list):
            logger.warning(
                f"顔数の不一致: bbox={len(bboxes)}, "
                f"landmarks={len(landmarks_list)}。"
                f"{num_faces} 件のみ使用します。"
            )

        # Face オブジェクトを生成
        faces = []
        for i in range(num_faces):
            face = Face(
                face_id=i,
                bbox=bboxes[i],
                landmarks=landmarks_list[i],
                context=None  # ImageContext 側で設定される
            )
            faces.append(face)

        # ImageContext を生成
        ctx = ImageContext(image, faces, self)

        logger.info(f"{len(faces)} 人の顔を含む ImageContext を生成しました")
        return ctx

    def open_path(self,image_source: str) -> ImageContext:
        """
        画像ファイルのパスから画像を読み込み、ImageContext を生成する。

        Args:
            image_source: 画像ファイルのパス

        Returns:
            ImageContext
        """
        
        path = Path(image_source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_source}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_source}")

        # PIL を使用して画像を RGB 形式で読み込む
        pil_image = Image.open(path).convert("RGB")
        image_np = np.array(pil_image)
        return self.open(image_np)
    
    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """入力画像の妥当性を検証する"""
        if image is None or image.size == 0:
            raise ValueError("無効な画像です")

        if image.dtype != np.uint8:
            raise ValueError(f"画像の dtype は uint8 である必要があります: {image.dtype}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"画像の形状は (H, W, 3) である必要があります: {image.shape}")