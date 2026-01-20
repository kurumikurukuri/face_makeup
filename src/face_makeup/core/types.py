"""
データ構造定義
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np


@dataclass
class FaceBBox:
    """顔検出結果"""
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass
class FaceLandmarks:
    """ランドマーク（正規化座標）"""
    points: np.ndarray  # shape: (478, 2), dtype: float32, range: 0..1
    points_3d: Optional[np.ndarray] = None  # shape: (478, 3), dtype: float32, 正規化3D座標

    def to_image_coords(self, img_width: int, img_height: int) -> np.ndarray:
        """画像座標系に変換"""
        coords = self.points.copy()
        coords[:, 0] *= img_width
        coords[:, 1] *= img_height
        return coords.astype(np.int32)


@dataclass
class ColorParams:
    """色補正パラメータ"""
    target_rgb: Optional[Tuple[int, int, int]] = None  # RGBで直接指定


# 部位ごとのパラメータ
@dataclass
class PartParams:
    """部位共通の基底クラス"""
    color_params: ColorParams
    opacity: float  # 0..1


@dataclass
class LipsParams(PartParams):
    """リップ用パラメータ

    将来的にグロス効果、輪郭強調などの追加パラメータを定義可能
    現状は基底クラスのパラメータのみ使用
    """
    pass


@dataclass
class CheekParams(PartParams):
    """チーク用パラメータ

    楕円形のグラデーションマスクを生成するためのパラメータ
    """
    center: Tuple[float, float]  # 正規化座標 0..1 (マスク領域内での相対位置)
    radius: Tuple[float, float]  # (radius_x, radius_y) 正規化サイズ 0..1
    falloff: float = 0.5  # 減衰の強さ (0..1, 小さいほど中心が濃く、大きいほど広く薄く)


@dataclass
class EyeShadowParams(PartParams):
    """アイシャドウ用パラメータ

    将来的にグラデーション機能を追加予定
    現状は基底クラスのパラメータのみ使用
    """
    pass

@dataclass
class EyeLinerParams(PartParams):
    """アイライン用パラメータ
    太さを指定する。
    """
    thickness: int = 2