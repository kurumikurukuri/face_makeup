"""
顔部位セグメンテーション
ランドマークからマスク生成
"""
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2

from .types import FaceLandmarks
from src.face_makeup.constants import PART_LANDMARK_MAP


class FaceSegmenter:
    """ランドマークから顔部位マスクを生成"""
    def __init__(self):
        pass


    def _expand_indices_with_midpoints(self,
                                       indices: List,
                                       coords: np.ndarray) -> np.ndarray:
        """
        インデックスリストから座標を取得し、中点を展開する

        Args:
            indices: ランドマークインデックスのリスト
                     通常のint、または ('mid', idx1, idx2, ...) タプル
            coords: 全ランドマーク座標 (N, 2)

        Returns:
            np.ndarray: 展開された座標 (M, 2)

        例:
            indices = [33, ('mid', 33, 133), 159]
            → [coord[33], (coord[33]+coord[133])/2, coord[159]]
        """
        expanded_coords = []

        for idx in indices:
            if isinstance(idx, tuple) and idx[0] == 'mid':
                # 中点の計算
                midpoint_indices = idx[1:]
                if len(midpoint_indices) < 2:
                    raise ValueError(f"Midpoint requires at least 2 indices, got {len(midpoint_indices)}")

                # 指定されたインデックスの座標を取得
                points = coords[list(midpoint_indices)]
                # 平均を計算（重心）
                midpoint = np.mean(points, axis=0)
                expanded_coords.append(midpoint)
            elif isinstance(idx, int):
                # 通常のランドマーク
                expanded_coords.append(coords[idx])
            else:
                raise ValueError(f"Invalid index type: {type(idx)}. Expected int or ('mid', ...) tuple.")

        return np.array(expanded_coords, dtype=coords.dtype)

    def segment(self,
                image_shape: Tuple[int, int],
                landmarks: FaceLandmarks,
                parts: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        顔部位のセグメンテーション

        Args:
            image_shape: (H, W)
            landmarks: FaceLandmarks
            parts: 生成する部位リスト（Noneなら全部位）

        Returns:
            Dict[str, np.ndarray]: 部位名 -> ハードマスク (H, W) uint8 0/255
        """
        if parts is None:
            parts = list(PART_LANDMARK_MAP.keys())

        h, w = image_shape
        coords = landmarks.to_image_coords(w, h)
        masks = {}

        for part in parts:
            if part not in PART_LANDMARK_MAP:
                continue

            part_config = PART_LANDMARK_MAP[part]

            # インデックスとタイプを取得
            if isinstance(part_config, tuple) and len(part_config) == 2:
                indices, seg_type = part_config
            else:
                indices = part_config
                seg_type = 'convex'  # デフォルト

            # 唇・アイライナーは特別処理（上下を別々に処理して結合）
            if isinstance(indices, dict):
                mask = self._create_combined_mask(coords, indices, image_shape, seg_type)
            else:
                # 中点を展開して座標を取得
                part_points = self._expand_indices_with_midpoints(indices, coords)
                mask = self._create_mask_from_points(part_points, image_shape, seg_type)

            masks[part] = mask

        return masks

    def _create_combined_mask(self,
                              coords: np.ndarray,
                              indices_dict: Dict[str, List],
                              image_shape: Tuple[int, int],
                              seg_type: str = 'polygon') -> np.ndarray:
        """
        複数パーツを結合したマスクを生成（唇やアイライナー用）

        Args:
            coords: 全ランドマーク座標
            indices_dict: {'upper': [...], 'lower': [...]} 等のインデックス辞書
            image_shape: (H, W)
            seg_type: セグメンテーションタイプ ('polygon', 'curve' 等)

        Returns:
            mask: (H, W) uint8 0/255
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for key, indices in indices_dict.items():
            part_points = self._expand_indices_with_midpoints(indices, coords)
            part_mask = self._create_mask_from_points(part_points, image_shape, seg_type)
            mask = cv2.bitwise_or(mask, part_mask)

        return mask

    def _create_mask_from_points(self,
                                  points: np.ndarray,
                                  image_shape: Tuple[int, int],
                                  seg_type: str = 'polygon') -> np.ndarray:
        """
        点群からマスクを生成

        Args:
            points: (N, 2) のランドマーク点
            image_shape: (H, W)
            seg_type: セグメンテーションタイプ
                      'polygon' - 多角形
                      'convex' - 凸包
                      'curve' - 曲線

        Returns:
            mask: (H, W) uint8 0/255
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(points) < 2:
            # 点が2つ未満の場合は何もしない
            return mask
        
        elif seg_type == 'polygon':
            # 多角形（Polygon）として塗りつぶし
            # 点を順番通りに結んだ多角形（凹形状も可能）
            if len(points) < 3:
                return mask
            points_int = points.astype(np.int32)
            cv2.fillPoly(mask, [points_int], 255)

        elif seg_type == 'convex':
            # 凸包（Convex Hull）を計算して塗りつぶし
            if len(points) < 3:
                return mask
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

        elif seg_type == 'curve':
            # 曲線（curve）として描画
            # アイラインなど細い線用
            if len(points) < 2:
                return mask
            points_int = points.astype(np.int32)
            # 線の太さを画像サイズに応じて調整
            thickness = max(2, min(h, w) // 500)
            cv2.polylines(mask, [points_int], isClosed=False, color=255, thickness=thickness)

        else:
            raise ValueError(f"Unknown segmentation type: {seg_type}")

        return mask