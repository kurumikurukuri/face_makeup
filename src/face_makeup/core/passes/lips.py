"""
Lip makeup pass — polygon mask + HSV-aware blend.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from face_makeup.constants import LIPS_INDICES
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import standard_blend
from face_makeup.core.types import LipsParams


class LipMaskStrategy(MaskStrategy):
    """Polygon masks for upper and lower lip, combined via bitwise OR."""

    def generate(
        self,
        coords: np.ndarray,
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
        landmarks: object | None = None,
        segmenter: object | None = None,
    ) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        shift = 4
        scale = 1 << shift

        for indices in LIPS_INDICES.values():
            points = coords[indices]
            if len(points) < 3:
                continue
            points_shifted = (points * scale).astype(np.int32)
            part_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(part_mask, [points_shifted], 255, shift=shift)
            mask = cv2.bitwise_or(mask, part_mask)

        return mask


class LipBlendStrategy(BlendStrategy):
    """Standard HSV-aware blend for lips."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: LipsParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: object | None = None,
    ) -> np.ndarray:
        return standard_blend(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
            params.blur_strength,
        )


class LipsPass(MakeupPass):
    """Lip makeup pass with polygon-based mask."""

    def __init__(self, params: LipsParams):
        super().__init__(
            pass_type="lips",
            priority=20,
            params=params,
            landmark_indices=list(LIPS_INDICES["upper"]) + list(LIPS_INDICES["lower"]),
            mask_strategy=LipMaskStrategy(),
            blend_strategy=LipBlendStrategy(),
        )
