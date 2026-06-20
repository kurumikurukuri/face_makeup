"""
Eye shadow makeup pass — polygon mask + HSV-aware blend.
"""
from __future__ import annotations

import cv2
import numpy as np

from face_makeup.constants import (
    EYELID_EDGE_LEFT_INDICES,
    EYELID_EDGE_RIGHT_INDICES,
    EYELID_LEFT_INDICES,
    EYELID_RIGHT_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.passes.blend_utils import standard_blend
from face_makeup.core.types import EyeShadowParams


class EyeShadowMaskStrategy(MaskStrategy):
    """Polygon mask from eyelid landmark indices."""

    def __init__(self, indices: list[int]):
        self._indices = indices

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

        points = coords[self._indices]
        if len(points) < 3:
            return mask

        shift = 4
        scale = 1 << shift
        points_shifted = (points * scale).astype(np.int32)
        cv2.fillPoly(mask, [points_shifted], 255, shift=shift)

        return mask


class EyeShadowBlendStrategy(BlendStrategy):
    """Standard HSV-aware blend for eye shadow."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: EyeShadowParams,
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


_INDICES_MAP = {
    "eyelid_left": EYELID_LEFT_INDICES,
    "eyelid_right": EYELID_RIGHT_INDICES,
    "eyelid_edge_left": EYELID_EDGE_LEFT_INDICES,
    "eyelid_edge_right": EYELID_EDGE_RIGHT_INDICES,
}


class EyeShadowPass(MakeupPass):
    """Eye shadow makeup pass with polygon-based mask."""

    def __init__(self, params: EyeShadowParams, part_name: str):
        indices = _INDICES_MAP[part_name]
        super().__init__(
            pass_type=part_name,
            priority=25,
            params=params,
            landmark_indices=list(indices),
            mask_strategy=EyeShadowMaskStrategy(list(indices)),
            blend_strategy=EyeShadowBlendStrategy(),
        )
