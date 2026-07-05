"""
Eye shadow makeup pass — polygon mask + HSV-aware blend.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from face_makeup.constants import (
    EYELID_CENTER_LEFT_INDICES,
    EYELID_CENTER_RIGHT_INDICES,
    EYELID_EDGE_CENTER_LEFT_INDICES,
    EYELID_EDGE_CENTER_RIGHT_INDICES,
    EYELID_EDGE_INNER_LEFT_INDICES,
    EYELID_EDGE_INNER_RIGHT_INDICES,
    EYELID_EDGE_LEFT_INDICES,
    EYELID_EDGE_OUTER_LEFT_INDICES,
    EYELID_EDGE_OUTER_RIGHT_INDICES,
    EYELID_EDGE_RIGHT_INDICES,
    EYELID_INNER_LEFT_INDICES,
    EYELID_INNER_RIGHT_INDICES,
    EYELID_LEFT_INDICES,
    EYELID_OUTER_LEFT_INDICES,
    EYELID_OUTER_RIGHT_INDICES,
    EYELID_RIGHT_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import apply_glitter, standard_blend
from face_makeup.core.types import EyeShadowParams

if TYPE_CHECKING:
    from face_makeup.core.types import FaceLandmarks

# OpenCV sub-pixel precision bits for anti-aliased fillPoly.
_SUBPIXEL_BITS = 4


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

        shift = _SUBPIXEL_BITS
        scale = 1 << shift
        points_shifted = (points * scale).astype(np.int32)
        cv2.fillPoly(mask, [points_shifted], 255, shift=shift)

        return mask


class EyeShadowBlendStrategy(BlendStrategy):
    """Standard HSV-aware blend with optional glitter (lamé) overlay."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: EyeShadowParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> np.ndarray:
        result = standard_blend(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
            params.blur_strength,
        )
        if params.glitter > 0:
            rng = np.random.default_rng(params.seed) if params.seed is not None else None
            result = apply_glitter(result, mask, params.glitter, rng=rng)
        return result


_INDICES_MAP = {
    "eyelid_left": EYELID_LEFT_INDICES,
    "eyelid_right": EYELID_RIGHT_INDICES,
    "eyelid_edge_left": EYELID_EDGE_LEFT_INDICES,
    "eyelid_edge_right": EYELID_EDGE_RIGHT_INDICES,
    # Overall band zones (outer/center/inner).
    "eyelid_outer_left": EYELID_OUTER_LEFT_INDICES,
    "eyelid_center_left": EYELID_CENTER_LEFT_INDICES,
    "eyelid_inner_left": EYELID_INNER_LEFT_INDICES,
    "eyelid_outer_right": EYELID_OUTER_RIGHT_INDICES,
    "eyelid_center_right": EYELID_CENTER_RIGHT_INDICES,
    "eyelid_inner_right": EYELID_INNER_RIGHT_INDICES,
    # Edge band zones (outer/center/inner).
    "eyelid_edge_outer_left": EYELID_EDGE_OUTER_LEFT_INDICES,
    "eyelid_edge_center_left": EYELID_EDGE_CENTER_LEFT_INDICES,
    "eyelid_edge_inner_left": EYELID_EDGE_INNER_LEFT_INDICES,
    "eyelid_edge_outer_right": EYELID_EDGE_OUTER_RIGHT_INDICES,
    "eyelid_edge_center_right": EYELID_EDGE_CENTER_RIGHT_INDICES,
    "eyelid_edge_inner_right": EYELID_EDGE_INNER_RIGHT_INDICES,
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
