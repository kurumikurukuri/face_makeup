"""
Lip makeup pass — polygon mask + HSV-aware blend.
"""
from __future__ import annotations

import cv2
import numpy as np

from face_makeup.constants import LIPS_INDICES
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import (
    blend_matte,
    feather_px_from_softness,
    soften_mask,
)
from face_makeup.core.types import LipsParams

# OpenCV sub-pixel precision bits for anti-aliased fillPoly.
_SUBPIXEL_BITS = 4


class LipMaskStrategy(MaskStrategy):
    """Soft lip mask: upper+lower polygons OR'd together, then edge-softened.

    Returns the final soft mask (float32 0-1) with the fade baked in, so the
    pipeline can apply exclusions to it before blending.
    """

    def __init__(self, softness: float):
        self._softness = softness

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

        feather_px = feather_px_from_softness(self._softness, coords, image_shape)

        shift = _SUBPIXEL_BITS
        scale = 1 << shift

        for indices in LIPS_INDICES.values():
            points = coords[indices]
            if len(points) < 3:
                continue
            points_shifted = (points * scale).astype(np.int32)
            part_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(part_mask, [points_shifted], 255, shift=shift)
            mask = cv2.bitwise_or(mask, part_mask)

        return soften_mask(mask, feather_px)


class LipBlendStrategy(BlendStrategy):
    """Pure matte colour application over a ready soft lip mask."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: LipsParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: object | None = None,
    ) -> np.ndarray:
        return blend_matte(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
        )


class LipsPass(MakeupPass):
    """Lip makeup pass with polygon-based mask."""

    def __init__(self, params: LipsParams):
        super().__init__(
            pass_type="lips",
            priority=20,
            params=params,
            landmark_indices=list(LIPS_INDICES["upper"]) + list(LIPS_INDICES["lower"]),
            mask_strategy=LipMaskStrategy(params.softness),
            blend_strategy=LipBlendStrategy(),
        )
